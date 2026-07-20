import itertools
import math
from typing import List, Tuple

import cupy as cp
import numpy as np

from .encoder import PolarEncoder, PolarCode
from .generator import _bit_reverse, build_generator

def _int_to_bits(value: int, num_bits: int) -> np.ndarray:
    # print(type(value), type(num_bits))
    return np.array([(value >> i) & 1 for i in range(num_bits - 1, -1, -1)], dtype=int)

def _bit_reverse_perm(N: int) -> list[int]:
    """
    Maps between the graph's node order (based on F^n) and the encoder's
    codeword order (based on G = F^n B_n) so LLRs/beliefs on the x-side of the
    graph line up with `encoder.encode()`'s output.
    """
    n_stages: int = int(math.log2(N))
    return [_bit_reverse(i, n_stages) for i in range(N)]

def _build_bit_indices(N: int) -> list[np.ndarray]:
    indices: list[np.ndarray] = []
    n_stages: int = int(math.log2(N))
    for idx in range(N):
        indices.append(_int_to_bits(idx, n_stages))
    return indices

def _boxplus(x: float | cp.ndarray, y: float | cp.ndarray) -> float:
    #*cp.get_array_module dispatches to numpy or cupy depending on what it is
    #*handed, assuming both x and y are of the same type
    xp = cp.get_array_module(x)
    return xp.sign(x) * xp.sign(y) * xp.minimum(xp.abs(x), xp.abs(y))

def _build_graph(N: int, bit_indices: list[np.ndarray] | None = None) -> List[List[Tuple[str, int]]]:
    """
    Builds an N x (log2(N)) matrix whose entries are tuples that follow the
    structure (node type, node partner), where node type is either "+" or "="
    and node partner is the corresponding "=" or "+" node, respectively, which
    forms a butterfly pattern with the current node.

    Parameters
    ----------
    - N: int
        Codeword length (also the number of LLRs).
    - bit_indices: list[np.ndarray], optional
        The list of indices converted to log2(N)-bits. If it is None,
        _build_bit_indices is called.
    """
    if bit_indices is None:
        bit_indices = _build_bit_indices(N)
    n_stages: int = int(math.log2(N))
    #*every (node, stage) entry will have its ("t" or "b", partner index) pair
    graph: List[List[Tuple[str, int]]] = [[] for i in range(N)]
    d_dec: int = 0
    d_bin: np.ndarray = 0
    zero_arr: np.ndarray = np.zeros(n_stages)
    for n_stage in range(n_stages):
        d_dec = 2**(n_stage)
        d_bin = _int_to_bits(d_dec, n_stages) #!Could be optimized, I think
        for node_i in range(N):
            is_plus: bool = ((bit_indices[node_i] & d_bin) == zero_arr).all()
            # import pdb; pdb.set_trace()
            if is_plus: #*it is a + node
                graph[node_i].append(('+', node_i + d_dec))
            else: #*it is an = node
                graph[node_i].append(('=', node_i - d_dec))
    return graph

def _initialize_R(N: int, frozen_bits: frozenset) -> np.ndarray:
    """
    Initialize left-to-right sweep matrix with the left-most column containing
    the estimate of u, of which, initially, we only know the frozen bits'
    values (the remaining entries are set to 0). It is an N x (log(N) + 1)
    matrix.

    Parameters
    ----------
    - N: int
        Codeword length.
    - frozen_bits: np.ndarray
        Indices of the frozen bits.
    """
    R: np.ndarray = np.zeros((N, int(math.log2(N)) + 1))
    for i in range(N):
        if i in frozen_bits:
            R[i][0] = 30
    return R

def _initialize_L(N: int, llr_1N: np.ndarray, perm: list[int]) -> np.ndarray:
    """
    Initialize right-to-left sweep matrix with the right-most column being the
    LLRs returned after transmitting u through the channel (the remaining
    entries are set to 0). It is an N x (log(N) + 1) matrix. It applies the
    bit-reversal permutation from `_bit_reverse_perm` to its last column, which
    is in the codeword side (the one after F^n has been applied and therefore
    B_N needs to be applied next).

    Parameters
    ----------
    - N: int
        Codeword length.
    - llr_1N: np.ndarray
        Indices of the frozen bits.
    - perm: list[int]
        Bit-reversal permutation (see _bit_reverse_perm) mapping codeword
        order to the graph's node order.
    """
    n_cols: int = int(math.log2(N)) + 1
    L: np.ndarray = np.zeros((N, n_cols))
    llr_1N = np.where(llr_1N == np.inf, 30, llr_1N)
    llr_1N = np.where(llr_1N == -np.inf, -30, llr_1N)
    L[:, n_cols - 1] = llr_1N[perm]
    return L

def _build_graph_arrays(graph: List[List[Tuple[str, int]]]) -> Tuple[List[np.ndarray]]:
    """
    Returns two lists:
    - Each element of the first one is an array with the partner of each
    corresponding node (arrs of partners)
    - Each element of the second one is a boolean array that is True where the
    corresponding node is a '+' node and False where it is an '=' one (arrs of
    are_plus).
    """
    graph_partners: List[np.ndarray] = []
    graph_are_p: List[np.ndarray] = []
    for j in range(len(graph[0])):
        graph_partners.append([])
        graph_are_p.append([])
        for bit_pos in graph:
            graph_partners[j].append(bit_pos[j][1])
            graph_are_p[j].append(bit_pos[j][0] == "+")

    for j in range(len(graph_partners)):
        graph_partners[j] = np.asarray(graph_partners[j], dtype=np.int64)
        graph_are_p[j] = np.asarray(graph_are_p[j], dtype=bool)

    return graph_partners, graph_are_p

def _right_sweep(R: np.ndarray, L: np.ndarray, graph: List[List[Tuple[str, int]]]) -> np.ndarray:
    """
    Left-to-right pass of u's estimate through the polarized channel's graph.
    """
    for n_stage in range(1, len(R[0])):
        for node_i in range(len(R)):
            node_j: int = graph[node_i][n_stage-1][1] #*partner of node i
            if graph[node_i][n_stage-1][0] == "+": #*the -1 because it has log2(N) - 1 cols
                R[node_i][n_stage] = _boxplus(R[node_i][n_stage-1],
                                              L[node_j][n_stage] + R[node_j][n_stage-1])
            else:
                R[node_i][n_stage] = _boxplus(R[node_j][n_stage-1], L[node_j][n_stage]) \
                                     + R[node_i][n_stage-1]
    return R

def _right_sweep_parallel(R: np.ndarray, L: np.ndarray, partner_arrs: List[np.ndarray], is_plus_arrs: List[np.ndarray]) -> np.ndarray:
    """
    Left-to-right pass of u's estimate through the polarized channel's graph
    using parallelization to traverse all paths simulteanously.
    """
    xp = cp.get_array_module(R) #!May not be necessary, unless we ask to run it on the GPU
    for n_stage in range(1, len(R[0])):
        nodes_j: np.ndarray = partner_arrs[n_stage-1]
        nodes_j_are_plus: np.ndarray = is_plus_arrs[n_stage-1]
        R[:,n_stage] = xp.where(nodes_j_are_plus,
                                 _boxplus(R[:,n_stage-1], L[nodes_j,n_stage] + R[nodes_j,n_stage-1]), # +
                                 _boxplus(R[nodes_j,n_stage-1], L[nodes_j,n_stage]) + R[:,n_stage-1]) # =
    return R

def _right_sweep_bpl(R: np.ndarray, L: np.ndarray, partner_arrs: List[np.ndarray], is_plus_arrs: List[np.ndarray]) -> np.ndarray:
    """
    Left-to-right pass of u's estimate through the multiple polarized channels'
    graph using parallelization to traverse all paths simulteanously.
    """
    xp = cp.get_array_module(R)
    #*(L, 1) so it broadcasts against nodes_j's (L, N): without it, indexing with a
    #*slice would pair every graph with *every* graph's partners instead of its own

    #*To iterate over nodes using the [ , , ] notation
    graphs_i: np.ndarray = xp.arange(R.shape[0])[:, None]
    for n_stage in range(1, len(R[0][0])):
        nodes_j: np.ndarray = partner_arrs[:, n_stage-1]
        nodes_j_are_plus: np.ndarray = is_plus_arrs[:, n_stage-1]
        #*Across all L graphs and across all their N bits
        R[:,:,n_stage] = xp.where(nodes_j_are_plus,
                                 _boxplus(R[:,:,n_stage-1], L[graphs_i,nodes_j,n_stage] + R[graphs_i,nodes_j,n_stage-1]), # +
                                 _boxplus(R[graphs_i,nodes_j,n_stage-1], L[graphs_i,nodes_j,n_stage]) + R[:,:,n_stage-1]) # =
    return R

def _left_sweep(R: np.ndarray, L: np.ndarray, graph: List[List[Tuple[str, int]]]) -> np.ndarray:
    """
    Right-to-left pass of L's last column (initially the LLRs) through the
    polarized channel's graph.
    """
    for n_stage in range(len(L[0]) - 1, 0, -1):
        for node_i in range(len(L)):
            node_j: int = graph[node_i][n_stage-1][1]
            if graph[node_i][n_stage-1][0] == "+":
                L[node_i][n_stage-1] = _boxplus(L[node_i][n_stage],
                                                L[node_j][n_stage] + R[node_j][n_stage-1])
            else:
                L[node_i][n_stage-1] = _boxplus(R[node_j][n_stage-1] , L[node_j][n_stage]) \
                                     + L[node_i][n_stage]
    return L

def _left_sweep_parallel(R: np.ndarray, L: np.ndarray, partner_arrs: List[np.ndarray], is_plus_arrs: List[np.ndarray]) -> np.ndarray:
    """
    Right-to-left pass of L's last column (initially the LLRs) through the
    polarized channel's graph using parallelization to traverse all paths
    simulteanously.
    """
    xp = cp.get_array_module(L) #!May not be necessary, unless we ask to run it on the GPU, since this will belong to BP only
    for n_stage in range(len(L[0]) - 1, 0, -1):
        nodes_j_are_plus: np.ndarray = is_plus_arrs[n_stage-1]
        nodes_j: np.ndarray = partner_arrs[n_stage-1]
        L[:,n_stage-1] = xp.where(nodes_j_are_plus,
                                _boxplus(L[:,n_stage], L[nodes_j,n_stage] + R[nodes_j,n_stage-1]),
                                _boxplus(R[nodes_j,n_stage-1] , L[nodes_j,n_stage]) + L[:,n_stage])
    return L

def _left_sweep_bpl(R: np.ndarray, L: np.ndarray, partner_arrs: List[np.ndarray], is_plus_arrs: List[np.ndarray]) -> np.ndarray:
    """
    Right-to-left pass of u's estimate through the multiple polarized channels'
    graph using parallelization to traverse all paths simulteanously.
    """
    xp = cp.get_array_module(L)
    #*see the note in _right_sweep_bpl on why the graph index cannot be a slice
    graphs_i: np.ndarray = xp.arange(L.shape[0])[:, None]
    #*L[0][0] = N
    for n_stage in range(len(L[0][0]) - 1, 0, -1):
        nodes_j_are_plus: np.ndarray = is_plus_arrs[:,n_stage-1]
        nodes_j: np.ndarray = partner_arrs[:,n_stage-1]
        #*Across all L graphs and across all their N bits
        L[:,:,n_stage-1] = xp.where(nodes_j_are_plus,
                                _boxplus(L[:,:,n_stage], L[graphs_i,nodes_j,n_stage] + R[graphs_i,nodes_j,n_stage-1]),
                                _boxplus(R[graphs_i,nodes_j,n_stage-1] , L[graphs_i,nodes_j,n_stage]) + L[:,:,n_stage])
    return L

def bp_decode(frozen_bits: frozenset, llr_1N: np.ndarray, graph: List[List[Tuple[str, int]]] = None, iter_cap: int = 1000, parallel: bool = False) -> Tuple[np.ndarray, bool]:
    """
    BP decoding algorithm with optional parallelization.
    
    Returns the proposed decodified word *u'* and whether it was successful on the
    decodification (if *u'*G is equal to the given codeword) or not.
    """
    i: int = 0
    u: list[int] = []
    N: int = llr_1N.size
    K: int = len(frozen_bits)
    n_stages: int = int(math.log2(N))
    bit_indices: list[np.ndarray] = _build_bit_indices(N)
    if graph is None:
        graph: List[List[Tuple[str, int]]] = _build_graph(N, bit_indices)
    perm: list[int] = _bit_reverse_perm(N)
    R: np.ndarray = _initialize_R(N, frozen_bits)
    L: np.ndarray = _initialize_L(N, llr_1N, perm)
    if parallel:
        part_arrs, is_plus_arrs = _build_graph_arrays(graph)

    # print("initial L:")
    # for row in L:
    #     print(row)
    # print("initial R:")
    # for row in R:
    #     print(row)
    # import pdb; pdb.set_trace()

    encoder = PolarEncoder(PolarCode(N=N, K=K, frozen_positions=frozen_bits))
    # print(N, K)
    while i < iter_cap:
        if not parallel:
            R = _right_sweep(R, L, graph)
            L = _left_sweep(R, L, graph)
        else:
            R = _right_sweep_parallel(R, L, part_arrs, is_plus_arrs)
            L = _left_sweep_parallel(R, L, part_arrs, is_plus_arrs)
        u_belief: np.ndarray = R[:,0] + L[:,0] #*taken from the sweeps of *LLRs*
        u_hat_lst: list[int] = []
        for j, u_i in enumerate(u_belief):
            if j in frozen_bits:
                pass
            else:
                if u_i < 0:
                    u_hat_lst.append(1)
                else:
                    u_hat_lst.append(0)
        u_hat: np.ndarray = np.array(u_hat_lst)
        # print(u_hat)
        x_hat: np.ndarray = encoder.encode(u_hat)
        #*R/L's last column is in the graph's node order (see _bit_reverse_perm);
        #*bit-reversal is an involution, so indexing by perm again undoes it
        #*and lines the belief up with x_hat's encoder-order codeword.
        x_belief_graph: np.ndarray = ((R[:,n_stages] + L[:,n_stages]) < 0).astype(int)
        x_belief: np.ndarray = x_belief_graph[perm]
        if (x_hat == x_belief).all():
            return u_hat, True
        i += 1
        # print(i)
        
    # print("Estimate:", u_hat)
    return u_hat, False

def _build_bpl_graphs(n_graphs: int, N: int) -> List[List[List[Tuple[str, int]]]]:
    """
    Builds `n_graphs` or `log2(N)!` (whichever is smaller) distinct permuted
    factor graphs for BPL decoding, using `_build_graph` with its stages
    (columns) reordered by a permutation of range(log2(N)). The identity
    permutation, i.e. the plain BP graph, is always the first one.
    """
    n_stages: int = int(math.log2(N))
    stages_id: tuple[int] = tuple(range(n_stages))
    n_perms: int = math.factorial(n_stages)

    #!Could be optimized: if n_graphs >= n_perms//2, sample the perms we are going to remove instead of the ones we are adding, and viceversa
    if n_graphs >= n_perms:
        stages_perms: List[tuple[int]] = list(itertools.permutations(stages_id))
    else:
        stages_perms: List[tuple[int]] = [stages_id] #*add the identity
        seen: set = {stages_id} #*tuples so lookups are done using a hash table
        while len(stages_perms) < n_graphs:
            candidate: tuple[int] = tuple(np.random.permutation(n_stages).tolist())
            if candidate not in seen:
                seen.add(candidate)
                stages_perms.append(candidate)

    #*Every graph is the base one with its stages reordered
    base_graph: List[List[Tuple[str, int]]] = _build_graph(N)
    graphs: List[List[List[Tuple[str, int]]]] = []
    for perm in stages_perms:
        permuted_graph: List[List[Tuple[str, int]]] = []
        for node in base_graph:
            permuted_graph.append([node[stage] for stage in perm])
        graphs.append(permuted_graph)
    return graphs

def bpl_decode(frozen_bits: frozenset, llr_1N: np.ndarray, n_graphs: int = 4, iter_cap: int = 1000, use_gpu: bool = True) -> Tuple[np.ndarray, bool]:
    """
    BPL algorithm with random graph selection (for now). Every graph is decoded
    at once as an (L, N, log2(N)+1) tensor rather than one after another, and the
    first graph to converge wins.

    Returns the proposed decodified word *u'* and whether it was successful on
    the decodification (if *u'*G is equal to the given codeword) or not.

    Parameters
    ----------
    - frozen_bits: frozenset
        Indices of the frozen bits.
    - llr_1N: np.ndarray
        LLRs returned after transmitting u through the channel.
    - n_graphs: int
        Number of permuted factor graphs to decode with (commonly known as the
        list size L). Clamped to log2(N)!, the number of distinct stage orders.
    - iter_cap: int
        Maximum number of BP iterations.
    - use_gpu: bool
        Whether to run the sweeps on the GPU through CuPy.
    """
    xp = cp if use_gpu else np
    N: int = llr_1N.size
    n_stages: int = int(math.log2(N))
    graphs_permuted: List[List[List[Tuple[str, int]]]] = _build_bpl_graphs(n_graphs, N)
    n_graphs = len(graphs_permuted) #*_build_bpl_graphs clamps, so it may be fewer

    perm: list[int] = _bit_reverse_perm(N)
    graph_arrs: List[Tuple[np.ndarray, np.ndarray]] = [_build_graph_arrays(graph) for graph in graphs_permuted]
    partner_arrs: np.ndarray = xp.asarray([arrs[0] for arrs in graph_arrs]) #*(L, log2(N), N)
    is_plus_arrs: np.ndarray = xp.asarray([arrs[1] for arrs in graph_arrs]) #*(L, log2(N), N)

    #*Every graph starts from the same R and L; [None] adds the graph axis so the
    #*repeat stacks whole matrices instead of interleaving their rows
    R: np.ndarray = xp.repeat(xp.asarray(_initialize_R(N, frozen_bits))[None], n_graphs, axis=0) #*(L, N, log2(N)+1)
    L: np.ndarray = xp.repeat(xp.asarray(_initialize_L(N, llr_1N, perm))[None], n_graphs, axis=0) #*(L, N, log2(N)+1)

    #*Non-frozen positions (where decisions matter)
    info_positions: np.ndarray = xp.asarray([i for i in range(N) if i not in frozen_bits])
    perm_dev: np.ndarray = xp.asarray(perm)
    G: np.ndarray = xp.asarray(build_generator(n_stages), dtype=xp.int32)

    i: int = 0
    u_all: np.ndarray = xp.zeros((n_graphs, N), dtype=xp.int32)
    while i < iter_cap:
        R = _right_sweep_bpl(R, L, partner_arrs, is_plus_arrs)
        L = _left_sweep_bpl(R, L, partner_arrs, is_plus_arrs)

        u_belief: np.ndarray = R[:,:,0] + L[:,:,0] #*taken from the sweeps of *LLRs*, (L, N)
        u_all[:, info_positions] = (u_belief[:, info_positions] < 0) #*boolean hard-rule casted to int
        x_hat_all: np.ndarray = (u_all @ G) % 2 #*one batched encode for all L graphs

        #*R's and L's last columns are in the graph's node order 
        #*(see _bit_reverse_perm); bit-reversal is an involution, so it matches
        #*x_hat's order after applying it again
        x_belief_graph: np.ndarray = ((R[:,:,n_stages] + L[:,:,n_stages]) < 0).astype(xp.int32)
        x_belief_all: np.ndarray = x_belief_graph[:, perm_dev]

        #*Per graph, not across them: any single graph converging is a success.
        converged: np.ndarray = (x_hat_all == x_belief_all).all(axis=1) #*(L,)
        if bool(converged.any()):
            u_hat: np.ndarray = u_all[int(converged.argmax())][info_positions]
            #*GPU to CPU conversion if necessary
            return (cp.asnumpy(u_hat) if use_gpu else u_hat), True
        i += 1

    u_hat: np.ndarray = u_all[0][info_positions]
    return (cp.asnumpy(u_hat) if use_gpu else u_hat), False