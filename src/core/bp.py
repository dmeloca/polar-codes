from typing import List, Tuple

import numpy as np

from .encoder import PolarEncoder, PolarCode
from .generator import _bit_reverse

def _int_to_bits(value: int, num_bits: int) -> np.ndarray:
    # print(type(value), type(num_bits))
    return np.array([(value >> i) & 1 for i in range(num_bits - 1, -1, -1)], dtype=int)

def _bit_reverse_perm(N: int) -> list[int]:
    """
    Maps between the graph's node order (based on F^n) and the encoder's
    codeword order (based on G = F^n B_n) so LLRs/beliefs on the x-side of the
    graph line up with `encoder.encode()`'s output.
    """
    n_stages: int = int(np.log2(N))
    return [_bit_reverse(i, n_stages) for i in range(N)]

def _build_bit_indices(N: int) -> list[np.ndarray]:
    indices: list[np.ndarray] = []
    n_PEs: int = int(np.log2(N))
    for idx in range(N):
        indices.append(_int_to_bits(idx, n_PEs))
    return indices

def _boxplus(x: float, y: float) -> float:
    return np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))

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
    n_stages: int = int(np.log2(N))
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
    R: np.ndarray = np.zeros((N, int(np.log2(N)) + 1))
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
    n_cols: int = int(np.log2(N) + 1)
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
    - Each element of the second one is an array with type of node each
    corresponding node is (arrs of types, i.e., '+' or '=').
    """
    graph_partners: List[np.ndarray] = []
    graph_types: List[np.ndarray] = []
    for j in range(len(graph[0])):
        graph_partners.append([])
        graph_types.append([])
        for bit_pos in graph:
            graph_partners[j].append(bit_pos[j][1])
            graph_types[j].append(bit_pos[j][0])

    for j in range(len(graph_partners)):
        graph_partners[j] = np.asarray(graph_partners[j], dtype=np.int64)
        graph_types[j] = np.asarray(graph_types[j], dtype=np.str_)

    return graph_partners, graph_types

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

def _right_sweep_parallel(R: np.ndarray, L: np.ndarray, partner_arrs: List[np.ndarray], type_arrs: List[np.ndarray]) -> np.ndarray:
    """
    Left-to-right pass of u's estimate through the polarized channel's graph
    using parallelization to traverse all paths simulteanously.
    """
    for n_stage in range(1, len(R[0])):
        nodes_j: np.ndarray = partner_arrs[n_stage-1]
        nodes_j_types: np.ndarray = type_arrs[n_stage-1]
        R[:,n_stage] = np.where(nodes_j_types == "+",
                                 _boxplus(R[:,n_stage-1], L[nodes_j,n_stage] + R[nodes_j,n_stage-1]), # +
                                 _boxplus(R[nodes_j,n_stage-1], L[nodes_j,n_stage]) + R[:,n_stage-1]) # =
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

def _left_sweep_parallel(R: np.ndarray, L: np.ndarray, partner_arrs: List[np.ndarray], type_arrs: List[np.ndarray]) -> np.ndarray:
    """
    Right-to-left pass of L's last column (initially the LLRs) through the
    polarized channel's graph using parallelization to traverse all paths
    simulteanously.
    """
    for n_stage in range(len(L[0]) - 1, 0, -1):
        nodes_j_types: np.ndarray = type_arrs[n_stage-1]
        nodes_j: np.ndarray = partner_arrs[n_stage-1]
        L[:,n_stage-1] = np.where(nodes_j_types == "+",
                                _boxplus(L[:,n_stage], L[nodes_j,n_stage] + R[nodes_j,n_stage-1]),
                                _boxplus(R[nodes_j,n_stage-1] , L[nodes_j,n_stage]) + L[:,n_stage])
    return L

def bp_decode(frozen_bits: frozenset, llr_1N: np.ndarray, iter_cap: int = 1000, parallel: bool = False) -> np.ndarray:
    i: int = 0
    u: list[int] = []
    N: int = llr_1N.size
    K: int = len(frozen_bits)
    bit_indices: list[np.ndarray] = _build_bit_indices(N)
    graph: List[List[Tuple[str, int]]] = _build_graph(N, bit_indices)
    perm: list[int] = _bit_reverse_perm(N)
    R: np.ndarray = _initialize_R(N, frozen_bits)
    L: np.ndarray = _initialize_L(N, llr_1N, perm)
    if parallel:
        part_arrs, type_arrs = _build_graph_arrays(graph)
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
            R = _right_sweep_parallel(R, L, part_arrs, type_arrs)
            L = _left_sweep_parallel(R, L, part_arrs, type_arrs)
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
        x_belief_graph: np.ndarray = ((R[:,int(np.log2(N))] + L[:,int(np.log2(N))]) < 0).astype(int)
        x_belief: np.ndarray = x_belief_graph[perm]
        if (x_hat == x_belief).all():
            break
        i += 1
        # print(i)
        
    # print("Estimate:", u_hat)
    return u_hat