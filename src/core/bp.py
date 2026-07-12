from typing import List, Tuple

import numpy as np

from .encoder import PolarEncoder, PolarCode

def _int_to_bits(value: int, num_bits: int) -> np.ndarray:
    return np.array([(value >> i) & 1 for i in range(num_bits - 1, -1, -1)], dtype=int)

def _build_bit_indices(N: int) -> list[np.ndarray]:
    indices: list[np.ndarray] = []
    for idx in range(N):
        indices.append(_int_to_bits(idx, int(np.log2(N))))
    return indices

def _boxplus(x: float, y: float) -> float:
    return np.sign(x) * np.sign(y) * min(np.abs(x), np.abs(y))

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
    values (the remaining entries are set to 0).

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

def _initialize_L(N: int, llr_1N: np.ndarray) -> np.ndarray:
    """
    Initialize right-to-left sweep matrix with the right-most column being the
    LLRs returned after transmitting u through the channel (the remaining
    entries are set to 0).

    Parameters
    ----------
    - N: int
        Codeword length.
    - llr_1N: np.ndarray
        Indices of the frozen bits.
    """
    n_cols: int = int(np.log2(N) + 1)
    L: np.ndarray = np.zeros((N, n_cols))
    llr_1N = np.where(llr_1N == np.inf, 30, llr_1N)
    llr_1N = np.where(llr_1N == -np.inf, -30, llr_1N)
    L[:, n_cols - 1] = llr_1N
    return L

def _right_sweep(R: np.ndarray, L: np.ndarray, graph: List[List[Tuple[str, int]]]) -> np.ndarray:
    """
    Left-to-right pass of u's estimate through the polarized channel's graph.
    """
    for n_stage in range(1, len(R[0])):
        for node_i in range(len(R)):
            node_j: int = graph[node_i][n_stage-1][1] #*partner of node i
            # import pdb; pdb.set_trace()
            if graph[node_i][n_stage-1][0] == "+": #*the -1 because it has log2(N) - 1 cols
                R[node_i][n_stage] = _boxplus(R[node_i][n_stage-1],
                                              L[node_j][n_stage] + R[node_j][n_stage-1])
            else:
                R[node_i][n_stage] = _boxplus(R[node_j][n_stage-1], L[node_j][n_stage]) \
                                     + R[node_i][n_stage-1]
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
                                                L[node_j][n_stage] + R[node_j][n_stage-1] )
            else:
                L[node_i][n_stage-1] = _boxplus(R[node_j][n_stage-1] , L[node_j][n_stage]) \
                                     + L[node_i][n_stage]
    return L

def bp_decode(frozen_bits: frozenset, llr_1N: np.ndarray, iter_cap: int = 1000) -> np.ndarray:
    i: int = 0
    u: list[int] = []
    N: int = llr_1N.size
    K: int = len(frozen_bits)
    bit_indices: list[np.ndarray] = _build_bit_indices(N)
    graph: List[List[Tuple[str, int]]] = _build_graph(N, bit_indices)
    R: np.ndarray = _initialize_R(N, frozen_bits)
    L: np.ndarray = _initialize_L(N, llr_1N)
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
        R = _right_sweep(R, L, graph)
        L = _left_sweep(R, L, graph)
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
        x_belief: np.ndarray = ((R[:,int(np.log2(N))-1] + L[:,int(np.log2(N))-1]) < 0).astype(int)
        if (x_hat == x_belief).all():
            break
        i += 1
        # print(i)
        
    # print("Estimate:", u_hat)
    return u_hat