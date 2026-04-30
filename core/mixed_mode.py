import numpy as np


# ── 慣例 A：同極性同組（Jim 的 VNA 配置）──────────────────
#   Port 1 (index 0) → D+ TX
#   Port 2 (index 1) → D+ RX
#   Port 3 (index 2) → D- TX
#   Port 4 (index 3) → D- RX
#
#   d_TX = (P1 - P3) / √2
#   d_RX = (P2 - P4) / √2

# ── 慣例 B：同端點同組（PCIe CEM / JEDEC）────────────────
#   Port 1 (index 0) → D+ TX
#   Port 2 (index 1) → D- TX
#   Port 3 (index 2) → D+ RX
#   Port 4 (index 3) → D- RX
#
#   d_TX = (P1 - P2) / √2
#   d_RX = (P3 - P4) / √2

_s = 1 / np.sqrt(2)

_M_A = np.array([
    [ _s,  0., -_s,  0.],   # d_TX = (P1 - P3) / √2
    [ 0.,  _s,  0., -_s],   # d_RX = (P2 - P4) / √2
    [ _s,  0.,  _s,  0.],   # c_TX = (P1 + P3) / √2
    [ 0.,  _s,  0.,  _s],   # c_RX = (P2 + P4) / √2
])

_M_B = np.array([
    [ _s, -_s,  0.,  0.],   # d_TX = (P1 - P2) / √2
    [ 0.,  0.,  _s, -_s],   # d_RX = (P3 - P4) / √2
    [ _s,  _s,  0.,  0.],   # c_TX = (P1 + P2) / √2
    [ 0.,  0.,  _s,  _s],   # c_RX = (P3 + P4) / √2
])

_MATRICES = {'A': _M_A, 'B': _M_B}


def single_to_mixed_mode_npairs(s_se: np.ndarray, n_pairs: int, mapping: str = 'A') -> np.ndarray:
    """
    Convert (4*n_pairs)-port S-matrix to mixed-mode.
    Output ordering per pair: [d_TX, d_RX, c_TX, c_RX].
    """
    M_single = _MATRICES[mapping]
    n = 4 * n_pairs
    M = np.zeros((n, n))
    for i in range(n_pairs):
        M[4*i:4*(i+1), 4*i:4*(i+1)] = M_single
    return np.einsum('ij,fjk,lk->fil', M, s_se, M)


def get_pair_fd(s_mm: np.ndarray, freq: np.ndarray, pair_idx: int, label: str) -> dict:
    """Frequency-domain mixed-mode params for one pair (0-based index)."""
    i = pair_idx
    return {
        "freq":  freq,
        "sdd21": s_mm[:, 4*i+1, 4*i],
        "sdd11": s_mm[:, 4*i,   4*i],
        "sdd22": s_mm[:, 4*i+1, 4*i+1],
        "scd21": s_mm[:, 4*i+3, 4*i],
        "sdc21": s_mm[:, 4*i+1, 4*i+2],
        "label": label,
    }


def get_next_datasets(s_mm: np.ndarray, freq: np.ndarray, victim: int, n_pairs: int) -> list:
    """NEXT datasets for victim pair: sdd21 = NEXT term from each aggressor."""
    return [
        {"freq": freq, "sdd21": s_mm[:, 4*victim, 4*agg], "label": f"Pair {agg+1}→{victim+1}"}
        for agg in range(n_pairs) if agg != victim
    ]


def get_fext_datasets(s_mm: np.ndarray, freq: np.ndarray, victim: int, n_pairs: int) -> list:
    """FEXT datasets for victim pair: sdd21 = FEXT term from each aggressor."""
    return [
        {"freq": freq, "sdd21": s_mm[:, 4*victim+1, 4*agg], "label": f"Pair {agg+1}→{victim+1}"}
        for agg in range(n_pairs) if agg != victim
    ]


def single_to_mixed_mode(s_se: np.ndarray, mapping: str = 'A') -> np.ndarray:
    """
    將單端 S-matrix 轉換為 mixed-mode S-matrix。

    s_se:    shape [freq, 4, 4]
    mapping: 'A'（慣例 A，同極性同組）或 'B'（慣例 B，同端點同組）
    回傳:    shape [freq, 4, 4]，port 順序 [d_TX, d_RX, c_TX, c_RX]
    """
    M = _MATRICES[mapping]
    return np.einsum('ij,fjk,lk->fil', M, s_se, M)


def get_SDD21(s_mm: np.ndarray) -> np.ndarray:
    """差分 insertion loss：differential input (TX) → differential output (RX)"""
    return s_mm[:, 1, 0]


def get_SDD11(s_mm: np.ndarray) -> np.ndarray:
    """差分 return loss at TX"""
    return s_mm[:, 0, 0]


def get_SDD22(s_mm: np.ndarray) -> np.ndarray:
    """差分 return loss at RX"""
    return s_mm[:, 1, 1]


def get_SCD21(s_mm: np.ndarray) -> np.ndarray:
    """Mode conversion：differential input (TX) → common mode output (RX)"""
    return s_mm[:, 3, 0]


def get_SDC21(s_mm: np.ndarray) -> np.ndarray:
    """Mode conversion：common mode input (TX) → differential output (RX)"""
    return s_mm[:, 1, 2]
