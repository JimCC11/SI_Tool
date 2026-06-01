import numpy as np


def weighting_function(freq_hz: np.ndarray, symbol_rate_hz: float,
                       rise_time_s: float, nyquist_factor: float = 1.5) -> np.ndarray:
    """
    W(f) = sinc²(f/fb) · 1/(1+(f/ft)⁴) · 1/(1+(f/fr)⁸)
    ft = 0.2365 / Tr,  fr = nyquist_factor × (fb/2)
    numpy sinc is normalized: sinc(x) = sin(πx)/(πx)
    """
    ft = 0.2365 / rise_time_s
    fr = nyquist_factor * (symbol_rate_hz / 2)
    sinc_val = np.sinc(freq_hz / symbol_rate_hz)
    return (sinc_val ** 2
            / (1 + (freq_hz / ft) ** 4)
            / (1 + (freq_hz / fr) ** 8))


def compute_irl(freq_hz: np.ndarray, sdd11: np.ndarray, sdd22: np.ndarray,
                symbol_rate_hz: float = 32e9, rise_time_s: float = 25e-12,
                nyquist_factor: float = 1.5) -> tuple:
    """
    iRL = dB( sqrt( 1/N · Σ W(fi) · RL_avg²(fi) ) )

    Returns:
        irl_db  : scalar [dB], more negative = better RL
        w       : weighting function array
        rl_avg  : (|SDD11| + |SDD22|) / 2, linear magnitude
    """
    w = weighting_function(freq_hz, symbol_rate_hz, rise_time_s, nyquist_factor)
    rl_avg = (np.abs(sdd11) + np.abs(sdd22)) / 2
    irl_linear = np.sqrt(np.sum(w * rl_avg ** 2) / len(freq_hz))
    irl_db = 20 * np.log10(irl_linear + 1e-15)
    return irl_db, w, rl_avg
