import numpy as np


def compute_ccicn(
    traces: list,
    mode: str,
    fb_hz: float,
    a_mv: float,
    il_post_db: float,
    tr_s: float,
    il_pre_db: float = 0.0,
    nyquist_factor: float = 1.5,
    df_hz: float = 10e6,
) -> tuple:
    """
    Compute ccICN RMS noise by frequency-domain integration.

    traces : list of (freq_hz_array, s_complex_or_magnitude_array)
             May come from different files with different frequency grids.
    mode   : 'NEXT' or 'FEXT'

    IL_pre and IL_post are fixed dB values (no frequency scaling).
    FEXT: ch_power = 10^((IL_pre + IL_post) / 10)
    NEXT: ch_power = 10^(2 × IL_post / 10)

    Returns: (ccicn_mv_rms, freq_ghz_grid, integrand)
    """
    nyquist_hz = fb_hz / 2
    fr_hz      = nyquist_factor * nyquist_hz
    ft_hz      = 0.2365 / tr_s

    freqs = np.arange(df_hz, fr_hz + df_hz / 2, df_hz)

    # Power-sum: interpolate each trace individually, then sum
    ps_power = np.zeros(len(freqs))
    for freq_snp_hz, s in traces:
        ps_power += np.interp(freqs, freq_snp_hz, np.abs(s) ** 2)

    # IL as fixed constants (user-defined)
    if mode == 'FEXT':
        ch_power = 10 ** ((il_pre_db + il_post_db) / 10)
    else:
        ch_power = 10 ** (2 * il_post_db / 10)

    # Signal PSD (σ_x² = 1)
    a_v = a_mv * 1e-3
    psd = (a_v ** 2) / fb_hz * np.sinc(freqs / fb_hz) ** 2

    # TX 4th-order + RX 8th-order Butterworth magnitude²
    h_tx2 = 1.0 / (1 + (freqs / ft_hz) ** 4)
    h_rx2 = 1.0 / (1 + (freqs / fr_hz) ** 8)

    integrand = psd * ch_power * h_tx2 * h_rx2 * ps_power
    ccicn_mv  = float(np.sqrt(0.5 * df_hz * np.sum(integrand)) * 1000)
    return ccicn_mv, freqs / 1e9, integrand
