import numpy as np


def _raised_cosine_gate(t: np.ndarray, t_start: float, t_stop: float, rise_frac: float = 0.1) -> np.ndarray:
    span = t_stop - t_start
    dt = float(t[1] - t[0]) if len(t) > 1 else 1e-4
    rise = max(span * rise_frac, dt * 2)
    r0, r1 = t_start - rise / 2, t_start + rise / 2
    r2, r3 = t_stop  - rise / 2, t_stop  + rise / 2
    gate = np.zeros(len(t))
    m1 = (t >= r0) & (t < r1)
    gate[m1] = 0.5 * (1 - np.cos(np.pi * (t[m1] - r0) / rise))
    gate[(t >= r1) & (t <= r2)] = 1.0
    m2 = (t > r2) & (t <= r3)
    gate[m2] = 0.5 * (1 + np.cos(np.pi * (t[m2] - r2) / rise))
    return gate


def apply_time_gate(s11_freq: np.ndarray, freqs_hz: np.ndarray,
                    t_start_ns: float, t_stop_ns: float,
                    rise_frac: float = 0.1, n_pad_factor: int = 8,
                    exclude: bool = False, strength: float = 1.0):
    """
    Apply time-domain gate to S11 and return gated S11 in frequency domain.

    exclude=True  → remove selected region
    exclude=False → keep only selected region
    strength      → 0.0 = no gating, 1.0 = full gating

    Returns:
        s11_gated : gated S11, same length as s11_freq
        t_ns      : time axis (ns)
        impulse   : original impulse response (same length as t_ns)
        gate      : blended gate window (same length as t_ns)
    """
    s11_ext = np.concatenate([[s11_freq[0].real + 0j], s11_freq])
    n_orig = len(s11_ext)
    n_fft = n_orig * n_pad_factor

    padded = np.zeros(n_fft, dtype=complex)
    padded[:n_orig] = s11_ext

    impulse = np.fft.irfft(padded)
    n_time = len(impulse)

    df_hz = float(freqs_hz[1] - freqs_hz[0])
    T_ns = 1e9 / df_hz
    t_ns = np.linspace(0.0, T_ns, n_time, endpoint=False)

    gate_full = _raised_cosine_gate(t_ns, t_start_ns, t_stop_ns, rise_frac)
    if exclude:
        gate_full = 1.0 - gate_full
    gate = (1.0 - strength) + strength * gate_full   # blend: 0%→all-pass, 100%→full gate

    s11_full = np.fft.rfft(impulse * gate)
    s11_gated = s11_full[1:n_orig]

    return s11_gated, t_ns, impulse, gate


def compute_tdr_gated(s11_freq: np.ndarray, freqs_hz: np.ndarray,
                      t_start_ns: float, t_stop_ns: float,
                      z0: float = 100.0, rise_time_ps: float = 35.0,
                      n_pad_factor: int = 8, rise_frac: float = 0.1,
                      exclude: bool = False, strength: float = 1.0):
    """
    Gate the TDR impulse response directly inside the compute_tdr pipeline.
    Applies the same Gaussian + Hanning processing as compute_tdr, then
    gates the impulse response before integration → avoids double-windowing.

    Returns:
        t_ps : time axis (ps)
        z    : impedance (Ω)
        t_ns : time axis (ns), same length as t_ps
        gate : gate window (same length)
    """
    tr_s = rise_time_ps * 1e-12
    fc   = 0.35 / tr_s

    freqs = np.concatenate([[0.0], freqs_hz])
    s11   = np.concatenate([[s11_freq[0].real + 0j], s11_freq])

    gauss = np.exp(-0.5 * (freqs / fc) ** 2)
    win   = np.hanning(len(s11))
    s11_proc = s11 * gauss * win

    n_orig = len(s11_proc)
    n_fft  = n_orig * n_pad_factor
    padded = np.zeros(n_fft, dtype=complex)
    padded[:n_orig] = s11_proc

    impulse = np.fft.irfft(padded)
    n_time  = len(impulse)

    df_hz = float(freqs_hz[1] - freqs_hz[0])
    T_total_ps = 1e12 / df_hz
    t_ps = np.linspace(0.0, T_total_ps, n_time, endpoint=False)
    t_ns = t_ps / 1000.0

    gate_full = _raised_cosine_gate(t_ns, t_start_ns, t_stop_ns, rise_frac)
    if exclude:
        gate_full = 1.0 - gate_full
    gate = (1.0 - strength) + strength * gate_full

    step  = np.cumsum(impulse * gate)
    gamma = np.clip(np.real(step), -0.9999, 0.9999)
    z     = z0 * (1 + gamma) / (1 - gamma)

    return t_ps, z, t_ns, gate


def gate_tdr_csv(t_ns: np.ndarray, z: np.ndarray,
                 t_start_ns: float, t_stop_ns: float,
                 z0: float = 100.0, rise_frac: float = 0.1,
                 exclude: bool = False, strength: float = 1.0):
    """
    Apply time-domain gate directly to CSV TDR data (Z vs t).

    Process: Z → Γ (step response) → gradient → impulse
             → gate → cumsum → Γ_gated → Z_gated
    Also computes S11(f) via FFT for RL plot.

    Returns:
        z_gated   : gated impedance (Ω), same length as z
        s11_orig  : original S11 in freq domain (complex)
        s11_gated : gated S11 in freq domain (complex)
        freq_ghz  : frequency axis (GHz)
        gate      : gate window (same length as t_ns)
    """
    dt_ns = float(t_ns[1] - t_ns[0])

    # Z → Γ (step response)
    gamma = (z - z0) / (z + z0 + 1e-15)

    # Γ → impulse (dΓ/dt, scaled by dt so FFT units are consistent)
    impulse = np.gradient(gamma, dt_ns)   # units: 1/ns

    # Gate
    gate_full = _raised_cosine_gate(t_ns, t_start_ns, t_stop_ns, rise_frac)
    if exclude:
        gate_full = 1.0 - gate_full
    gate = (1.0 - strength) + strength * gate_full

    impulse_gated = impulse * gate

    # Integrate → gated step response (keep same starting value)
    gamma_gated = np.cumsum(impulse_gated) * dt_ns
    gamma_gated += gamma[0] - gamma_gated[0]
    gamma_gated = np.clip(gamma_gated, -0.9999, 0.9999)

    z_gated = z0 * (1 + gamma_gated) / (1 - gamma_gated)

    # FFT → S11(f) for RL plot
    n = len(impulse)
    dt_s = dt_ns * 1e-9
    s11_orig  = np.fft.rfft(impulse)  * dt_ns
    s11_gated = np.fft.rfft(impulse_gated) * dt_ns
    freq_ghz  = np.fft.rfftfreq(n, d=dt_s) / 1e9

    return z_gated, s11_orig, s11_gated, freq_ghz, gate
