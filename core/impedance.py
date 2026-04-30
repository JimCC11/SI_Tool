import numpy as np


def compute_tdr(s11_freq: np.ndarray, freqs_hz: np.ndarray,
                z0: float = 50.0, rise_time_ps: float = 35.0,
                n_pad_factor: int = 8):
    """
    從 S11 計算 TDR 阻抗曲線。

    步驟：
    1. 補 DC 點
    2. 套 Gaussian rise time filter（模擬 TDR 儀器的步階激勵）
    3. 套 Kaiser window（抑制 time domain ringing）
    4. Zero-padding（提升時間解析度）
    5. irfft → impulse response
    6. 積分 → step response Γ(t)
    7. Z(t) = Z0 × (1 + Γ) / (1 - Γ)

    回傳:
        t_ps: 時間軸（ps）
        z:    阻抗（Ω）
    """
    tr_s = rise_time_ps * 1e-12
    fc   = 0.35 / tr_s  # Gaussian filter -3dB 頻率（20%~80% rise time）

    # Step 1: 補 DC（用最低頻率的值外插）
    freqs = np.concatenate([[0.0], freqs_hz])
    s11   = np.concatenate([[s11_freq[0]], s11_freq])

    # Step 2: Gaussian rise time filter
    gauss = np.exp(-0.5 * (freqs / fc) ** 2)

    # Step 3: Hanning window
    win = np.hanning(len(s11))

    s11_proc = s11 * gauss * win

    # Step 4: Zero-padding
    n_orig = len(s11_proc)
    n_fft  = n_orig * n_pad_factor
    s11_padded = np.zeros(n_fft, dtype=complex)
    s11_padded[:n_orig] = s11_proc

    # Step 5: irfft → impulse response
    impulse = np.fft.irfft(s11_padded)

    # Step 6: 積分 → step response
    step = np.cumsum(impulse)

    # 時間軸：總時窗 = 1/df_orig
    df_orig   = freqs_hz[1] - freqs_hz[0]
    n_time    = len(impulse)
    T_total_ps = 1.0 / df_orig * 1e12
    t_ps = np.linspace(0, T_total_ps, n_time, endpoint=False)

    # Step 7: Γ → Z
    gamma = np.clip(np.real(step), -0.9999, 0.9999)
    z = z0 * (1 + gamma) / (1 - gamma)

    return t_ps, z


def compute_tdr_single(s_se: np.ndarray, freqs_hz: np.ndarray,
                       rise_time_ps: float = 35.0, forward: bool = True) -> tuple:
    """單端 TDR（Forward: S11，Reverse: S22，Z0 = 50 Ω）"""
    s11 = s_se[:, 0, 0] if forward else s_se[:, 1, 1]
    return compute_tdr(s11, freqs_hz, z0=50.0, rise_time_ps=rise_time_ps)


def compute_tdr_diff(s_mm: np.ndarray, freqs_hz: np.ndarray,
                     rise_time_ps: float = 35.0, forward: bool = True) -> tuple:
    """差分 TDR（Forward: SDD11，Reverse: SDD22，Z0_diff = 100 Ω）"""
    sdd = s_mm[:, 0, 0] if forward else s_mm[:, 1, 1]
    return compute_tdr(sdd, freqs_hz, z0=100.0, rise_time_ps=rise_time_ps)
