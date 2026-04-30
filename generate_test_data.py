"""
產生合成的 PCIe connector S4P 練習檔案。

Port mapping（對應 SI Tool 設定）:
    Port 1 → D+ TX
    Port 2 → D+ RX
    Port 3 → D− TX
    Port 4 → D− RX

模型特性（參考 PCIe Gen 3/4 connector 典型值）:
    - 頻率範圍：0.01 ~ 20 GHz，201 點
    - 電氣延遲：80 ps
    - Insertion loss：skin effect + dielectric loss，約 -1 dB @ 8 GHz
    - Return loss：parasitic 電容 ~0.08 pF，約 -25 dB @ 8 GHz
    - Mode conversion（SCD21/SDC21）：-42 ~ -50 dB（差分對輕微不對稱）
    - Differential impedance：~98 Ω（nominal 100 Ω with slight bump）
"""

import numpy as np

# ── 頻率設定 ──────────────────────────────────────────────
freqs_ghz = np.linspace(0.01, 20.0, 201)
freqs_hz  = freqs_ghz * 1e9
n         = len(freqs_ghz)

# ── Connector 參數 ────────────────────────────────────────
Z0    = 50.0       # 單端參考阻抗 [Ω]
tau   = 80e-12     # 電氣延遲 [s]，80 ps
a     = 0.10       # skin effect 係數 [dB/GHz^0.5]
b     = 0.008      # dielectric loss 係數 [dB/GHz]
C_p   = 0.08e-12   # parasitic 電容 [F]，約 0.08 pF

# D+ / D− 輕微不對稱，讓 mode conversion 不為零（更真實）
asym_amp   = 0.008   # 振幅不對稱比例
asym_phase = 3e-12   # 相位（延遲差）[s]

# Crosstalk 參數
xt_next_k = 2.5e-3   # NEXT 強度係數
xt_fext_k = 1.2e-3   # FEXT 強度係數

# ── 建立 S-matrix [n_freq, 4, 4] ─────────────────────────
S = np.zeros((n, 4, 4), dtype=complex)

for i, (f_hz, f_ghz) in enumerate(zip(freqs_hz, freqs_ghz)):
    w = 2 * np.pi * f_hz

    # 傳輸損耗
    il_db   = -(a * np.sqrt(f_ghz) + b * f_ghz)
    il      = 10 ** (il_db / 20.0)
    phase   = -w * tau
    T       = il * np.exp(1j * phase)

    # Parasitic 電容造成的反射
    Gamma   = (1j * w * C_p * Z0) / (2.0 + 1j * w * C_p * Z0)
    T_eff   = T * np.sqrt(max(1.0 - abs(Gamma)**2, 0.0))

    # ── D+ 路徑（Port 1 <-> Port 2）────────────────────────
    S[i, 0, 0] = Gamma      # S11
    S[i, 1, 1] = Gamma      # S22
    S[i, 1, 0] = T_eff      # S21
    S[i, 0, 1] = T_eff      # S12

    # ── D− 路徑（Port 3 <-> Port 4），加入輕微不對稱 ────────
    dT    = asym_amp * np.exp(1j * (-w * asym_phase + 0.3))
    T_neg = T_eff * (1.0 + dT)
    G_neg = Gamma * (1.0 + asym_amp * 0.5)

    S[i, 2, 2] = G_neg          # S33
    S[i, 3, 3] = G_neg          # S44
    S[i, 3, 2] = T_neg          # S43
    S[i, 2, 3] = T_neg          # S34

    # ── NEXT（近端串擾）：Port 1 <-> Port 3, Port 2 <-> Port 4 ──
    xt_n = xt_next_k * (f_ghz / 10.0) ** 0.7 * np.exp(1j * (phase * 0.25 + np.pi / 3))
    S[i, 2, 0] = xt_n   # S31
    S[i, 0, 2] = xt_n   # S13
    S[i, 3, 1] = xt_n   # S42
    S[i, 1, 3] = xt_n   # S24

    # ── FEXT（遠端串擾）：Port 1 <-> Port 4, Port 2 <-> Port 3 ──
    xt_f = xt_fext_k * (f_ghz / 10.0) ** 0.5 * np.exp(1j * (phase + np.pi / 5))
    S[i, 3, 0] = xt_f   # S41
    S[i, 0, 3] = xt_f   # S14
    S[i, 2, 1] = xt_f   # S32
    S[i, 1, 2] = xt_f   # S23

# ── 寫入 Touchstone S4P（RI 格式）────────────────────────
output_path = "test_data/pcie_connector_practice.s4p"

lines = [
    "! Synthetic PCIe Connector S4P — Practice File",
    "! Port mapping:",
    "!   Port 1 = D+ TX    Port 2 = D+ RX",
    "!   Port 3 = D- TX    Port 4 = D- RX",
    "! Frequency: 0.01 ~ 20 GHz, 201 points",
    "! IL @ 8 GHz ~ -1.0 dB | RL ~ -25 dB | Zdiff ~ 98 ohm",
    "# GHz S RI R 50",
]

for i, f_ghz in enumerate(freqs_ghz):
    # Touchstone 4-port: 每個頻率點寫 4 行，每行對應一個 input port（column）
    # 行內順序：Sj1 Sj2 Sj3 Sj4（j=1..4），以 real imag 表示
    for col in range(4):
        row_parts = []
        if col == 0:
            row_parts.append(f"{f_ghz:.6f}")  # 第一行才寫頻率
        else:
            row_parts.append(" " * 12)        # 後三行用空格對齊
        for row in range(4):
            v = S[i, row, col]
            row_parts.append(f"{v.real:14.9f} {v.imag:14.9f}")
        lines.append(" ".join(row_parts))

with open(output_path, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"Generated: {output_path}")

# ── 簡單驗證：印出幾個關鍵數值 ───────────────────────────
# 找 8 GHz 最近的 index
idx_8g = np.argmin(np.abs(freqs_ghz - 8.0))

s21_db  = 20 * np.log10(abs(S[idx_8g, 1, 0]) + 1e-15)
s11_db  = 20 * np.log10(abs(S[idx_8g, 0, 0]) + 1e-15)
s31_db  = 20 * np.log10(abs(S[idx_8g, 2, 0]) + 1e-15)

print(f"\n@ {freqs_ghz[idx_8g]:.1f} GHz:")
print(f"  S21 (Insertion loss)   = {s21_db:.2f} dB")
print(f"  S11 (Return loss)      = {s11_db:.2f} dB")
print(f"  S31 (NEXT)             = {s31_db:.2f} dB")
