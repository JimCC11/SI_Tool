import os
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.irl import compute_irl, weighting_function
from core.mixed_mode import single_to_mixed_mode, single_to_mixed_mode_npairs
from core.parser import get_frequency_ghz, load_snp

st.set_page_config(page_title="iRL Calculator", layout="wide")
st.title("SI Tool — iRL Calculator")

_AX = dict(showgrid=True, gridcolor="#888888", gridwidth=1, showline=False,
           mirror=False, zeroline=False,
           title_font=dict(size=20, family="Arial"), tickfont=dict(size=16, family="Arial"))
_LY = dict(hovermode="x unified", font=dict(size=16, family="Arial"),
           title_font=dict(size=22, family="Arial"),
           legend=dict(orientation="v", yanchor="bottom", y=0.02, xanchor="right", x=0.98,
                       font=dict(size=16, family="Arial")),
           margin=dict(t=42, b=58, l=60, r=20),
           plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
_BL  = dict(color="#888888", width=1)
_EPS = 1e-9
_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#d97706",
           "#7c3aed", "#0891b2", "#be185d", "#059669"]

_TABLE_STYLE = """
<style>
.pm-table { width:100%; border-collapse:collapse; text-align:center; font-size:0.85rem; }
.pm-table td { padding: 4px 6px; border: 1px solid #444; }
.pm-diff { font-weight:bold; background:#dbeafe; color:#1e40af; }
</style>
"""


def _n_pairs_from_name(name: str) -> int:
    ext = name.lower().rsplit('.', 1)[-1]
    if ext.startswith('s') and ext.endswith('p'):
        try:
            return max(1, int(ext[1:-1]) // 4)
        except ValueError:
            pass
    return 1


def _process_sp(uf, mapping: str, pair_idx: int,
                fb_hz: float, tr_s: float, nyquist_factor: float) -> dict:
    ext = uf.name.lower().rsplit('.', 1)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as f:
        f.write(uf.getvalue())
        tmp = f.name
    try:
        nw = load_snp(tmp)
        n_pairs = nw.number_of_ports // 4
        idx = min(pair_idx, n_pairs - 1)

        s_se = nw.s
        if n_pairs == 1:
            s_mm = single_to_mixed_mode(s_se, mapping=mapping)
        else:
            s_mm = single_to_mixed_mode_npairs(s_se, n_pairs=n_pairs, mapping=mapping)

        sdd11 = s_mm[:, 4 * idx,     4 * idx]
        sdd22 = s_mm[:, 4 * idx + 1, 4 * idx + 1]
        freq_hz  = nw.f
        freq_ghz = get_frequency_ghz(nw)
        irl_db, w, rl_avg = compute_irl(freq_hz, sdd11, sdd22, fb_hz, tr_s, nyquist_factor)
        return dict(label=uf.name, freq_ghz=freq_ghz, freq_hz=freq_hz,
                    sdd11=sdd11, sdd22=sdd22, irl_db=irl_db, w=w, rl_avg=rl_avg,
                    n_pairs=n_pairs)
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("設定")

    src      = st.radio("資料來源", ["S-parameter", "CSV file"], horizontal=True, key="irl_src")
    vna_mode = src == "S-parameter"

    # ── S-parameter ──
    if vna_mode:
        uploaded_sp = st.file_uploader(
            "上傳 S-parameter 檔案",
            type=["s4p", "s8p", "s12p", "s16p", "s20p", "s24p"],
            accept_multiple_files=True,
            key="irl_sp",
        )

        pair_idx  = 0
        max_pairs = max((_n_pairs_from_name(f.name) for f in uploaded_sp), default=1) if uploaded_sp else 1
        if max_pairs > 1:
            pair_labels = [f"Pair {i + 1}" for i in range(max_pairs)]
            sel      = st.selectbox("選擇差分對", pair_labels, key="irl_pair")
            pair_idx = pair_labels.index(sel)

        st.divider()
        st.subheader("Port Mapping")
        mapping_choice = st.radio("Port Mapping", ["Odd-Even", "N+1"], index=0,
                                  horizontal=True, label_visibility="hidden", key="irl_pm")
        mapping = 'A' if mapping_choice == 'Odd-Even' else 'B'
        if mapping == 'A':
            st.markdown(_TABLE_STYLE + """
<table class="pm-table">
  <tr>
    <td class="pm-diff" rowspan="2">Diff 1</td>
    <td>Port 1</td><td>────</td><td>Port 2</td>
    <td class="pm-diff" rowspan="2">Diff 2</td>
  </tr>
  <tr><td>Port 3</td><td>────</td><td>Port 4</td></tr>
</table>""", unsafe_allow_html=True)
        else:
            st.markdown(_TABLE_STYLE + """
<table class="pm-table">
  <tr>
    <td class="pm-diff" rowspan="2">Diff 1</td>
    <td>Port 1</td><td>────</td><td>Port 3</td>
    <td class="pm-diff" rowspan="2">Diff 2</td>
  </tr>
  <tr><td>Port 2</td><td>────</td><td>Port 4</td></tr>
</table>""", unsafe_allow_html=True)

    # ── CSV file ──
    else:
        uploaded_csv = st.file_uploader("上傳 RL CSV 檔案", type=["csv"], key="irl_csv")
        st.caption("格式：Frequency_GHz, RL11_dB [, RL22_dB]")

    st.divider()

    # ── iRL 參數 ──
    st.subheader("iRL 參數")

    _irl_defaults = pd.DataFrame({
        "Parameter":   ["Symbol Rate", "Rise Time Tᵣ", "Nyquist factor"],
        "Value":       [32.0,          25.0,            1.5],
        "Unit":        ["GBaud",       "ps",            "×Nyquist"],
    })
    _edited = st.data_editor(
        _irl_defaults,
        hide_index=True,
        use_container_width=True,
        key="irl_params_tbl",
        disabled=["Parameter", "Unit"],
        column_config={
            "Value": st.column_config.NumberColumn(
                label="Value", min_value=0.01, format="%.3g", step=1.0,
            )
        },
    )

    symbol_rate_gbaud = max(1.0,  float(_edited.iloc[0]["Value"]))
    rise_time_ps_irl  = max(1.0,  float(_edited.iloc[1]["Value"]))
    nyquist_factor    = max(0.1,  float(_edited.iloc[2]["Value"]))

    fb_hz  = symbol_rate_gbaud * 1e9
    tr_s   = rise_time_ps_irl * 1e-12
    ft_ghz = 0.2365 / tr_s / 1e9
    fr_ghz = nyquist_factor * (symbol_rate_gbaud / 2)

    st.markdown(f"""
<table style="width:100%;border-collapse:collapse;font-size:0.82rem;margin-top:2px">
  <tr>
    <td style="padding:4px 8px;border:1px solid #555;">ft = 0.2365 / Tᵣ</td>
    <td style="padding:4px 8px;border:1px solid #555;text-align:right;
               font-weight:bold;color:#2563eb;">{ft_ghz:.2f}</td>
    <td style="padding:4px 8px;border:1px solid #555;color:#666;">GHz</td>
  </tr>
  <tr>
    <td style="padding:4px 8px;border:1px solid #555;">fr = Nf × Nyquist</td>
    <td style="padding:4px 8px;border:1px solid #555;text-align:right;
               font-weight:bold;color:#2563eb;">{fr_ghz:.2f}</td>
    <td style="padding:4px 8px;border:1px solid #555;color:#666;">GHz</td>
  </tr>
</table>
""", unsafe_allow_html=True)

    st.divider()
    st.subheader("Gate 設定")
    st.caption("（功能開發中）")


# ── Plot helpers ──────────────────────────────────────────────────────────────
def _fig_w(freq_ghz, w, ft_g, fr_g):
    f_max = float(freq_ghz[-1])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freq_ghz, y=w, name="W(f)", line=dict(color="#2563eb"),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.08)",
    ))
    # shade excluded region (fr ~ f_max)
    if fr_g < f_max:
        fig.add_vrect(x0=fr_g, x1=f_max,
                      fillcolor="rgba(160,160,160,0.18)", layer="below", line_width=0)
        fig.add_annotation(x=(fr_g + f_max) / 2, y=0.5, yref="paper",
                           text="計算範圍外", showarrow=False,
                           font=dict(color="#888", size=12))
    for xv, label, color in [(ft_g, f"ft={ft_g:.2f}", "#dc2626"),
                              (fr_g, f"fr={fr_g:.2f} (上限)", "#16a34a")]:
        if xv <= f_max:
            fig.add_vline(x=xv, line=dict(color=color, dash="dash", width=1.5))
            fig.add_annotation(x=xv, y=1.02, yref="paper", text=label,
                               showarrow=False, font=dict(color=color, size=13))
    fig.add_vline(x=0,     line=_BL)
    fig.add_vline(x=f_max, line=_BL)
    fig.update_layout(
        title=dict(text="Weighting Function W(f)", x=0.5, xanchor="center", pad=dict(t=20)),
        xaxis_title="Frequency (GHz)", yaxis_title="W(f)",
        xaxis=dict(**_AX, range=[0 - _EPS, f_max + _EPS]),
        yaxis=dict(**_AX, range=[-_EPS, 1.05]),
        **_LY,
    )
    return fig


def _fig_rl(datasets, ft_g, fr_g):
    f_max = max(float(d["freq_ghz"][-1]) for d in datasets)
    fig   = go.Figure()
    # shade excluded region
    if fr_g < f_max:
        fig.add_vrect(x0=fr_g, x1=f_max,
                      fillcolor="rgba(160,160,160,0.18)", layer="below", line_width=0)
    for i, d in enumerate(datasets):
        clr = _COLORS[i % len(_COLORS)]
        rl_db = 20 * np.log10(d["rl_avg"] + 1e-15)
        fig.add_trace(go.Scatter(
            x=d["freq_ghz"], y=rl_db, name=d["label"],
            line=dict(color=clr),
        ))
    for xv, label, color in [(ft_g, f"ft={ft_g:.2f}", "#dc2626"),
                              (fr_g, f"fr={fr_g:.2f} (上限)", "#16a34a")]:
        if xv <= f_max:
            fig.add_vline(x=xv, line=dict(color=color, dash="dash", width=1.5))
            fig.add_annotation(x=xv, y=1.02, yref="paper", text=label,
                               showarrow=False, font=dict(color=color, size=13))
    fig.add_hline(y=0, line=_BL)
    fig.add_vline(x=0,     line=_BL)
    fig.add_vline(x=f_max, line=_BL)
    fig.update_layout(
        title=dict(text="RL_avg(f) = (|RL₁₁| + |RL₂₂|) / 2", x=0.5, xanchor="center", pad=dict(t=20)),
        xaxis_title="Frequency (GHz)", yaxis_title="Magnitude (dB)",
        xaxis=dict(**_AX, range=[0 - _EPS, f_max + _EPS]),
        yaxis=dict(**_AX),
        **_LY,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# S-PARAMETER MODE
# ══════════════════════════════════════════════════════════════════════════════
if vna_mode:
    if not uploaded_sp:
        st.info("← 請從左側上傳 S-parameter 檔案")
        st.stop()

    results = []
    errors  = []
    for uf in uploaded_sp:
        try:
            r = _process_sp(uf, mapping, pair_idx, fb_hz, tr_s, nyquist_factor)
            results.append(r)
            st.success(
                f"**{uf.name}** — {r['n_pairs']} 差分對 | "
                f"{r['freq_ghz'][0]:.3f} ~ {r['freq_ghz'][-1]:.3f} GHz"
            )
        except Exception as e:
            errors.append((uf.name, str(e)))

    for name, msg in errors:
        st.error(f"**{name}** 讀取失敗：{msg}")

    if not results:
        st.stop()

    # ── iRL 結果 ──
    st.subheader("iRL 計算結果")
    cols = st.columns(min(len(results), 4))
    for i, r in enumerate(results):
        with cols[i % 4]:
            st.metric(r["label"], f"{r['irl_db']:.2f} dB")

    if len(results) > 1:
        df_tbl = pd.DataFrame([
            {"檔案": r["label"], "iRL (dB)": f"{r['irl_db']:.2f}"}
            for r in results
        ])
        st.dataframe(df_tbl, use_container_width=True, hide_index=True)

    # ── 圖表 ──
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(_fig_w(results[0]["freq_ghz"], results[0]["w"], ft_ghz, fr_ghz),
                        use_container_width=True)
    with col2:
        datasets = [{"freq_ghz": r["freq_ghz"], "rl_avg": r["rl_avg"], "label": r["label"]}
                    for r in results]
        st.plotly_chart(_fig_rl(datasets, ft_ghz, fr_ghz), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# CSV FILE MODE
# ══════════════════════════════════════════════════════════════════════════════
else:
    if uploaded_csv is None:
        st.info("← 請從左側上傳 RL CSV 檔案")
        st.stop()

    try:
        df       = pd.read_csv(uploaded_csv)
        freq_ghz = df.iloc[:, 0].to_numpy(dtype=float)
        freq_hz  = freq_ghz * 1e9

        if df.shape[1] >= 3:
            rl11_lin = 10 ** (df.iloc[:, 1].to_numpy(dtype=float) / 20)
            rl22_lin = 10 ** (df.iloc[:, 2].to_numpy(dtype=float) / 20)
        else:
            rl11_lin = 10 ** (df.iloc[:, 1].to_numpy(dtype=float) / 20)
            rl22_lin = rl11_lin

        irl_db, w, rl_avg = compute_irl(freq_hz, rl11_lin, rl22_lin, fb_hz, tr_s, nyquist_factor)

    except Exception as e:
        st.error(f"CSV 讀取失敗：{e}")
        st.stop()

    # ── iRL 結果 ──
    st.subheader("iRL 計算結果")
    st.metric(uploaded_csv.name, f"{irl_db:.2f} dB")

    # ── 圖表 ──
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(_fig_w(freq_ghz, w, ft_ghz, fr_ghz), use_container_width=True)
    with col2:
        datasets = [{"freq_ghz": freq_ghz, "rl_avg": rl_avg, "label": uploaded_csv.name}]
        st.plotly_chart(_fig_rl(datasets, ft_ghz, fr_ghz), use_container_width=True)
