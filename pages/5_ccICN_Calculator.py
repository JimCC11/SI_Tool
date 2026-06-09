import os
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.ccicn import compute_ccicn
from core.mixed_mode import single_to_mixed_mode, single_to_mixed_mode_npairs
from core.parser import get_frequency_ghz, load_snp

st.set_page_config(page_title="ccICN Calculator", layout="wide")
st.title("SI Tool — ccICN Calculator")

# ── Chart style (same as iRL) ─────────────────────────────────────────────────
_AXIS_BASE = dict(
    showgrid=True, gridcolor="#888888", gridwidth=1,
    showline=False, mirror=False, zeroline=False,
    title_font=dict(size=20, family="Arial"),
    tickfont=dict(size=16, family="Arial"),
)
_LAYOUT = dict(
    hovermode="x unified", font=dict(size=16, family="Arial"),
    title_font=dict(size=22, family="Arial"),
    legend=dict(orientation="v", yanchor="bottom", y=0.02, xanchor="right", x=0.98,
                font=dict(size=16, family="Arial")),
    margin=dict(t=42, b=58, l=60, r=20),
    plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
)
_BL  = dict(color="#888888", width=1)
_EPS = 1e-9

_SNP_TYPES = ["s4p", "s8p", "s12p", "s16p", "s20p", "s24p", "s28p", "s32p"]
_TABLE_STYLE = """
<style>
.pm-table { width:100%; border-collapse:collapse; text-align:center; font-size:0.85rem; }
.pm-table td { padding: 4px 6px; border: 1px solid #444; }
.pm-diff { font-weight:bold; background:#dbeafe; color:#1e40af; }
</style>
"""
_COLORS = [
    "#2563eb", "#dc2626", "#16a34a", "#d97706",
    "#7c3aed", "#0891b2", "#be185d", "#854d0e",
    "#64748b", "#065f46", "#92400e", "#1e3a8a",
]
_PS_NEXT_COLOR = "#1d4ed8"
_PS_FEXT_COLOR = "#b91c1c"


def _xax(**kw): return {**_AXIS_BASE, "title": dict(standoff=5), **kw}
def _yax(**kw): return {**_AXIS_BASE, **kw}


def _n_pairs_from_name(name: str) -> int:
    ext = name.lower().rsplit('.', 1)[-1]
    if ext.startswith('s') and ext.endswith('p'):
        try:
            n = int(ext[1:-1])
            if n % 4 == 0 and 4 <= n <= 32:
                return n // 4
        except ValueError:
            pass
    return 1


def _port_map_html(n_pairs: int, mapping: str) -> str:
    rows = [_TABLE_STYLE + '<table class="pm-table">']
    for k in range(n_pairs):
        p1, p2, p3, p4 = 4*k+1, 4*k+2, 4*k+3, 4*k+4
        d1, d2 = 2*k+1, 2*k+2
        a1, a2, b1, b2 = (p1, p2, p3, p4) if mapping == 'A' else (p1, p3, p2, p4)
        rows.append(
            f'<tr><td class="pm-diff" rowspan="2">Diff {d1}</td>'
            f'<td>Port {a1}</td><td>────</td><td>Port {a2}</td>'
            f'<td class="pm-diff" rowspan="2">Diff {d2}</td></tr>'
            f'<tr><td>Port {b1}</td><td>────</td><td>Port {b2}</td></tr>'
        )
    rows.append("</table>")
    return "".join(rows)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("設定")

    src      = st.radio("資料來源", ["S-parameter", "CSV"], horizontal=True, key="ccicn_src")
    csv_mode = src == "CSV"

    # ════════════════════ S-parameter mode ════════════════════
    if not csv_mode:
        uploaded_sp = st.file_uploader(
            "上傳 S-parameter 檔案",
            type=_SNP_TYPES,
            accept_multiple_files=True,
            key="ccicn_sp",
        )

        n_pairs = _n_pairs_from_name(uploaded_sp[0].name) if uploaded_sp else 1

        next_options, next_map = [], {}
        fext_options, fext_map = [], {}
        for a in range(n_pairs):
            for v in range(n_pairs):
                if a != v:
                    nl = f"Pair {2*a+1} → Pair {2*v+1}"
                    fl = f"Pair {2*a+1} → Pair {2*v+2}"
                    next_options.append(nl); next_map[nl] = (a, v)
                    fext_options.append(fl); fext_map[fl] = (a, v)

        file_next_sel, file_fext_sel = {}, {}
        if uploaded_sp and n_pairs >= 2:
            st.caption(f"偵測：{n_pairs * 4}-port → {n_pairs} 差分對")
            for uf in uploaded_sp:
                st.divider()
                st.caption(f"📄 **{uf.name}**")
                file_next_sel[uf.name] = st.multiselect(
                    "NEXT", next_options, default=[],
                    key=f"ccicn_nx_{uf.name}",
                )
                file_fext_sel[uf.name] = st.multiselect(
                    "FEXT", fext_options, default=[],
                    key=f"ccicn_fx_{uf.name}",
                )
        elif uploaded_sp and n_pairs < 2:
            st.caption("需要 ≥ 2 差分對")

        st.divider()
        st.subheader("Port Mapping")
        mapping_choice = st.radio("Port Mapping", ["Odd-Even", "N+1"], index=0,
                                  horizontal=True, label_visibility="hidden", key="ccicn_pm")
        mapping = 'A' if mapping_choice == 'Odd-Even' else 'B'
        st.markdown(_port_map_html(n_pairs if uploaded_sp else 1, mapping),
                    unsafe_allow_html=True)

    # ════════════════════ CSV mode ════════════════════
    else:
        st.subheader("NEXT CSV")
        next_csv_files = st.file_uploader(
            "上傳 NEXT CSV 檔案", type=["csv"],
            accept_multiple_files=True, key="ccicn_nx_csv",
        )
        st.caption("格式：Frequency_GHz, Magnitude_dB")

        st.subheader("FEXT CSV")
        fext_csv_files = st.file_uploader(
            "上傳 FEXT CSV 檔案", type=["csv"],
            accept_multiple_files=True, key="ccicn_fx_csv",
        )
        st.caption("格式：Frequency_GHz, Magnitude_dB")

    # ════════════════════ ccICN 參數 (共用) ════════════════════
    st.divider()
    st.subheader("ccICN 參數")

    # Fixed constants
    FB_GBAUD = 32.0
    A_FT_MV  = 800.0
    A_NT_MV  = 1000.0
    TR_PS    = 7.5

    fb_hz  = FB_GBAUD * 1e9
    tr_s   = TR_PS * 1e-12
    ft_ghz = 0.2365 / tr_s / 1e9
    fr_ghz = 1.5 * (FB_GBAUD / 2)

    _td = "padding:4px 8px;border:1px solid #555;"
    st.markdown(f"""
<table style="width:100%;border-collapse:collapse;font-size:0.82rem;margin-bottom:6px">
  <tr>
    <td style="{_td}">Symbol Rate fb</td>
    <td style="{_td};text-align:right;color:#555;">{FB_GBAUD:.4g}</td>
    <td style="{_td};color:#888;">GBaud</td>
  </tr>
  <tr>
    <td style="{_td}">A_FT</td>
    <td style="{_td};text-align:right;color:#555;">{A_FT_MV:.4g}</td>
    <td style="{_td};color:#888;">mVpp</td>
  </tr>
  <tr>
    <td style="{_td}">A_NT</td>
    <td style="{_td};text-align:right;color:#555;">{A_NT_MV:.4g}</td>
    <td style="{_td};color:#888;">mVpp</td>
  </tr>
  <tr>
    <td style="{_td}">Rise Time Tᵣ</td>
    <td style="{_td};text-align:right;color:#555;">{TR_PS:.4g}</td>
    <td style="{_td};color:#888;">ps</td>
  </tr>
  <tr>
    <td style="{_td}">ft = 0.2365 / Tᵣ</td>
    <td style="{_td};text-align:right;color:#555;">{ft_ghz:.2f}</td>
    <td style="{_td};color:#888;">GHz</td>
  </tr>
  <tr>
    <td style="{_td}">fr = 1.5 × Nyquist</td>
    <td style="{_td};text-align:right;color:#555;">{fr_ghz:.2f}</td>
    <td style="{_td};color:#888;">GHz</td>
  </tr>
</table>
""", unsafe_allow_html=True)

    # Editable: IL_pre / IL_post only
    _il_defaults = pd.DataFrame({
        "Parameter": ["IL_pre", "IL_post"],
        "Value":     [-20.0,    -6.0],
        "Unit":      ["dB",     "dB"],
    })
    _il_edited = st.data_editor(
        _il_defaults, hide_index=True, use_container_width=True,
        key="ccicn_params",
        disabled=["Parameter", "Unit"],
        column_config={
            "Value": st.column_config.NumberColumn(label="Value", format="%.4g")
        },
    )

    il_pre_db  = float(_il_edited.iloc[0]["Value"])
    il_post_db = float(_il_edited.iloc[1]["Value"])


# ── Main ──────────────────────────────────────────────────────────────────────
# all_next_traces / all_fext_traces: (display_label, freq_hz, freq_ghz, s)
all_next_traces: list = []
all_fext_traces: list = []

# ══════════════════════ S-parameter mode ══════════════════════
if not csv_mode:
    if not uploaded_sp:
        st.info("← 請從左側上傳 S-parameter 檔案（S8P 以上）")
        st.stop()

    if n_pairs < 2:
        st.warning("需要 ≥ 2 對差分對的 SNP 檔案（S8P 以上）")
        st.stop()

    tmp_paths = []
    try:
        for uf in uploaded_sp:
            ext = uf.name.lower().rsplit('.', 1)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as f:
                f.write(uf.getvalue())
                tmp_paths.append(f.name)
            try:
                nw       = load_snp(tmp_paths[-1])
                freq_hz  = nw.f
                freq_ghz = get_frequency_ghz(nw)
                n_p      = nw.number_of_ports // 4
                s_se     = nw.s
                s_mm     = (single_to_mixed_mode(s_se, mapping=mapping) if n_p == 1
                            else single_to_mixed_mode_npairs(s_se, n_pairs=n_p, mapping=mapping))

                multi = len(uploaded_sp) > 1
                for lbl in file_next_sel.get(uf.name, []):
                    a, v = next_map[lbl]
                    if a < n_p and v < n_p:
                        s    = s_mm[:, 4*v, 4*a]
                        disp = f"{uf.name}  {lbl}" if multi else lbl
                        all_next_traces.append((disp, freq_hz, freq_ghz, s))

                for lbl in file_fext_sel.get(uf.name, []):
                    a, v = fext_map[lbl]
                    if a < n_p and v < n_p:
                        s    = s_mm[:, 4*v+1, 4*a]
                        disp = f"{uf.name}  {lbl}" if multi else lbl
                        all_fext_traces.append((disp, freq_hz, freq_ghz, s))

                st.success(
                    f"**{uf.name}** — {n_p} 差分對 | "
                    f"{freq_ghz[0]:.3f}~{freq_ghz[-1]:.3f} GHz | "
                    f"NEXT: {len(file_next_sel.get(uf.name,[]))} 路徑  "
                    f"FEXT: {len(file_fext_sel.get(uf.name,[]))} 路徑"
                )
            except Exception as e:
                st.error(f"**{uf.name}** 讀取失敗：{e}")
    finally:
        for p in tmp_paths:
            try: os.unlink(p)
            except Exception: pass

# ══════════════════════ CSV mode ══════════════════════
else:
    has_csv = bool(next_csv_files or fext_csv_files)
    if not has_csv:
        st.info("← 請從左側上傳 NEXT / FEXT CSV 檔案")
        st.stop()

    def _load_csv_traces(csv_files, kind: str) -> list:
        out = []
        for uf in (csv_files or []):
            try:
                df       = pd.read_csv(uf)
                freq_ghz = df.iloc[:, 0].to_numpy(dtype=float)
                freq_hz  = freq_ghz * 1e9
                mag_db   = df.iloc[:, 1].to_numpy(dtype=float)
                s_mag    = 10 ** (mag_db / 20)
                out.append((uf.name, freq_hz, freq_ghz, s_mag))
                st.success(f"**{uf.name}** [{kind}] — {freq_ghz[0]:.3f}~{freq_ghz[-1]:.3f} GHz")
            except Exception as e:
                st.error(f"**{uf.name}** [{kind}] 讀取失敗：{e}")
        return out

    all_next_traces = _load_csv_traces(next_csv_files, "NEXT")
    all_fext_traces = _load_csv_traces(fext_csv_files, "FEXT")

# ── 共同：無路徑時停止 ────────────────────────────────────────────────────────
if not all_next_traces and not all_fext_traces:
    st.info("← 請選擇 NEXT / FEXT 路徑")
    st.stop()

# ── ccICN 計算結果 ────────────────────────────────────────────────────────────
ccicn_next_mv = ccicn_fext_mv = None

if all_next_traces:
    traces_next = [(fhz, s) for _, fhz, _, s in all_next_traces]
    ccicn_next_mv, _, _ = compute_ccicn(
        traces_next, 'NEXT', fb_hz, a_nt_mv, il_post_db, tr_s,
    )

if all_fext_traces:
    traces_fext = [(fhz, s) for _, fhz, _, s in all_fext_traces]
    ccicn_fext_mv, _, _ = compute_ccicn(
        traces_fext, 'FEXT', fb_hz, a_ft_mv, il_post_db, tr_s,
        il_pre_db=il_pre_db,
    )

st.subheader("ccICN 計算結果")
n_metrics = (1 if ccicn_next_mv is not None else 0) + (1 if ccicn_fext_mv is not None else 0)
if n_metrics:
    mcols = st.columns(n_metrics)
    ci = 0
    if ccicn_next_mv is not None:
        with mcols[ci]:
            st.metric("ccICN_NEXT", f"{ccicn_next_mv:.3f} mV",
                      help=f"{len(all_next_traces)} 條路徑 Power Sum")
        ci += 1
    if ccicn_fext_mv is not None:
        with mcols[ci]:
            st.metric("ccICN_FEXT", f"{ccicn_fext_mv:.3f} mV",
                      help=f"{len(all_fext_traces)} 條路徑 Power Sum")


# ── Plot helper ────────────────────────────────────────────────────────────────
def _make_xt_fig(title: str, all_traces: list, ps_color: str) -> go.Figure:
    fig = go.Figure()
    if not all_traces:
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            annotations=[dict(text="尚未選擇路徑", xref="paper", yref="paper",
                              x=0.5, y=0.5, showarrow=False, font=dict(size=18))],
            **_LAYOUT,
        )
        return fig

    f_max = min(float(t[2][-1]) for t in all_traces)

    for i, (lbl, _, freq_ghz, s) in enumerate(all_traces):
        md_db = 20 * np.log10(np.abs(s) + 1e-15)
        fig.add_trace(go.Scatter(
            x=freq_ghz, y=md_db, name=lbl,
            line=dict(color=_COLORS[i % len(_COLORS)], width=1.5),
        ))

    # Combined power sum on first trace's freq grid
    ref_fhz  = all_traces[0][1]
    ref_fghz = all_traces[0][2]
    ps_power = np.zeros(len(ref_fhz))
    for _, fhz, _, s in all_traces:
        ps_power += np.interp(ref_fhz, fhz, np.abs(s) ** 2)
    ps_db = 20 * np.log10(np.sqrt(ps_power) + 1e-15)
    fig.add_trace(go.Scatter(
        x=ref_fghz, y=ps_db,
        name=f"PS{title} (Combined)",
        line=dict(color=ps_color, width=2.5),
    ))

    fig.add_hline(y=0,     line=_BL)
    fig.add_hline(y=-80,   line=_BL)
    fig.add_vline(x=0,     line=_BL)
    fig.add_vline(x=f_max, line=_BL)
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", pad=dict(t=20)),
        xaxis_title="Frequency (GHz)", yaxis_title="Magnitude (dB)",
        xaxis=_xax(range=[0 - _EPS, f_max + _EPS]),
        yaxis=_yax(range=[-80 - _EPS, 0 + _EPS], tick0=0),
        **_LAYOUT,
    )
    return fig


col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(_make_xt_fig("PSNEXT", all_next_traces, _PS_NEXT_COLOR),
                    use_container_width=True)
with col2:
    st.plotly_chart(_make_xt_fig("PSFEXT", all_fext_traces, _PS_FEXT_COLOR),
                    use_container_width=True)
