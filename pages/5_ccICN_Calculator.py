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

    uploaded_sp = st.file_uploader(
        "上傳 S-parameter 檔案",
        type=_SNP_TYPES,
        accept_multiple_files=True,
        key="ccicn_sp",
    )

    n_pairs = _n_pairs_from_name(uploaded_sp[0].name) if uploaded_sp else 1

    # Build NEXT / FEXT option lists from n_pairs
    next_options, next_map = [], {}
    fext_options, fext_map = [], {}
    for a in range(n_pairs):
        for v in range(n_pairs):
            if a != v:
                nl = f"Pair {2*a+1} → Pair {2*v+1}"
                fl = f"Pair {2*a+1} → Pair {2*v+2}"
                next_options.append(nl); next_map[nl] = (a, v)
                fext_options.append(fl); fext_map[fl] = (a, v)

    if uploaded_sp:
        st.caption(f"偵測：{n_pairs * 4}-port → {n_pairs} 差分對")

    st.subheader("NEXT")
    if n_pairs < 2:
        st.caption("需要 ≥ 2 差分對")
        sel_next = []
    else:
        sel_next = st.multiselect("NEXT 路徑", next_options,
                                  default=next_options, key="ccicn_nx")

    st.subheader("FEXT")
    if n_pairs < 2:
        st.caption("需要 ≥ 2 差分對")
        sel_fext = []
    else:
        sel_fext = st.multiselect("FEXT 路徑", fext_options,
                                  default=fext_options, key="ccicn_fx")

    st.divider()
    st.subheader("Port Mapping")
    mapping_choice = st.radio("Port Mapping", ["Odd-Even", "N+1"], index=0,
                              horizontal=True, label_visibility="hidden", key="ccicn_pm")
    mapping = 'A' if mapping_choice == 'Odd-Even' else 'B'
    st.markdown(_port_map_html(n_pairs, mapping), unsafe_allow_html=True)

    st.divider()
    st.subheader("ccICN 參數")
    _defaults = pd.DataFrame({
        "Parameter": ["Symbol Rate fb", "A_FT", "A_NT",
                      "IL_pre @ Nyquist", "IL_post @ Nyquist", "Rise Time Tᵣ"],
        "Value":     [32.0,              800.0,  1000.0,
                      -20.0,             -6.0,                  7.5],
        "Unit":      ["GBaud",           "mVpp", "mVpp",
                      "dB",              "dB",                  "ps"],
    })
    _edited = st.data_editor(
        _defaults, hide_index=True, use_container_width=True,
        key="ccicn_params",
        disabled=["Parameter", "Unit"],
        column_config={
            "Value": st.column_config.NumberColumn(label="Value", format="%.4g")
        },
    )

    fb_gbaud   = float(_edited.iloc[0]["Value"])
    a_ft_mv    = float(_edited.iloc[1]["Value"])
    a_nt_mv    = float(_edited.iloc[2]["Value"])
    il_pre_db  = float(_edited.iloc[3]["Value"])
    il_post_db = float(_edited.iloc[4]["Value"])
    tr_ps      = float(_edited.iloc[5]["Value"])

    fb_hz  = fb_gbaud * 1e9
    tr_s   = tr_ps * 1e-12
    ft_ghz = 0.2365 / tr_s / 1e9
    fr_ghz = 1.5 * (fb_gbaud / 2)

    st.markdown(f"""
<table style="width:100%;border-collapse:collapse;font-size:0.82rem;margin-top:2px">
  <tr>
    <td style="padding:4px 8px;border:1px solid #555;">ft = 0.2365 / Tᵣ</td>
    <td style="padding:4px 8px;border:1px solid #555;text-align:right;
               font-weight:bold;color:#2563eb;">{ft_ghz:.2f}</td>
    <td style="padding:4px 8px;border:1px solid #555;color:#666;">GHz</td>
  </tr>
  <tr>
    <td style="padding:4px 8px;border:1px solid #555;">fr = 1.5 × Nyquist</td>
    <td style="padding:4px 8px;border:1px solid #555;text-align:right;
               font-weight:bold;color:#2563eb;">{fr_ghz:.2f}</td>
    <td style="padding:4px 8px;border:1px solid #555;color:#666;">GHz</td>
  </tr>
</table>
""", unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
if not uploaded_sp:
    st.info("← 請從左側上傳 S-parameter 檔案（S8P 以上）")
    st.stop()

if n_pairs < 2:
    st.warning("需要 ≥ 2 對差分對的 SNP 檔案（S8P 以上）")
    st.stop()

tmp_paths = []
results   = []
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

            next_traces, fext_traces = {}, {}
            for lbl in sel_next:
                a, v = next_map[lbl]
                if a < n_p and v < n_p:
                    next_traces[lbl] = s_mm[:, 4*v, 4*a]
            for lbl in sel_fext:
                a, v = fext_map[lbl]
                if a < n_p and v < n_p:
                    fext_traces[lbl] = s_mm[:, 4*v+1, 4*a]

            ccicn_next_mv = ccicn_fext_mv = None
            if next_traces:
                ccicn_next_mv, _, _ = compute_ccicn(
                    freq_hz, list(next_traces.values()), 'NEXT',
                    fb_hz, a_nt_mv, il_post_db, tr_s,
                )
            if fext_traces:
                ccicn_fext_mv, _, _ = compute_ccicn(
                    freq_hz, list(fext_traces.values()), 'FEXT',
                    fb_hz, a_ft_mv, il_post_db, tr_s,
                    il_pre_nyquist_db=il_pre_db,
                )

            results.append({
                "label": uf.name, "freq_hz": freq_hz, "freq_ghz": freq_ghz,
                "next_traces": next_traces, "fext_traces": fext_traces,
                "ccicn_next_mv": ccicn_next_mv, "ccicn_fext_mv": ccicn_fext_mv,
            })
            st.success(
                f"**{uf.name}** — {n_p} 差分對 | "
                f"{freq_ghz[0]:.3f}~{freq_ghz[-1]:.3f} GHz"
            )
        except Exception as e:
            st.error(f"**{uf.name}** 讀取失敗：{e}")
finally:
    for p in tmp_paths:
        try: os.unlink(p)
        except Exception: pass

if not results:
    st.stop()

# ── ccICN 計算結果 ────────────────────────────────────────────────────────────
st.subheader("ccICN 計算結果")
n_metrics = sum(
    (1 if r["ccicn_next_mv"] is not None else 0) +
    (1 if r["ccicn_fext_mv"] is not None else 0)
    for r in results
)
if n_metrics:
    mcols = st.columns(n_metrics)
    ci    = 0
    multi = len(results) > 1
    for r in results:
        pfx = f"{r['label']} " if multi else ""
        if r["ccicn_next_mv"] is not None:
            with mcols[ci]:
                st.metric(f"{pfx}ccICN_NEXT", f"{r['ccicn_next_mv']:.3f} mV")
            ci += 1
        if r["ccicn_fext_mv"] is not None:
            with mcols[ci]:
                st.metric(f"{pfx}ccICN_FEXT", f"{r['ccicn_fext_mv']:.3f} mV")
            ci += 1

f_max = float(results[0]["freq_ghz"][-1])


# ── Plot helper ────────────────────────────────────────────────────────────────
def _make_xt_fig(title: str, traces_key: str, ps_color: str) -> go.Figure:
    fig = go.Figure()
    ci  = 0
    multi = len(results) > 1

    for r in results:
        tdict = r[traces_key]
        if not tdict:
            continue
        pfx = f"{r['label']} " if multi else ""
        for lbl, s in tdict.items():
            md_db = 20 * np.log10(np.abs(s) + 1e-15)
            fig.add_trace(go.Scatter(
                x=r["freq_ghz"], y=md_db,
                name=f"{pfx}{lbl}",
                line=dict(color=_COLORS[ci % len(_COLORS)], width=1.5),
            ))
            ci += 1
        # Power sum
        ps    = np.sqrt(np.sum([np.abs(s) ** 2 for s in tdict.values()], axis=0))
        ps_db = 20 * np.log10(ps + 1e-15)
        fig.add_trace(go.Scatter(
            x=r["freq_ghz"], y=ps_db,
            name=f"{pfx}PS{title}",
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
    st.plotly_chart(_make_xt_fig("PSNEXT", "next_traces", _PS_NEXT_COLOR),
                    use_container_width=True)
with col2:
    st.plotly_chart(_make_xt_fig("PSFEXT", "fext_traces", _PS_FEXT_COLOR),
                    use_container_width=True)
