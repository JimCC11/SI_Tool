import numpy as np
import plotly.graph_objects as go

_AXIS_BASE = dict(
    showgrid=True,
    gridcolor="#888888",
    gridwidth=1,
    showline=False,
    mirror=False,
    zeroline=False,
    title_font=dict(size=20, family="Arial"),
    tickfont=dict(size=16, family="Arial"),
)

_LAYOUT = dict(
    hovermode="x unified",
    font=dict(size=16, family="Arial"),
    title_font=dict(size=22, family="Arial"),
    legend=dict(
        orientation="v", yanchor="bottom", y=0.02, xanchor="right", x=0.98,
        font=dict(size=16, family="Arial"),
    ),
    margin=dict(t=42, b=58, l=60, r=20),
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
)

_BL = dict(color="#888888", width=1)

# 每個檔案的顏色對 (color_a, color_b)
_FILE_COLORS = [
    ("#2563eb", "#dc2626"),  # blue, red
    ("#16a34a", "#d97706"),  # green, orange
    ("#7c3aed", "#0891b2"),  # purple, cyan
    ("#be185d", "#854d0e"),  # pink, brown
]


def _xaxis(**kwargs) -> dict:
    return {**_AXIS_BASE, "title": dict(standoff=5), **kwargs}


def _yaxis(**kwargs) -> dict:
    return {**_AXIS_BASE, **kwargs}


def _to_db(s: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.abs(s) + 1e-15)


def _trace_name(ds: dict, name: str, multi: bool) -> str:
    return f"{ds['label']} {name}" if multi else name


def _tlabel(ds: dict, key: str, default: str, multi: bool) -> str:
    """Use per-trace label from ds['trace_labels'] if present, else fall back to _trace_name."""
    tl = ds.get("trace_labels")
    if tl and key in tl:
        return tl[key]
    return _trace_name(ds, default, multi)


_EPS = 1e-9


def plot_insertion_loss(datasets: list,
                        x_min: float = 0, x_max: float = None, x_step: float = None,
                        y_min: float = None, y_max: float = 0, y_step: float = None,
                        ) -> go.Figure:
    multi = len(datasets) > 1
    fig = go.Figure()
    all_y = []
    for idx, ds in enumerate(datasets):
        color = _FILE_COLORS[idx % len(_FILE_COLORS)][0]
        y = _to_db(ds["sdd21"])
        all_y.append(y)
        fig.add_trace(go.Scatter(x=ds["freq"], y=y,
                                 name=_tlabel(ds, "sdd21", "SDD21", multi),
                                 line=dict(color=color)))

    if x_max is None:
        x_max = float(max(ds["freq"][-1] for ds in datasets))
    if y_min is None:
        y_min = float(np.floor(np.min(np.concatenate(all_y))))

    x_kw: dict = dict(range=[x_min - _EPS, x_max + _EPS])
    if x_step is not None:
        x_kw['dtick'] = x_step
    y_kw: dict = dict(range=[y_min - _EPS, y_max + _EPS], tick0=y_max)
    if y_step is not None:
        y_kw['dtick'] = y_step

    fig.add_hline(y=y_max, line=_BL)
    fig.add_hline(y=y_min, line=_BL)
    fig.add_vline(x=x_min, line=_BL)
    fig.add_vline(x=x_max, line=_BL)
    fig.update_layout(
        title=dict(text="Insertion Loss (SDD21)", x=0.5, xanchor="center", pad=dict(t=20)),
        xaxis_title="Frequency (GHz)",
        yaxis_title="Magnitude (dB)",
        xaxis=_xaxis(**x_kw),
        yaxis=_yaxis(**y_kw),
        **_LAYOUT,
    )
    return fig


def plot_return_loss(datasets: list,
                     x_min: float = 0, x_max: float = None, x_step: float = None,
                     y_min: float = -80, y_max: float = 0, y_step: float = None,
                     show_sdd11: bool = True, show_sdd22: bool = True,
                     ) -> go.Figure:
    multi = len(datasets) > 1
    fig = go.Figure()
    for idx, ds in enumerate(datasets):
        ca, cb = _FILE_COLORS[idx % len(_FILE_COLORS)]
        if show_sdd11:
            fig.add_trace(go.Scatter(x=ds["freq"], y=_to_db(ds["sdd11"]),
                                     name=_tlabel(ds, "sdd11", "SDD11", multi),
                                     line=dict(color=ca)))
        if show_sdd22:
            fig.add_trace(go.Scatter(x=ds["freq"], y=_to_db(ds["sdd22"]),
                                     name=_tlabel(ds, "sdd22", "SDD22", multi),
                                     line=dict(color=cb)))

    if x_max is None:
        x_max = float(max(ds["freq"][-1] for ds in datasets))

    x_kw: dict = dict(range=[x_min - _EPS, x_max + _EPS])
    if x_step is not None:
        x_kw['dtick'] = x_step
    y_kw: dict = dict(range=[y_min - _EPS, y_max + _EPS], tick0=y_max)
    if y_step is not None:
        y_kw['dtick'] = y_step

    fig.add_hline(y=y_max, line=_BL)
    fig.add_hline(y=y_min, line=_BL)
    fig.add_vline(x=x_min, line=_BL)
    fig.add_vline(x=x_max, line=_BL)
    fig.update_layout(
        title=dict(text="Return Loss", x=0.5, xanchor="center", pad=dict(t=20)),
        xaxis_title="Frequency (GHz)",
        yaxis_title="Magnitude (dB)",
        xaxis=_xaxis(**x_kw),
        yaxis=_yaxis(**y_kw),
        **_LAYOUT,
    )
    return fig


def plot_psnext(datasets: list,
                ps_datasets: list = None,
                x_min: float = 0, x_max: float = None, x_step: float = None,
                y_min: float = -80, y_max: float = 0, y_step: float = None,
                ) -> go.Figure:
    """
    datasets:    individual NEXT terms → dashed lines
    ps_datasets: pre-computed per-victim PSNEXT → solid lines
                 if None, computes one grand PSNEXT from datasets (S4P behaviour)
    """
    fig = go.Figure()
    if datasets:
        freq = datasets[0]["freq"]
        multi = len(datasets) > 1
        for idx, ds in enumerate(datasets):
            color = _FILE_COLORS[idx % len(_FILE_COLORS)][0]
            fig.add_trace(go.Scatter(
                x=freq, y=20 * np.log10(np.abs(ds["sdd21"]) + 1e-15),
                name=_trace_name(ds, "NEXT", multi),
                line=dict(color=color, dash="dash"),
            ))
        if ps_datasets:
            for idx, ps in enumerate(ps_datasets):
                color = _FILE_COLORS[idx % len(_FILE_COLORS)][0]
                fig.add_trace(go.Scatter(
                    x=freq, y=10 * np.log10(np.abs(ps["sdd21"])**2 + 1e-30),
                    name=ps["label"], line=dict(color=color),
                ))
        else:
            ps = np.sum(np.array([np.abs(ds["sdd21"])**2 for ds in datasets]), axis=0)
            fig.add_trace(go.Scatter(x=freq, y=10 * np.log10(ps + 1e-30),
                                     name="PSNEXT", line=dict(color=_FILE_COLORS[0][0])))
        if x_max is None:
            x_max = float(freq[-1])
    else:
        if x_max is None:
            x_max = 20.0

    x_kw: dict = dict(range=[x_min - _EPS, x_max + _EPS])
    if x_step is not None: x_kw['dtick'] = x_step
    y_kw: dict = dict(range=[y_min - _EPS, y_max + _EPS], tick0=y_max)
    if y_step is not None: y_kw['dtick'] = y_step

    fig.add_hline(y=y_max, line=_BL); fig.add_hline(y=y_min, line=_BL)
    fig.add_vline(x=x_min, line=_BL); fig.add_vline(x=x_max, line=_BL)
    fig.update_layout(
        title=dict(text="PSNEXT", x=0.5, xanchor="center", pad=dict(t=20)),
        xaxis_title="Frequency (GHz)", yaxis_title="Magnitude (dB)",
        xaxis=_xaxis(**x_kw), yaxis=_yaxis(**y_kw), **_LAYOUT,
    )
    return fig


def plot_psfext(datasets: list,
                ps_datasets: list = None,
                x_min: float = 0, x_max: float = None, x_step: float = None,
                y_min: float = -80, y_max: float = 0, y_step: float = None,
                ) -> go.Figure:
    fig = go.Figure()
    if datasets:
        freq = datasets[0]["freq"]
        multi = len(datasets) > 1
        for idx, ds in enumerate(datasets):
            color = _FILE_COLORS[idx % len(_FILE_COLORS)][1]
            fig.add_trace(go.Scatter(
                x=freq, y=20 * np.log10(np.abs(ds["sdd21"]) + 1e-15),
                name=_trace_name(ds, "FEXT", multi),
                line=dict(color=color, dash="dash"),
            ))
        if ps_datasets:
            for idx, ps in enumerate(ps_datasets):
                color = _FILE_COLORS[idx % len(_FILE_COLORS)][1]
                fig.add_trace(go.Scatter(
                    x=freq, y=10 * np.log10(np.abs(ps["sdd21"])**2 + 1e-30),
                    name=ps["label"], line=dict(color=color),
                ))
        else:
            ps = np.sum(np.array([np.abs(ds["sdd21"])**2 for ds in datasets]), axis=0)
            fig.add_trace(go.Scatter(x=freq, y=10 * np.log10(ps + 1e-30),
                                     name="PSFEXT", line=dict(color=_FILE_COLORS[0][1])))
        if x_max is None:
            x_max = float(freq[-1])
    else:
        if x_max is None:
            x_max = 20.0

    x_kw: dict = dict(range=[x_min - _EPS, x_max + _EPS])
    if x_step is not None: x_kw['dtick'] = x_step
    y_kw: dict = dict(range=[y_min - _EPS, y_max + _EPS], tick0=y_max)
    if y_step is not None: y_kw['dtick'] = y_step

    fig.add_hline(y=y_max, line=_BL); fig.add_hline(y=y_min, line=_BL)
    fig.add_vline(x=x_min, line=_BL); fig.add_vline(x=x_max, line=_BL)
    fig.update_layout(
        title=dict(text="PSFEXT", x=0.5, xanchor="center", pad=dict(t=20)),
        xaxis_title="Frequency (GHz)", yaxis_title="Magnitude (dB)",
        xaxis=_xaxis(**x_kw), yaxis=_yaxis(**y_kw), **_LAYOUT,
    )
    return fig


def plot_mode_conversion(datasets: list,
                         x_min: float = 0, x_max: float = None, x_step: float = None,
                         y_min: float = -80, y_max: float = 0, y_step: float = None,
                         ) -> go.Figure:
    multi = len(datasets) > 1
    fig = go.Figure()
    for idx, ds in enumerate(datasets):
        ca, cb = _FILE_COLORS[idx % len(_FILE_COLORS)]
        fig.add_trace(go.Scatter(x=ds["freq"], y=_to_db(ds["scd21"]),
                                 name=_tlabel(ds, "scd21", "SCD21", multi),
                                 line=dict(color=ca)))
        fig.add_trace(go.Scatter(x=ds["freq"], y=_to_db(ds["sdc21"]),
                                 name=_tlabel(ds, "sdc21", "SDC21", multi),
                                 line=dict(color=cb)))

    if x_max is None:
        x_max = float(max(ds["freq"][-1] for ds in datasets))

    x_kw: dict = dict(range=[x_min - _EPS, x_max + _EPS])
    if x_step is not None:
        x_kw['dtick'] = x_step
    y_kw: dict = dict(range=[y_min - _EPS, y_max + _EPS], tick0=y_max)
    if y_step is not None:
        y_kw['dtick'] = y_step

    fig.add_hline(y=y_max, line=_BL)
    fig.add_hline(y=y_min, line=_BL)
    fig.add_vline(x=x_min, line=_BL)
    fig.add_vline(x=x_max, line=_BL)
    fig.update_layout(
        title=dict(text="Mode Conversion", x=0.5, xanchor="center", pad=dict(t=20)),
        xaxis_title="Frequency (GHz)",
        yaxis_title="Magnitude (dB)",
        xaxis=_xaxis(**x_kw),
        yaxis=_yaxis(**y_kw),
        **_LAYOUT,
    )
    return fig


def plot_impedance(datasets: list,
                   rise_time_ps: float = 35.0,
                   show_zdiff: bool = True, show_zse: bool = False,
                   show_forward: bool = True, show_reverse: bool = False,
                   x_min: float = 0, x_max: float = 1000, x_step: float = None,
                   y_min: float = 0, y_max: float = 150, y_step: float = None) -> go.Figure:
    multi = len(datasets) > 1
    x_kw: dict = dict(range=[x_min - _EPS, x_max + _EPS])
    if x_step is not None:
        x_kw['dtick'] = x_step
    y_kw: dict = dict(range=[y_min - _EPS, y_max + _EPS], tick0=y_min)
    if y_step is not None:
        y_kw['dtick'] = y_step

    fig = go.Figure()
    for idx, ds in enumerate(datasets):
        ca, cb = _FILE_COLORS[idx % len(_FILE_COLORS)]
        t_fwd_ns = ds["t_fwd"] / 1000
        t_rev_ns = ds["t_rev"] / 1000
        if show_forward:
            if show_zse:
                fig.add_trace(go.Scatter(x=t_fwd_ns, y=ds["z11_fwd"],
                                         name=_trace_name(ds, "Z_SE Fwd", multi),
                                         line=dict(color=ca, dash="solid")))
            if show_zdiff:
                fig.add_trace(go.Scatter(x=t_fwd_ns, y=ds["zdiff_fwd"],
                                         name=_trace_name(ds, "Z_Diff Fwd", multi),
                                         line=dict(color=cb, dash="solid")))
        if show_reverse:
            if show_zse:
                fig.add_trace(go.Scatter(x=t_rev_ns, y=ds["z11_rev"],
                                         name=_trace_name(ds, "Z_SE Rev", multi),
                                         line=dict(color=ca, dash="dash")))
            if show_zdiff:
                fig.add_trace(go.Scatter(x=t_rev_ns, y=ds["zdiff_rev"],
                                         name=_trace_name(ds, "Z_Diff Rev", multi),
                                         line=dict(color=cb, dash="dash")))

    fig.add_hline(y=y_min, line=_BL)
    fig.add_hline(y=y_max, line=_BL)
    fig.add_vline(x=x_min, line=_BL)
    fig.add_vline(x=x_max, line=_BL)
    fig.update_layout(
        title=dict(text=f"Impedance@Tr={rise_time_ps}ps(20%~80%)", x=0.5, xanchor="center", pad=dict(t=20)),
        xaxis_title="Time (ns)",
        yaxis_title="Impedance (Ω)",
        xaxis=_xaxis(**x_kw),
        yaxis=_yaxis(**y_kw),
        **_LAYOUT,
    )
    return fig
