"""Reusable presentation components for the Student AutoML dashboard."""

from __future__ import annotations

from html import escape
from typing import Sequence

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    px = None
    go = None


NAVIGATION = (
    "Dataset Overview",
    "Experiment Setup",
    "Model Training & Comparison",
    "Evaluation Results",
    "Explainable AI",
    "Saved Experiments",
    "Rigorous Validation",
)

_PLOTLY_TEMPLATE = "plotly_white"
_COLORS = ("#2563eb", "#0f9f9a", "#64748b", "#7c3aed", "#f59e0b", "#dc5470")


def apply_dashboard_style() -> None:
    """Apply a consistent navy, white, and blue visual system."""
    st.markdown(
        """
        <style>
        :root {
            --sa-navy: #10233f;
            --sa-blue: #2563eb;
            --sa-teal: #0f9f9a;
            --sa-ink: #172b4d;
            --sa-muted: #64748b;
            --sa-border: #e2e8f0;
            --sa-surface: #ffffff;
            --sa-canvas: #f4f7fb;
        }
        html, body, [class*="css"] {
            font-family: Inter, "Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--sa-ink);
        }
        .stApp {
            background: var(--sa-canvas);
        }
        [data-testid="stHeader"] {
            background: rgba(244, 247, 251, 0.9);
        }
        [data-testid="stSidebar"] {
            background: var(--sa-navy);
            border-right: 1px solid rgba(255,255,255,.08);
        }
        [data-testid="stSidebar"] > div:first-child {
            background: var(--sa-navy);
        }
        [data-testid="stSidebar"] * {
            color: #e7eefb;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            border-radius: 9px;
            padding: .18rem .35rem;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background: rgba(255,255,255,.08);
        }
        [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
            background: rgba(255,255,255,.06);
            border: 1px dashed rgba(231,238,251,.35);
        }
        [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
            color: var(--sa-navy);
        }
        .main .block-container {
            max-width: 1440px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        h1, h2, h3 {
            color: var(--sa-ink);
            letter-spacing: -.025em;
        }
        h1 { font-size: 2rem !important; font-weight: 720 !important; }
        h2 { font-size: 1.35rem !important; font-weight: 680 !important; }
        h3 { font-size: 1.02rem !important; font-weight: 650 !important; }
        [data-testid="stMetric"] {
            background: var(--sa-surface);
            border: 1px solid var(--sa-border);
            border-radius: 14px;
            padding: 1rem 1.05rem;
            box-shadow: 0 4px 14px rgba(16, 35, 63, .045);
        }
        [data-testid="stMetricLabel"] {
            color: var(--sa-muted);
            font-size: .82rem;
        }
        [data-testid="stMetricValue"] {
            color: var(--sa-ink);
            font-size: 1.65rem;
            font-weight: 700;
        }
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            border: 1px solid var(--sa-border);
            border-radius: 12px;
            overflow: hidden;
            background: white;
        }
        [data-testid="stPlotlyChart"], [data-testid="stVegaLiteChart"] {
            border: 1px solid var(--sa-border);
            border-radius: 14px;
            background: white;
            padding: .35rem;
        }
        [data-testid="stAlert"] {
            border-radius: 12px;
        }
        .sa-brand {
            display: flex;
            align-items: center;
            gap: .75rem;
            padding: .35rem .15rem 1.15rem;
            margin-bottom: .65rem;
            border-bottom: 1px solid rgba(231,238,251,.15);
        }
        .sa-mark {
            display: grid;
            place-items: center;
            flex: 0 0 42px;
            height: 42px;
            border: 1px solid rgba(255,255,255,.22);
            border-radius: 12px;
            background: rgba(37,99,235,.25);
            color: white;
            font-size: .86rem;
            font-weight: 750;
            letter-spacing: .04em;
        }
        .sa-brand-title {
            color: #fff;
            font-size: 1rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .sa-brand-subtitle {
            color: #a9bad3;
            font-size: .72rem;
            margin-top: .2rem;
        }
        .sa-kicker {
            color: var(--sa-blue);
            font-size: .76rem;
            font-weight: 700;
            letter-spacing: .09em;
            text-transform: uppercase;
            margin-bottom: .3rem;
        }
        .sa-page-description {
            color: var(--sa-muted);
            font-size: .96rem;
            line-height: 1.55;
            max-width: 900px;
            margin: -.25rem 0 1.25rem;
        }
        .sa-hero {
            background: linear-gradient(120deg, #10233f 0%, #17365e 100%);
            border: 1px solid rgba(255,255,255,.08);
            border-radius: 18px;
            padding: 1.7rem 1.85rem;
            color: white;
            box-shadow: 0 12px 30px rgba(16,35,63,.12);
            margin: .25rem 0 1.35rem;
        }
        .sa-hero .sa-kicker { color: #8edbd7; }
        .sa-hero h2 {
            color: white;
            font-size: 1.55rem !important;
            line-height: 1.3;
            margin: .25rem 0 .55rem;
        }
        .sa-hero p {
            color: #d3deec;
            font-size: .94rem;
            line-height: 1.55;
            margin: 0;
            max-width: 880px;
        }
        .sa-panel {
            background: white;
            border: 1px solid var(--sa-border);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            box-shadow: 0 4px 14px rgba(16,35,63,.035);
            margin: .25rem 0 .85rem;
        }
        .sa-panel-label {
            color: var(--sa-muted);
            font-size: .76rem;
            font-weight: 650;
            letter-spacing: .055em;
            text-transform: uppercase;
            margin-bottom: .4rem;
        }
        .sa-panel-value {
            color: var(--sa-ink);
            font-size: 1.1rem;
            font-weight: 650;
            line-height: 1.45;
        }
        .sa-panel-note {
            color: var(--sa-muted);
            font-size: .84rem;
            line-height: 1.45;
            margin-top: .35rem;
        }
        .sa-status {
            display: inline-flex;
            align-items: center;
            gap: .4rem;
            border: 1px solid #bfe9df;
            border-radius: 999px;
            background: #edf9f6;
            color: #087c72;
            font-size: .77rem;
            font-weight: 650;
            padding: .32rem .65rem;
        }
        div.stButton > button[kind="primary"],
        div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
            border-radius: 9px;
            font-weight: 650;
            box-shadow: 0 4px 12px rgba(37,99,235,.16);
        }
        div[data-testid="stTabs"] { display: none !important; }
        @media (max-width: 760px) {
            .main .block-container { padding: 1rem .85rem 2rem; }
            h1 { font-size: 1.65rem !important; }
            .sa-hero { padding: 1.25rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_brand() -> None:
    st.sidebar.markdown(
        """
        <div class="sa-brand">
            <div class="sa-mark">SA</div>
            <div>
                <div class="sa-brand-title">Student AutoML</div>
                <div class="sa-brand-subtitle">Academic analytics platform</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(section: str, title: str, description: str) -> None:
    st.markdown(f'<div class="sa-kicker">{escape(section)}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(
        f'<div class="sa-page-description">{escape(description)}</div>',
        unsafe_allow_html=True,
    )


def render_hero(title: str, description: str, kicker: str = "STUDENT AUTOML") -> None:
    st.markdown(
        f"""
        <section class="sa-hero">
            <div class="sa-kicker">{escape(kicker)}</div>
            <h2>{escape(title)}</h2>
            <p>{escape(description)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_panel(label: str, value: str, note: str | None = None) -> None:
    note_markup = f'<div class="sa-panel-note">{escape(note)}</div>' if note else ""
    st.markdown(
        f"""
        <div class="sa-panel">
            <div class="sa-panel-label">{escape(label)}</div>
            <div class="sa-panel-value">{escape(value)}</div>
            {note_markup}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(items: Sequence[tuple[str, str]], columns: int = 4) -> None:
    slots = st.columns(min(columns, len(items)))
    for index, (label, value) in enumerate(items):
        with slots[index % len(slots)]:
            st.metric(label, value)


def render_bar_chart(
    frame: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    horizontal: bool = False,
    color: str | None = None,
    height: int = 340,
) -> None:
    """Render a Plotly chart when installed, with a built-in chart fallback."""
    if frame.empty or x not in frame.columns or y not in frame.columns:
        st.info("There are no chartable results for this view.")
        return
    if px is not None:
        if horizontal:
            fig = px.bar(
                frame,
                x=y,
                y=x,
                orientation="h",
                title=title,
                color=color,
                color_discrete_sequence=list(_COLORS),
            )
        else:
            fig = px.bar(
                frame,
                x=x,
                y=y,
                title=title,
                color=color,
                color_discrete_sequence=list(_COLORS),
            )
        fig.update_layout(
            template=_PLOTLY_TEMPLATE,
            height=height,
            margin=dict(l=12, r=18, t=55, b=20),
            font=dict(family="Inter, Segoe UI, sans-serif", color="#172b4d", size=12),
            title_font_size=15,
            showlegend=bool(color),
            xaxis_title=None,
            yaxis_title=None,
        )
        fig.update_xaxes(showgrid=True, gridcolor="#edf1f6", zeroline=False)
        fig.update_yaxes(showgrid=False, automargin=True)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        return
    st.caption(title)
    chart_frame = frame.set_index(x)[[y]]
    st.bar_chart(chart_frame, horizontal=horizontal, height=height)


def render_confusion_matrix(
    matrix: list[list[int]],
    labels: Sequence[str],
) -> None:
    if go is not None and px is not None:
        frame = pd.DataFrame(
            matrix,
            index=[f"Actual {label}" for label in labels],
            columns=[f"Predicted {label}" for label in labels],
        )
        fig = px.imshow(
            frame,
            text_auto=True,
            color_continuous_scale=["#eff6ff", "#93c5fd", "#2563eb", "#10233f"],
            aspect="auto",
        )
        fig.update_layout(
            template=_PLOTLY_TEMPLATE,
            height=320,
            margin=dict(l=10, r=10, t=25, b=10),
            coloraxis_showscale=False,
            font=dict(family="Inter, Segoe UI, sans-serif", color="#172b4d", size=12),
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        return
    st.dataframe(
        pd.DataFrame(
            matrix,
            index=[f"Actual {label}" for label in labels],
            columns=[f"Predicted {label}" for label in labels],
        ),
        width="stretch",
    )


def render_line_chart(
    frame: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    height: int = 320,
) -> None:
    if frame.empty or x not in frame.columns or y not in frame.columns:
        st.info("There are no chartable results for this view.")
        return
    if px is not None:
        fig = px.line(
            frame,
            x=x,
            y=y,
            title=title,
            color_discrete_sequence=[_COLORS[0]],
        )
        fig.update_layout(
            template=_PLOTLY_TEMPLATE,
            height=height,
            margin=dict(l=12, r=18, t=55, b=20),
            font=dict(family="Inter, Segoe UI, sans-serif", color="#172b4d", size=12),
            title_font_size=15,
            xaxis_title=None,
            yaxis_title=None,
        )
        fig.update_xaxes(showgrid=True, gridcolor="#edf1f6", zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor="#edf1f6", zeroline=False)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        return
    st.caption(title)
    st.line_chart(frame.set_index(x)[[y]], height=height)


def render_scatter_chart(
    frame: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    height: int = 340,
) -> None:
    if frame.empty or x not in frame.columns or y not in frame.columns:
        st.info("There are no chartable results for this view.")
        return
    if px is not None:
        fig = px.scatter(
            frame,
            x=x,
            y=y,
            title=title,
            color_discrete_sequence=[_COLORS[1]],
            opacity=0.72,
        )
        fig.update_layout(
            template=_PLOTLY_TEMPLATE,
            height=height,
            margin=dict(l=12, r=18, t=55, b=20),
            font=dict(family="Inter, Segoe UI, sans-serif", color="#172b4d", size=12),
            title_font_size=15,
            xaxis_title=None,
            yaxis_title=None,
        )
        fig.update_xaxes(showgrid=True, gridcolor="#edf1f6", zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor="#edf1f6", zeroline=False)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        return
    st.caption(title)
    st.scatter_chart(frame, x=x, y=y, height=height)
