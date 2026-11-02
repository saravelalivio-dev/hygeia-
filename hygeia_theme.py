"""Colours, fonts, CSS and page layout helpers for the HYGEIA site."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# --- Tokens ---

# Ink
INK = "#241018"            # near-black plum, body text
INK_SOFT = "#5E4350"       # secondary text
INK_MUTED = "#8B7280"      # meta, axis labels
INK_INVERSE = "#FFFFFF"

# Brand
CRIMSON = "#C2185B"        # primary
CRIMSON_DEEP = "#8E0F3F"   # pressed, headings on light
PLUM_DARK = "#2E0F1D"      # hero and footer ground
PLUM_MID = "#4A1530"       # gradient partner
ROSE_LIGHT = "#F06FA3"     # highlight on dark
BRASS = "#C79A3C"          # classical accent, the serpent and rules
BRASS_LIGHT = "#E6C878"

# Surfaces
SURFACE = "#FFFFFF"
SURFACE_TINT = "#FDF5F8"   # faint, not a wash
PETAL = "#F2CEDC"          # borders on light
BLUSH = "#FCE8F0"          # chip fills
HAIRLINE = "#EADDE4"

# Older names, kept so the rest of the code still works.
ROSE = CRIMSON
ROSE_DEEP = CRIMSON_DEEP

# Chart series colours, used in this order. Checked for colourblind safety.
CATEGORICAL = [
    "#C2185B",  # crimson  - primary series, brand hue
    "#0FA3A0",  # teal     - contrast partner
    "#7B3FB8",  # violet
    "#B06F12",  # bronze
    "#3D56C4",  # indigo
    "#E0457F",  # rose
]

# Light to dark ramp, for charts that show amounts.
ROSE_SEQUENTIAL = [
    "#FCE2EC", "#F7BBD0", "#EE87AC", "#DE4C84", "#C2185B", "#7E0B37",
]

# For values that can be positive or negative, e.g. SHAP contributions.
DIVERGING = ["#0FA3A0", "#86CFCD", "#ECE6E9", "#EE87AC", "#C2185B"]

# Status colours. Always shown with a text label too, not colour alone.
STATUS = {
    "good": "#157F55",
    "warning": "#A8650F",
    "serious": "#C2185B",
    "critical": "#7E0B37",
}

GRID = "#F3DEE7"

DISPLAY_FONT = "'Playfair Display', Georgia, 'Times New Roman', serif"
BODY_FONT = "'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif"
FONT_STACK = BODY_FONT  # charts use the body face


# --- Logo: the Bowl of Hygieia ---

def bowl_of_hygieia(size: int = 42, bowl: str = BRASS_LIGHT,
                    serpent: str = "#FFFFFF", eye: str = CRIMSON_DEEP) -> str:
    """Bowl of Hygieia as an SVG, drawn so it still reads at masthead size."""
    return f"""
<svg width="{size}" height="{size}" viewBox="0 0 64 64" fill="none"
     xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="The Bowl of Hygieia">
  <!-- bowl -->
  <path d="M12 24 H52 C52 37 44 44 32 44 C20 44 12 37 12 24 Z"
        fill="none" stroke="{bowl}" stroke-width="3.2" stroke-linejoin="round"/>
  <!-- stem and foot -->
  <path d="M32 44 v7" stroke="{bowl}" stroke-width="3.2" stroke-linecap="round"/>
  <path d="M20 56 H44" stroke="{bowl}" stroke-width="3.2" stroke-linecap="round"/>
  <!-- serpent: coils the foot and stem, rises through the bowl, head reared -->
  <path d="M44 55 C36 58 24 54 27 49 C29 46 38 46 40 42
           C42 37 36 33 38 27 C40 21 47 20 50 15"
        fill="none" stroke="{serpent}" stroke-width="3.4"
        stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="51" cy="13.5" r="3.4" fill="{serpent}"/>
  <circle cx="52.3" cy="12.6" r="1" fill="{eye}"/>
</svg>
"""


# --- Plotly Template ---

def register_template(name: str = "hygeia_bold") -> str:
    """Set up the chart styling that every plot inherits."""
    template = go.layout.Template()
    template.layout = go.Layout(
        colorway=CATEGORICAL,
        font=dict(family=BODY_FONT, size=13, color=INK),
        title=dict(font=dict(family=DISPLAY_FONT, size=17, color=CRIMSON_DEEP),
                   x=0.01, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=64, r=28, t=56, b=52),
        xaxis=dict(
            gridcolor=GRID, zerolinecolor=PETAL, linecolor=PETAL,
            tickfont=dict(color=INK_MUTED, size=12),
            title=dict(font=dict(color=INK_SOFT, size=12.5)),
        ),
        yaxis=dict(
            gridcolor=GRID, zerolinecolor=PETAL, linecolor=PETAL,
            tickfont=dict(color=INK_MUTED, size=12),
            title=dict(font=dict(color=INK_SOFT, size=12.5)),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.82)", bordercolor=PETAL, borderwidth=1,
            font=dict(size=12, color=INK), orientation="h",
            yanchor="bottom", y=1.02, xanchor="right", x=1,
        ),
        hoverlabel=dict(
            bgcolor=SURFACE, bordercolor=CRIMSON,
            font=dict(family=BODY_FONT, size=12, color=INK),
        ),
        colorscale=dict(sequential=ROSE_SEQUENTIAL, diverging=DIVERGING),
    )
    pio.templates[name] = template
    pio.templates.default = name
    return name


def style_3d(fig: go.Figure) -> go.Figure:
    """Style a 3-D plot. Plotly templates don't cover 3-D scenes."""
    axis = dict(
        backgroundcolor="rgba(253,245,248,0.65)",
        gridcolor=GRID, zerolinecolor=PETAL, showbackground=True,
        tickfont=dict(color=INK_MUTED, size=10),
        title=dict(font=dict(color=INK_SOFT, size=12)),
    )
    fig.update_layout(scene=dict(xaxis=axis, yaxis=axis, zaxis=axis))
    return fig


# --- Css ---

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800;900&family=Inter:wght@400;500;600;700&display=swap');

/* ---------- Strip Streamlit's chrome ---------- */
#MainMenu, footer, header[data-testid="stHeader"] {{ display: none !important; }}
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] {{ display: none !important; }}

html, body, [class*="css"], .stMarkdown, p, li, td, th, label, input, select {{
    font-family: {BODY_FONT};
    color: {INK};
}}
h1, h2, h3, h4 {{ font-family: {DISPLAY_FONT}; }}

.stApp {{ background: {SURFACE_TINT}; }}

/* Streamlit puts every element side by side in .block-container, so a
   wrapper div won't contain anything. The container is the white page, and
   the dark bands use negative margins to reach the edges. */
.block-container {{
    padding: 0 48px 0 48px !important;
    max-width: 1300px;
    background: {SURFACE};
    box-shadow: 0 0 0 1px {HAIRLINE};
}}
.hy-utility, .hy-masthead, .hy-hero, .hy-statband,
.hy-footer, .hy-colophon {{ margin-left: -48px; margin-right: -48px; }}

/* Breathing room between major sections. */
.hy-gap {{ height: 30px; }}

/* ---------- Utility bar ---------- */
.hy-utility {{
    background: {PLUM_DARK};
    color: rgba(255,255,255,0.72);
    font-size: 0.73rem; font-weight: 500; letter-spacing: 0.05em;
    padding: 8px 24px;
    border-radius: 0 0 4px 4px;
    display: flex; gap: 18px; align-items: center; flex-wrap: wrap;
}}
.hy-utility .sep {{ color: {BRASS}; }}

/* ---------- Masthead ---------- */
.hy-masthead {{
    background: {SURFACE};
    border-bottom: 3px solid {CRIMSON};
    padding: 18px 26px;
    display: flex; align-items: center; gap: 16px;
    box-shadow: 0 1px 0 {HAIRLINE};
}}
.hy-mark {{
    width: 54px; height: 54px; flex: none; border-radius: 6px;
    background: linear-gradient(150deg, {CRIMSON_DEEP} 0%, {CRIMSON} 100%);
    display: flex; align-items: center; justify-content: center;
}}
.hy-wordmark {{
    font-family: {DISPLAY_FONT}; font-weight: 900; font-size: 2rem;
    letter-spacing: 0.1em; color: {CRIMSON_DEEP}; line-height: 1;
}}
.hy-tagline {{
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.26em;
    color: {BRASS}; text-transform: uppercase; margin-top: 7px;
}}
.hy-masthead-right {{
    margin-left: auto; text-align: right;
    font-size: 0.76rem; color: {INK_MUTED}; line-height: 1.6;
}}
.hy-masthead-right strong {{ color: {INK}; font-weight: 600; }}

/* ---------- Hero ---------- */
.hy-hero {{
    background:
      linear-gradient(104deg, {PLUM_DARK} 0%, {PLUM_MID} 52%, {CRIMSON_DEEP} 100%);
    color: {INK_INVERSE};
    padding: 56px 52px 50px 52px;
    position: relative; overflow: hidden;
}}
.hy-hero:before {{
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(760px 300px at 86% 8%,
                rgba(194,24,91,0.46) 0%, rgba(194,24,91,0) 68%);
}}
.hy-hero > * {{ position: relative; }}
.hy-hero .eyebrow {{
    font-size: 0.71rem; font-weight: 700; letter-spacing: 0.3em;
    text-transform: uppercase; color: {BRASS_LIGHT}; margin-bottom: 20px;
}}
.hy-hero h1 {{
    font-family: {DISPLAY_FONT}; font-weight: 800;
    font-size: clamp(2.1rem, 4.4vw, 3.5rem); line-height: 1.08;
    margin: 0 0 20px 0; color: {INK_INVERSE}; max-width: 17ch;
}}
.hy-hero h1 em {{ font-style: italic; color: {ROSE_LIGHT}; }}
.hy-hero .sub {{
    font-size: 1.1rem; line-height: 1.68; max-width: 60ch;
    color: rgba(255,255,255,0.93); font-weight: 400;
}}
.hy-hero .byline {{
    margin-top: 28px; padding-top: 20px;
    border-top: 1px solid rgba(255,255,255,0.22);
    font-size: 0.84rem; color: rgba(255,255,255,0.8); max-width: 64ch;
    line-height: 1.7;
}}
.hy-hero .byline b {{ color: {INK_INVERSE}; font-weight: 600; }}

/* ---------- Stat band ---------- */
.hy-statband {{
    background: {PLUM_DARK};
    display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    border-top: 3px solid {BRASS};
}}
.hy-statband .cell {{
    padding: 22px 20px; border-right: 1px solid rgba(255,255,255,0.1);
}}
.hy-statband .cell:last-child {{ border-right: none; }}
.hy-statband .v {{
    font-family: {DISPLAY_FONT}; font-weight: 800; font-size: 1.85rem;
    color: {INK_INVERSE}; line-height: 1.05;
}}
.hy-statband .k {{
    font-size: 0.68rem; letter-spacing: 0.13em; text-transform: uppercase;
    color: {BRASS_LIGHT}; font-weight: 700; margin-top: 8px;
}}

/* ---------- Pillars ---------- */
.hy-pillars {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 0; border: 1px solid {HAIRLINE}; border-radius: 3px; overflow: hidden;
}}
.hy-pillar {{
    padding: 28px 26px; background: {SURFACE};
    border-right: 1px solid {HAIRLINE};
}}
.hy-pillar:last-child {{ border-right: none; }}
.hy-pillar .num {{
    font-family: {DISPLAY_FONT}; font-size: 2.4rem; font-weight: 800;
    color: #EE87AC; line-height: 1; margin-bottom: 10px;
}}
.hy-pillar h4 {{
    font-size: 1.12rem; font-weight: 700; color: {CRIMSON_DEEP};
    margin: 0 0 9px 0; line-height: 1.3;
}}
.hy-pillar p {{
    font-size: 0.92rem; line-height: 1.68; color: {INK_SOFT}; margin: 0;
}}

/* ---------- Goddess panel ---------- */
.hy-goddess {{
    display: grid; grid-template-columns: 136px 1fr; gap: 30px;
    align-items: start;
    background: linear-gradient(100deg, {PLUM_DARK} 0%, {PLUM_MID} 100%);
    color: {INK_INVERSE}; padding: 36px 40px; border-radius: 3px;
    border-left: 5px solid {BRASS};
}}
.hy-goddess .emblem {{
    display: flex; align-items: center; justify-content: center;
    padding: 20px; border: 1px solid rgba(199,154,60,0.45); border-radius: 3px;
    background: rgba(0,0,0,0.18);
}}
.hy-goddess .eyebrow {{
    font-size: 0.69rem; font-weight: 700; letter-spacing: 0.26em;
    text-transform: uppercase; color: {BRASS_LIGHT}; margin-bottom: 11px;
}}
.hy-goddess h3 {{
    font-family: {DISPLAY_FONT}; font-size: 1.75rem; font-weight: 800;
    color: {INK_INVERSE}; margin: 0 0 13px 0; line-height: 1.22;
}}
.hy-goddess p {{
    font-size: 0.96rem; line-height: 1.74; color: rgba(255,255,255,0.9);
    margin: 0 0 11px 0; max-width: 72ch;
}}
.hy-goddess b {{ color: {BRASS_LIGHT}; font-weight: 600; }}

/* ---------- Section headings ---------- */
.hy-h2 {{
    font-family: {DISPLAY_FONT}; font-size: 2rem; font-weight: 800;
    color: {CRIMSON_DEEP}; margin: 8px 0 6px 0; line-height: 1.18;
}}
.hy-h2 small {{
    display: block; font-family: {BODY_FONT}; font-size: 0.73rem;
    font-weight: 700; color: {INK_MUTED}; letter-spacing: 0.15em;
    text-transform: uppercase; margin-top: 10px;
}}
.hy-lede {{
    font-size: 1.04rem; line-height: 1.74; color: {INK_SOFT};
    max-width: 76ch; margin: 12px 0 18px 0;
}}
.hy-rule {{ display: flex; align-items: center; gap: 13px; margin: 30px 0 15px 0; }}
.hy-rule span {{
    font-size: 0.71rem; text-transform: uppercase; letter-spacing: 0.18em;
    color: {CRIMSON_DEEP}; font-weight: 700; white-space: nowrap;
}}
.hy-rule:after {{
    content: ''; flex: 1; height: 2px;
    background: linear-gradient(90deg, {CRIMSON} 0%, {PETAL} 28%, rgba(242,206,220,0) 100%);
}}

/* ---------- Cards ---------- */
.hy-card {{
    background: {SURFACE}; border: 1px solid {HAIRLINE};
    border-top: 3px solid {CRIMSON};
    border-radius: 3px; padding: 22px 24px; margin-bottom: 14px;
}}
.hy-card h3 {{
    margin: 0 0 10px 0; font-size: 1.1rem; font-weight: 700;
    color: {CRIMSON_DEEP}; font-family: {DISPLAY_FONT};
}}
.hy-card p {{ color: {INK_SOFT}; font-size: 0.92rem; line-height: 1.7; margin: 0.5rem 0; }}
.hy-card a {{ color: {CRIMSON}; font-weight: 600; text-decoration: none; }}
.hy-card a:hover {{ text-decoration: underline; }}

.hy-verdict {{
    background: {PLUM_DARK}; color: {INK_INVERSE};
    border-left: 5px solid {BRASS}; border-radius: 3px;
    padding: 24px 26px; margin-bottom: 14px;
}}
.hy-verdict .big {{
    font-family: {DISPLAY_FONT}; font-size: 3rem; font-weight: 800;
    color: {INK_INVERSE}; line-height: 1; margin: 6px 0;
}}
.hy-verdict .cap {{
    font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.16em;
    color: {BRASS_LIGHT}; font-weight: 700;
}}

.hy-seq {{
    font-family: 'SFMono-Regular', 'Consolas', ui-monospace, monospace;
    font-size: 0.87rem; letter-spacing: 0.11em; word-break: break-all;
    background: {SURFACE_TINT}; border: 1px solid {PETAL};
    border-left: 3px solid {CRIMSON};
    border-radius: 2px; padding: 12px 15px; color: {CRIMSON_DEEP};
    line-height: 1.75;
}}

/* ---------- Chips ---------- */
.chip {{
    display: inline-block; padding: 5px 13px; border-radius: 2px;
    font-size: 0.72rem; font-weight: 700; margin: 3px 5px 3px 0;
    letter-spacing: 0.03em;
    border: 1px solid {PETAL}; background: {BLUSH}; color: {CRIMSON_DEEP};
}}
.chip-pass {{ background: #E2F4EC; border-color: #A8DCC4; color: #0F5C3E; }}
.chip-fail {{ background: #FBE0E9; border-color: #F0B6CC; color: #7E0B37; }}
.chip-warn {{ background: #FAEBD8; border-color: #EDCCA0; color: #7A3D0B; }}
.chip-lead {{ background: {CRIMSON}; border-color: {CRIMSON_DEEP}; color: #FFFFFF; }}

/* ---------- Tabs as a navigation bar ---------- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0; background: {SURFACE};
    border-bottom: 2px solid {PETAL}; border-radius: 0;
    padding: 0; position: sticky; top: 0; z-index: 99; overflow-x: auto;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent; border: none; border-radius: 0;
    padding: 15px 24px; font-weight: 600; font-size: 0.9rem;
    color: {INK_SOFT}; white-space: nowrap;
    border-bottom: 3px solid transparent; margin-bottom: -2px;
}}
.stTabs [data-baseweb="tab"]:hover {{ color: {CRIMSON_DEEP}; background: {SURFACE_TINT}; }}
.stTabs [aria-selected="true"] {{
    color: {CRIMSON_DEEP} !important; font-weight: 700;
    border-bottom: 3px solid {CRIMSON} !important;
    background: transparent !important;
}}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {{ display: none; }}

/* ---------- Metrics ---------- */
[data-testid="stMetric"] {{
    background: {SURFACE}; border: 1px solid {HAIRLINE};
    border-left: 3px solid {CRIMSON};
    border-radius: 2px; padding: 15px 18px;
}}
[data-testid="stMetricLabel"] {{
    color: {INK_MUTED} !important; font-weight: 700; font-size: 0.72rem;
    letter-spacing: 0.08em; text-transform: uppercase;
}}
[data-testid="stMetricValue"] {{
    color: {CRIMSON_DEEP} !important; font-weight: 800;
    font-family: {DISPLAY_FONT};
}}
[data-testid="stMetricDelta"] {{ font-size: 0.76rem; }}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {{
    background: {SURFACE}; border-right: 1px solid {HAIRLINE};
}}
[data-testid="stSidebar"] .block-container {{ padding-top: 1.6rem; }}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
    color: {CRIMSON_DEEP}; font-family: {DISPLAY_FONT};
}}

/* ---------- Controls ---------- */
.stButton > button {{
    background: {CRIMSON}; color: #FFFFFF; border: none; border-radius: 2px;
    padding: 0.72rem 1.7rem; font-weight: 700; font-size: 0.88rem;
    font-family: {BODY_FONT}; letter-spacing: 0.05em; text-transform: uppercase;
    transition: background 0.15s ease, transform 0.12s ease;
}}
.stButton > button:hover {{ background: {CRIMSON_DEEP}; transform: translateY(-1px); }}
.stButton > button:focus {{ box-shadow: 0 0 0 3px {PETAL} !important; }}
.stSlider [data-baseweb="slider"] [role="slider"] {{ background: {CRIMSON} !important; }}
.stExpander {{
    border: 1px solid {HAIRLINE} !important; border-radius: 2px !important;
    background: {SURFACE_TINT};
}}
.stDataFrame {{ border: 1px solid {HAIRLINE}; border-radius: 2px; overflow: hidden; }}
.stAlert {{ border-radius: 2px; border-left-width: 4px; }}
[data-testid="stFileUploader"] {{
    background: {SURFACE_TINT}; border: 1px dashed {PETAL};
    border-radius: 2px; padding: 10px;
}}

/* ---------- Footer ---------- */
.hy-footer {{
    background: {PLUM_DARK}; color: rgba(255,255,255,0.82);
    padding: 44px 48px 34px 48px;
    border-top: 3px solid {BRASS};
    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 34px;
}}
.hy-footer h4 {{
    font-family: {BODY_FONT};
    font-size: 0.69rem; text-transform: uppercase; letter-spacing: 0.17em;
    color: {BRASS_LIGHT}; margin: 0 0 13px 0; font-weight: 700;
}}
.hy-footer p, .hy-footer li {{
    font-size: 0.86rem; color: rgba(255,255,255,0.78); line-height: 1.72;
    margin: 0 0 5px 0;
}}
.hy-footer ul {{ list-style: none; padding: 0; margin: 0; }}
.hy-footer a {{ color: {ROSE_LIGHT}; text-decoration: none; font-weight: 500; }}
.hy-footer a:hover {{ text-decoration: underline; color: {BRASS_LIGHT}; }}
.hy-footer b {{ color: {INK_INVERSE}; font-weight: 600; }}
.hy-colophon {{
    background: #1C0912; color: rgba(255,255,255,0.56);
    text-align: center; font-size: 0.76rem; letter-spacing: 0.04em;
    padding: 18px 20px;
}}
.hy-colophon .brass {{ color: {BRASS}; font-weight: 600; letter-spacing: 0.2em; }}

/* ---------- Narrow screens ---------- */
@media (max-width: 820px) {{
    .block-container {{ padding: 0 20px 0 20px !important; }}
    .hy-utility, .hy-masthead, .hy-hero, .hy-statband,
    .hy-footer, .hy-colophon {{ margin-left: -20px; margin-right: -20px; }}
    .hy-hero {{ padding: 36px 24px 32px 24px; }}
    .hy-masthead {{ flex-wrap: wrap; padding: 16px 20px; }}
    .hy-masthead-right {{ margin-left: 0; text-align: left; width: 100%; margin-top: 10px; }}
    .hy-goddess {{ grid-template-columns: 1fr; padding: 26px 22px; }}
    .hy-footer {{ padding: 32px 22px 26px 22px; }}
}}
</style>
"""


def inject(st) -> None:
    """Inject the stylesheet and activate the Plotly template. Call once, first."""
    st.markdown(CSS, unsafe_allow_html=True)
    register_template()


# --- Page Furniture ---

def utility_bar(st, items: list[str]) -> None:
    """Thin dark strip at the very top of the page."""
    inner = f' <span class="sep">|</span> '.join(items)
    st.markdown(f'<div class="hy-utility">{inner}</div>', unsafe_allow_html=True)


def masthead(st, wordmark: str, tagline: str, right_html: str = "") -> None:
    """Logo mark, wordmark, tagline, and a right-hand identification block."""
    st.markdown(
        f"""
        <div class="hy-masthead">
            <div class="hy-mark">{bowl_of_hygieia(34)}</div>
            <div>
                <div class="hy-wordmark">{wordmark}</div>
                <div class="hy-tagline">{tagline}</div>
            </div>
            <div class="hy-masthead-right">{right_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero(st, eyebrow: str, headline: str, sub: str, byline: str) -> None:
    """Hero banner. Put <em> around a word in the headline to highlight it."""
    st.markdown(
        f"""
        <div class="hy-hero">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{headline}</h1>
            <div class="sub">{sub}</div>
            <div class="byline">{byline}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_band(st, stats: list[tuple[str, str]]) -> None:
    """Row of key numbers under the hero. Each entry is (value, label)."""
    cells = "".join(
        f'<div class="cell"><div class="v">{v}</div><div class="k">{k}</div></div>'
        for v, k in stats
    )
    st.markdown(f'<div class="hy-statband">{cells}</div>', unsafe_allow_html=True)


def pillars(st, items: list[tuple[str, str, str]]) -> None:
    """Numbered explainer columns. Each entry is (number, heading, body)."""
    cells = "".join(
        f'<div class="hy-pillar"><div class="num">{n}</div>'
        f"<h4>{h}</h4><p>{b}</p></div>"
        for n, h, b in items
    )
    st.markdown(f'<div class="hy-pillars">{cells}</div>', unsafe_allow_html=True)


def goddess_panel(st, eyebrow: str, title: str, body_html: str) -> None:
    """Panel explaining the Hygeia name, with the emblem beside it."""
    st.markdown(
        f"""
        <div class="hy-goddess">
            <div class="emblem">{bowl_of_hygieia(84)}</div>
            <div>
                <div class="eyebrow">{eyebrow}</div>
                <h3>{title}</h3>
                {body_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def gap(st, height: int = 30) -> None:
    """Vertical breathing room between major sections."""
    st.markdown(f'<div class="hy-gap" style="height:{height}px"></div>',
                unsafe_allow_html=True)


def heading(st, title: str, subtitle: str = "", lede: str = "") -> None:
    """Heading, with the formal term underneath and an intro paragraph."""
    sub = f"<small>{subtitle}</small>" if subtitle else ""
    st.markdown(f'<div class="hy-h2">{title}{sub}</div>', unsafe_allow_html=True)
    if lede:
        st.markdown(f'<p class="hy-lede">{lede}</p>', unsafe_allow_html=True)


def lede(st, text: str) -> None:
    """Standalone lead paragraph."""
    st.markdown(f'<p class="hy-lede">{text}</p>', unsafe_allow_html=True)


def rule(st, label: str) -> None:
    """Labelled section divider."""
    st.markdown(f'<div class="hy-rule"><span>{label}</span></div>',
                unsafe_allow_html=True)


def card(st, title: str, body_html: str) -> None:
    """A bordered content card."""
    st.markdown(f'<div class="hy-card"><h3>{title}</h3>{body_html}</div>',
                unsafe_allow_html=True)


def chip(label: str, kind: str = "") -> str:
    """Return a pill span. ``kind`` in {'', 'pass', 'fail', 'warn', 'lead'}."""
    cls = f"chip chip-{kind}" if kind else "chip"
    return f'<span class="{cls}">{label}</span>'


def site_footer(st, columns: list[tuple[str, str]], colophon: str,
                tagline: str = "") -> None:
    """Footer with several columns."""
    cols = "".join(f"<div><h4>{h}</h4>{body}</div>" for h, body in columns)
    tag = f'<div class="brass">{tagline}</div>' if tagline else ""
    st.markdown(
        f'<div class="hy-footer">{cols}</div>'
        f'<div class="hy-colophon">{tag}{colophon}</div>',
        unsafe_allow_html=True,
    )
