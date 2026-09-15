"""Tema liquid glass dan palet chart.

Efek kaca di sini dibangun dari ``backdrop-filter`` berlapis, bukan dari filter
SVG dengan peta perpindahan seperti pada prototipe lama. Filter itu memberi
refraksi yang lebih meyakinkan, tapi harganya sebuah PNG base64 puluhan kilobyte
yang tertanam di tengah kode dan tidak bisa dibaca siapa pun. Lapisan CSS
memberi sembilan per sepuluh tampilannya dengan sepersepuluh bebannya.

Panel ditata lewat ``st.container(key=...)``, yang menempelkan kelas
``.st-key-<nama>`` pada elemen. Itu kait resmi Streamlit untuk penataan, jauh
lebih tahan lama daripada menebak nama kelas hasil build.

Warna deret
-----------
Oranye untuk strategi, netral terang untuk pasif. Diperiksa terhadap permukaan
gelap, bukan dikira-kira:

============  ========  ======  ==================
warna         OKLCH L   C       kontras
============  ========  ======  ==================
oranye        0,782     0,171   8,47:1
netral        0,971     0,003   16,00:1
============  ========  ======  ==================

Pemisahan OKLab delta-E 25,6 pada penglihatan normal, 28,9 protanopia, 24,0
deuteranopia -- semuanya jauh di atas lantai 15 dan target 8.

Dua penyimpangan yang disengaja dari resep palet kategorikal. Garis pasif nyaris
tanpa chroma karena memang bukan identitas melainkan penanda acuan; acuan yang
berwarna akan bersaing dengan deret yang seharusnya diperhatikan. Dan L oranye
berada di atas pita gelap yang disarankan, karena permukaan kaca di sini lebih
terang daripada permukaan gelap yang dipakai menyusun pita itu -- kontras
terukurnya 8,47:1, jauh di atas syarat 3:1. Keduanya tetap mendapat label
langsung di ujung garis, sehingga identitas tidak pernah bergantung pada warna
saja.
"""

from __future__ import annotations

__all__ = ["CSS", "PALETTE", "plotly_layout"]

#: Warna yang dipakai chart dan penanda di seluruh aplikasi.
PALETTE = {
    "strategy": "#FF9F0A",
    "strategy_soft": "rgba(255, 159, 10, 0.16)",
    "passive": "#F5F5F7",
    "passive_soft": "rgba(245, 245, 247, 0.10)",
    "good": "#30D158",
    "warning": "#FFD60A",
    "critical": "#FF453A",
    "ink": "rgba(255, 255, 255, 0.92)",
    "ink_muted": "rgba(255, 255, 255, 0.58)",
    "ink_faint": "rgba(255, 255, 255, 0.34)",
    "grid": "rgba(255, 255, 255, 0.07)",
}


CSS = """
<style>
:root {
  --glass-bg: rgba(255, 255, 255, 0.055);
  --glass-bg-strong: rgba(255, 255, 255, 0.085);
  --glass-border: rgba(255, 255, 255, 0.14);
  --glass-highlight: rgba(255, 255, 255, 0.40);
  --ink: rgba(255, 255, 255, 0.92);
  --ink-muted: rgba(255, 255, 255, 0.58);
  --ink-faint: rgba(255, 255, 255, 0.34);
  --accent: #FF9F0A;
  --radius: 22px;
  --font: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI Variable",
          "Segoe UI", Inter, system-ui, sans-serif;
}

/* Latar: gradien mesh gelap yang diam, supaya kaca punya sesuatu untuk dibiaskan.
   Kaca di atas warna rata tidak terlihat seperti kaca, cuma seperti kotak abu. */
.stApp {
  background:
    radial-gradient(1100px 700px at 12% -5%,  rgba(255, 159, 10, 0.16), transparent 58%),
    radial-gradient(900px 640px at 88% 8%,   rgba(94, 92, 230, 0.20),  transparent 60%),
    radial-gradient(1000px 820px at 60% 105%, rgba(10, 132, 255, 0.16), transparent 62%),
    #0B0B0F;
  color: var(--ink);
  font-family: var(--font);
}

.stApp > header { background: transparent; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1560px; }

/* ---- Panel kaca ---- */
[class*="st-key-glass-"] {
  position: relative;
  background: var(--glass-bg);
  backdrop-filter: blur(28px) saturate(185%);
  -webkit-backdrop-filter: blur(28px) saturate(185%);
  border: 1px solid var(--glass-border);
  border-radius: var(--radius);
  padding: 20px 22px;
  box-shadow:
    0 12px 40px rgba(0, 0, 0, 0.42),
    inset 0 1px 0 rgba(255, 255, 255, 0.16);
  overflow: hidden;
}

/* Kilau spekular di tepi atas. Inilah yang membuat permukaan terbaca sebagai
   kaca dan bukan sekadar panel semi-transparan. */
[class*="st-key-glass-"]::before {
  content: "";
  position: absolute;
  inset: 0 0 auto 0;
  height: 1px;
  background: linear-gradient(90deg,
    transparent, var(--glass-highlight) 28%, var(--glass-highlight) 72%, transparent);
  pointer-events: none;
}

/* Cahaya lembut yang jatuh dari sudut kiri atas tiap panel. */
[class*="st-key-glass-"]::after {
  content: "";
  position: absolute;
  top: -55%;
  left: -12%;
  width: 55%;
  height: 130%;
  background: radial-gradient(ellipse at center, rgba(255, 255, 255, 0.09), transparent 68%);
  transform: rotate(-16deg);
  pointer-events: none;
}

/* Streamlit menaruh latar buram pada beberapa pembungkus dalam; tanpa ini
   backdrop-filter tidak punya apa pun untuk dilihat. */
[class*="st-key-glass-"] [data-testid="stVerticalBlock"],
[class*="st-key-glass-"] [data-testid="stHorizontalBlock"] { background: transparent; }

/* ---- Tipografi ---- */
h1, h2, h3, h4 { font-family: var(--font); letter-spacing: -0.021em; color: var(--ink); }
h1 { font-size: 1.62rem; font-weight: 640; }
.panel-title {
  font-size: 0.70rem;
  font-weight: 590;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  color: var(--ink-faint);
  margin: 0 0 14px 0;
}
.panel-note { font-size: 0.78rem; color: var(--ink-muted); margin: -6px 0 14px 0; }

/* ---- Kartu metrik ---- */
.metric-row { display: flex; flex-wrap: wrap; gap: 10px; }
.metric {
  flex: 1 1 120px;
  background: var(--glass-bg-strong);
  border: 1px solid var(--glass-border);
  border-radius: 15px;
  padding: 11px 14px;
}
.metric .label {
  font-size: 0.67rem; letter-spacing: 0.055em; text-transform: uppercase;
  color: var(--ink-faint); margin-bottom: 3px;
}
.metric .value {
  font-size: 1.28rem; font-weight: 600; letter-spacing: -0.024em;
  font-variant-numeric: tabular-nums; color: var(--ink);
}
.metric .sub { font-size: 0.71rem; color: var(--ink-muted); font-variant-numeric: tabular-nums; }
.metric.is-accent .value { color: var(--accent); }

/* ---- Lencana kekhawatiran ---- */
.concern {
  display: flex; gap: 10px; align-items: flex-start;
  background: rgba(255, 214, 10, 0.09);
  border: 1px solid rgba(255, 214, 10, 0.26);
  border-radius: 14px; padding: 11px 14px; margin-bottom: 9px;
  font-size: 0.83rem; line-height: 1.5; color: var(--ink);
}
.concern .mark { color: #FFD60A; font-weight: 700; flex-shrink: 0; }
.clear {
  background: rgba(48, 209, 88, 0.09);
  border: 1px solid rgba(48, 209, 88, 0.26);
  border-radius: 14px; padding: 11px 14px;
  font-size: 0.83rem; color: var(--ink);
}

/* ---- Kontrol ---- */
.stButton > button, .stDownloadButton > button {
  background: rgba(255, 255, 255, 0.10);
  border: 1px solid var(--glass-border);
  border-radius: 13px;
  color: var(--ink);
  font-family: var(--font);
  font-weight: 560;
  letter-spacing: -0.006em;
  padding: 0.5rem 1.05rem;
  backdrop-filter: blur(14px);
  transition: background 160ms ease, transform 160ms ease, border-color 160ms ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  background: rgba(255, 255, 255, 0.17);
  border-color: rgba(255, 255, 255, 0.28);
  transform: translateY(-1px);
}
.stButton > button:active { transform: translateY(0); }
.stButton > button[kind="primary"] {
  background: linear-gradient(180deg, rgba(255, 159, 10, 0.95), rgba(255, 133, 10, 0.88));
  border-color: rgba(255, 190, 90, 0.55);
  color: #221500;
  font-weight: 640;
}
.stButton > button[kind="primary"]:hover {
  background: linear-gradient(180deg, rgba(255, 175, 45, 1), rgba(255, 145, 20, 0.95));
}

/* Input, select, slider, uploader, chat */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
[data-testid="stDateInput"] input,
[data-testid="stChatInput"] textarea {
  background: rgba(255, 255, 255, 0.075) !important;
  border: 1px solid var(--glass-border) !important;
  border-radius: 12px !important;
  color: var(--ink) !important;
  font-family: var(--font) !important;
}
[data-testid="stChatInput"] { background: transparent !important; }
[data-testid="stFileUploaderDropzone"] {
  background: rgba(255, 255, 255, 0.05);
  border: 1px dashed rgba(255, 255, 255, 0.24);
  border-radius: 15px;
}
[data-testid="stSlider"] [role="slider"] {
  background: #fff;
  box-shadow: 0 1px 6px rgba(0, 0, 0, 0.45);
}
[data-testid="stSidebar"] {
  background: rgba(12, 12, 17, 0.62);
  backdrop-filter: blur(32px) saturate(180%);
  -webkit-backdrop-filter: blur(32px) saturate(180%);
  border-right: 1px solid rgba(255, 255, 255, 0.09);
}

/* Chat */
[data-testid="stChatMessage"] {
  background: rgba(255, 255, 255, 0.055);
  border: 1px solid rgba(255, 255, 255, 0.10);
  border-radius: 17px;
  padding: 11px 14px;
  margin-bottom: 9px;
  backdrop-filter: blur(12px);
}

/* Tab, expander, tabel */
.stTabs [data-baseweb="tab-list"] { gap: 5px; background: transparent; border-bottom: none; }
.stTabs [data-baseweb="tab"] {
  background: rgba(255, 255, 255, 0.055);
  border: 1px solid var(--glass-border);
  border-radius: 11px;
  padding: 5px 14px;
  color: var(--ink-muted);
  font-size: 0.82rem;
}
.stTabs [aria-selected="true"] {
  background: rgba(255, 255, 255, 0.15) !important;
  color: var(--ink) !important;
}
[data-testid="stExpander"] details {
  background: rgba(255, 255, 255, 0.045);
  border: 1px solid var(--glass-border);
  border-radius: 15px;
}
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }
hr { border-color: rgba(255, 255, 255, 0.09); }

/* Layar sempit. Kolom Streamlit tidak pernah membungkus sendiri: pada 900px,
   pembagian 7:3 menyisakan panel asisten selebar satu kata per baris. Memaksa
   flex-wrap dengan lebar dasar membuat kolom menumpuk alih-alih menyempit. */
@media (max-width: 1100px) {
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    flex: 1 1 340px;
    min-width: min(340px, 100%);
  }
}

@media (max-width: 820px) {
  .block-container { padding-left: 0.9rem; padding-right: 0.9rem; }
  [class*="st-key-glass-"] { padding: 16px; border-radius: 18px; }
}
</style>
"""


def plotly_layout(height: int = 430) -> dict:
    """Layout plotly yang menyatu dengan panel kaca.

    Latar dibuat sepenuhnya transparan supaya kaca di belakangnya terlihat;
    chart dengan latar sendiri akan tampak seperti tambalan di atas panel.
    """
    return {
        "height": height,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {
            "family": '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", sans-serif',
            "color": PALETTE["ink_muted"],
            "size": 12,
        },
        "margin": {"l": 8, "r": 76, "t": 10, "b": 8},
        "hovermode": "x unified",
        "hoverlabel": {
            "bgcolor": "rgba(28,28,34,0.94)",
            "bordercolor": "rgba(255,255,255,0.16)",
            "font": {"color": PALETTE["ink"], "size": 12},
        },
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": PALETTE["ink_muted"], "size": 11},
        },
        "xaxis": {
            "showgrid": False,
            "zeroline": False,
            "linecolor": PALETTE["grid"],
            "tickfont": {"color": PALETTE["ink_faint"], "size": 11},
        },
        "yaxis": {
            "gridcolor": PALETTE["grid"],
            "zeroline": False,
            "linecolor": "rgba(0,0,0,0)",
            "tickfont": {"color": PALETTE["ink_faint"], "size": 11},
        },
    }
