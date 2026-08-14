"""Shared style for all figures: registers Noto CJK so Chinese renders."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
from matplotlib import rcParams

fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
rcParams.update({
    "font.family": ["DejaVu Sans", "Noto Sans CJK JP"],
    "mathtext.fontset": "cm",
    "axes.linewidth": 1.0,
    "font.size": 10,
    "figure.facecolor": "white",
})

INK  = "#1a1a1a"   # main line / axes
ACC  = "#2f6f9f"   # blue  — "assumptions hold" / L2 / Ridge
WARN = "#c1553b"   # rust  — violation / L1 / Lasso
MUTE = "#8a8a8a"   # scattered data points
GRID = "#dcdcdc"
