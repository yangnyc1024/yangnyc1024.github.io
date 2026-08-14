import numpy as np
from figstyle import *
import matplotlib.pyplot as plt

# ---------- fig 3: soft-thresholding vs proportional shrinkage ----------
fig, ax = plt.subplots(figsize=(7.4, 5.6))
lam = 1.0
b = np.linspace(-3.4, 3.4, 700)
ax.plot(b, b, color=MUTE, lw=1.3, ls=(0,(5,4)), label="No penalty  $\\hat\\beta^{\\,\\mathrm{OLS}}$")
ax.plot(b, b/(1+lam), color=ACC, lw=2.4, label="Ridge   $\\hat\\beta/(1+\\lambda)$")
ax.plot(b, np.sign(b)*np.maximum(np.abs(b)-lam, 0), color=WARN, lw=2.4,
        label="Lasso   soft-threshold")

ax.axhline(0, color=INK, lw=0.9); ax.axvline(0, color=INK, lw=0.9)
ax.fill_between([-lam, lam], -3.4, 3.4, color=WARN, alpha=0.07, zorder=0)
ax.annotate("", xy=(-lam, -2.55), xytext=(lam, -2.55),
            arrowprops=dict(arrowstyle="<->", color=WARN, lw=1.2))
ax.text(0, -2.92, r"$|\hat\beta^{\,\mathrm{OLS}}|<\lambda$", ha="center",
        fontsize=11, color=WARN)
ax.annotate("exactly zero", xy=(0.55, 0), xytext=(1.35, -1.55), fontsize=11,
            color=WARN, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=WARN, lw=1.2,
                            connectionstyle="arc3,rad=0.3"))
ax.set_xlim(-3.4, 3.4); ax.set_ylim(-3.4, 3.4); ax.set_aspect("equal")
ax.set_xlabel(r"$\hat\beta^{\,\mathrm{OLS}}$", fontsize=12)
ax.set_ylabel("penalized estimate", fontsize=11.5)
ax.set_xticks([-lam, 0, lam]); ax.set_xticklabels([r"$-\lambda$", "0", r"$\lambda$"], fontsize=11)
ax.set_yticks([0]); ax.set_yticklabels(["0"], fontsize=11)
ax.grid(color=GRID, lw=0.6, alpha=0.7)
ax.set_axisbelow(True)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.legend(loc="upper left", fontsize=10.5, frameon=False)
ax.set_title("Orthonormal design: what each penalty does to a coefficient",
             fontsize=12.5, pad=26, color=INK)
ax.text(0.5, 1.035, "L1 有一段被压平在 0 上；L2 只是一条过原点的斜线",
        transform=ax.transAxes, ha="center", fontsize=10.5, color="#555")
fig.tight_layout()
fig.savefig("figures/fig3-soft-thresholding.png", dpi=200, bbox_inches="tight",
            facecolor="white")
print("fig3 ok")

# ---------- fig 4: Ridge shrinkage factor across SVD directions ----------
fig, ax = plt.subplots(figsize=(7.8, 5.2))
d = np.linspace(0.05, 4.0, 500)
for lam_, c, a in [(0.1, ACC, 0.35), (0.5, ACC, 0.6), (2.0, ACC, 1.0)]:
    ax.plot(d, d**2/(d**2+lam_), color=c, lw=2.3, alpha=a,
            label=rf"$\lambda={lam_}$")

ax.axvspan(0.05, 0.95, color=WARN, alpha=0.07, zorder=0)
ax.text(0.5, 0.135, "low-variance directions\nshrunk hardest\n低方差方向收缩最狠",
        ha="center", fontsize=10.5, color=WARN, fontweight="bold", linespacing=1.6)
ax.text(3.0, 0.30, "high-variance directions\nbarely touched\n高方差方向几乎不动",
        ha="center", fontsize=10.5, color="#555", linespacing=1.5)

ax.set_xlim(0, 4.05); ax.set_ylim(0, 1.05)
ax.set_xlabel(r"singular value  $d_j$   (data variance along that direction)", fontsize=11.5)
ax.set_ylabel(r"shrinkage factor  $\dfrac{d_j^2}{d_j^2+\lambda}$", fontsize=12)
ax.grid(color=GRID, lw=0.6, alpha=0.7); ax.set_axisbelow(True)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.legend(loc="lower right", fontsize=11, frameon=False)
ax.set_title("Ridge shrinks hardest exactly where collinearity lives",
             fontsize=12.5, pad=26, color=INK)
ax.text(0.5, 1.04, "共线性造成的不稳定方向，正是 Ridge 压得最狠的方向",
        transform=ax.transAxes, ha="center", fontsize=10.5, color="#555")
fig.tight_layout()
fig.savefig("figures/fig4-ridge-svd-shrinkage.png", dpi=200, bbox_inches="tight",
            facecolor="white")
print("fig4 ok")
