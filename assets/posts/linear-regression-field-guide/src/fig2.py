import numpy as np
from figstyle import *
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

bhat = np.array([1.30, 3.20])          # OLS solution
A = np.array([[1.0, 0.70], [0.70, 1.0]])  # contour shape

def contours(ax, levels, color="#9aa5ad"):
    g = np.linspace(-2.4, 3.6, 500)
    Xg, Yg = np.meshgrid(g, g)
    D = np.stack([Xg - bhat[0], Yg - bhat[1]])
    Z = A[0,0]*D[0]**2 + 2*A[0,1]*D[0]*D[1] + A[1,1]*D[1]**2
    ax.contour(Xg, Yg, Z, levels=levels, colors=color, linewidths=0.9, alpha=0.75)

def touch_point(t, norm):
    """First contour touch = constrained argmin."""
    g = np.linspace(-0.4, 3.4, 900)
    Xg, Yg = np.meshgrid(g, g)
    D = np.stack([Xg - bhat[0], Yg - bhat[1]])
    Z = A[0,0]*D[0]**2 + 2*A[0,1]*D[0]*D[1] + A[1,1]*D[1]**2
    R = np.abs(Xg) + np.abs(Yg) if norm == 1 else np.sqrt(Xg**2 + Yg**2)
    Z = np.where(R <= t + 1e-9, Z, np.inf)
    i = np.unravel_index(np.argmin(Z), Z.shape)
    return Xg[i], Yg[i], Z[i]

fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.4))
t = 1.55

for ax, norm, name, col, cn in [
        (axes[0], 1, "Lasso   L1", WARN, "菱形有尖角落在坐标轴上 → 稀疏"),
        (axes[1], 2, "Ridge   L2", ACC, "圆球处处光滑 → 系数都非零")]:

    if norm == 1:
        ax.add_patch(Polygon([[t,0],[0,t],[-t,0],[0,-t]], closed=True,
                             fc=col, alpha=0.16, ec=col, lw=1.9, zorder=2))
    else:
        ax.add_patch(Circle((0,0), t, fc=col, alpha=0.16, ec=col, lw=1.9, zorder=2))

    px, py, zt = touch_point(t, norm)
    contours(ax, levels=sorted({zt*0.18, zt*0.5, zt, zt*1.9, zt*3.2}))
    ax.plot(*bhat, 'o', ms=8, color=INK, zorder=6)
    ax.text(bhat[0]+0.16, bhat[1]+0.14, r"$\hat\beta^{\,\mathrm{OLS}}$",
            fontsize=12, color=INK)
    ax.plot(px, py, 'o', ms=9, color=col, zorder=7, mec="white", mew=1.4)
    lab = (r"$\hat\beta^{\,\mathrm{lasso}}$" if norm==1 else r"$\hat\beta^{\,\mathrm{ridge}}$")
    ax.annotate(lab, xy=(px, py), xytext=(px-1.55, py+0.85), fontsize=12, color=col,
                arrowprops=dict(arrowstyle="->", color=col, lw=1.2,
                                connectionstyle="arc3,rad=0.25"))
    if norm == 1:
        ax.annotate(r"$\beta_1=0$", xy=(px, py), xytext=(px+0.55, py-0.75),
                    fontsize=11, color=col, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.1))

    ax.axhline(0, color=INK, lw=1.0, zorder=1)
    ax.axvline(0, color=INK, lw=1.0, zorder=1)
    ax.set_xlim(-2.3, 3.3); ax.set_ylim(-2.3, 4.1); ax.set_aspect("equal")
    ax.set_title(name, fontsize=13, color=col, pad=9, fontweight="bold")
    ax.text(0.5, -0.105, cn, transform=ax.transAxes, ha="center",
            fontsize=10, color="#444")
    ax.set_xlabel(r"$\beta_1$", fontsize=12); ax.set_ylabel(r"$\beta_2$", fontsize=12, rotation=0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

fig.suptitle("Constraint region + loss contours: where they first touch is the solution",
             fontsize=12.5, y=1.10, color=INK)
fig.text(0.5, 1.035, "等高线第一次碰到可行域的地方就是解", ha="center",
         fontsize=10.5, color="#555")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig("figures/fig2-l1-l2-constraint.png", dpi=200, bbox_inches="tight",
            facecolor="white")
print("fig2 ok")
