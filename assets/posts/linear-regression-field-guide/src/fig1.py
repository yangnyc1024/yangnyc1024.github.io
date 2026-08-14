import numpy as np
from figstyle import *
import matplotlib.pyplot as plt

def bell(ax, x0, mu, sigma, scale=0.42, color=ACC):
    y = np.linspace(mu - 3.4*sigma, mu + 3.4*sigma, 300)
    d = np.exp(-0.5*((y-mu)/sigma)**2)
    ax.plot(x0 + d*scale, y, color=color, lw=1.6, zorder=5)
    ax.fill_betweenx(y, x0, x0 + d*scale, color=color, alpha=0.14, zorder=4)
    ax.plot([x0, x0], [y[0], y[-1]], color=color, lw=0.9, alpha=0.5, zorder=4)
    ax.plot([x0, x0+scale], [mu, mu], color=color, lw=1.0, alpha=0.7, zorder=4)
    ax.plot(x0, mu, 'o', ms=4, color=color, zorder=6)

def panel(ax, sigmas, title, sub, color):
    b0, b1 = 0.9, 0.62
    xs = np.array([1.5, 3.5, 5.5])
    xl = np.linspace(0.4, 6.7, 200)
    ax.plot(xl, b0 + b1*xl, color=INK, lw=1.8, zorder=3)

    rng = np.random.default_rng(11)
    gx = np.linspace(0.7, 6.4, 40)
    gs = np.interp(gx, xs, sigmas)
    ax.plot(gx, b0 + b1*gx + rng.normal(0, gs), 'o', ms=3.6, color=MUTE,
            alpha=0.65, zorder=2)

    for xi, si in zip(xs, sigmas):
        bell(ax, xi, b0 + b1*xi, si, color=color)

    ax.annotate(r"$\mathbb{E}[Y\,|\,X\!=\!x]=\beta_0+\beta_1x$",
                xy=(4.8, b0+b1*4.8), xytext=(0.75, 6.3), fontsize=11, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.0,
                                connectionstyle="arc3,rad=-0.2"))
    for xi, lb in zip(xs, ["$x_1$", "$x_2$", "$x_3$"]):
        ax.plot([xi, xi], [0, -0.13], color=INK, lw=0.9, clip_on=False)
        ax.text(xi, -0.46, lb, ha="center", fontsize=11, color=INK)

    ax.set_title(title, fontsize=12.5, color=color, pad=10, fontweight="bold")
    ax.text(0.5, -0.20, sub, transform=ax.transAxes, ha="center",
            fontsize=9.8, color="#444", linespacing=1.5)
    ax.set_xlim(0, 7.3); ax.set_ylim(-0.65, 7.2)
    ax.set_xlabel("$x$", fontsize=12, labelpad=2)
    ax.set_ylabel("$y$", fontsize=12, rotation=0, labelpad=10)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(INK)
    ax.set_xticks([]); ax.set_yticks([])

fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2))
panel(axes[0], [0.62]*3, "Assumptions hold",
      "All bells share one width", ACC)
axes[0].text(0.5, -0.255, r"同方差", transform=axes[0].transAxes, ha="right",
             fontsize=9.8, color="#444")
axes[0].text(0.51, -0.255, r"$\mathrm{Var}(\epsilon\,|\,X)=\sigma^2 I$",
             transform=axes[0].transAxes, ha="left", fontsize=9.8, color="#444")
panel(axes[1], [0.28, 0.58, 0.95], "Heteroscedasticity",
      "Centres still on the line, but widths grow", WARN)
axes[1].text(0.5, -0.255, "中心仍在线上（无偏），宽度在变（标准误算错）",
             transform=axes[1].transAxes, ha="center", fontsize=9.8, color="#444")
fig.suptitle(r"$y\mid x\;\sim\;\mathcal{N}(\beta_0+\beta_1x,\;\sigma^2)$"
             "        the line is a string of conditional means",
             fontsize=13, y=1.12, color=INK)
fig.text(0.5, 1.045, "回归线就是一串条件均值串起来的", ha="center",
         fontsize=11, color="#555")
fig.tight_layout(rect=[0, 0.05, 1, 0.97])
fig.savefig("figures/fig1-conditional-distribution.png", dpi=200,
            bbox_inches="tight", facecolor="white")
print("fig1 ok")
