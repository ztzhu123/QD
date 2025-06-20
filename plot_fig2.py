from fractions import Fraction
from functools import partial

import h5py
import matplotlib as mpl
from matplotlib.patches import (
    Arc,
    BoxStyle,
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    PathPatch,
    Polygon,
    Wedge,
)
import numpy as np
from qutip import *

from fig_utils import AxesGroup, add_ax, annot_alphabet, fig_in_a4
from output_ctrl import print_info, wrap_emph
from path import DATA_DIR
from plot_bloch import plot_bloch_2d
from plot_toolbox import bar
from plot_toolbox import plot1d as _plot1d
from styles import (
    FONTSIZE,
    LEGEND_SIZE,
    LW,
    MEW,
    MS,
    USE_TEX,
    article_style,
    format_legend,
)

plot1d = partial(_plot1d, constrained_layout=False)


def as_pi_str(angle, frac=True, precision=None, must_reduce_pi=False, wrap_dollar=1):
    if np.iterable(angle):
        strs = []
        for a in angle:
            strs.append(as_pi_str(a, frac, precision, must_reduce_pi, wrap_dollar))
        return strs
    sign = "-" if angle < 0 else ""
    coeff = round(np.abs(angle) / np.pi, 7)
    ret = None
    if np.isclose(coeff, 0):
        ret = "0"
    elif np.isclose(coeff, 1):
        ret = rf"{sign}\pi"

    elif np.isclose(coeff, 1 / 3):
        ret = r"%s\frac{\pi}{3}" % sign
    else:
        for i in range(2, 6):
            if np.isclose(coeff, i / 3):
                ret = r"%s\frac{%d}{3}\pi" % (sign, i)
                break
    if ret is None:
        numerator, denominator = Fraction(str(coeff)).as_integer_ratio()
        if numerator > 10 or denominator > 100:
            if must_reduce_pi:
                if precision is None:
                    coeff = round(coeff, 2)
                else:
                    coeff = f"{coeff:.{precision}f}"
                ret = rf"{sign}{coeff}\pi"
            else:
                if precision is None:
                    angle = round(angle, 2)
                else:
                    angle = f"{angle:.{precision}f}"
                ret = str(angle)
        elif numerator == 1:
            if frac:
                ret = r"%s\frac{\pi}{%d}" % (sign, denominator)
            else:
                ret = rf"{sign}\pi/{denominator}"
        elif frac:
            ret = r"%s\frac{%d}{%d}\pi" % (sign, numerator, denominator)
        else:
            ret = rf"{sign}{numerator}/{denominator}*\pi"
    if wrap_dollar:
        ret = f"${ret}$"
    return ret


def collect_result(N=10):
    result = {}
    if N == 10:
        envs = list(range(1, 11))
    elif N == 6:
        envs = [2, 4, 6]
    elif N == 2:
        envs = [1, 2]
    else:
        raise
    for env in envs:
        result[env] = {}
        filename = (
            DATA_DIR / f"bloch_N={N}_ptrace_env={env}_corr=1_valid=1_tq_error_0.363.h5"
        )
        with h5py.File(filename, "r") as f:
            for k in f.keys():
                result[env][k] = f[k][()]
    result_MI = {}
    for N in [2, 6, 10]:
        result_MI[N] = {}
        filename = DATA_DIR / "fig2_12q_MI_discord.h5"
        with h5py.File(filename, "r") as f:
            for k in f[f"N={N}"].keys():
                result_MI[N][k] = f[f"N={N}"][k][()]
                if k != "ms":
                    result_MI[N][k] = result_MI[N][k][:, :10]
    return result, result_MI


def plot_4x3(ax=None):
    ax.set_aspect("equal")
    ax.axis("off")

    radius = 0.85
    thetas = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    x0 = 0.2
    xs = np.cos(thetas) * radius
    ys = np.sin(thetas) * radius
    ax.scatter(
        xs.tolist() + [x0, -x0],
        ys.tolist() + [0, 0],
        zorder=np.inf,
        ec="#315b92",
        fc="white",
        s=100,
        linewidths=0.9,
    )
    for i in range(len(xs)):
        ax.plot([0, xs[i]], [0, ys[i]], lw=1, color="k", zorder=-np.inf)
        index = {
            0: 3,
            1: 2,
            2: 1,
            3: 12,
            4: 11,
            5: 10,
            6: 9,
            7: 8,
            8: 5,
            9: 4,
        }[i]
        ax.text(
            xs[i],
            ys[i],
            # "Q$_{%d}$" % index,
            index,
            ha="center",
            va="center",
            fontdict={"fontsize": FONTSIZE},
            zorder=np.inf,
        )

    ax.add_patch(
        Ellipse(
            (0, 0),
            0.8,
            0.8 / 3 * 2,
            # zorder=np.inf,
            linewidth=0.9,
            ec="k",
            fc="#faf7cf",
        )
    )
    for i in [6, 7]:
        y = 0
        x = x0
        if i == 6:
            x = -x
        ax.text(
            x,
            y,
            i,
            ha="center",
            va="center",
            fontdict={"fontsize": FONTSIZE},
            zorder=np.inf,
        )

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)


@article_style()
def plot_all(save=False):
    fig = fig_in_a4(1, 0.45, dpi=200)
    gs = fig.add_gridspec(
        100, 100, left=0.13, right=0.95, bottom=0.06, top=0.99, wspace=3, hspace=0.2
    )
    ax_grid = fig.add_subplot(gs[:25, :20])
    ax_circuit = fig.add_subplot(gs[:25, 40:])
    ax_dist = fig.add_subplot(gs[33:70, :47])
    ax_MI = fig.add_subplot(gs[33:70, 52:])
    axes_bloch = [fig.add_subplot(gs[33:55, i * 16 : (i + 1) * 16]) for i in range(6)]
    axes_state = [
        fig.add_subplot(gs[78:97, 8 + i * 21 : 8 + (i + 1) * 21]) for i in range(4)
    ]

    # ----- plot the 1st row -----
    ratio = 1.25
    pos = ax_grid.get_position()
    ax_grid.set_position(
        [pos.xmin - 0.01, pos.ymin - 0.07, pos.width * ratio, pos.height * ratio]
    )

    pos = ax_circuit.get_position()
    ax_circuit.set_position([pos.xmin - 0.1, pos.ymin - 0.06, 0.56, 0.29])

    plot_4x3(ax=ax_grid)
    plot_circuit(ax=ax_circuit)

    # ----- plot the 2nd row -----
    for ax in [ax_dist, ax_MI]:
        pos = ax.get_position()
        ax.set_position([pos.xmin, pos.ymin + 0.02, pos.width * 0.93, pos.height * 0.9])

    # ----- plot part distribution -----
    Ns = [2, 6, 10]
    colors = ["#da514f", "#4157a7"]
    colors_MI = ["#4a4747", "#f14c4f", "#3b6790"]

    ax = ax_dist
    pos = ax.get_position()

    for i in range(3):
        plot1d(
            [0],
            [0],
            label=f"$N$ = {Ns[i]}",
            marker="o",
            ls="--",
            ax=ax,
            mec=colors_MI[i],
            mfc="white",
            color=colors_MI[i],
            hollow=0,
            ms=MS - 0.4,
            mew=0.8,
            lw=LW - 0.3,
        )
    handles, labels = ax.get_legend_handles_labels()
    ax.clear()

    for i, N in enumerate(Ns):
        result, result_MI = collect_result(N=N)
        m = N
        plot1d(
            result[m]["thetas_sim_noisy"][::10],
            (result[m]["mu0s_mean_sim_noisy"] + result[m]["mu1s_mean_sim_noisy"])[::10],
            ax=ax,
            lw=LW - 0.3,
            color=colors_MI[i],
            # zorder=np.inf,
            ls="--",
        )
        plot1d(
            result[m]["thetas_exp"][::10],
            (result[m]["mu0s_mean_exp"] + result[m]["mu1s_mean_exp"])[::10],
            ax=ax,
            lw=LW,
            color=colors_MI[i],
            # zorder=np.inf,
            label=f"$N$ = {N}",
            marker="o",
            ls="",
            hollow=1,
            mew=MEW + 0.1,
            ms=MS - 0.5,
        )

        if N == 10:
            MI = result_MI[N]["MI"]
            ms = result_MI[N]["ms"]
            plot1d(
                ms,
                MI.mean(1),
                MI.std(1, ddof=1),
                ax=ax_MI,
                lw=LW,
                mew=MEW,
                ms=MS,
                label=r"$I$",
                errorbar=False,
                color=colors_MI[0],
            )

            discord = result_MI[N]["discords"]
            plot1d(
                ms,
                (MI - discord).mean(1),
                (MI - discord).std(1, ddof=1),
                ax=ax_MI,
                lw=LW,
                mew=MEW,
                ms=MS,
                label=r"$\chi$",
                errorbar=False,
                color=colors_MI[1],
            )
            plot1d(
                ms,
                discord.mean(1),
                discord.std(1, ddof=1),
                ax=ax_MI,
                lw=LW,
                mew=MEW,
                ms=MS,
                label=r"$D$",
                errorbar=False,
                color=colors_MI[2],
            )
    # ax_inset.grid(False)

    if 1:
        ag = AxesGroup(1, cols_or_rows=3, init=False).init(fig, ax_dist)

        xlim = [0, np.pi]
        ag.set_xticks(
            [0, np.pi / 2, np.pi],
            labels=as_pi_str([0, np.pi / 2, np.pi], precision=0, frac=0),
            xlim=xlim,
            axes=ax_dist,
            sharex=0,
            xlim_pad_ratio=0.05,
            fontsize=FONTSIZE,
        )
        ag.set_yticks(
            [0, 0.5, 1],
            ylim=[0, 1],
            axes=ax_dist,
            sharey=1,
            ylim_pad_ratio=0.05,
            fontsize=FONTSIZE,
        )
        ag.set_xlabel(
            r"$\theta$",
            fontsize=FONTSIZE,
            axes=ax_dist,
            sharex=0,
            labelpad=3,
        )
        ag.set_ylabel(
            r"$P(\theta)$",
            fontsize=FONTSIZE,
            axes=ax_dist,
            sharey=1,
        )

        ag.tick_params(direction="out")
        ag.grid(False)
    if 2:
        xlim = [0, 10]
        ag_MI = AxesGroup(1, cols_or_rows=1, init=False).init(fig, ax_MI)
        ag_MI.set_xticks(
            [0, 5, 10],
            # labels=[0, 0.5, 0.7, 1],
            xlim=xlim,
            axes=ax_MI,
            sharex=0,
            xlim_pad_ratio=0.05,
        )
        ag_MI.set_yticks(
            [0, 1, 2], ylim=[0, 2], axes=ax_MI, sharey=0, ylim_pad_ratio=0.05
        )
        ag_MI.set_xlabel(
            r"$m$",
            fontsize=FONTSIZE,
            clean_others=0,
            axes=ax_MI,
            sharex=0,
            labelpad=0,
        )
        ag_MI.set_ylabel(
            # r"$I\mathcal{(S:F)}$",
            r"Information",
            fontsize=FONTSIZE,
            clean_others=0,
            axes=ax_MI,
            sharey=0,
            # usetex=USE_TEX,
        )

        ag_MI.tick_params(direction="out")
        ag_MI.grid(False)

    format_legend(ax_dist, handles=handles, labels=labels, loc="lower right")
    format_legend(ax_MI, usetex=1, size=LEGEND_SIZE - 1, loc="lower right")
    plot_env(ax_dist, ax_MI)

    # ----- plot the 3rd row -----
    plot_blochs(axes_bloch)

    # ----- plot the 4th row -----
    plot_hists(axes_state)
    ag = AxesGroup(len(axes_state), cols_or_rows=len(axes_state), init=False).init(
        fig, axes_state
    )

    xlim = [-1, 1]
    ag.set_xticks(
        [-1, 0, 1],
        xlim=xlim,
        sharex=1,
        xlim_pad_ratio=0.05,
        fontsize=FONTSIZE,
    )
    # for ax in ag.axes:
    #     ax.set_yscale("log")
    #     ax.minorticks_off()
    ag.set_yticks(
        [-0.5, 0, 0.5],
        ylim=[-0.6, 0.6],
        sharey=0,
        ylim_pad_ratio=0.0,
        axes=0,
        fontsize=FONTSIZE,
    )
    ag.set_yticks(
        [-1.5, 0, 1.5],
        ylim=[-1.6, 1.6],
        sharey=0,
        ylim_pad_ratio=0.0,
        axes=1,
        fontsize=FONTSIZE,
    )
    ag.set_yticks(
        [-2, 0, 2],
        ylim=[-2.05, 2.05],
        sharey=0,
        ylim_pad_ratio=0.0,
        axes=2,
        fontsize=FONTSIZE,
    )
    ag.set_yticks(
        [-2.5, 0, 2.5],
        ylim=[-2.65, 2.65],
        sharey=0,
        ylim_pad_ratio=0.0,
        axes=3,
        fontsize=FONTSIZE,
    )
    ag.set_xlabel(
        "$z$",
        fontsize=FONTSIZE,
        sharex=1,
        clean_others=1,
        labelpad=0,
    )
    ag.set_ylabel(
        r"$\mathcal{A}$",
        fontsize=FONTSIZE,
        sharey=1,
        clean_others=1,
        axes=["row0"],
        usetex=USE_TEX,
    )
    for ax in ag.axes:
        ax.hlines(0, *ax.get_xlim(), color="k", lw=1, ls="-")
        ax.vlines(0, *ax.get_ylim(), ls="--", color="k", lw=0.8)
        ax.text(
            0.35,
            0.03,
            r"$|1_{\mathcal{S}}\mathsf{\rangle}$",
            va="bottom",
            ha="center",
            fontdict={"fontsize": FONTSIZE},
            usetex=USE_TEX,
            transform=ax.transAxes,
        )
        ax.text(
            0.65,
            0.97,
            r"$|0_{\mathcal{S}}\mathsf{\rangle}$",
            va="top",
            ha="center",
            fontdict={"fontsize": FONTSIZE},
            usetex=USE_TEX,
            transform=ax.transAxes,
        )

    ag.tick_params(direction="out")
    ag.grid(False)

    ratio = 1.4
    annot_alphabet(
        [
            ax_grid,
            ax_circuit,
            ax_dist,
            ax_MI,
            axes_state[0],
        ],
        fontsize=FONTSIZE,
        dx=-0.05,
        dy=0.01,
        dx_dict={"b": 0.02},
        dy_dict={0: -0.02, 1: -0.02, "e": 0.02},
        top_bond_dict={1: 0},
        left_bond_dict={"e": "c", "a": "c"},
        transform="fig",
        upper=1,
    )

    if save:
        name = "fig2"
        fig_name = DATA_DIR / "fig2" / f"{name}.pdf"
        fig.savefig(fig_name, pad_inches=0, transparent=True)
        print_info("Saved ", wrap_emph(fig_name.as_posix()))


@article_style()
def plot_circuit(ax=None, save=False):
    total_layer = 14
    numq = 12
    dy = 1
    circle_size = 18
    circuit_start = 0.08
    gate_width = 0.03
    gate_height = 0.7
    circuit_end = 1
    gate_start = 0.13
    gate_end = circuit_end - gate_width / 2 - (circuit_start - gate_width / 2) + 0.02
    L = gate_end - gate_start
    interval = L / (total_layer - 1) - gate_width

    def get_x(layer):
        return gate_start + layer * (interval + gate_width)

    if ax is None:
        fig = fig_in_a4(0.7, 0.3)
        ax = fig.add_subplot(111)
        fig.subplots_adjust(bottom=0.05, top=0.95, left=0.07, right=0.97)
    ax.axis("off")
    lines = {}
    for i in range(1, numq + 1, 1):
        lines[i] = (numq - i) * dy
        ax.text(
            0,
            lines[i],
            r"Q${}_{%d}$" % i,
            va="center",
            ha="left",
            fontdict={"fontsize": FONTSIZE},
        )
    for i in range(1, numq + 1, 1):
        y = lines[i]
        ax.plot(
            [circuit_start, circuit_end], [y, y], color="#7f7f7f", lw=1, zorder=-np.inf
        )

    H_color = "#ffdf62"
    U_color = "#8faadc"
    TOMO_color = "#8eb387"

    def plot_gate(
        layer, index, color, custom_xy=None, width=None, height=None, zorder=np.inf
    ):
        if custom_xy is not None:
            x, y = custom_xy
        else:
            x = get_x(layer)
            y = lines[index]

        if width is None:
            width = gate_width
        if height is None:
            height = gate_height
        xy = (x - width / 2, y - height / 2)
        box = FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle=BoxStyle.Round(pad=0, rounding_size=0.00),
            facecolor=color,
            zorder=zorder,
        )
        ax.add_patch(box)

    def plot_circle(layer, index, hollow=False, custom_xy=None, s=None):
        if custom_xy is not None:
            x, y = custom_xy
        else:
            x = get_x(layer)
            y = lines[index]
        if hollow:
            fc = "white"
            ec = (0, 0, 0, 1)
        else:
            fc = "k"
            ec = (0, 0, 0, 1)
        if s is None:
            s = circle_size
        ax.scatter(
            x,
            y,
            zorder=np.inf,
            ec=ec,
            fc=fc,
            s=s,
            linewidths=0.9,
        )

    def plot_cz(layer, index0, index1, hollow=False, custom_xy=None, s=None):
        if custom_xy is not None:
            x, y0, y1 = custom_xy
        else:
            x = get_x(layer)
            y0 = lines[index0]
            y1 = lines[index1]
        plot_circle(layer, index0, hollow, custom_xy=(x, y0), s=s)
        plot_circle(layer, index1, hollow, custom_xy=(x, y1), s=s)
        ax.plot([x, x], [y0, y1], color="k", lw=1, zorder=-np.inf)

    def plot_cu(
        layer, index0, index1, hollow=False, custom_xy=None, width=None, height=None
    ):
        if custom_xy is not None:
            x, y0, y1 = custom_xy
        else:
            x = get_x(layer)
            y0 = lines[index0]
            y1 = lines[index1]
        plot_circle(layer, index0, hollow, custom_xy=(x, y0))
        ax.plot([x, x], [y0, y1], color="k", lw=1, zorder=-np.inf)
        plot_gate(layer, index1, U_color, custom_xy=(x, y1), width=width, height=height)

    plot_gate(0, 6, H_color)
    plot_gate(0, 7, H_color)
    plot_cz(1, 6, 7)
    plot_gate(2, 6, H_color)

    i = 3
    for index in [5, 4, 3, 2, 1]:
        for hollow in [True, False]:
            plot_cu(i, 6, index, hollow=hollow)
            plot_cu(i, 13 - 6, 13 - index, hollow=hollow)
            i += 1
    plot_gate(
        i,
        7,
        TOMO_color,
        custom_xy=(get_x(i), np.mean([lines[6], lines[7]])),
        height=gate_height * 2.3,
    )

    # ------- legend -------
    # ------- cz -------
    x = 0.28
    y0 = lines[1] + dy * 2.6
    y1 = lines[1] + dy * 3.65
    y2 = y0 - dy * 1.23
    fd = {"fontsize": FONTSIZE - 1}
    plot_cz(
        0,
        0,
        H_color,
        custom_xy=(x, y0, y1),
        s=circle_size / 2,
    )
    ax.text(
        x,
        y2,
        "CZ",
        ha="center",
        va="center",
        zorder=np.inf,
        fontdict=fd,
    )
    # ------- h -------
    x = 0.16
    y = (y0 + y1) / 2 * 0.97
    plot_gate(
        0,
        0,
        H_color,
        custom_xy=(x, y),
        width=gate_width * 1.3,
        height=gate_height * 1.3,
    )
    ax.text(
        x,
        y2,
        "Hadamard",
        ha="center",
        va="center",
        zorder=np.inf,
        fontdict=fd,
    )
    # ------- u -------
    for j in [0, 1]:
        x = 0.45 + 0.26 * j
        plot_cu(
            0,
            0,
            0,
            hollow=j == 0,
            custom_xy=(x, y1, y0),
            width=gate_width * 1.0,
            height=gate_height * 1.0,
        )
        if j == 0:
            s1 = r"$|0\mathsf{\rangle}\mathsf{\langle}0|\otimes U_k^0$"
            s2 = r"+$|1\mathsf{\rangle}\mathsf{\langle}1|\otimes \mathbb{I}$"
        else:
            s1 = r"$|0\mathsf{\rangle}\mathsf{\langle}0|\otimes \mathbb{I}$"
            s2 = r"+$|1\mathsf{\rangle}\mathsf{\langle}1|\otimes U_k^1$"
        ax.text(
            x,
            y2,
            s1 + s2,
            ha="center",
            va="center",
            zorder=np.inf,
            fontdict=fd,
        )
    # ------- tomo -------
    x = 0.93
    plot_gate(
        0,
        0,
        TOMO_color,
        custom_xy=(x, y),
        width=gate_width * 1.3,
        height=gate_height * 1.3,
    )
    ax.text(
        x,
        y2,
        r"$U_{\rm tomo.}$",
        ha="center",
        va="center",
        zorder=np.inf,
        fontdict=fd,
    )

    ax.set_xlim(0, circuit_end)
    ax.set_ylim(-1, (numq + 3) * dy)

    if save:
        fig_name_part = DATA_DIR / "fig2" / "fig2_circuit.pdf"
        fig.savefig(fig_name_part, pad_inches=0, transparent=True)
        print_info("Saved ", wrap_emph(fig_name_part.as_posix()))


def add_cmap(fig, pos, vmin, vmax, color, dx=0, xlabel=None):
    if pos is None:
        cax = add_ax(fig, 0.805 + dx, 0.6, 0.005, 0.03)
    else:
        cax = add_ax(fig, pos.xmin + 0.02, pos.ymax + 0.01, 0.03, 0.01)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        None, [mpl.colors.to_rgba(color, 0.25), color], N=256
    )
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap
        ),
        cax=cax,
        orientation="vertical",
    )
    assert vmin == 0e-4
    assert vmax == 6e-2
    # cbar.ax.set_xticks([vmin, vmax], [r"$2\times 10^{-4}$", r"$6\times 10^{-2}$"])
    cbar.ax.set_yticks([vmin, vmax], [0, 0.06])
    cbar.ax.tick_params(axis="both", which="major", length=2, pad=1)
    if xlabel is None:
        xlabel = r"$X_{\alpha}$"
    cbar.ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    cbar.ax.xaxis.set_label_position("top")
    # cbar.ax.xaxis.set_label_coords(0.5, 8.0)


def plot_hists(axes):
    ms = np.arange(2, 10, 2)
    results = collect_result(N=10)[0]
    for i, m in enumerate(ms):
        ax = axes[i]
        zs_group, Ps_group, zs, Ps = sort_P(results, m)
        xs = []
        ys = []
        bins = np.linspace(-1, 1, 11)
        for k in range(len(bins) - 1):
            left = bins[k]
            right = bins[k + 1]
            if k != len(bins) - 2:
                mask = (zs >= left) & (zs < right)
            else:
                mask = (zs >= left) & (zs <= right)
            Ps_mask = np.ma.masked_array(Ps, mask=~mask)
            num0 = np.zeros_like(Ps)
            num1 = np.zeros_like(Ps)
            for l in range(len(zs)):
                num0[l] = m - np.sum(state_index_number([2] * m, l))
                num1[l] = np.sum(state_index_number([2] * m, l))
            num0 = np.ma.masked_array(num0, mask=~mask)
            num1 = np.ma.masked_array(num1, mask=~mask)
            y = ((num0 - num1) * Ps_mask).sum()
            ys.append(y)
            xs.append(np.mean([left, right]))
        bar(
            ys,
            xs=xs,
            ax=ax,
            width=bins[1] - bins[0],
            color="#f7a760",
            linewidth=0.8,
            edgecolor="k",
        )
        ax.text(
            0.5,
            1.05,
            f"$m$ = {m}",
            ha="center",
            va="bottom",
            fontsize=FONTSIZE,
            transform=ax.transAxes,
        )


def plot_error_rate(ax=None):
    results = collect_result(N=10)[0]
    all_error_rates = []
    all_ms = []
    for start in [1, 2]:
        error_rates = []
        ms = np.arange(start, 10, 2)
        for m in ms:
            # zs_group, Ps_group, zs, Ps = sort_P(results, m)
            # zs_group2, Ps_group2, zs2, Ps2 = sort_P(results, m + 1)
            zs_group, Ps_group, zs, Ps = sort_P(results, 10)
            total_error_rate = 0
            Ps = Ps.sum(1)
            Ps2 = ptrace_prob(Ps, np.arange(m + 1))
            for i in range(2**m):
                state = state_index_number([2] * m, i)
                state_plus0 = list(state) + [0]
                state_plus1 = list(state) + [1]
                index0 = state_number_index([2] * (m + 1), state_plus0)
                index1 = state_number_index([2] * (m + 1), state_plus1)
                if m % 2 == 1:
                    can_be_error = np.sum(state) in [m // 2, m // 2 + 1]
                else:
                    # can_be_error = np.sum(state) in [m // 2 - 1, m // 2, m // 2 + 1]
                    can_be_error = np.sum(state) in [m // 2]
                if not can_be_error:
                    continue
                if m % 2 == 1:
                    if np.sum(state) == m // 2:
                        error_rate = Ps2[index1]
                    else:
                        error_rate = Ps2[index0]
                else:
                    if np.sum(state) == m // 2:
                        error_rate = Ps2[index0] + Ps2[index1]
                    else:
                        error_rate = 0
                total_error_rate += error_rate
            error_rates.append(total_error_rate)
        # ax = plot1d(ms, error_rates, marker="o", ax=ax, ms=MS, lw=LW, mew=MEW)
        all_error_rates.append(error_rates)
        all_ms.append(ms)
    ax = bar(
        all_error_rates[0],
        xs=all_ms[0],
        constrained_layout=False,
        ax=ax,
        color="tab:blue",
        label="odd",
        width=0.5,
    )
    ax = bar(
        all_error_rates[1],
        xs=all_ms[1],
        constrained_layout=False,
        ax=ax,
        color="tab:orange",
        label="even",
        width=0.5,
    )


def sort_P(results, m):
    result = results[m]
    zs = result["zs_exp"].T
    Ps = result["Xs_exp"].T
    Ps = Ps / Ps.sum(axis=0, keepdims=True)
    Ps = Ps / np.sum(Ps)
    # for i in range(zs.shape[0]):
    #     indexes = np.argsort(zs[i])
    #     zs[i] = zs[i][indexes]
    #     Ps[i] = Ps[i][indexes]

    zs_group = [[], [], []]
    Ps_group = [[], [], []]
    for j in range(zs.shape[0]):
        state = state_index_number([2] * m, j)
        num1 = np.sum(state)
        num0 = m - num1
        if num0 > num1:
            zs_group[0].append(zs[j])
            Ps_group[0].append(Ps[j])
        elif num0 < num1:
            zs_group[2].append(zs[j])
            Ps_group[2].append(Ps[j])
        else:
            zs_group[1].append(zs[j])
            Ps_group[1].append(Ps[j])
    return zs_group, Ps_group, zs, Ps


def plot_blochs(axes_bloch):
    ratio = 0.53
    fig = axes_bloch[0].figure
    for j, axes in enumerate([axes_bloch[:3], axes_bloch[3:]]):
        for i, ax in enumerate(axes):
            pos = ax.get_position()
            dx = 0.007 - 0.05 * i + j * 0.036

            ax.set_position(
                [
                    pos.xmin + dx,
                    pos.ymin + 0.047,
                    pos.width * ratio,
                    pos.height * ratio,
                ]
            )

    colors = ["#d9504e", "#4157a7"]
    color = colors[0]
    threshold = 0.0000
    cmap_prob = mpl.colors.LinearSegmentedColormap.from_list(
        None, [mpl.colors.to_rgba(color, 0.0), color], N=2560
    )

    Ns = [2, 6, 10]
    for i, N in enumerate(Ns):
        m = N
        result = collect_result(N=N)[0]
        bloch_Xs = result[m]["Xs_exp"].flatten()
        xs = result[m]["xs_exp"].flatten()
        ys = result[m]["ys_exp"].flatten()
        zs = result[m]["zs_exp"].flatten()
        bloch_Xs = bloch_Xs / bloch_Xs.sum()
        # if N==10:
        #     xs = xs[:10]
        #     ys = ys[:10]
        #     zs = zs[:10]
        #     bloch_Xs = bloch_Xs[:10]
        ax = axes_bloch[i]
        # if i == 1:
        #     # threshold = 0
        #     a = 0.005
        #     b = 0.001
        #     mask = (bloch_Xs >= b) & (bloch_Xs <= a)
        #     bloch_Xs[mask] = a
        #     bloch_Xs = bloch_Xs / bloch_Xs.sum()
        vmin, vmax = plot_bloch_2d(
            xs,
            ys,
            zs,
            bloch_Xs,
            ax=ax,
            threshold=threshold,
            color=cmap_prob,
            # threshold=0,
            # color=color,
            s=5,
            constrained_layout=False,
        )
        # pos = ax.get_position()
        # add_cmap(fig, pos, vmin, vmax, color)

        ax.set_title(f"$N$ = {N}", fontdict={"fontsize": FONTSIZE})

    vmin = threshold
    vmax = 0.06
    add_cmap(fig, None, vmin, vmax, colors[0], dx=-0.43, xlabel=r"$X_{\alpha\beta}$")
    add_cmap(fig, None, vmin, vmax, colors[1])

    color = colors[1]
    cmap_prob = mpl.colors.LinearSegmentedColormap.from_list(
        None, [mpl.colors.to_rgba(color, 0.0), color], N=2560
    )

    ms = [2, 5, 8]
    N = 10
    result = collect_result(N=N)[0]

    for i, m in enumerate(ms):
        bloch_Xs = result[m]["Xs_exp"].flatten()
        xs = result[m]["xs_exp"].flatten()
        ys = result[m]["ys_exp"].flatten()
        zs = result[m]["zs_exp"].flatten()
        bloch_Xs = bloch_Xs / bloch_Xs.sum()
        N = len(bloch_Xs)
        ax = axes_bloch[3 + i]
        vmin, vmax = plot_bloch_2d(
            xs,
            ys,
            zs,
            bloch_Xs,
            ax=ax,
            threshold=threshold,
            color=cmap_prob,
            # threshold=0,
            # color=color,
            s=5,
            constrained_layout=False,
        )
        pos = ax.get_position()

        if m == 5:
            ax.set_title(f"$m$ = {m}", fontdict={"fontsize": FONTSIZE}, pad=6.9)
        else:
            ax.set_title(f"$m$ = {m}", fontdict={"fontsize": FONTSIZE})


def plot_env(ax_dist, ax_MI):
    color_whole_env = "#dc8686"
    colors = ["#babbbd", "#6eaadb"]
    r = 1
    pos = ax_dist.get_position()
    ax_dist = add_ax(
        ax_dist.figure,
        (pos.xmin + (pos.xmax - pos.xmin) * 0.4),
        (pos.ymin + (pos.ymax - pos.ymin) * 0.15),
        pos.width * 0.25,
        pos.height * 0.25,
    )
    pos = ax_MI.get_position()
    ax_MI = add_ax(
        ax_MI.figure,
        (pos.xmin + (pos.xmax - pos.xmin) * 0.4),
        (pos.ymin + (pos.ymax - pos.ymin) * 0.15),
        pos.width * 0.25,
        pos.height * 0.25,
    )
    for i, ax in enumerate([ax_dist, ax_MI]):
        if i == 0:
            center = [0, 0]
            lw = 0.6
            ax.add_patch(
                Circle(
                    center,
                    radius=r,
                    facecolor=color_whole_env,
                    edgecolor="k",
                    linewidth=0.5,
                    zorder=-np.inf,
                )
            )
            ax.add_patch(
                Circle(
                    center,
                    radius=r * 0.6,
                    facecolor="white",
                    edgecolor="k",
                    linewidth=0.5,
                )
            )
            ax.add_patch(
                Circle(
                    center,
                    radius=r * 0.4,
                    facecolor="#b9d8da",
                    edgecolor="k",
                    linewidth=0.5,
                )
            )
            ax.text(
                *center,
                r"$\mathcal{S}$",
                fontsize=FONTSIZE,
                fontname="Arial",
                fontweight="bold",
                ha="center",
                va="center",
                usetex=USE_TEX,
                zorder=np.inf,
            )
            ax.text(
                1.1,
                0.8,
                r"$\mathcal{E}$",
                fontsize=FONTSIZE,
                fontname="Arial",
                fontweight="bold",
                ha="center",
                va="center",
                usetex=USE_TEX,
                zorder=np.inf,
            )
            for theta in np.linspace(0, np.pi, 3, endpoint=False):
                x0 = center[0] + r * np.cos(theta)
                x1 = center[0] - r * np.cos(theta)
                y0 = center[1] + r * np.sin(theta)
                y1 = center[1] - r * np.sin(theta)
                # ax.plot([x0, x1], [y0, y1], color="k", lw=lw, zorder=-100)
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
        else:
            center = [0, 0]
            for k in range(2):
                if k == 0:
                    dx = -0.1
                    angle = [90, 270]
                else:
                    dx = 0.1
                    angle = [-90, 90]
                ax.add_patch(
                    Wedge(
                        [dx, 0],
                        r,
                        angle[0],
                        angle[1],
                        facecolor=colors[k],
                        edgecolor="k",
                        linewidth=0.5,
                        zorder=-np.inf,
                    )
                )
                ax.add_patch(
                    Wedge(
                        [dx, 0],
                        r * 0.6,
                        angle[0],
                        angle[1],
                        facecolor="white",
                        edgecolor="k",
                        linewidth=0.5,
                        zorder=1000,
                    )
                )
                ax.add_patch(
                    Circle(
                        [dx, 0],
                        r * 0.585,
                        facecolor="white",
                        edgecolor="none",
                        linewidth=0.5,
                        zorder=1000,
                    )
                )
            ax.add_patch(
                Circle(
                    center,
                    radius=r * 0.4,
                    facecolor="#b9d8da",
                    edgecolor="k",
                    linewidth=0.5,
                    zorder=np.inf,
                )
            )
            ax.text(
                *center,
                r"$\mathcal{S}$",
                fontsize=FONTSIZE,
                fontname="Arial",
                fontweight="bold",
                ha="center",
                va="center",
                usetex=USE_TEX,
                zorder=np.inf,
            )
            ax.text(
                1.30,
                0.75,
                r"$\mathcal{F}$",
                fontsize=FONTSIZE,
                fontname="Arial",
                fontweight="bold",
                ha="center",
                va="center",
                usetex=USE_TEX,
                zorder=np.inf,
            )
            for theta in np.linspace(0, np.pi, 3, endpoint=False):
                x0 = center[0] + r * np.cos(theta)
                x1 = center[0] - r * np.cos(theta)
                y0 = center[1] + r * np.sin(theta)
                y1 = center[1] - r * np.sin(theta)
                # ax.plot([x0, x1], [y0, y1], color="k", lw=lw, zorder=-100)
            ax.set_xlim(-1.25, 1.25)
            ax.set_ylim(-1.25, 1.25)
        ax.axis("off")
        ax.set_aspect("equal")


def ptrace_prob(probs, sel):
    """
    Parameters
    ----------
    probs: (2**num_q, ...)

    Return
    ----------
    ptraced_prob: (2**(num_q - len(sel)), ...)
    """
    if not np.iterable(sel):
        sel = [sel]
    probs = np.asfarray(probs)
    sel = np.array(sel, dtype=np.int32)
    num_q = int(np.log2(probs.shape[0]))
    if (1 << num_q) != probs.shape[0]:
        raise ValueError()
    if np.any(sel < 0) or np.any(sel >= num_q):
        raise ValueError()

    other_shape = probs.shape[1:]
    sum_axes = tuple([i for i in range(num_q) if i not in sel])
    if len(sum_axes) != 0:
        probs = probs.reshape((2,) * num_q + other_shape).sum(axis=sum_axes)
    return probs.reshape(-1, *other_shape)
