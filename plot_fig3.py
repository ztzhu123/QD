from fractions import Fraction
from functools import partial

import h5py
from matplotlib.patches import (
    Arc,
    BoxStyle,
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
)
import numpy as np

from fig_utils import AxesGroup, add_fix_ax, annot_alphabet, fig_in_a4
from output_ctrl import print_info, wrap_emph
from path import DATA_DIR
from plot_toolbox import plot1d as _plot1d
from plot_toolbox import plot3d
from styles import (
    CAPSIZE,
    DASHES,
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


def get_legend(ax, colors, labels, std=False, **kwargs):
    for i, color in enumerate(colors):
        label = labels[i]
        plot1d(
            [1], [1], 0.1 if std else None, ax=ax, color=color, label=label, **kwargs
        )
    handles, labels = ax.get_legend_handles_labels()
    ax.clear()
    return handles, labels


def collect_discord_result(norm=False):
    filename = DATA_DIR / "MI_valid=1_corr=1_tq_error_0.304.h5"
    result = {}
    with h5py.File(filename, "r") as f:
        for key in f.keys():
            result[key] = f[key][()]
            if "_exp" in key and "theta" not in key and "S_center" not in key:
                if norm:
                    S_center_exp = f["S_center_exp"][()][..., None]
                    result[key + "_mean"] = (f[key][()] / S_center_exp).mean(0)
                    result[key + "_std"] = (f[key][()] / S_center_exp).std(0, ddof=1)
                else:
                    result[key + "_mean"] = f[key][()].mean(0)
                    result[key + "_std"] = f[key][()].std(0, ddof=1)
    if norm:
        for key in ["MIs", "chi", "discords"]:
            for suffix in ["_sim_ideal", "_sim_noisy"]:
                result[key + suffix] = (
                    result[key + suffix] / result["S_center" + suffix][..., None]
                )
    return result


@article_style()
def plot_all(save=False, norm=False):
    result = collect_discord_result(norm=norm)
    fig = fig_in_a4(1, 0.3, dpi=200)

    ax_grid = add_fix_ax(fig, [-0.03, 0.6, 0.3, 0.4])
    ax_circuit = add_fix_ax(fig, [0.23, 0.6, 0.48, 0.4])
    MI_axes = [
        add_fix_ax(fig, [0.76, 0.05 + 0.5 * i + 0.02 * (1 - i), 0.2, 0.4], is_3d=1)
        for i in range(2)
    ]
    MI_axes = MI_axes[::-1]
    x = 0.06
    y = 0.05
    width = 0.15
    axes_chi = [add_fix_ax(fig, [x + 0.17 * i, y + 0.3, width, 0.2]) for i in range(4)]
    axes_discord = [
        add_fix_ax(fig, [x + 0.17 * i, y + 0.05, width, 0.2]) for i in range(4)
    ]
    if norm:
        suffix = r" / $H_{\mathcal{S}}$"
    else:
        suffix = ""

    # ratio = 1.2
    # pos = ax_grid.get_position()
    # ax_grid.set_position(
    #     [pos.xmin - 0.04, pos.ymin - 0.1, pos.width * ratio, pos.height * ratio]
    # )

    # pos = ax_circuit.get_position()
    # ax_circuit.set_position([pos.xmin, pos.ymin - 0.14, 0.5, 0.5])

    # ----- lattice -----
    plot_3x3(ax=ax_grid)

    # ----- circuit -----
    plot_circuit(ax=ax_circuit)
    # ----- plot chi -----
    colors = ["#1a73e8", "#f5a21e", "#d44746", "#534e50"]
    for i, num_env in enumerate([1, 2, 3, 4]):
        thetas_sim_ideal = result["thetas_sim_ideal"]
        thetas_sim_noisy = result["thetas_sim_noisy"]
        thetas_exp = result["thetas_exp"]
        color = colors[i]

        plot1d(
            thetas_exp,
            result["chis_exp_mean"][:, i],
            result["chis_exp_std"][:, i],
            ax=axes_chi[i],
            lw=LW,
            mew=MEW,
            ms=MS,
            capsize=CAPSIZE,
            marker="o",
            ls="",
            zorder=100,
            clip_on=False,
            color=color,
            # label="Exp." if i == 3 else None,
        )
        # plot1d(
        #     thetas_sim_ideal,
        #     result["chi_sim_ideal"][:, i],
        #     ax=axes_chi[i],
        #     lw=LW,
        #     color=color,
        #     alpha=0.5,
        #     # label="Sim.(ideal)" if i == 3 else None,
        # )
        plot1d(
            thetas_sim_noisy,
            result["chi_sim_noisy"][:, i],
            ax=axes_chi[i],
            lw=LW,
            # ls="--",
            color=color,
            # label="Sim.(noisy)" if i == 3 else None,
        )
        axes_chi[i].set_title("$m=%d$" % num_env, fontsize=FONTSIZE)

    # ax = axes_chi[-1]
    # handles, labels = ax.get_legend_handles_labels()
    # handles = np.array(handles)[[1, 2, 0]].tolist()
    # labels = np.array(labels)[[1, 2, 0]].tolist()
    # format_legend(ax, loc="lower center", size=6)

    # ----- plot discords -----
    # colors = ["#a3a279", "#569cab", "#195d7b", "#002a5c"]
    for i, num_env in enumerate([1, 2, 3, 4]):
        thetas_sim_ideal = result["thetas_sim_ideal"]
        thetas_sim_noisy = result["thetas_sim_noisy"]
        thetas_exp = result["thetas_exp"]
        color = colors[i]

        ax = axes_discord[i]
        plot1d(
            thetas_exp,
            result["discords_exp_mean"][:, i],
            result["discords_exp_std"][:, i],
            ax=ax,
            lw=LW,
            mew=MEW,
            ms=MS,
            capsize=CAPSIZE,
            marker="o",
            ls="",
            zorder=100,
            clip_on=False,
            color=color,
            label="Experiment" if i == 0 else None,
        )
        # plot1d(
        #     thetas_sim_ideal,
        #     result["discords_sim_ideal"][:, i],
        #     ax=ax,
        #     lw=LW,
        #     color=color,
        #     alpha=0.5,
        #     label="Sim.(ideal)" if i == 0 else None,
        # )
        plot1d(
            thetas_sim_noisy,
            result["discords_sim_noisy"][:, i],
            ax=ax,
            lw=LW,
            # ls="--",
            color=color,
            label="Simulation" if i == 0 else None,
        )

    ax = axes_discord[0]
    handles, labels = ax.get_legend_handles_labels()
    # handles = np.array(handles)[[1, 0]].tolist()
    # labels = np.array(labels)[[1, 0]].tolist()
    format_legend(ax, loc="upper center", size=LEGEND_SIZE)

    xlim = [thetas_sim_ideal.min(), thetas_sim_ideal.max()]
    ag = AxesGroup(8, 4, init=False)
    ag.inited = True
    ag.axes = axes_chi + axes_discord
    xticks = xlim.copy()
    xticks.insert(1, np.mean(xticks))
    ag.set_xticks(
        xticks,
        labels=as_pi_str(xticks, precision=0, frac=0),
        xlim=xlim,
        sharex=1,
        xlim_pad_ratio=0.02,
    )
    ag.set_yticks(
        [0, 0.5, 1], labels=[0, 0.5, 1], ylim=[0, 1], sharey=1, ylim_pad_ratio=0.02
    )

    ag.set_xlabel(
        r"$\theta$",
        fontsize=FONTSIZE,
        clean_others=1,
        sharex=1,
        labelpad=3,
    )
    ag.set_ylabel(
        r"$\chi(\mathcal{S}\!:\!\check{\mathcal{F}})$" + suffix,
        fontsize=FONTSIZE,
        clean_others=1,
        sharey=1,
        usetex=USE_TEX,
        axes=[0, 1, 2, 3],
    )
    ag.set_ylabel(
        r"$\mathcal{D}(\mathcal{S}\!:\!\check{\mathcal{F}})$" + suffix,
        fontsize=FONTSIZE,
        clean_others=0,
        sharey=1,
        usetex=USE_TEX,
        axes=[4, 5, 6, 7],
    )

    # ----- plot 3d MI -----
    for i, ax in enumerate(MI_axes):
        pos = ax.get_position()
        if i == 0:
            dy = 0.04
        else:
            dy = 0.12
        # ax.set_position(
        #     [pos.xmin - 0.035, pos.ymin - dy, pos.width * 2.3, pos.height * 2.3]
        # )

        if i == 0:
            MIs = [np.zeros_like(result["MIs_sim_noisy"][:, 0])]
            for m in range(1, 5):
                MIs.append(result["MIs_sim_noisy"][:, m - 1])
            MIs = np.asarray(MIs)
            thetas = result["thetas_sim_noisy"]
            ax.set_title(f"Simulation", fontdict={"fontsize": FONTSIZE}, pad=0)
        else:
            MIs = [np.zeros_like(result["MIs_exp_mean"][:, 0])]
            for m in range(1, 5):
                MIs.append(result["MIs_exp_mean"][:, m - 1])
            MIs = np.asarray(MIs)
            thetas = result["thetas_exp"]
            ax.set_title(f"Experiment", fontdict={"fontsize": FONTSIZE}, pad=0)
        y = np.arange(5)
        plot3d(thetas, y, MIs, ax=ax, cmap="RdBu_r", constrained_layout=False)

        ax.grid(False)
        ax.view_init(elev=29.25, azim=-64.7, roll=0)

        xlim = [thetas_sim_ideal.min(), thetas_sim_ideal.max()]
        xticks = xlim.copy()
        xticks.insert(1, np.mean(xticks))
        ag.set_xticks(
            xticks,
            labels=as_pi_str(xticks, precision=0, frac=0),
            xlim=xlim,
            axes=ax,
            sharex=0,
            xlim_pad_ratio=0.02,
        )
        ylim = [y.min(), y.max()]
        yticks = ylim.copy()
        yticks.insert(1, np.mean(yticks))
        ag.set_yticks(yticks, ylim=ylim, axes=ax, sharey=0, ylim_pad_ratio=0.02)

        labelpad = -8
        ax.set_xlabel(
            r"$\theta$",
            fontdict={"fontsize": FONTSIZE},
            labelpad=labelpad,
        )
        ax.set_ylabel(
            r"$m$",
            fontdict={"fontsize": FONTSIZE},
            labelpad=labelpad,
        )
        ax.tick_params(axis="both", pad=-4)

        ax.set_zlabel(
            r"$I\mathcal{(S\!:\!F}{})$" + suffix,
            usetex=USE_TEX,
            fontdict={"fontsize": FONTSIZE},
            labelpad=labelpad,
        )
        ax.set_zlim(0, 2)
        ax.set_zticks([0, 1, 2])

    # ----- general -----
    ag.tick_params(direction="out")
    ag.grid(False)
    annot_alphabet(
        [ax_grid, ax_circuit, MI_axes[0], axes_discord[0]],
        fontsize=FONTSIZE,
        dx=-0.05,
        dy=-0.025,
        transform="fig",
        top_bond_dict={2: 0, 1: 0},
        left_bond_dict={3: 0},
        dx_dict={0: 0.1, 1: 0.05, 2: 0.05},
        dy_dict={3: 0.3},
        zorder=np.inf,
        upper=1,
    )

    if save:
        fig_name = DATA_DIR / "fig3" / f"fig3.pdf"
        fig.savefig(fig_name, pad_inches=0, transparent=True)
        print_info("Saved ", wrap_emph(fig_name.as_posix()))


def plot_3x3(ax=None):
    ax.set_aspect("equal")
    ax.axis("off")

    xs = [0, 1, 2] * 3
    ys = [0] * 3 + [1] * 3 + [2] * 3
    xs.pop(4)
    ys.pop(4)
    ax.scatter(
        xs,
        ys,
        zorder=np.inf,
        ec="#315b92",
        fc="white",
        s=130,
        linewidths=1.1,
    )
    ax.scatter(
        [1],
        [1],
        zorder=np.inf,
        ec="#315b92",
        fc="#ffdf62",
        s=130,
        linewidths=1.1,
    )
    for i in range(3):
        ax.plot([i, i], [0, 2], lw=1.2, color="k", zorder=-np.inf)
        ax.plot([0, 2], [i, i], lw=1.2, color="k", zorder=-np.inf)
    i = 1
    for y, x in [
        (1, 1),
        (2, 1),
        (1, 2),
        (0, 1),
        (1, 0),
        (2, 2),
        (0, 2),
        (0, 0),
        (2, 0),
    ]:
        ax.text(
            x,
            y,
            i,
            ha="center",
            va="center",
            fontdict={"fontsize": FONTSIZE},
            zorder=np.inf,
        )
        i += 1

    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)


@article_style()
def plot_circuit(ax=None):
    total_layer = 14
    numq = 9
    dy = 1
    circle_size = 18
    circuit_start = 0.08
    gate_width = 0.04
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
            0.03,
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
    U_color2 = "#f79979"
    TOMO_color = "#8eb387"

    def plot_gate(layer, index, color, custom_xy=None, width=None, height=None):
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
            zorder=np.inf,
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
        layer,
        index0,
        index1,
        hollow=False,
        custom_xy=None,
        width=None,
        height=None,
        isU2=False,
    ):
        if custom_xy is not None:
            x, y0, y1 = custom_xy
        else:
            x = get_x(layer)
            y0 = lines[index0]
            y1 = lines[index1]
        plot_circle(layer, index0, hollow, custom_xy=(x, y0))
        ax.plot([x, x], [y0, y1], color="k", lw=1, zorder=-np.inf)
        if isU2:
            color = U_color2
        else:
            color = U_color
        plot_gate(layer, index1, color, custom_xy=(x, y1), width=width, height=height)

    plot_gate(0, 1, H_color)

    i = 1
    for index in [2, 3, 4, 5]:
        plot_cu(i, 1, index)
        i += 1
    j = 4
    for index in [6, 9, 6, 7, 7, 8, 8, 9]:
        plot_cu(i, j // 2, index, isU2=1)
        j += 1
        i += 1

    for k in range(1, 6):
        plot_gate(i, k, TOMO_color)

    # ------- legend -------
    y0 = lines[1] + dy * 2.6
    y1 = lines[1] + dy * 3.65
    y2 = y0 - dy * 1.23
    fd = {"fontsize": FONTSIZE - 1}
    # ------- h -------
    x = 0.37
    y = (y0 + y1) / 2 * 0.97
    plot_gate(
        0,
        0,
        H_color,
        custom_xy=(x, y),
        width=gate_width * 1.3,
        height=gate_height * 1.3,
    )
    ax.text(x, y2, "Hadamard", ha="center", va="center", zorder=np.inf, fontdict=fd)
    # ------- u -------
    for j in [0, 1]:
        x = 0.52 + 0.13 * j
        plot_cu(
            0,
            0,
            0,
            custom_xy=(x, y1, y0),
            width=gate_width * 1.0,
            height=gate_height * 1.0,
            isU2=j,
        )
        if j == 0:
            label = r"$U_{\oslash}(2\theta)$"
        else:
            label = r"$U_{\oslash}(\theta+3\pi/2)$"
        ax.text(x, y2, label, ha="center", va="center", zorder=np.inf, fontdict=fd)
    # ----- tomo -----
    x = 0.78
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
