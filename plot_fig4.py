from fractions import Fraction
from functools import partial

import h5py
import numpy as np

from fig_utils import AxesGroup, fig_in_a4
from output_ctrl import print_info, wrap_emph
from path import DATA_DIR
import plot_fig3
from plot_toolbox import plot1d as _plot1d
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


def collect_ob_result(norm=False):
    filename = DATA_DIR / "measure_O_corr=1_tq_error_0.304.h5"
    result_O = {}
    with h5py.File(filename, "r") as f:
        for key in f.keys():
            result_O[key] = f[key][()]
    result_O["Os_exp_mean"] = result_O["Os_exp"].mean(0)
    result_O["Os_exp_std"] = result_O["Os_exp"].std(0, ddof=1)

    filename = DATA_DIR / "MI_valid=1_corr=1_tq_error_0.304.h5"
    result_MI = plot_fig3.collect_discord_result(norm=norm)
    return result_O, result_MI


@article_style()
def plot_all(save=False, norm=False):
    result_O, result_MI = collect_ob_result(norm=norm)

    fig = fig_in_a4("1col", 0.16, dpi=200)
    ag = AxesGroup(1, 3, figs=fig)

    fig.subplots_adjust(bottom=0.2, wspace=0.3, left=0.21, right=0.79)

    colors = ["#08519c", "#a63603"]
    if norm:
        suffix = r" / $H_{\mathcal{S}}$"
    else:
        suffix = ""

    # ----- plot mutual information -----
    thetas_sim_ideal = result_O["thetas_sim_ideal"]
    thetas_sim_noisy = result_O["thetas_sim_noisy"]
    thetas_exp = result_O["thetas_exp"]
    ax = ag.axes[0]
    color = colors[0]

    plot1d(
        thetas_exp,
        result_O["Os_exp_mean"],
        result_O["Os_exp_std"],
        ax=ax,
        lw=LW,
        mew=MEW,
        ms=MS,
        capsize=CAPSIZE,
        marker="o",
        ls="",
        label="Exp.",
        zorder=100,
        clip_on=False,
        color=color,
    )
    plot1d(
        thetas_sim_noisy,
        result_O["Os_sim_noisy"],
        ax=ax,
        lw=LW,
        label="Sim.",
        color=color,
        alpha=0.6,
    )

    format_legend(ax, loc="upper left")

    xlim = [thetas_sim_noisy.min(), thetas_sim_noisy.max()]
    xticks = xlim.copy()
    xticks.insert(0, np.mean(xticks))
    for i in range(1):
        ag.set_xticks(
            xticks,
            labels=as_pi_str(xticks, precision=0, frac=0),
            xlim=xlim,
            axes=ag.axes[i],
            sharex=0,
            xlim_pad_ratio=0.02,
            fontsize=FONTSIZE,
        )
        ag.set_xlabel(
            r"$\theta$",
            fontsize=FONTSIZE,
            clean_others=0,
            axes=ag.axes[i],
            sharex=0,
        )

    ag.set_yticks(
        [0, 0.5, 1],
        labels=[0, 0.5, 1],
        ylim=[0, 1],
        axes=ax,
        sharey=0,
        ylim_pad_ratio=0.02,
    )

    ag.set_ylabel(
        r"$\mathsf{\langle}\mathcal{O}\mathsf{\rangle}$",
        fontsize=FONTSIZE,
        clean_others=0,
        axes=ax,
        sharey=0,
        usetex=USE_TEX,
    )

    # ----- colors -----
    ax.yaxis.label.set_color(color)
    ax.tick_params(axis="y", colors=color)

    # ----- plot mutual information -----
    ax = ax.twinx()
    ax.spines["left"].set_color(color)
    color = colors[1]
    ax.spines["right"].set_color(color)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["right"].set_linewidth(1.5)
    ax.yaxis.label.set_color(color)
    ax.tick_params(axis="y", colors=color)

    # for i in range(1, 9):
    for env in [2]:
        plot1d(
            result_MI["thetas_exp"],
            result_MI["MIs_exp_mean"][:, env - 1],
            result_MI["MIs_exp_std"][:, env - 1],
            ax=ax,
            lw=LW,
            mew=MEW,
            ms=MS,
            capsize=CAPSIZE,
            marker="o",
            ls="",
            label="Exp.",
            zorder=100,
            clip_on=False,
            color=color,
        )
        plot1d(
            result_MI["thetas_sim_noisy"],
            result_MI["MIs_sim_noisy"][:, env - 1],
            ax=ax,
            lw=LW,
            label="Sim.",
            color=color,
            alpha=0.6,
        )

    handles, labels = ax.get_legend_handles_labels()
    format_legend(
        ax,
        loc="upper right",
        # bbox_to_anchor=(0.5, 0.87),
        handles=handles,
        labels=labels,
    )

    xlim = [result_MI["thetas_sim_noisy"].min(), result_MI["thetas_sim_noisy"].max()]
    ag.set_yticks(
        [0, 0.5, 1],
        labels=[0, 0.5, 1],
        ylim=[0, 1],
        axes=ax,
        sharey=0,
        ylim_pad_ratio=0.02,
    )

    ax.set_ylabel(
        r"$I\mathcal{(S:F}{})$" + suffix,
        fontdict={"fontsize": FONTSIZE},
        usetex=USE_TEX,
    )
    ax.set_title("$N=4,~m=2$", fontsize=FONTSIZE, color=color)

    ag.tick_params(direction="out")
    ax.grid(False)
    ag.grid(False)

    if save:
        fig_name = DATA_DIR / "fig4" / f"fig4.pdf"
        fig.savefig(fig_name, pad_inches=0)
        print_info("Saved ", wrap_emph(fig_name.as_posix()))


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
