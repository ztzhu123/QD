from functools import partial
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from fig_utils import AxesGroup, annot_alphabet, fig_in_a4, num_to_alphabet
from output_ctrl import print_info, wrap_emph
from path import DATA_DIR
from plot_toolbox import plot1d as _plot1d
from plot_toolbox import plot_ecdf as _plot_ecdf
from styles import CAPSIZE, FONTSIZE, LEGEND_SIZE, LW, MS, USE_TEX, article_style

plot1d = partial(_plot1d, constrained_layout=False)

DEVICE_INFO_FILENAME = DATA_DIR.joinpath("device_info.xlsx")

def is_qubit(qname):
    return qname.startswith("q")


def collect_result(full=True):
    full = bool(full)
    _9q_result = {}
    _12q_result = {}
    for numq in [9, 12]:
        result = eval(f"_{numq}q_result")
        if numq == 12 and full:
            xeb_name = f"xeb_{numq}q_full.h5"
            read_name = f"meas_fid_{numq}q_full"
        else:
            xeb_name = f"xeb_{numq}q.h5"
            read_name = f"meas_fid_{numq}q"
        with h5py.File(DATA_DIR / xeb_name, "r") as f:
            for key in f.keys():
                result[key] = f[key][()]
        df = pd.read_excel(DEVICE_INFO_FILENAME, f"t1t2_{numq}q", index_col=0)
        df_meas_fid = pd.read_excel(DEVICE_INFO_FILENAME, read_name, index_col=0)
        qnames = df.loc["T1"].keys()
        result["t1s"] = []
        result["t2s"] = []
        result["read_errors"] = []
        for qname in qnames:
            if not is_qubit(qname):
                continue
            result["t1s"].append(df.loc["T1", qname])
            result["t2s"].append(df.loc["T2_SE", qname])
            result["read_errors"].append((1 - df_meas_fid.loc["mean", qname]))
        assert len(result["t1s"]) == numq
        assert len(result["t2s"]) == numq
        assert len(result["read_errors"]) == numq
        np.testing.assert_almost_equal(np.mean(result["t1s"]), df.loc["T1", "mean"])
        np.testing.assert_almost_equal(np.median(result["t1s"]), df.loc["T1", "median"])
        np.testing.assert_almost_equal(np.mean(result["t2s"]), df.loc["T2_SE", "mean"])
        np.testing.assert_almost_equal(
            np.median(result["t2s"]), df.loc["T2_SE", "median"]
        )
    return _9q_result, _12q_result


@article_style()
def plot_all(save=False, full=True):
    fig = fig_in_a4(1, 0.34, dpi=150)
    ag = AxesGroup(4, 2, max_rows_or_cols_per_fig=6, figs=fig)
    fig.subplots_adjust(
        hspace=0.5, wspace=0.3, top=0.93, bottom=0.12, left=0.15, right=0.9
    )
    _9q_result, _12q_result = collect_result(full=full)
    colors_t1 = ["#095db1", "#fa6d2a"]
    colors = ["#0b5fb1", "#c72c2e", "#00676a"]
    for i, numq in enumerate([9, 12]):
        ax = ag.axes[i]
        result = eval(f"_{numq}q_result")
        dx = 0.3
        dy = -0.3
        plot_ecdf(
            result["t1s"],
            ax=ax,
            ag=ag,
            precision=0,
            color=colors_t1[0],
        )
        plot_ecdf(
            result["t2s"],
            ax=ax,
            xlabel=r"Coherence Time $(\mu{\rm s})$",
            ag=ag,
            xlim=[0, 200],
            precision=0,
            color=colors_t1[1],
        )
        ax.set_title(f"{numq}Q", fontsize=FONTSIZE)

        fontdict = {"fontsize": FONTSIZE}
        ax.text(
            0.02 + dx,
            0.98 + dy - 0.02,
            f"Median",
            ha="left",
            va="center",
            transform=ax.transAxes,
            fontdict=fontdict,
            color="k",
        )
        ax.text(
            0.02 + dx,
            0.98 + dy - 0.15,
            r"$T_1$  : %d $\mu$s" % np.median(result["t1s"]).round(),
            ha="left",
            va="center",
            transform=ax.transAxes,
            fontdict=fontdict,
            color=colors_t1[0],
        )
        ax.text(
            0.02 + dx,
            0.98 + dy - 0.3,
            r"$T_2^{\rm SE}$: %d $\mu$s" % np.median(result["t2s"]).round(),
            ha="left",
            va="center",
            transform=ax.transAxes,
            fontdict=fontdict,
            color=colors_t1[1],
        )

        ax = ag.axes[i + 2]
        dx = 0.62
        dy = -0.55
        keys = ["sq_errors", "cz_errors", "read_errors"]
        for j, key in enumerate(keys):
            plot_ecdf(
                np.array(result[key]) * 100,
                ax=ax,
                ag=ag,
                color=colors[j],
                xlim=[0.01, 10],
                xlabel="Error (%)",
            )
            if key == "sq_errors":
                prefix = "SQ:"
            elif key == "cz_errors":
                prefix = "CZ:"
            elif key == "read_errors":
                prefix = "Readout:"
            ax.text(
                0.02 + dx,
                0.98 + dy - 0.1 * (j + 1),
                f"{prefix} {np.median(result[key])*100:.3f} %",
                ha="left",
                va="center",
                transform=ax.transAxes,
                fontdict=fontdict,
                color=colors[j],
            )
        ax.set_xscale("log")
        ax.set_title(f"{numq}Q", fontsize=FONTSIZE)

        fontdict = {"fontsize": FONTSIZE}
        ax.text(
            0.02 + dx,
            0.98 + dy,
            f"Median",
            ha="left",
            va="center",
            transform=ax.transAxes,
            fontdict=fontdict,
            color="k",
        )

    ag.set_ylabel("Integrated histogram", fontsize=FONTSIZE, clean_others=1, sharey=1)
    ag.set_yticks([0, 0.5, 1], ylim=[0, 1], sharey=1, ylim_pad_ratio=0.0)

    annot_alphabet(ag.axes, dx=-0.03, fontsize=FONTSIZE, upper=1)
    ag.tick_params(direction="out")
    ag.grid(False)
    if save:
        name = "sm_device"
        fig_name = DATA_DIR / f"{name}.pdf"
        fig.savefig(fig_name, pad_inches=0, transparent=True)
        print_info("Saved ", wrap_emph(fig_name.as_posix()))


def plot_ecdf(
    values,
    ax,
    xlim=None,
    ylim=[0, 1],
    xlabel=None,
    ylabel=None,
    xticks=None,
    yticks=None,
    precision=3,
    dash_color=None,
    annot_color="k",
    xlabelpad=None,
    unit="$\mu$s",
    ag=None,
    **kwargs,
):
    ax = _plot_ecdf(
        values,
        ax=ax,
        unit=unit,
        lw=LW,
        annot=False,
        annot_size=FONTSIZE,
        mean=0,
        median=1,
        precision=precision,
        sep="\n",
        xlim=xlim,
        ylim=ylim,
        dash_color=dash_color,
        annot_color=annot_color,
        constrained_layout=False,
        **kwargs,
    )
    if xlabel:
        ax.set_xlabel(xlabel, fontdict={"size": FONTSIZE}, labelpad=xlabelpad)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    if ag is not None:
        if not ag.is_left(ax):
            ax.tick_params(labelleft=False)
            ax.set_ylabel(None)
        # if not ag.is_bottom(ax_ecdf):
        #     ax.tick_params(labelbottom=False)
        #     ax.set_xlabel(None)
    return ax
