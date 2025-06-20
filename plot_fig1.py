from functools import partial

import h5py
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import Circle, PathPatch, Wedge
from matplotlib.path import Path
import numpy as np

from fig_utils import AxesGroup, annot_alphabet, fig_in_a4
from output_ctrl import *
from path import DATA_DIR
from plot_bloch import plot_bloch_2d
from plot_toolbox import plot1d as _plot1d
from styles import FONTSIZE, LW, MEW, MS, USE_TEX, article_style, format_legend

plot1d = partial(_plot1d, constrained_layout=False)


@article_style()
def plot_random_part_env(save=False, vary_N=False, num_seed=20):
    m = 3
    N = 10
    if vary_N:
        xlabel = "N"
        suffix = "_vary_N"
    else:
        xlabel = "m"
        suffix = ""

    path = DATA_DIR / f"fig1_part_env{suffix}_seed={num_seed}.h5"
    start = 0
    seeds = list(range(start, start + num_seed))

    fig = fig_in_a4(0.3, 0.13, dpi=200)
    ag = AxesGroup(1, 3, figs=fig)
    ax = ag.axes[0]

    fig.subplots_adjust(bottom=0.2, top=0.85, left=0.27, right=0.83)

    colors = ["#08519b", "#a63603"]
    labels = [
        r"$I(\mathcal{S}\!:\!\mathcal{F})$",
        r"$\mathcal{D}(\mathcal{S}\!:\!\check{\mathcal{F}})$",
    ]
    handles, labels = get_legend(
        ax,
        lw=LW,
        mew=MEW,
        ms=MS,
        labels=labels,
        colors=colors,
    )

    all_Is = []
    all_Ds = []
    for i, seed in enumerate(seeds):
        with h5py.File(path, "r") as f:
            ms = f[f"seed={seed}/ms"][()]
            Is = f[f"seed={seed}/Is"][()]
            chi = f[f"seed={seed}/chi"][()]
            D = f[f"seed={seed}/D"][()]
            S_center = f[f"seed={seed}/S_center"][()]
        all_Is.append(Is)
        all_Ds.append(D)
        y = np.array([Is, D])
        plot1d(
            ms,
            y,
            ax=ax,
            lw=LW,
            mew=MEW,
            ms=MS,
            alpha=0.15,
            color=colors,
        )
    y = np.array([np.mean(all_Is, 0), np.mean(all_Ds, 0)])
    plot1d(
        ms,
        y,
        ax=ax,
        lw=LW,
        mew=MEW,
        ms=MS,
        color=colors,
    )
    if vary_N:
        ax.set_title(f"$m=${m}", fontdict={"fontsize": FONTSIZE})
    else:
        ax.set_title(f"$N=${N}", fontdict={"fontsize": FONTSIZE})

    if not vary_N:
        format_legend(ax, handles=handles, labels=labels, usetex=1)

    xlim = [ms.min(), ms.max()]
    ag.set_xticks(
        xlim,
        xlim=xlim,
        axes=ax,
        sharex=0,
        xlim_pad_ratio=0.02,
    )
    ag.set_yticks([0, 1, 2], ylim=[0, 2], axes=ax, sharey=0, ylim_pad_ratio=0.02)

    ag.set_xlabel(
        "$%s$" % xlabel,
        fontsize=FONTSIZE,
        clean_others=0,
        axes=ax,
        sharex=0,
        labelpad=1,
    )
    ag.set_ylabel(
        r"Information",
        fontsize=FONTSIZE,
        clean_others=0,
        axes=ax,
        sharey=0,
        labelpad=0,
    )
    ax.hlines(1, *ax.get_xlim(), color="k", ls="--", lw=0.6)

    ag.tick_params(direction="out")
    ag.grid(False)

    if save:
        path = DATA_DIR / "fig1" / f"fig1_part_env{suffix}.pdf"
        ax.figure.savefig(path, pad_inches=0, transparent=True)
        print_info("Saved ", wrap_emph(path.as_posix()))


@article_style()
def plot_env(save=False):
    science = 1
    inch2cm = 2.54
    A4_xsize = 18 / inch2cm
    A4_ysize = 29.7 / inch2cm

    xsize = A4_xsize * 1.0
    ysize = A4_ysize * 0.3

    fig = plt.figure(figsize=(xsize, ysize), dpi=250)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=None)
    ax.axis("off")

    filenames = [
        [
            "fig1_bloch_center=2_env=2_haar_p1=0.5_seeds=20_angle=0.5",
            "fig1_bloch_center=2_env=2_haar_p1=0.5_seeds=20_angle=0.5_ptrace",
        ],
        [
            "fig1_bloch_center=2_env=6_haar_p1=0.5_seeds=20_angle=0.5",
            "fig1_bloch_center=2_env=6_haar_p1=0.5_seeds=20_angle=0.5_ptrace",
        ],
        [
            "fig1_bloch_center=2_env=10_haar_p1=0.5_seeds=20_angle=0.5",
            "fig1_bloch_center=2_env=10_haar_p1=0.5_seeds=20_angle=0.5_ptrace",
        ],
        [
            "fig1_bloch_center=2_env=inf_haar_p1=0.5_seeds=20_angle=0.5",
            "fig1_bloch_center=2_env=inf_haar_p1=0.5_seeds=20_angle=0.5_ptrace",
        ],
    ]

    axs_pos = {}
    for n_i, filename_ in enumerate(filenames):
        for filename in filename_:
            ax_width = 0.09
            ax_height = ax_width * xsize / ysize

            ax_pos_1 = [
                [0.18 + n_i * 0.2 + i * ax_width * 1.01, 0.02, ax_width, ax_height]
                for i in range(2)
            ]

            ax_width = 0.11
            ax_height = ax_width * xsize / ysize
            ax_pos_2 = [
                [
                    ax_pos_1[2 * i][0]
                    + 0.5
                    * (
                        ax_pos_1[2 * i + 1][0]
                        + ax_pos_1[2 * i + 1][2]
                        - ax_pos_1[2 * i][0]
                        - ax_width
                    ),
                    ax_pos_1[2 * i][1] + 0.35,
                    ax_width,
                    ax_height,
                ]
                for i in range(1)
            ]

            des = filename.split("_")[-1]
            if des == "ptrace":
                axs_pos[filename] = ax_pos_1
            else:
                axs_pos[filename] = ax_pos_2

    ax_width = (
        axs_pos[filenames[-1][0]][0][0]
        + axs_pos[filenames[-1][0]][0][2]
        - axs_pos[filenames[0][0]][0][0]
    )
    ax_height = (
        axs_pos[filenames[0][0]][0][1]
        + axs_pos[filenames[0][0]][0][3]
        - axs_pos[filenames[0][1]][0][1]
    )
    coeff_ = 1.2
    ax_pos = [
        axs_pos[filenames[0][0]][0][0] - 0.5 * ax_width * (coeff_ - 1),
        axs_pos[filenames[0][1]][0][1],
        ax_width * coeff_,
        ax_height,
    ]
    ax = fig.add_axes(ax_pos, facecolor=[0, 0, 0, 0])
    xlims = [0, ax_pos[2]]
    ylims = [-1.4, 1]
    xs = [
        (axs_pos[filenames[i][0]][0][0] + axs_pos[filenames[i][0]][0][2] * 0.5)
        - ax_pos[0]
        for i in range(len(filenames))
    ]
    des = [" = 2", " = 6", " = 10", r"$\rightarrow$"]
    for i, x_ in enumerate(xs):
        ax.plot([x_] * 2, [0, 0.08], linewidth=0.5, color="k", ls="-")
        ax.text(
            x_,
            1.18,
            r"$N$" + des[i],
            fontsize=FONTSIZE,
            fontname="Arial",
            horizontalalignment="center",
            verticalalignment="center",
        )
        if i == 3:
            ax.text(
                x_ + 0.023,
                1.2,
                "$\infty$",
                fontsize=FONTSIZE,
                fontname="Arial",
                horizontalalignment="center",
                verticalalignment="center",
                usetex=1,
            )

        px_ = [x_, x_ - 0.1, x_ + 0.1][::-1]
        py_ = [0, -1.1, -1.1][::-1]

        path = Path(np.array([px_, py_]).transpose())
        patch = PathPatch(path, facecolor="none", edgecolor="none")
        ax.add_patch(patch)
        _, yv = np.meshgrid(
            np.linspace(xlims[0], xlims[1], 100), np.linspace(ylims[0], ylims[1], 100)
        )
        cmap = LinearSegmentedColormap.from_list(
            "mycmap", [[0.4, 0.4, 0.4, 0.5], [1, 1, 1, 0]]
        )
        im = ax.imshow(
            yv,
            cmap=cmap,
            extent=[xlims[0], xlims[1], min(py_), max(py_)],
            aspect="auto",
        )
        im.set_clip_path(patch)

    ax.arrow(
        xlims[0] + np.diff(xlims)[0] * 0.05,
        0,
        np.diff(xlims)[0] - np.diff(xlims)[0] * 0.1,
        0,
        color="k",
        linewidth=0.5,
        width=0.008,
        head_width=0.05,
        head_length=0.01,
    )
    ax.text(
        xlims[0] + np.diff(xlims)[0] * 0.05,
        0.1,
        "Quantum",
        fontsize=FONTSIZE,
        fontname="Arial",
        fontweight="bold",
        horizontalalignment="center",
        verticalalignment="center",
    )
    ax.text(
        np.diff(xlims)[0] - np.diff(xlims)[0] * 0.05,
        0.1,
        "Classical",
        fontsize=FONTSIZE,
        fontname="Arial",
        fontweight="bold",
        horizontalalignment="center",
        verticalalignment="center",
    )
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.grid(False)
    ax.axis("off")

    color_whole_env = "#dc8686"
    colors = ["#fdca81", "#6eaadb"]
    ax_whole_env = None
    for ax_i in [1, 0]:
        ax_width = (
            axs_pos[filenames[-1][0]][0][0]
            + axs_pos[filenames[-1][0]][0][2]
            - axs_pos[filenames[0][0]][0][0]
        )
        ax_height = (
            axs_pos[filenames[0][0]][0][1]
            + axs_pos[filenames[0][0]][0][3]
            - axs_pos[filenames[0][1]][0][1]
        )
        ax_width = 1 - ax_width * 1.18
        ax_height = ax_width * xsize / ysize
        ax_pos = [0, axs_pos[filenames[0][ax_i]][0][1], ax_width, ax_height]
        ax = fig.add_axes(ax_pos, facecolor=[0, 0, 0, 0])
        if ax_i == 1:
            center_x = [-0.1, 0.1]
        else:
            center_x = [0, 0]
        center_y = [-0.2, -0.2]
        r = [0.83, 0.5]
        for i, r_ in enumerate(r):
            for j, x_, y_ in zip(range(len(center_x)), center_x, center_y):
                names = (
                    ["N", "N-1", "...", "N/2+3", "N/2+2", "N/2+1"]
                    if j == 0
                    else ["1", "2", "3", "...", "N/2-1", "N/2"][::-1]
                )
                phis = [90, 120, 210, 240, 270] if j == 0 else [270, 300, 390, 420, 450]
                phis_dash = [150, 180] if j == 0 else [330, 360]
                phis_text = (
                    np.linspace(90, 240, 6) if j == 0 else np.linspace(270, 420, 6)
                )
                c_1 = 1 if j == 0 else -1
                c_2 = 1 if j == 0 else 3
                color_ = colors[j] if i == 0 else "w"
                if i == 0:
                    for p_i, phi in enumerate(phis[:-1]):
                        ax.add_patch(
                            Wedge(
                                (x_, y_),
                                r_,
                                phi,
                                phis[p_i + 1],
                                facecolor=color_ if ax_i == 1 else color_whole_env,
                                edgecolor="k",
                                linewidth=0.5,
                            )
                        )
                        if ax_i == 0:
                            ax_whole_env = ax

                    for p_i, phi in enumerate(phis_dash):
                        phi_1 = phi / 180 * np.pi
                        ax.plot(
                            [x_ + r[1] * np.cos(phi_1), x_ + r[0] * np.cos(phi_1)],
                            [y_ + r[1] * np.sin(phi_1), y_ + r[0] * np.sin(phi_1)],
                            color="k",
                            linewidth=0.5,
                            ls="--",
                        )

                    for p_i, phi in enumerate(phis_text):
                        phi_1 = (phi + 90 / len(phis_text)) / 180 * np.pi
                        if names[p_i] in ["..."]:
                            fontsize = FONTSIZE
                            ax.text(
                                x_ + r_ * 0.8 * np.cos(phi_1) + 0.01 * c_1 * c_2,
                                y_ + r_ * 0.8 * np.sin(phi_1) - 0.03 * c_1,
                                "$%s$" % names[p_i],
                                fontsize=fontsize,
                                fontname="Arial",
                                fontweight="normal",
                                horizontalalignment="center",
                                verticalalignment="center",
                                rotation=phi + 100,
                            )
                        else:
                            fontsize = 6 * (len(names[p_i]) < 3) + 4 * (
                                len(names[p_i]) >= 3
                            )
                            ax.text(
                                x_ + r_ * 0.8 * np.cos(phi_1),
                                y_ + r_ * 0.8 * np.sin(phi_1),
                                "$%s$" % names[p_i],
                                fontsize=fontsize,
                                fontname="Arial",
                                fontweight="normal",
                                horizontalalignment="center",
                                verticalalignment="center",
                            )
                else:
                    if np.diff(center_x)[0] != 0:
                        ax.add_patch(
                            Wedge(
                                (x_, y_),
                                r_,
                                phis[0],
                                phis[0] + 180,
                                facecolor="w",
                                edgecolor="k",
                                linewidth=0.5,
                            )
                        )

        if np.diff(center_x)[0] != 0:
            ax.add_patch(
                Circle(
                    (np.mean(center_x), np.mean(center_y)),
                    radius=r[1],
                    facecolor="w",
                    edgecolor="none",
                    linewidth=0.5,
                )
            )
        else:
            ax.add_patch(
                Circle(
                    (np.mean(center_x), np.mean(center_y)),
                    radius=r[1],
                    facecolor="w",
                    edgecolor="k",
                    linewidth=0.5,
                )
            )
        ax.add_patch(
            Circle(
                (np.mean(center_x), np.mean(center_y)),
                radius=0.35,
                edgecolor="k",
                linewidth=0.5,
                facecolor="#b9d8da",
            )
        )
        ax.text(
            np.mean(center_x),
            np.mean(center_y),
            r"$\mathcal{S}$",
            fontsize=FONTSIZE,
            fontname="Arial",
            fontweight="bold",
            horizontalalignment="center",
            verticalalignment="center",
            usetex=USE_TEX,
        )
        if ax_i == 0:
            ax.text(
                0.4,
                0.7,
                r"$\mathcal{E}$",
                fontsize=FONTSIZE,
                fontname="Arial",
                fontweight="normal",
                horizontalalignment="center",
                verticalalignment="center",
                usetex=USE_TEX,
            )
        else:
            ax.text(
                -0.5,
                0.7,
                r"$\mathcal{F}_A$",
                fontsize=FONTSIZE,
                fontname="Arial",
                fontweight="normal",
                horizontalalignment="center",
                verticalalignment="center",
                usetex=USE_TEX,
            )
            ax.text(
                +0.5,
                0.7,
                r"$\mathcal{F}_B$",
                fontsize=FONTSIZE,
                fontname="Arial",
                fontweight="normal",
                horizontalalignment="center",
                verticalalignment="center",
                usetex=USE_TEX,
            )

        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_aspect("equal")
        ax.grid(False)
        ax.axis("off")

    colors = ["#ecaf58", "#095db1"]
    for n_i, filename_ in enumerate(filenames):
        for filename in filename_:
            des = filename.split("_")[-1]
            datas = {}
            with h5py.File(DATA_DIR / f"{filename}.h5", "r") as f:
                if des == "ptrace":
                    for k in f.keys():
                        g = f[k]
                        datas[k] = np.vstack(
                            [
                                g["xs"][()].flatten(),
                                g["ys"][()].flatten(),
                                g["zs"][()].flatten(),
                                g["Xs"][()].flatten(),
                            ]
                        )
                else:
                    datas["result"] = np.vstack(
                        [
                            f["xs"][()].flatten(),
                            f["ys"][()].flatten(),
                            f["zs"][()].flatten(),
                            f["Xs"][()].flatten(),
                        ]
                    )
            ax_pos = axs_pos[filename]

            num = filename.split("env=")[1].split("_")[0]
            if des == "ptrace":
                keys = [key for key in datas]
                datatype = "scatter"
                steps = [
                    datas[keys[0]].shape[1] / np.min([50, datas[keys[0]].shape[1]]),
                    datas[keys[0]].shape[1] / np.min([400, datas[keys[0]].shape[1]]),
                    datas[keys[0]].shape[1] / np.min([800, datas[keys[0]].shape[1]]),
                    datas[keys[0]].shape[1] / np.min([1, datas[keys[0]].shape[1]]),
                ]
                step = 1 if num == "inf" else steps[n_i]
                idxs = np.arange(0, datas[keys[0]].shape[1], 1, dtype=int)
            else:
                keys = ["result"]
                datatype = "arrow"  #'arrow'
                steps = [
                    1,
                    1,
                    datas[keys[0]].shape[1] / np.min([800, datas[keys[0]].shape[1]]),
                    datas[keys[0]].shape[1] / np.min([1, datas[keys[0]].shape[1]]),
                ]
                step = 1 if num == "inf" else steps[n_i]
                idxs = np.arange(0, datas[keys[0]].shape[1], step, dtype=int)

            axs = [
                fig.add_axes(ax_pos_, facecolor=[0, 0, 0, 0])
                for ax_pos_ in ax_pos[: len(keys)]
            ]
            for i, key in enumerate(keys):
                data = datas[key]
                xs = data[0, idxs].flatten()
                ys = data[1, idxs].flatten()
                zs = data[2, idxs].flatten()
                p1s = data[3, idxs].flatten()

                plot_bloch_2d(
                    xs,
                    ys,
                    zs,
                    s=6,
                    Xs=p1s,
                    ax=axs[i],
                    color=colors[i],
                    datatype=datatype,
                )
    annot_alphabet(
        [ax_whole_env],
        fontsize=FONTSIZE,
        dx=0.00,
        dy=-0.08,
        transform="fig",
        offset=3,
        upper=science,
    )

    if save:
        name = "fig1_env"
        if science:
            name += "_science"
        path = DATA_DIR / "fig1" / f"{name}.pdf"
        ax.figure.savefig(path, pad_inches=0, transparent=True)
        print_info("Saved ", wrap_emph(path.as_posix()))


def get_legend(ax, colors, labels, **kwargs):
    for i, color in enumerate(colors):
        label = labels[i]
        plot1d([1], [1], ax=ax, color=color, label=label, **kwargs)
    handles, labels = ax.get_legend_handles_labels()
    ax.clear()
    return handles, labels
