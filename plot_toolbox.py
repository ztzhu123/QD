from copy import deepcopy
from itertools import cycle

import matplotlib as mpl
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from colors import color_palette
from fig_utils import subplot, write_footnote

footnote = write_footnote  # for compatibility


mpl.rcParams.update({"font.size": 14, "lines.linewidth": 2})


def plot1d(
    x,
    y=None,
    yerr=None,
    xsim=None,
    ysim=None,
    xlabel=None,
    ylabel=None,
    xlim="auto",
    ylim="auto",
    title=None,
    fig_title=None,
    grid=True,
    errorbar=True,
    ax=None,
    index_ls_map={},
    figsize=[6.5, 4],
    xtick_unit_pi=False,
    fill_alpha=0.2,
    sim_label=None,
    sim_same_color=False,
    sim_kwargs=None,
    legend_size=11,
    legend_ncol=1,
    hollow=True,
    constrained_layout=True,
    legend_loc="best",
    title_color=None,
    title_size=None,
    **line_kwargs,
):
    if "color" in line_kwargs and line_kwargs["color"] is None:
        line_kwargs.pop("color")
    if ax is None:
        ax = subplot(figsize=figsize)
    fig = ax.figure
    fig.set_constrained_layout(constrained_layout)
    if y is None:
        y = x
        x = np.arange(np.shape(y)[-1])
    x, y = np.asarray(x), np.asarray(y)
    multi_lines = len(y.shape) > 1
    n = y.shape[0] if multi_lines else 1
    sim_kwargs = sim_kwargs or {}

    # record the min and max values of x and y for this axis
    lim_vars = {
        "xmin": x.min(),
        "xmax": x.max(),
        "ymin": y.min(),
        "ymax": y.max(),
    }
    if yerr is not None:
        yerr = np.asarray(yerr)
        lim_vars["ymin"] = min((y - yerr).min(), lim_vars["ymin"])
        lim_vars["ymax"] = max((y + yerr).max(), lim_vars["ymax"])
    if ysim is not None:
        ysim = np.asarray(ysim)
        lim_vars["ymin"] = min(ysim.min(), lim_vars["ymin"])
        lim_vars["ymax"] = max(ysim.max(), lim_vars["ymax"])
    if xsim is not None:
        xsim = np.asarray(xsim)
        lim_vars["xmin"] = min(xsim.min(), lim_vars["xmin"])
        lim_vars["xmax"] = max(xsim.max(), lim_vars["xmax"])
    else:
        xsim = x
    for key in lim_vars:
        full_key = f"_custom_key_{key}"
        v_local = lim_vars[key]
        if hasattr(ax, full_key):
            v = getattr(ax, full_key)
            if "min" in key:
                v = min(v, v_local)
            else:
                v = max(v, v_local)
        else:
            v = v_local
        setattr(ax, full_key, v)
        lim_vars[key] = v

    legend_on = False
    if "marker" in line_kwargs and line_kwargs["marker"] == "o":
        if hollow:
            line_kwargs.setdefault("mfc", "none")
        line_kwargs.setdefault("mew", "1.5")
        line_kwargs.setdefault("ms", "7")

    if not multi_lines:
        if ysim is not None:
            line_kwargs.setdefault("label", "exp.")
            if sim_label is None:
                sim_label = "sim."
            sim_kwargs.setdefault("label", sim_label)
            sim_kwargs.setdefault("ls", "--")
            if not sim_same_color:
                ax.plot(xsim, ysim, **sim_kwargs)
        if yerr is None or not errorbar:
            ax.plot(x, y, **line_kwargs)
            if yerr is not None:
                ax.fill_between(
                    x,
                    y - yerr,
                    y + yerr,
                    color=line_kwargs.get("color", None),
                    ec="None",
                    alpha=fill_alpha,
                )
        else:
            line_kwargs.setdefault("marker", "o")
            if hollow:
                line_kwargs.setdefault("mfc", "None")
            line_kwargs.setdefault("capsize", 5)
            ax.errorbar(x, y, yerr, **line_kwargs)
        if sim_same_color:
            sim_kwargs["color"] = ax.get_lines()[-1].get_color()
            sim_kwargs["alpha"] = ax.get_lines()[-1].get_alpha()
            ax.plot(xsim, ysim, **sim_kwargs)
        legend_on = line_kwargs.get("label", None) is not None
    else:
        label = line_kwargs.pop("label", None)
        if label is None:
            label = [""] * n
        else:
            legend_on = True
            if isinstance(label, str):
                label = [label] * n
            for i in range(n):
                label[i] = str(label[i])

        if len(x.shape) == 1:
            x = np.array([x] * n)
        colors = line_kwargs.pop("color", None)
        if isinstance(colors, str) or colors is None:
            colors = [colors]
        colors = cycle(colors)

        if "ls" in line_kwargs:
            ls = line_kwargs.pop("ls")
            index_ls_map = {i: ls for i in range(n)}

        for i in range(n):
            color = next(colors)
            ls = index_ls_map.get(i, None)
            if yerr is None or not errorbar:
                ax.plot(
                    x[i],
                    y[i],
                    color=color,
                    label=label[i],
                    ls=ls,
                    **line_kwargs,
                )
                if yerr is not None:
                    ax.fill_between(
                        x[i],
                        y[i] - yerr[i],
                        y[i] + yerr[i],
                        color=color,
                        ec="None",
                        alpha=0.2,
                    )
            else:
                line_kwargs.setdefault("marker", "o")
                if hollow:
                    line_kwargs.setdefault("mfc", "None")
                line_kwargs.setdefault("capsize", 5)
                ax.errorbar(
                    x[i],
                    y[i],
                    yerr[i],
                    color=color,
                    label=label[i],
                    ls=ls,
                    **line_kwargs,
                )

    if legend_on:
        ax.legend(
            labelspacing=0.1,
            framealpha=0.4,
            prop={"size": legend_size},
            loc=legend_loc,
            ncol=legend_ncol,
        )

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        fontdict = {}
        if title_color:
            fontdict["color"] = title_color
        if title_size:
            fontdict["fontsize"] = title_size
        ax.set_title(title, fontdict=fontdict)
    if fig_title is not None:
        fig.suptitle(fig_title)
    if xlim == "auto":
        xmin, xmax = lim_vars["xmin"], lim_vars["xmax"]
        dx = xmax - xmin
        xlim = xmin - 0.01 * dx, xmax + 0.01 * dx
    if ylim == "auto":
        ymin, ymax = lim_vars["ymin"], lim_vars["ymax"]
        dy = ymax - ymin
        ylim = ymin - 0.02 * dy, ymax + 0.02 * dy
    if xtick_unit_pi:
        ticks = ax.get_xticks()
        ax.set_xticks(ticks, list(map(lambda x: rf"{x:.2f}$\pi$", ticks / np.pi)))
    if xlim is not None and not np.isclose(xlim[0], xlim[1]):
        ax.set_xlim(*xlim)
    if ylim is not None and not np.isclose(ylim[0], ylim[1]):
        ax.set_ylim(*ylim)
    ax.grid(grid)
    ax.tick_params(axis="both", which="both", bottom=False, left=False)
    return ax


def plot2d(
    data,
    cmap="Spectral_r",
    ax=None,
    figsize=[7, 5],
    xlim=[0, 1],
    ylim=[0, 1],
    xlabel=None,
    ylabel=None,
    title=None,
    plot_cbar=True,
    clabel=None,
    vmin=None,
    vmax=None,
    interp="none",
    fontsize=None,
    convert_lim=True,
    horizontal=False,
    constrained_layout=True,
    return_cbar=False,
    log=False,
    norm=None,
):
    if not hasattr(data, "mask"):
        data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError()
    xlim = [np.min(xlim), np.max(xlim)]
    ylim = [np.min(ylim), np.max(ylim)]
    if convert_lim:
        Ny, Nx = data.shape
        dx = (xlim[1] - xlim[0]) / max(Nx - 1, 1)
        dy = (ylim[1] - ylim[0]) / max(Ny - 1, 1)
        xlim = xlim.copy()
        ylim = ylim.copy()
        xlim[0] -= dx / 2
        ylim[0] -= dy / 2
        xlim[1] += dx / 2
        ylim[1] += dy / 2

    if ax is None:
        ax = subplot(figsize=figsize)
    fig = ax.figure
    if constrained_layout:
        fig.set_constrained_layout(True)

    if vmin is None:
        vmin = vmin or np.nanmin(data)
    if vmax is None:
        vmax = vmax or np.nanmax(data)
    if vmin > data.min():
        print_warn("data.min() < vmin, the plotting result can be misleading")
    if vmax < data.max():
        print_warn("data.max() > vmax, the plotting result can be misleading")
    if log:
        norm = LogNorm(vmin=vmin, vmax=vmax)
        vmin = None
        vmax = None
    im = ax.imshow(
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        extent=list(xlim) + list(ylim),
        aspect="auto",
        interpolation=interp,
        norm=norm,
    )

    ax.set_xlabel(xlabel, {"fontsize": fontsize})
    ax.set_ylabel(ylabel, {"fontsize": fontsize})
    ax.set_title(title, fontsize=fontsize)

    cbar = None
    if plot_cbar:
        if horizontal:
            orientation = "horizontal"
        else:
            orientation = "vertical"
        cbar = fig.colorbar(im, ax=ax, orientation=orientation)
        cbar.set_label(clabel)
    if return_cbar:
        return ax, cbar

    return ax


def plot3d(
    x,
    y,
    z,
    ax=None,
    figsize=[7, 5],
    xlabel=None,
    ylabel=None,
    zlabel=None,
    xlim="auto",
    ylim="auto",
    zlim="auto",
    title=None,
    fontsize=None,
    cmap="Blues",
    constrained_layout=True,
):
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    assert z.shape == (len(y), len(x))
    Ny, Nx = z.shape

    indexes = np.argsort(x)
    x = x[indexes]
    z = z[:, indexes]
    indexes = np.argsort(y)
    y = y[indexes]
    z = z[indexes]

    if ax is None:
        ax = subplot(figsize=figsize, is_3d=True)
    fig = ax.figure
    if constrained_layout:
        fig.set_constrained_layout(True)

    dxs = np.diff(np.sort(x))
    # assert len(np.unique(dxs)) == 1
    dx = dxs[0]

    dys = np.diff(np.sort(y))
    # assert len(np.unique(dys)) == 1
    dy = dys[0]

    # new_x = np.append(x, x[-1] + dx) - dx / 2
    # new_y = np.append(y, y[-1] + dy) - dy / 2

    # X, Y = np.meshgrid(new_x, new_y)
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, z, cmap=cmap)

    if xlim == "auto":
        xlim = [x.min() - 0.01 * dx, x.max() + 0.01 * dx]
    if ylim == "auto":
        ylim = [y.min() - 0.01 * dy, y.max() + 0.01 * dy]
    if zlim == "auto":
        zlim = [z.min(), z.max()]
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if zlim is not None:
        ax.set_zlim(*zlim)

    if xlabel is not None:
        ax.set_xlabel(xlabel, {"fontsize": fontsize})
    if ylabel is not None:
        ax.set_ylabel(ylabel, {"fontsize": fontsize})
    if zlabel is not None:
        ax.set_zlabel(zlabel, {"fontsize": fontsize}, labelpad=10)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    return ax


def plot_ecdf(
    x,
    mean=True,
    median=False,
    title=None,
    xlabel=None,
    ylabel="Integrated histogram",
    xlim=None,
    ylim="auto",
    ax=None,
    figsize=(6, 4),
    annot=True,
    annot_size=None,
    annot_color="k",
    annot_yshift=0,
    annot_suffix=None,
    unit=None,
    lw=1.5,
    precision=2,
    dash_color=None,
    constrained_layout=True,
    sep=" ",
    **kwargs,
):
    if annot_suffix:
        annot_suffix = f"({annot_suffix})"
    else:
        annot_suffix = ""
    if ax is None:
        ax = subplot(figsize=figsize)
    x = np.sort(x)
    y = np.arange(1, len(x) + 1) / len(x)
    ax.step(np.hstack([x[0], x]), np.hstack([0, y]), where="post", lw=lw, **kwargs)
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        if ylim == "auto":
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(*ylim)
    ax.grid("on", ls="--")
    ylim = ax.get_ylim()
    color = ax.lines[-1].get_c()
    alpha = ax.lines[-1].get_alpha()
    if dash_color is None:
        dash_color = color

    if mean:
        mean_value = np.mean(x)
        ax.vlines(
            mean_value,
            *ylim,
            ls="--",
            color=dash_color,
            alpha=alpha,
            lw=lw,
        )
        if annot:
            s = "%.*f" % (precision, mean_value)
            if unit:
                s += f" {unit}"
            if hasattr(ax.figure, "renderer"):
                rainbow_text(
                    ax,
                    0.02,
                    0.98 + annot_yshift,
                    [f"Mean{annot_suffix}:", s],
                    ["k", color],
                    ha="left",
                    va="top",
                    transform="ax",
                )
            else:
                if annot_size is not None:
                    fontdict = {"fontsize": annot_size}
                else:
                    fontdict = None
                ax.text(
                    0.02,
                    0.98 + annot_yshift,
                    f"Mean{annot_suffix}:{sep}{s}",
                    ha="left",
                    va="top",
                    transform=ax.transAxes,
                    fontdict=fontdict,
                    color=annot_color,
                )
        ax.set_ylim(*ylim)
    if median:
        median_value = np.median(x)
        ax.vlines(
            median_value,
            *ylim,
            ls="-.",
            color=dash_color,
            alpha=alpha,
        )
        if annot:
            s = "%.*f" % (precision, median_value)
            if unit:
                s += f" {unit}"
            rainbow_text(
                ax,
                0.02,
                0.93 + annot_yshift,
                [f"Median{annot_suffix}:", s],
                ["k", color],
                ha="left",
                va="top",
                transform="ax",
            )
        ax.set_ylim(*ylim)
    if constrained_layout:
        ax.figure.set_constrained_layout(1)
    return ax


def bar(
    y,
    yerr=None,
    xlabel=None,
    ylabel=None,
    ylim=None,
    x_tick_labels=None,
    x_tick_label_rotation=0,
    label=None,
    width=0.25,
    vline_lw=0,
    title=None,
    xs=None,
    ax=None,
    palette="default",
    color=None,
    grid=True,
    offset=0,
    constrained_layout=True,
    **kwargs,
):
    if ax is None:
        ax = subplot(figsize=[10, 4])
    if constrained_layout:
        ax.figure.set_constrained_layout(True)
    y = np.array(y)
    if len(y.shape) == 1:
        y = y.reshape(1, -1)
    n = len(y)
    if label is None:
        label = [""] * n
        legend_on = False
    else:
        legend_on = True
        if isinstance(label, str):
            label = [label] * n
    if isinstance(palette, str):
        palette = color_palette(palette)
    offsets = np.arange(-(n - 1) / 2, (n - 2) / 2 + 1, 1) * width + offset

    x = np.arange(y.shape[1])
    if yerr is None:
        yerr = [None] * n
    for i in range(n):
        if xs is None:
            _x = x + offsets[i]
        else:
            _x = xs
        if color is None:
            c = palette[i]
        else:
            c = color
            c = color
        ax.bar(
            _x,
            y[i],
            width,
            yerr=yerr[i],
            label=label[i],
            color=c,
            **kwargs,
        )
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if legend_on:
        ax.legend()
    if ylim is None:
        ylim = ax.get_ylim()
    if x_tick_labels is not None:
        ax.set_xticks(x, labels=x_tick_labels, rotation=x_tick_label_rotation)
    if vline_lw > 0:
        ax.vlines(x, *ylim, ls="--", color="k", alpha=0.5, lw=vline_lw)
    ax.set_ylim(*ylim)
    ax.grid(grid)
    if title is not None:
        ax.set_title(title)
    return ax
