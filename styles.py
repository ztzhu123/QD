from functools import wraps
import os

import matplotlib as mpl

USE_TEX = True
LW = 1.2
MS = 3.5
MEW = 0.7
CAPSIZE = 3
LEGEND_SIZE = 7
FONTSIZE = 10
AX_TITLE_SIZE = 10
AX_LABEL_SIZE = AX_TITLE_SIZE - 2
LEGEND_FONT_SIZE = AX_LABEL_SIZE
DASHES = (3, 2.5)


def article_style(**rcparams):
    def decorator(plot_func):
        @wraps(plot_func)
        def wrapper(*args, **kwargs):
            original_params = mpl.rcParams.copy()
            mpl.rcParams.update(
                {
                    "font.size": FONTSIZE,
                    "axes.titlesize": AX_TITLE_SIZE,
                    "xtick.labelsize": AX_LABEL_SIZE,
                    "ytick.labelsize": AX_LABEL_SIZE,
                    "legend.fontsize": LEGEND_FONT_SIZE,
                    "font.family": "Arial",
                    "mathtext.fontset": "custom",
                    "mathtext.rm": "Arial",
                    "mathtext.it": "Arial:italic",
                }
            )
            actual_rcparams = {}
            for key in rcparams:
                actual_key = key.replace("_", ".")
                actual_rcparams[actual_key] = rcparams[key]
            mpl.rcParams.update(actual_rcparams)
            result = plot_func(*args, **kwargs)
            mpl.rcParams.update(original_params)
            return result

        return wrapper

    return decorator


def get_circuit_style(gate_colors=None, texts=None):
    gate_colors = gate_colors or {}
    texts = texts or {}

    gate_colors.setdefault("cz", "#3180bd")
    gate_colors.setdefault("_measure", "M")

    style = {"displaycolor": {}, "displaytext": {}}
    for gate_name, color in gate_colors.items():
        style["displaycolor"][gate_name] = (color, "#000000")
    for gate_name, text in texts.items():
        style["displaytext"][gate_name] = text

    return style
