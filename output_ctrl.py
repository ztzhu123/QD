import re

import colorama

colorama.init(autoreset=True)

INFO_COLOR = colorama.ansi.code_to_chars("1;34")
EMPH_COLOR = colorama.ansi.code_to_chars("1;35")
HEALTH_COLOR = colorama.ansi.code_to_chars("1;32")
WARN_COLOR = colorama.ansi.code_to_chars("1;33")
ERROR_COLOR = colorama.ansi.code_to_chars("1;31")
RESET_COLOR = colorama.Style.RESET_ALL

_COLOR_TYPES = ["info", "emph", "health", "warn", "error"]
# _GLOBAL_VARS = globals()
__all__ = ['print_info_emph']

for color_type in _COLOR_TYPES:
    color_var_name = f"{color_type.upper()}_COLOR"
    wrap_var_name = f"wrap_{color_type}"
    print_var_name = f"print_{color_type}"

    # TODO (ztzhu): set variables in the global dict
    # directly will cause IDE intellisense broken.
    # I haven't found a way to fix it.

    # color = _GLOBAL_VARS[color_var_name]
    # _GLOBAL_VARS[wrap_var_name] = partial(wrap_color, color=color)
    # _GLOBAL_VARS[print_var_name] = partial(print_color, color=color)

    __all__.extend([color_var_name, wrap_var_name, print_var_name])


def wrap_color(*values, color):
    if len(values) == 1:
        value = str(values[0])
        return f"{color}{value}{RESET_COLOR}"
    string = ""
    for v in values:
        string += wrap_color(v, color=color)
    return string


def wrap_info(*values):
    return wrap_color(*values, color=INFO_COLOR)


def wrap_emph(*values, **kwargs):
    quote = kwargs.get("quote", "`")
    values = list(values)
    for i in range(len(values)):
        values[i] = f"{quote}{values[i]}{quote}"
    return wrap_color(*values, color=EMPH_COLOR)


def wrap_health(*values):
    return wrap_color(*values, color=HEALTH_COLOR)


def wrap_warn(*values):
    return wrap_color(*values, color=WARN_COLOR)


def wrap_error(*values):
    return wrap_color(*values, color=ERROR_COLOR)


def print_color(*strings, color=RESET_COLOR, **kwargs):
    string = ""
    for s in strings:
        string += wrap_color(s, color=color)
    return print(string, **kwargs)


def print_info(*values, **kwargs):
    return print_color(*values, color=INFO_COLOR, **kwargs)


def print_emph(*values, **kwargs):
    return print_color(*values, color=EMPH_COLOR, **kwargs)


def print_info_emph(info, emph, end_str=".", quote="`", **kwargs):
    string = wrap_info(info) + " " + wrap_emph(emph, quote=quote) + wrap_info(end_str)
    return print(string, **kwargs)


def print_health(*values, **kwargs):
    return print_color(*values, color=HEALTH_COLOR, **kwargs)


def print_warn(*values, **kwargs):
    return print_color(*values, color=WARN_COLOR, **kwargs)


def print_error(*values, **kwargs):
    return print_color(*values, color=ERROR_COLOR, **kwargs)
