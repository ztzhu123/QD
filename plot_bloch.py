from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import Circle, PathPatch, Wedge
import numpy as np
import pandas as pd

from plot_toolbox import plot1d


def plot_bloch_2d(
    x,
    y,
    z,
    Xs,
    threshold=0,
    s=3,
    color="tab:blue",
    elev=-40,
    azim=-160,
    roll=60.3,
    norm_value=None,
    ax=None,
    datatype="scatter",
    **kwargs,
):
    r = 1
    theta = np.linspace(0, 2 * np.pi, 101)
    x1 = r * np.cos(theta)
    y1 = r * np.sin(theta)
    z1 = np.zeros_like(x1)
    ax = plot_3d_on_2d(
        x1,
        y1,
        z1,
        color="gray",
        zorder=-np.inf,
        lw=0.6,
        elev=elev,
        azim=azim,
        roll=roll,
        ax=ax,
        **kwargs,
    )
    x2 = r * np.cos(theta)
    z2 = r * np.sin(theta)
    y2 = np.zeros_like(x2)
    plot_3d_on_2d(
        x2,
        y2,
        z2,
        color="gray",
        ax=ax,
        zorder=-np.inf,
        lw=0.6,
        elev=elev,
        azim=azim,
        roll=roll,
        **kwargs,
    )

    x_2d, y_2d = project_3d_to_2d(x1, y1, z1, elev, azim, roll)
    r1 = np.sqrt(np.max(x_2d**2 + y_2d**2))
    x_2d, y_2d = project_3d_to_2d(x2, y2, z2, elev, azim, roll)
    r2 = np.sqrt(np.max(x_2d**2 + y_2d**2))

    r = max([r1, r2])
    ax.add_patch(Circle((0, 0), r, zorder=-np.inf, alpha=0.1))

    def plot_arrow(x, y, z):
        arrow_3d_on_2d(
            0,
            0,
            0,
            x,
            y,
            z,
            ax,
            color="k",
            width=0.001,
            head_width=0.07,
            elev=elev,
            azim=azim,
            roll=roll,
        )

    l = 1.3
    plot_arrow(l, 0, 0)
    plot_arrow(0, l, 0)
    plot_arrow(0, 0, l)

    text_3d_on_2d(1.8, 0, 0, "x", ax, elev=elev, azim=azim, roll=roll, fontsize=7)
    text_3d_on_2d(0, 1.5, 0, "y", ax, elev=elev, azim=azim, roll=roll, fontsize=7)
    text_3d_on_2d(0, 0, 1.5, "z", ax, elev=elev, azim=azim, roll=roll, fontsize=7)

    min_value, max_value = plot_data_2d(
        x,
        y,
        z,
        s,
        ax,
        Xs,
        threshold=threshold,
        color=color,
        datatype=datatype,
        norm_value=norm_value,
    )

    ax.grid(False)
    ax.axis("off")

    ax.set_aspect("equal")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.3, 1.4)
    return min_value, max_value


def plot_3d_on_2d(x, y, z, elev=-40, azim=-160, roll=60.3, ax=None, **kwargs):
    x_2d, y_2d = project_3d_to_2d(x, y, z, elev, azim, roll)
    ax = plot1d(x_2d, y_2d, ax=ax, **kwargs)
    return ax


def project_3d_to_2d(x, y, z, elev=-40, azim=-160, roll=60.3):
    if not np.iterable(x):
        x = [x]
    if not np.iterable(y):
        y = [y]
    if not np.iterable(z):
        z = [z]
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    rotation_matrix = euler_to_rotation_matrix(elev, azim, roll)
    sphere_points = np.vstack((x.flatten(), y.flatten(), z.flatten()))
    rotated_points = rotation_matrix @ sphere_points

    x_2d = rotated_points[0, :]
    y_2d = rotated_points[1, :]
    return x_2d, y_2d


def euler_to_rotation_matrix(elevation, azimuth, roll):
    elevation = np.radians(elevation)
    azimuth = np.radians(azimuth)
    roll = np.radians(roll)

    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ]
    )

    R_y = np.array(
        [
            [np.cos(elevation), 0, np.sin(elevation)],
            [0, 1, 0],
            [-np.sin(elevation), 0, np.cos(elevation)],
        ]
    )

    R_z = np.array(
        [
            [np.cos(azimuth), -np.sin(azimuth), 0],
            [np.sin(azimuth), np.cos(azimuth), 0],
            [0, 0, 1],
        ]
    )

    R = R_z @ R_y @ R_x

    return R


def text_3d_on_2d(x, y, z, text, ax, elev=-40, azim=-160, roll=60.3, **kwargs):
    x_2d, y_2d = project_3d_to_2d(x, y, z, elev, azim, roll)
    ax.text(x_2d, y_2d, text, **kwargs)


def plot_data_2d(
    x,
    y,
    z,
    s,
    ax,
    Xs,
    threshold=0,
    color="#8c312b",
    datatype="scatter",
    norm_value=None,
):
    Xs = np.asarray(Xs)
    if Xs is None:
        Xs = np.ones_like(x)
    assert np.ndim(Xs) == 1
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    mask = Xs > 0
    x = x[mask]
    y = y[mask]
    z = z[mask]
    Xs = Xs[mask]
    N = len(Xs)
    # assert np.all(x**2 + y**2 + z**2 >= 0.9999)
    coordinates = np.asarray([x, y, z])  # (3, N)
    encode = np.apply_along_axis(
        lambda x: hash(tuple(x)), 0, coordinates
    )  # (N,), encode coordinates
    category, unsorted_encode = pd.factorize(encode)
    probs = np.bincount(category, Xs)
    _, indexes = np.unique(encode, return_index=True)
    unsorted_encode_order = np.argsort(unsorted_encode)
    raw_indexes = np.zeros_like(indexes)
    np.put(raw_indexes, unsorted_encode_order, indexes)

    points = np.asarray([x[raw_indexes], y[raw_indexes], z[raw_indexes]]).T
    probs = np.asarray(probs)
    mask = (probs / probs.sum()) > threshold
    probs = probs[mask]
    points = points[mask]
    min_value = probs.min()
    max_value = probs.max()
    if norm_value is None:
        norm_value = max_value
    probs = probs / norm_value

    indexes = np.argsort(probs)
    probs = probs[indexes]
    points = points[indexes]
    x, y = project_3d_to_2d(points[:, 0], points[:, 1], points[:, 2])

    colors = []
    for p in probs:
        if not isinstance(color, str):
            colors.append(color(np.clip(p, 0, 0.9999)))
        else:
            colors.append(to_rgba(color, alpha=p))

    if datatype == "scatter":
        ax.scatter(
            x,
            y,
            # alpha=probs,
            zorder=np.inf,
            c=colors,
            ec=(0, 0, 0, 0),
            s=s,
        )
    elif datatype == "arrow":
        for i, x_, y_, p_ in zip(np.arange(0, len(x)), x, y, probs):
            if points[i, 2] >= 0:
                color_ = "#50557b"
            else:
                color_ = "#A76F6F"
            color_ = "#A76F6F"
            ax.arrow(
                0,
                0,
                x_ * 0.85,
                y_ * 0.85,
                color=colors[i],
                linewidth=1.0,
                width=0.005,
                head_width=0.08,
            )

    return min_value, max_value


def arrow_3d_on_2d(
    x, y, z, dx, dy, dz, ax, elev=-40, azim=-160, roll=60.3, *args, **kwargs
):
    x_2d, y_2d = project_3d_to_2d(x, y, z, elev, azim, roll)
    dx_2d, dy_2d = project_3d_to_2d([0, dx], [0, dy], [0, dz], elev, azim, roll)
    ax.arrow(x_2d[0], y_2d[0], dx_2d[1], dy_2d[1], *args, **kwargs)
