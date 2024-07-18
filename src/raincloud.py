import typing

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from util import lighten_color, make_color_transparent


def raincloud_plot(
    data: pd.DataFrame,
    x: str,
    y: str,
    ids: str,
    cols: str,
    rows: str,
    order: typing.List[str] = None,
    col_order: typing.List[str] = None,
    row_order: typing.List[str] = None,
    fig_size: typing.Tuple[float, float] = None,
    y_lims: typing.Union[
        typing.Tuple[float, float],
        typing.List[typing.Optional[typing.Tuple[float, float]]],
    ] = None,
    jitter: float = 0.1,
    dodge_bar: bool = True,
    show_lines: bool = True,
    show_kde: bool = False,
    line_alpha: float = 0.5,
    colors=None,
) -> typing.Tuple[plt.Figure, typing.List[plt.Axes]]:
    if colors is None:
        colors = sns.color_palette()

    main_plot_ratio = 0.8
    kde_plot_ratio = 0.3
    separator_ratio = 0.2

    col_values = data[cols].unique()
    row_values = data[rows].unique()

    if col_order is not None:
        col_values = sorted(col_values, key=lambda val: col_order.index(val))
    if row_order is not None:
        row_values = sorted(row_values, key=lambda val: row_order.index(val))

    num_cols = len(col_values)
    num_rows = len(row_values)

    if show_kde:
        width_ratios = [main_plot_ratio, kde_plot_ratio, separator_ratio] * num_cols
    else:
        width_ratios = [main_plot_ratio, separator_ratio] * num_cols
    width_ratios = width_ratios[:-1]

    fig, axs = plt.subplots(
        num_rows,
        len(width_ratios),
        gridspec_kw={
            "width_ratios": width_ratios,
            "wspace": 0.0,
        },
    )
    if fig_size is None:
        fig_size = (num_cols * 3, num_rows * 4)
    fig.set_figwidth(fig_size[0])
    fig.set_figheight(fig_size[1])

    axs = list(axs.flat)

    num_subplots_per_col = 2 if show_kde else 1
    # hide padding "plots"
    for row_id in reversed(range(num_rows)):
        for i in reversed(
            list(
                range(num_subplots_per_col, len(width_ratios), num_subplots_per_col + 1)
            )
        ):
            ax_idx = row_id * len(width_ratios) + i
            axs[ax_idx].grid(False)
            axs[ax_idx].set_xticks([])
            axs[ax_idx].set_yticks([])
            axs[ax_idx].set_facecolor((0, 0, 0, 0))
            del axs[ax_idx]

    assert len(axs) % num_subplots_per_col == 0

    for row_id, row_value in enumerate(row_values):
        # select only axes of this row (which are two for each column)
        axs_row = axs[
            row_id
            * num_cols
            * num_subplots_per_col : (row_id + 1)
            * num_cols
            * num_subplots_per_col
        ]
        # group into groups of two (main plot axes and density plot axes)
        axs_row = [
            tuple(axs_row[i * num_subplots_per_col : (i + 1) * num_subplots_per_col])
            for i in range(len(axs_row) // num_subplots_per_col)
        ]

        for col_idx, (sub_plot_axs, col_value) in enumerate(zip(axs_row, col_values)):
            row_data = data[(data[rows] == row_value) & (data[cols] == col_value)]
            main_ax: plt.Axes = sub_plot_axs[0]

            sns.boxplot(
                row_data,
                x=x,
                y=y,
                gap=0.5,
                width=0.5,
                fliersize=0,
                order=order,
                legend=False,
                ax=main_ax,
                zorder=2,
            )

            for ist, data_id in enumerate(row_data[ids].unique()):
                run_data = row_data[row_data[ids] == data_id]
                run_data = run_data.sort_values(
                    by=["task"], key=lambda col: np.array([order.index(c) for c in col])
                )
                ys = run_data[y].to_numpy()
                xs = main_ax.get_xticks()
                xs_offsets = np.random.normal(size=len(xs), scale=jitter)
                if dodge_bar:
                    xs_offsets = np.abs(xs_offsets)
                    xs_offsets += 0.25
                xs += xs_offsets

                if show_lines:
                    main_ax.plot(
                        xs, ys, color="grey", linewidth=1, alpha=line_alpha, zorder=1
                    )
                for i in range(len(xs)):
                    main_ax.plot(
                        xs[i],
                        ys[i],
                        linewidth=0,
                        alpha=0.3,
                        marker="o",
                        markersize=4,
                        color=colors[i],
                    )

            if show_kde:
                kde_ax: plt.Axes = sub_plot_axs[1]
                sns.kdeplot(
                    row_data,
                    y=y,
                    hue=x,
                    fill=True,
                    alpha=0.1,
                    hue_order=order,
                    legend=False,
                    ax=kde_ax,
                    bw_adjust=0.7,
                )

            # finally, set colors
            x_values = row_data[x].unique()
            x_values = sorted(x_values, key=lambda val: order.index(val))
            for i in range(len(x_values)):
                color = colors[i]

                # color of boxes
                face_color = lighten_color(color, amount=0.4)
                face_color = make_color_transparent(face_color, alpha=0.8)
                main_ax.patches[i].set_facecolor(face_color)
                # main_ax.patches[i].set_facecolor(make_color_transparent(color, alpha=1))
                main_ax.patches[i].set_edgecolor(color)

                # whiskers of boxes
                num_lines_per_box = 6
                for l in main_ax.lines[
                    i * num_lines_per_box : (i + 1) * num_lines_per_box
                ]:
                    l.set_color(color)

            main_ax.set_xticks(main_ax.get_xticks())
            main_ax.set_xticklabels(
                x_values,
                rotation=45,
                ha="right",
            )
            main_ax.set_xlabel("")
            if y_lims:
                if type(y_lims) is tuple:
                    main_ax.set_ylim(y_lims[0], y_lims[1])
                else:
                    col_lims = y_lims[col_idx]
                    if col_lims is not None:
                        main_ax.set_ylim(col_lims[0], col_lims[1])

            if row_id == 0:
                # label columns
                main_ax.set_title(col_value)
            if row_id != len(row_values) - 1:
                # remove x ticks and labels from all but last row
                main_ax.set_xticklabels([])

            if col_idx == 0 and len(row_values) > 1:
                # set row "labels" in first column,
                # if there are more than one row
                main_ax.set_ylabel(row_values[row_id])
            else:
                main_ax.set_ylabel("")

            if show_kde:
                kde_ax: plt.Axes = sub_plot_axs[1]
                # remove all ticks and labels from density plots
                kde_ax.set_axis_off()
                kde_ax.set_ylim(main_ax.get_ylim()[0], main_ax.get_ylim()[1])
                kde_ax.set_xlim(0.00025, kde_ax.get_xlim()[1] + 0.0025)
    return fig, axs
