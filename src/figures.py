import itertools
import math
import os
import textwrap
import typing

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import seaborn as sns
from matplotlib.patches import Patch
from scipy import stats

from raincloud import raincloud_plot
from util import make_color_transparent


def _set_theme():
    sns.set_theme(
        rc={
            "figure.autolayout": False,
            "font.family": ["Computer Modern", "CMU Serif", "cmu", "serif"],
            "font.serif": ["Computer Modern", "CMU Serif", "cmu"],
            #'text.usetex': True
        }
    )
    matplotlib.rcParams.update(
        {
            "figure.autolayout": False,
            "font.family": ["Computer Modern", "CMU Serif", "cmu", "serif"],
            "font.serif": ["Computer Modern", "CMU Serif", "cmu"],
            #'text.usetex': True
        }
    )
    sns.set_style(
        rc={
            "font.family": ["Computer Modern", "CMU Serif", "cmu", "serif"],
            "font.serif": ["Computer Modern", "CMU Serif", "cmu"],
            #'text.usetex': True
        }
    )
    sns.set(font="CMU Serif", font_scale=1.25)
    plt.rcParams["font.family"] = "CMU Serif"


def likert_alternative(
    df: pd.DataFrame,
    keys: typing.List[str],
    category_names: typing.List[str],
    key_labels: typing.Dict[str, str],
    gap: float = 0.1,
    symmetrical: bool = False,
    x_margin: float = 1,
    x_ticks_base: int = 5,
):
    """
    Parameters
    ----------
    """
    empty_data = pd.DataFrame(
        columns=["question", "value"],
        data=list(itertools.product(keys, [1, 2, 3, 4, 5])),
    )

    df = df[keys]
    df = df.melt(value_vars=keys, var_name="question")
    df = df.assign(count=1)
    df = df.groupby(["question", "value"]).count()
    df = df.reset_index()
    df = empty_data.merge(df, how="left").fillna(0)
    df = df.pivot_table(index="question", columns="value", values="count")

    data = df.values

    labels = [key_labels[key] if key in key_labels else key for key in keys]
    labels_lines = [textwrap.wrap(label, width=80) for label in labels]
    labels = ["\n".join(label_lines) for label_lines in labels_lines]
    data_cum = data.cumsum(axis=1)
    middle_index = data.shape[1] // 2
    offsets = data[:, range(middle_index)].sum(axis=1) + data[:, middle_index] / 2

    # Color Mapping
    category_colors = plt.get_cmap("coolwarm_r")(np.linspace(0.15, 0.85, data.shape[1]))
    category_hatches = ["\\\\\\", "\\\\\\", "", "///", "///"]

    bar_height_inches = 0.25

    line_height = 0.2
    max_lines = max([len(lines) for lines in labels_lines])
    max_line_height = max_lines * line_height

    max_height = max(max_line_height, bar_height_inches)

    bar_height_relative = bar_height_inches / max_height
    bar_height_relative = min(bar_height_relative, 1 - gap)
    gap = 1 - bar_height_relative

    y_margins = 0.2
    legend_height = 0.25
    x_ticks_height = 0.25
    fig_height = len(keys) * max_height
    fig_height += (len(keys) - 1) * max_height * gap
    fig_height += 2 * y_margins
    fig_height += legend_height
    fig_height += x_ticks_height
    fig, ax = plt.subplots(figsize=(12, fig_height))

    # Plot Bars
    for i, (colname, color, hatch) in enumerate(
        zip(category_names, category_colors, category_hatches)
    ):
        widths = data[:, i]
        starts = data_cum[:, i] - widths - offsets
        rects = ax.barh(
            labels,
            widths,
            left=starts,
            height=1 - gap,
            label=colname,
            color=color,
            hatch=hatch,
        )

    # Add Zero Reference Line
    ax.axvline(0, linestyle="--", color="black", alpha=0.25)

    # X Axis
    middle = data[:, 2].max() / 2
    half_middle = middle * 0.5

    left_extents = data[:, 0:2].sum(axis=1).max() + half_middle
    right_extents = data[:, 3:5].sum(axis=1).max() + half_middle

    left_extents = math.ceil(left_extents / x_ticks_base) * x_ticks_base
    right_extents = math.ceil(right_extents / x_ticks_base) * x_ticks_base

    left_extents += x_margin
    right_extents += x_margin

    max_extents = max(left_extents, right_extents)
    x_lim = (-left_extents, right_extents)

    if symmetrical:
        ax.set_xlim(-max_extents, max_extents)
        x_ticks = np.arange(5, max_extents, 5)
        x_ticks = np.concatenate((-np.flip(x_ticks), [0], x_ticks))
    else:
        ax.set_xlim(x_lim[0], x_lim[1])
        x_ticks = np.concatenate(
            (
                -np.flip(np.arange(5, left_extents, 5)),
                [0],
                np.arange(5, right_extents, 5),
            )
        )

    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(lambda x, pos: str(abs(int(x))))

    # Y Axis
    ax.invert_yaxis()

    # Remove spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Legend
    ax.legend(
        ncol=len(category_names),
        bbox_to_anchor=(1, 1),
        loc="lower right",
        fontsize="small",
    )

    # Set Background Color
    fig.set_facecolor("#FFFFFF")

    return fig, ax


def plot_likert_items(data: pd.DataFrame):
    key_labels = {
        #
        # demographics
        #
        "user-id": "ID",
        "start-time": "Start time",
        "end-time": "End time",
        "email": "E-Mail",
        "name": "Name",
        "age": "Age",
        "education": "Highest education",
        "years-bpmn-experience": "Years of working with BPMN",
        "field": "Field of work",
        #
        # tutorial
        #
        "tutorial-helpful": "The tutorial helped me with understanding the functionality of the tool",
        "tutorial-ready": "After the tutorial I felt read to use the tool",
        "tutorial-relevant": "The information presented in the tutorial was relevant for the tasks I had to perform in the tool",
        #
        # visualization
        #
        "visualization-understanding": "The visualization made understanding the process structure easier",
        "visualization-helpful": "The BPMN model was helpful for my annotation task",
        "visualization-time": "The BBPM Model reduced the time I needed for annotations",
        "visualization-confidence": "The visualization of the process with the BPMN model made me more confident in my annotation",
        "visualization-dependence": "I felt dependent on the BPMN model to annotate effectively",
        "visualization-certainty": "Using the BPMN model as a reference made me feel more confident in my annotations",
        "visualization-decisions": "The graphical representation influenced my decisions during annotation positively",
        #
        # combination
        #
        "combination-efficiency": "Die Kombination von Vorschlagsfeature und BPMN-Visualisierung erhöhte die Effizienz in der Bearbeitung meiner Aufgabe",
        "combination-understanding": "Die Kombination von Vorschlagsfeature und BPMN-Visualisierung hat mein Verständnis für die Prozesse deutlich verbessert",
        "combination-confidence": "Die Kombination von Vorschlagsfeature und BPMN-Visualisierung steigerte mein Vertrauen in meine Annotationen",
        "combination-overwhelming": "Die Menge an gleichzeitig dargebotenen Informationen aus Vorschlagsfeature und dem BPMN-Modell fühlte sich oft überwältigend an",
        "combination-too-much": "Es war manchmal schwierig, die aus den Vorschlagsfeature und der BPMN-Visualisierung stammenden Informationen schnell zu verarbeiten und zu verstehen",
        "combination-focus": "Ich hatte Schwierigkeiten, meinen Fokus zu behalten, wenn sowohl Vorschlagsfeature als auch BPMN-Visualisierungen aktiv waren",
        "combination-synergy": "Die Aufgabenbearbeitung mit der Kombination von Vorschlagsfeature und grafischer Unterstützung war effektiver als jede Unterstützung für sich genommen",
        #
        # recommendation
        #
        "recommendation-helpful": "Die Vorschläge waren hilfreich für meine Annotationen",
        "recommendation-time": "Das Vorschlagsfeature hat die Zeit, die ich für die Annotationen benötigte, reduziert",
        "recommendation-assistance": "Ohne das Vorschlagsfeature hätte ich mehr Schwierigkeiten gehabt, die Aufgabe zu erfüllen",
        "recommendation-confidence": "Ich konnte den Vorschlägen vertrauen, um korrekte Annotationen zu liefern",
        "recommendation-dependence": "Ich fühlte mich abhängig von den Vorschlägen, um effektiv zu annotieren",
        "recommendation-reliance": "Ich habe mich oft auf das Vorschlagsfeature verlassen, um Entscheidungen bei der Annotation zu treffen",
        "recommendation-decisions": "Die Vorschläge haben meine Entscheidungen während der Annotation positiv beeinflusst",
        "recommendation-suggestions": "Das Vorschlagsfeature hat mir neue Ansätze für die Annotation aufgezeigt, die ich allein möglicherweise nicht in Betracht gezogen hätte",
        #
        # comparisons
        #
        "comparisons-no-assistance": "Die Aufgabenbearbeitung ohne jegliche Unterstützung war anspruchsvoller im Vergleich mit Unterstützung",
        "comparisons-recommendation": "Die Aufgabenbearbeitung ohne Vorschlagsfeature war anspruchsvoller im Vergleich zur Bearbeitung mit Vorschlagsfeature",
        "comparisons-visualization": "Die Aufgabenbearbeitung ohne BPMN-Modell-Unterstützung war anspruchsvoller im Vergleich zur Bearbeitung mit BPMN-Modell",
        "comparisons-best-assistance": "Welches Hilfsmittel fanden Sie am nützlichsten für die Aufgabenbearbeitung",
        #
        # workflow
        #
        "workflow-speed": "Die Geschwindigkeit, mit der ich Aufgaben erledigen konnte, war zufriedenstellend",
        "workflow-intuitive": "Ich konnte leicht verstehen, was ich in jedem Schritt des Workflows tun musste",
        "workflow-suitable": "Ich fand, dass die Reihenfolge der Schritte im Workflow meine Aufgabe effizient unterstützt hat",
        "workflow-efficient": "Der Workflow des Tools führte zu einem reibungslosen und unterbrechungsfreien Arbeitsprozess",
        "workflow-task-mapping": "Die Gestaltung des Workflows entsprach gut der Aufgabe, die ich ausführen musste",
    }

    non_likert_items = [
        "user-id",
        "start-time",
        "end-time",
        "email",
        "name",
        "age",
        "education",
        "years-bpmn-experience",
        "field",
        "best-assistance-feature",
        "has-experience",
    ]
    likert_items: typing.List[str] = [
        i for i in data.columns if i not in non_likert_items
    ]
    likert_keys = {}
    for i in likert_items:
        category, _ = i.split("-", maxsplit=1)
        if category not in likert_keys:
            likert_keys[category] = []
        likert_keys[category].append(i)
    for category, keys in likert_keys.items():
        likert_alternative(
            data,
            keys=keys,
            category_names=[
                "Strongly disagree",
                "Disagree",
                "Neither agree nor disagree",
                "Agree",
                "Strongly agree",
            ],
            key_labels=key_labels,
        )
        plt.tight_layout()
        os.makedirs(os.path.join("..", "data", "figures", "preferences"), exist_ok=True)
        plt.savefig(os.path.join("..", "data", "figures", "preferences", f"{category}.png"))
        plt.savefig(os.path.join("..", "data", "figures", "preferences", f"{category}.pdf"))
        plt.close()

        for has_experience in [True, False]:
            likert_alternative(
                data[data["has-experience"] == has_experience],
                keys=keys,
                category_names=[
                    "Strongly disagree",
                    "Disagree",
                    "Neither agree nor disagree",
                    "Agree",
                    "Strongly agree",
                ],
                key_labels=key_labels,
            )
            group = "expert" if has_experience else "novice"
            plt.tight_layout()
            os.makedirs(os.path.join("..", "data", "figures", "preferences"), exist_ok=True)
            plt.savefig(
                os.path.join("..", "data", "figures", "preferences", f"{category}-{group}.png")
            )
            plt.savefig(
                os.path.join("..", "data", "figures", "preferences", f"{category}-{group}.pdf")
            )
            plt.close()


def plot_metrics(data: pd.DataFrame):
    # data = data.melt(
    #     id_vars=["user-id", "task-id"],
    #     value_vars=[
    #         "relation-r",
    #         "relation-p",
    #         "relation-f1",
    #         "mention-r",
    #         "mention-p",
    #         "mention-f1",
    #     ],
    #     var_name="metric",
    #     value_name="value",
    # )

    task_labels = {
        "1": "no assistance",
        "2": "recommendations",
        "3": "visualization",
        "4": "combined",
    }

    row_labels = {True: "with experience", False: "no experience"}

    col_labels = {
        "mention-f1": "mentions ($F_1$)",
        "entity-f1": "entities ($F_1$)",
        "relation-f1": "relations ($F_1$)",
        "time": "time (m)",
    }

    data["time"] = data.apply(lambda row: row["time"] / 60.0, axis=1)
    data["task"] = data.apply(lambda row: task_labels[row["task-id"]], axis=1)
    data["experience"] = data.apply(
        lambda row: row_labels[row["has-experience"]], axis=1
    )

    both_data = data.copy()
    both_data["experience"] = both_data.apply(lambda _: "either", axis=1)

    # data = pd.concat((data, both_data))
    data = both_data

    data = data.rename(columns=col_labels)

    data = data.melt(
        id_vars=["user-id", "experience", "task"],
        value_vars=list(col_labels.values()),
        var_name="metric",
        value_name="value",
    )

    data = data[data["metric"] != "entities"]

    fig, axs = raincloud_plot(
        data,
        x="task",
        y="value",
        ids="user-id",
        rows="experience",
        cols="metric",
        y_lims=[(-0.05, 1.05), (-0.05, 1.05), (-0.05, 1.05), None],
        order=["no assistance", "visualization", "recommendations", "combined"],
        col_order=list(col_labels.values()),
        dodge_bar=True,
        jitter=0.0,
        line_alpha=0.2,
        fig_size=(12, 4.5),
    )

    fig.tight_layout()

    os.makedirs(os.path.join("..", "data", "figures", "metrics"), exist_ok=True)

    fig.savefig(os.path.join("..", "data", "figures", "metrics", "combined.png"))
    fig.savefig(os.path.join("..", "data", "figures", "metrics", "combined.pdf"))

    plt.close(fig)


def plot_correlation(data: pd.DataFrame):
    data = data.loc[data["user-experience"] < 25]
    # data = data.loc[data["user-experience"] > 0]
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey="all", sharex="all")
    sns.regplot(
        data.loc[data["task-id"] == "1"], x="user-experience", y="relation-f1", ax=ax1
    )
    sns.regplot(
        data.loc[data["task-id"] == "2"], x="user-experience", y="relation-f1", ax=ax2
    )
    sns.regplot(
        data.loc[data["task-id"] == "3"], x="user-experience", y="relation-f1", ax=ax3
    )
    sns.regplot(
        data.loc[data["task-id"] == "4"], x="user-experience", y="relation-f1", ax=ax4
    )
    fig.tight_layout()
    fig.savefig(os.path.join("..", "data", "figures", "metrics", "correlation.png"))
    fig.savefig(os.path.join("..", "data", "figures", "metrics", "correlation.pdf"))

    plt.close(fig)


def plot_improvement(data: pd.DataFrame):
    data = data[["task-id", "has-experience", "overall-f1"]]
    data = data.groupby(by=["task-id", "has-experience"]).mean()
    sns.barplot(data, x="task-id", y="overall-f1", hue="has-experience")
    plt.tight_layout()

    plt.savefig(os.path.join("..", "data", "figures", "metrics", "improvement.png"))
    plt.savefig(os.path.join("..", "data", "figures", "metrics", "improvement.pdf"))

    plt.close()


def plot_pie(
    data: pd.DataFrame,
    col: str,
    ax: plt.Axes = None,
    start_angle=0.0,
    colors: typing.List[typing.Tuple[float, float, float, float]] = None,
):
    counts = data[[col]].assign(count=lambda _: 1).groupby(col).count()
    counts = counts.reset_index()
    if ax is None:
        _, ax = plt.subplots()

    if colors is None:
        colors = sns.color_palette()[0:6] * 10

    labels = counts[col]
    if col != "age":
        labels = labels.str.replace("-", " ")
    else:
        labels = labels.str.replace("-", "–")

    patches, texts, pcts = ax.pie(
        x=counts["count"],
        labels=labels,
        startangle=start_angle,
        autopct=lambda p: "{:.0f}".format(p * sum(counts["count"]) / 100),
        colors=colors,
        pctdistance=0.8,
        explode=[0.1 for c in counts["count"]],
    )
    # ax.legend(labels=counts[col])

    patch: Patch
    for i, patch in enumerate(patches):
        patch.set_facecolor(make_color_transparent(colors[i]))
        patch.set_edgecolor(colors[i])

    plt.setp(pcts, color=(0.15, 0.15, 0.15), fontweight="bold")
    ax.set_title(col)
    return ax


def plot_demographics(data: pd.DataFrame):
    def save_cur_fig(name: str):
        os.makedirs(os.path.join("..", "data", "figures", "demographics"), exist_ok=True)
        plt.savefig(os.path.join("..", "data", "figures", "demographics", f"{name}.png"))
        plt.savefig(os.path.join("..", "data", "figures", "demographics", f"{name}.pdf"))
        plt.close()

    plot_pie(data, "age")
    save_cur_fig("age")

    plot_pie(data, "education")
    save_cur_fig("education")

    plot_pie(data, "field")
    save_cur_fig("field")

    fig, axes = plt.subplots(1, 2, constrained_layout=True)
    fig.set_figheight(3.45)
    fig.set_figwidth(9)
    for ax, col, angle in zip(axes, ["education", "field"], [0, 30]):
        plot_pie(data, col, ax, start_angle=angle)
    # plt.tight_layout()
    save_cur_fig("combined")


def export_tlx_table(data: pd.DataFrame):
    tlx_metrics = [
        "mental-demand",
        "uncertainty",
        "effort",
        "frustration",
    ]

    data = data[["task-id", "has-experience"] + tlx_metrics]
    print(data)
    data = data.assign(count=lambda _: 1)

    aggregated_data = data.groupby(by=["task-id", "has-experience"]).agg(
        {
            **{m: "mean" for m in tlx_metrics},
            "count": "sum",
        }
    )
    aggregated_data = aggregated_data.reset_index()
    print(aggregated_data)

    # records = {
    #     "has-experience": [],
    #     ""
    # }
    for has_experience in [True, False]:
        for m in tlx_metrics:
            samples = [
                data[
                    (data["task-id"] == i) & (data["has-experience"] == has_experience)
                ][[m]].to_numpy()
                for i in range(1, 5)
            ]
            print(
                f"({'Expert' if has_experience else 'Non-Expert'}) {m}: {stats.f_oneway(*samples)}"
            )

    # print(aggregated_data)
    return aggregated_data


def plot_tlx(data: pd.DataFrame):
    task_labels = {
        1: "no assistance",
        2: "recommendations",
        3: "visualization",
        4: "combined",
    }

    row_labels = {True: "with experience", False: "no experience"}

    col_labels = {
        "mental-demand": "mental demand",
        "uncertainty": "uncertainty",
        "effort": "effort",
        "frustration": "frustration",
    }

    data["task"] = data.apply(lambda row: task_labels[row["task-id"]], axis=1)
    data["experience"] = data.apply(
        lambda row: row_labels[row["has-experience"]], axis=1
    )
    data = data.rename(columns=col_labels)

    data = data.melt(
        id_vars=["user-id", "experience", "task"],
        value_vars=list(col_labels.values()),
        var_name="metric",
        value_name="value",
    )
    data = data.assign(tmp=lambda x: 1)

    fig, axs = raincloud_plot(
        data,
        x="task",
        y="value",
        ids="user-id",
        cols="metric",
        rows="tmp",
        y_lims=(-5, 105),
        order=["no assistance", "visualization", "recommendations", "combined"],
        # row_order=["with experience", "no experience"],
        col_order=["mental demand", "uncertainty", "effort", "frustration"],
        line_alpha=0.2,
        jitter=0.0,
        dodge_bar=True,
        fig_size=(12, 4.5),
    )
    fig.tight_layout()
    os.makedirs(os.path.join("..", "data", "figures", "tlx"), exist_ok=True)
    fig.savefig(os.path.join("..", "data", "figures", "tlx", "combined.png"))
    fig.savefig(os.path.join("..", "data", "figures", "tlx", "combined.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    seaborn.set_theme()

    def main():
        _set_theme()

        data = pd.read_pickle("../data/results/questionnaire.pkl")
        plot_likert_items(data)
        plot_demographics(data)

        data = pd.read_pickle("../data/results/tlx.pkl")
        plot_tlx(data)

        data = pd.read_pickle("../data/results/results.pkl")

        plot_metrics(data)
        plot_correlation(data)
        plot_improvement(data)

    main()
