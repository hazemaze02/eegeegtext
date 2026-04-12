# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydantic
import tqdm

import neuralset
import neuralset as ns
import neuralset.studies


class StudySummary(pydantic.BaseModel):
    path: str
    studies: list[str] | tp.Literal["all"] = "all"
    n_timelines: int | tp.Literal["all"] = "all"
    cache: str
    infra: ns.infra.TaskInfra = ns.infra.TaskInfra(mode="retry")

    def model_post_init(self, __context):
        self.infra.folder = self.cache
        self.infra.apply_on(self.get_circles)
        super().model_post_init(__context)

    def get_events(self, keep_neuro_only: bool = True):
        if self.studies == "all":
            studies = []
            import importlib
            import pkgutil

            from neuralset.data import _validate_study_name

            for x in pkgutil.iter_modules(neuralset.studies.__path__):
                objs = importlib.import_module(f"neuralset.studies.{x.name}").__dir__()
                for obj in objs:
                    try:
                        _validate_study_name(obj)
                        studies.append(obj)
                    except:
                        pass
            print("Found studies:", studies)
        else:
            studies = self.studies

        events = []
        for name in studies:
            print(f"Loading study {name}")
            try:
                df = ns.data.StudyLoader(
                    name=name,
                    path=self.path,
                    cache=self.cache,
                    download=False,
                    install=True,
                    n_timelines=self.n_timelines,
                ).build()
                if keep_neuro_only:
                    df = df[df.type.isin({"Eeg", "Meg", "Fmri", "Fnirs"})]
                events.append(df)
            except:
                print(f"Failed to load study {name}")

        events = pd.concat(events)

        return events

    def df_to_tree(self, df, parent_id="all"):
        tree = []
        filtered_df = df[df["parent"] == parent_id]
        for _, row in filtered_df.iterrows():
            node = {
                "id": row["id"],
                "datum": row["datum"],
                "children": self.df_to_tree(df, parent_id=row["id"]),
            }
            if not node["children"]:
                del node["children"]
            tree.append(node)
        return tree

    def get_circles(self):
        events = self.get_events()
        circles_df = []
        for study_name, study in events.groupby("study"):
            n_hours = study.duration.sum() / 3600
            circles_df.append(
                {
                    "id": study_name,
                    "parent": "all",
                    "datum": n_hours,
                }
            )
            for subject_name, subject_group in study.groupby("subject"):
                n_hours = subject_group.duration.sum() / 3600
                circles_df.append(
                    {
                        "id": f"{study_name}_{subject_name}",
                        "parent": study_name,
                        "datum": n_hours,
                    }
                )
        circles_df = pd.DataFrame(circles_df)
        circle_tree = self.df_to_tree(circles_df)

        import circlify

        circles = circlify.circlify(
            circle_tree,
            show_enclosure=False,
            target_enclosure=circlify.Circle(x=0, y=0, r=1),
        )
        return circles

    def plot(self, colors=None, display_names=None):

        circles = self.get_circles()
        if not colors:
            datasets = np.unique(
                [circles_df["id"].split("_")[0] for circles_df in circles]
            )
            colors = dict(zip(datasets, plt.cm.tab20.colors))

        fig, ax = plt.subplots(figsize=(15, 15), dpi=200)
        ax.axis("off")

        xlim = max(abs(circle.x) + circle.r for circle in circles)
        ylim = max(abs(circle.y) + circle.r for circle in circles)
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-ylim, ylim)

        for circle in circles:
            x, y, r = circle
            if circle.level == 1:
                color = (0.9, 0.9, 0.9)
            else:
                dataset = circle.ex["id"].split("_")[0]
                color = colors[dataset]
            ax.add_patch(
                plt.Circle(
                    (x, y),
                    r,
                    linewidth=1,
                    facecolor=color,
                )
            )
            # add text
            if circle.level == 1:
                if display_names:
                    label = display_names[circle.ex["id"]]
                else:
                    label = circle.ex["id"]
                ax.text(
                    x,
                    y,
                    label,
                    ha="center",
                    va="center",
                    fontsize=24,
                    bbox=dict(facecolor="w", edgecolor="black", boxstyle="round,pad=.2"),
                )

        fig.tight_layout(pad=0)
        return fig


def add_sentences(
    events: pd.DataFrame,
    ratios: tp.Tuple[float, float, float] = (0.8, 0.1, 0.1),
    column_to_group: str = "sequence_id",
) -> pd.DataFrame:
    """
    Add sentence-level information to the events DataFrame based on the sequence_id column.
    """

    assert column_to_group in events.columns
    assert "Sentence" not in events.type.unique()

    if "timeline" in events.columns and len(events.timeline.unique()) > 1:
        # apply to each timeline
        timelines = []
        for _, df in tqdm.tqdm(events.groupby("timeline"), "Adding sentences"):
            df = add_sentences(df, ratios, column_to_group)
            timelines.append(df)
        return pd.concat(timelines)
    events["stop"] = events.start + events.duration

    words = events.query('type=="Word"')
    assert all(words.start.diff().dropna() >= 0)  # type: ignore

    # Add sentence-level information
    words = events.query('type=="Word"')
    sentences = []
    for _, sent in words.groupby(column_to_group, sort=False):
        # Find all events within the sentence
        duration = sent.stop.max() - sent.start.min() + 1e-8

        sentence = " ".join(sent.text)
        events.loc[sent.index, "sentence"] = sentence

        # Add Sentence event
        to_add = sent.iloc[0].to_dict()
        to_add["type"] = "Sentence"
        to_add["text"] = " ".join(sent.text)
        to_add["start"] = sent.start.min() - 1e-8
        to_add["duration"] = duration
        to_add["stop"] = sent.stop.max()
        sentences.append(to_add)

    events = pd.concat([events, pd.DataFrame(sentences)], ignore_index=True)
    events = events.sort_values("start")
    events = events.reset_index(drop=True)

    return events
