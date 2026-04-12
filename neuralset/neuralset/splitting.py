# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import random
import typing as tp
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pydantic

from . import events as event_module
from .features.base import BaseStatic


@dataclass
class DeterministicSplitter:
    ratios: tp.Dict[str, float]
    seed: float = 0.0

    def __post_init__(self) -> None:
        # check that the spliting ratios is valid
        assert all(ratio > 0 for ratio in self.ratios.values())
        assert np.allclose(
            sum(self.ratios.values()), 1.0
        ), f"the sum of ratios must be equal to 1. got {self.ratios}"

    def __call__(self, uid: str) -> str:
        hashed = int(hashlib.sha256(uid.encode()).hexdigest(), 16)
        rng = random.Random(hashed + self.seed)
        score = rng.random()

        cdf = np.cumsum(list(self.ratios.values()))
        names = list(self.ratios.keys())
        # associate a split to this deterministc hash
        for idx, cdf_val in enumerate(cdf):
            if score < cdf_val:
                return names[idx]
        raise ValueError


def set_event_split(
    events: pd.DataFrame,
    event_type_to_split: str = "Sound",
    event_type_to_use: str = "Word",
    min_duration: float | None = None,
    max_duration: float = np.inf,
):
    added_events: tp.List[tp.Dict] = []
    dropped_rows: tp.List[int] = []
    ns_event_type_to_split = getattr(event_module, event_type_to_split)
    assert hasattr(ns_event_type_to_split, "_split")
    assert "split" in events.columns

    for _, df in events.groupby("timeline"):
        df.sort_values("start", inplace=True)
        events_to_use = df.loc[events.type == event_type_to_use].copy()
        previous = events_to_use.copy().shift(1)
        # check has a split col
        split_change = events_to_use.split.astype(str) != previous.split.astype(str)
        events_to_use["section"] = np.cumsum(split_change.values)  # type: ignore
        timepoints: tp.List[float] = []
        for _, section in events_to_use.groupby("section"):
            start, end = (
                section.iloc[0].start,
                section.iloc[-1].start + section.iloc[-1].duration,
            )
            timepoints.extend(np.arange(start, end, max_duration))

        events_to_split = df.loc[events.type == event_type_to_split]
        dropped_rows.extend(events_to_split.index)
        for sound in events_to_split.itertuples():
            event = ns_event_type_to_split.from_dict(sound)
            new_events = event._split([t - sound.start for t in timepoints], min_duration)  # type: ignore
            for new_event in new_events:
                added_events.append(new_event.to_dict())

    out_events = events.copy()
    out_events.drop(dropped_rows, inplace=True)
    out_events = pd.concat([out_events, pd.DataFrame(added_events)])
    out_events.reset_index(drop=True, inplace=True)
    return out_events


class SimilaritySplitter(pydantic.BaseModel):
    """A class used to split events based on similarity clustering of static features.
    The class uses agglomerative clustering on precomputed embeddings of the events
    to ensure that same and similar events remain in the same split to avoid data leaking.

    Parameters
    ----------
    feature : BaseStatic
        A static feature extraction model that defines the type of event
        and provides methods to extract embeddings from events.
    ratios : Dict[str, float]
        A dictionary defining the proportion of events for each split.
        The sum of all ratios must equal 1.
    threshold : float
        The threshold for the distance used in the agglomerative clustering.
        Events with a distance below this threshold are grouped into clusters.

    """

    feature: BaseStatic
    ratios: dict[str, float] = {"train": 0.5, "val": 0.25, "test": 0.25}
    threshold: float = 0.2

    def model_post_init(self, log__) -> None:
        super().model_post_init(log__)
        if any(ratio <= 0 for ratio in self.ratios.values()):
            raise ValueError("All ratios must be greater than 0. Got: {self.ratios}")

        total_ratio = sum(self.ratios.values())
        if not np.isclose(total_ratio, 1.0, atol=1e-8):
            msg = f"The sum of ratios must be equal to 1.0. Got: {total_ratio}"
            raise ValueError(msg)

    def _similarity_clustering(self, embeddings) -> tp.List[int]:
        """Perform similarity-based clustering on the similarity matrix.

        Parameters
        ----------
        similarity_matrix: np.ndarray
            Precomputed cosine similarity matrix of dimension (number of events, number of events).


        Returns
        -------
        List[List[int]]
            Clusters of indices.
        """

        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics.pairwise import cosine_similarity

        similarity_matrix = cosine_similarity(embeddings)
        distance_matrix = 1 - similarity_matrix

        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            distance_threshold=self.threshold,
            linkage="complete",  # uses maximum distances between all observations of the 2 sets
        )

        return [label for label in clustering.fit_predict(distance_matrix)]

    def _cluster_assignment(self, clusters: list[int]) -> list[str]:
        """Assigns clusters to predefined splits (e.g., 'train', 'val', 'test') based on the
        ratios specified in `self.ratios`. Each cluster is assigned to a split such that
        the number of clusters in each split respects the specified ratio, and clusters are
        entirely allocated to a single split (no partial assignments).

        Parameters
        ----------
        clusters: list[int]
            A list of cluster IDs, where each cluster is represented by an integer,
            and the list contains the clusters to be assigned to splits.

        Returns
        -------
        list[str]
            A list of split labels ('train', 'val', 'test', etc.), corresponding
            to the same length as `clusters`, where each element indicates the split
            assignment for the corresponding cluster in the `clusters` input.
        """

        cluster_count = Counter(clusters)
        total_count = len(clusters)
        sorted_splits = sorted(self.ratios.items(), key=lambda x: x[1])
        split_sizes = {}
        cluster_split = {}

        # Split Assignment Strategy: littlest splits first, biggest split = remaining for no border effects
        remaining_count = total_count
        for split, ratio in sorted_splits[:-1]:
            split_sizes[split] = int(np.ceil(ratio * total_count))
            remaining_count -= int(np.ceil(ratio * total_count))
        largest_split = sorted_splits[-1][0]
        split_sizes[largest_split] = remaining_count

        # Assert all splits have at least one cluster
        if not all(split_sizes.values()):
            msg = "Some splits have no clusters"
            raise ValueError(msg)

        # Take all indexes of one cluster and assign them to a split
        for key, count in cluster_count.items():
            for split, size in split_sizes.items():
                if size >= count:
                    cluster_split[key] = split
                    split_sizes[split] -= count
                    break

        result = [cluster_split[cluster] for cluster in clusters]

        if not all(result):
            msg = "Some clusters were not assigned to any split"
            raise ValueError(msg)

        return result

    def __call__(self, events: pd.DataFrame) -> pd.DataFrame:
        """Splits a given DataFrame based on similarity clustering.

        Parameters
        ----------
        events: pd.DataFrame
            A DataFrame containing event data, with each row representing
            a single event and columns representing the event's attributes.

        Returns
        -------
        pd.DataFrame
            A copy of the input DataFrame with an additional 'split' column, which
            indicates the assigned split for each event.

        """

        self.feature.prepare(events)

        subclasses = [
            name
            for name, cls in event_module.Event._CLASSES.items()
            if issubclass(cls, self.feature.event_type)
        ]

        splitted_events = events[events["type"].isin(subclasses)]

        embeddings = []
        for _, row in splitted_events.iterrows():
            e = event_module.Event.from_dict(row.to_dict())
            embeddings.append(self.feature.get_static(e))

        embeddings = np.stack([embedding.numpy() for embedding in embeddings])
        clusters = self._similarity_clustering(embeddings)
        cluster_assignment = self._cluster_assignment(clusters)

        out_events = events.copy()
        out_events["split"] = ""
        out_events.loc[splitted_events.index, "split"] = cluster_assignment

        return out_events
