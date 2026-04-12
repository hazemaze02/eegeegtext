# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import pandas as pd
import tqdm

from neuralset.segments import Segment


def group_segments(
    segments: tp.Sequence[Segment],
    event_type: str = "Word",
    groupby: str = "text",
    group_subjects_together=True,
) -> tp.List[Segment]:
    """
    Group segments by attribute for a specific event type.
    Parameters:
    - segments: list of Segment instances
    - event_type: str, type of event to group
    - groupby: str, attribute to group by
    - group_subjects_together: bool, whether to group by subject as well
    Returns:
    - the grouped list of Segment instances
    """
    groups: dict = dict()
    for segment in tqdm.tqdm(
        segments, f"Grouping segments by attribute {groupby} for {event_type}"
    ):
        uid = getattr(segment._trigger, groupby)
        if not group_subjects_together:
            subject = getattr(segment._trigger, "subject")
            uid = f"{uid}_{subject}"
        if uid not in groups:
            # use the start as reference
            groups[uid] = {
                "start": segment.start,
                "duration": segment.duration,
                "events": [segment.events],
                "trigger": segment._trigger,
            }
        else:
            group = groups[uid]
            events = segment.events
            events.start -= (
                segment.start - group["start"]
            )  # time relative to the reference
            events = events.query(f"type!='{event_type}'")
            group["events"].append(events)

    grouped_segments = []
    for uid, group in groups.items():
        events = pd.concat(group["events"])
        events["uid"] = uid
        segment = Segment(start=group["start"], duration=group["duration"], events=events)
        segment._trigger = group["trigger"]
        grouped_segments.append(segment)
    return grouped_segments
