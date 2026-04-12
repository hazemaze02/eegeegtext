# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

from torch.utils.data import DataLoader

import neuralset as ns

# setup path
# typically used across users, can be read-only
path = "/large_experiments/brainai/shared/studies"
cache = ns.CACHE_FOLDER  # ~<userhome>/.cache/neuralset
infra: tp.Any = {"folder": cache}  # Any to deactivate type check

# load events for the first subject (2 session * 4 runs)
# This is only slow the first time you run it, then the
# dataframe is cached.
events = ns.data.StudyLoader(
    name="Gwilliams2022",
    path=path,
    cache=cache,
    download=False,
    install=False,
    n_timelines=8,
).build()
assert len(events) == 78712
assert events.subject.nunique() == 1

# Define the datasets
is_valid = events.text.apply(lambda x: isinstance(x, str))
# FIXME remove invalid events
events = events.loc[(events.type == "Meg") | is_valid]
segments = ns.segments.list_segments(
    events, idx=events.type == "Word", start=-0.5, duration=2
)

# define the feature
meg = ns.features.Meg(
    frequency=100.0, filter=(0.05, 20.0), baseline=(0.0, 0.5), infra=infra
)
wordemb = ns.features.SpacyEmbedding()

# retrieve data
dataset = ns.SegmentDataset({"neuro": meg, "text": wordemb}, segments)
dloader = DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=1)

for batch in dloader:
    print({x: y.shape for x, y in batch.data.items()})
    break
