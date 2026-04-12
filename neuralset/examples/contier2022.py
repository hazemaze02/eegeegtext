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

# load events for the first subject (12 session*10 runs)
# This is only slow the first time you run it, then the
# dataframe is cached.
events = ns.data.StudyLoader(
    name="Contier2022",
    path=path,
    cache=cache,
    download=False,
    install=False,
    n_timelines=12 * 10,
).build()
assert len(events) == 27168
assert events.subject.nunique() == 1

# Let's take only 2 recordings for speed
sel = events.timeline.unique()[:2]
events = events.loc[events.timeline.isin(sel)]

# define the feature
meg = ns.features.Meg(
    frequency=100.0, filter=(0.05, 20.0), baseline=(0.0, 0.5), infra=infra
)
image = ns.features.Image(infra=infra)

meg.prepare(events)
image.prepare(events)

# Define the dataset (e.g. event.split=="train")
segments = ns.segments.list_segments(
    events, idx=events.type == "Image", start=-0.5, duration=2
)

# define dataloader, here load everything at once
# note that you can use num_workers for speed, but this can conflict
# with the features that use cuda, if they have not already been cached
dataset = ns.SegmentDataset({"neuro": meg, "latent": image}, segments)
dloader = DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=10)
batch = next(iter(dloader))

print({x: y.shape for x, y in batch.data.items()})
