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

# define the feature
fmri = ns.features.Fmri()
dinov2 = ns.features.Image(infra=infra)

# load events for the first subject (8859+982 trials)
# This is only slow the first time you run it, then the
# dataframe is cached.
events = ns.data.StudyLoader(
    name="Allen2022",
    path=path,
    cache=cache,
    download=False,
    install=False,  # install pacakges required for this study
    n_timelines=8859 + 982,
).build()
assert len(events) == 19682
assert events.subject.nunique() == 1

# Let's take only two recordings, for speed
sel = events.timeline.unique()[:2]
events = events.loc[events.timeline.isin(sel)]

# preprocess image embedding to store in cache
dinov2.prepare(events)

# Define the datasets (e.g. split=="train")
is_image = events.type == "Image"
segments = ns.segments.list_segments(events, idx=is_image, duration=1)

# use num_workers>1 for speed (but can conflict with cuda
# if embedding have not been cached yet)
dataset = ns.SegmentDataset({"neuro": fmri, "latent": dinov2}, segments)
dloader = DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=10)
batch = next(iter(dloader))

for k, v in batch.data.items():
    print(k, v.shape)
