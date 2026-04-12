# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

from torch.utils.data import DataLoader

import neuralset as ns

# setup path
# typically used across users, can be read-only
path = Path("/large_experiments/brainai/shared/studies")
cache = ns.CACHE_FOLDER  # ~<userhome>/.cache/neuralset
infra: tp.Any = {"folder": cache}  # Any to deactivate type check

# define study
study = "Zhou2023"

# Define the features we want to read/compute
# this will store the resampled embedding
flow = ns.features.OpticalFlow(frequency=0.5, infra=infra)
fmri = ns.features.Fmri(frequency=0.5)
flow.install_requirements()
fmri.install_requirements()

# This is only slow the first time you run it, then the
# dataframe is cached.
events = ns.data.StudyLoader(
    name=study,
    path=path,
    cache=cache,
    download=False,
    install=False,  # install pacakges required for this study
).build()

# Define the dataset (e.g. event.split=="train")
segments = ns.segments.list_segments(
    events, idx=events.type == "Video", start=0.0, duration=10.0
)

# you can also use a striding approach:
# segments = ns.segments.list_segments(events, stride=4., duration=10.)

# or a combination
# videos = events.type == "video"
# segments = ns.segments.list_segments(events, idx=videos, stride=4., duration=.10))

# define dataloader
dataset = ns.SegmentDataset({"OpticalFlow": flow, "Fmri": fmri}, segments)
dloader = DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=1)

# Load an actual batch in memory
batch = next(iter(dloader))
print(batch.data["Fmri"].shape, batch.data["OpticalFlow"].shape)
