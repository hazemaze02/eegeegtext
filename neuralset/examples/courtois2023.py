# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from torch.utils.data import DataLoader

import neuralset as ns

# setup path
# typically used across users, can be read-only
path = Path("/large_experiments/brainai/data")
cache = ns.CACHE_FOLDER  # ~<userhome>/.cache/neuralset


# define study
study = "Courtois2023"
# courtois_freq = 1/ 1.49
courtois_freq = 0.5

# Define the features we want to read/compute
# this will store the resampled embedding
video = ns.features.Video(frequency=courtois_freq)
# FIXME: validate this
fmri = ns.features.Fmri(frequency=courtois_freq)
video.install_requirements()
fmri.install_requirements()

# This is only slow the first time you run it, then the
# dataframe is cached.
events = ns.data.StudyLoader(
    name=study,
    path=path / study,
    cache=cache / study,
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
dataset = ns.SegmentDataset({"vid": video, "fmri": fmri}, segments)
dloader = DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=1)

# Load an actual batch in memory
batch = next(iter(dloader))
print({x: y.shape for x, y in batch.data.items()})
