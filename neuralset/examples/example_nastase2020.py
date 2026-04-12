# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import numpy as np
from nilearn import datasets, plotting
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale

from neuralset.data import StudyLoader
from neuralset.dataloader import SegmentDataset
from neuralset.features.neuro import Fmri
from neuralset.features.text import WordFrequency
from neuralset.segments import list_segments

# setup path
data_path = Path("/datasets01/hasson_narratives/")
cache_path = Path.home() / ".cache/neuralset/tmp"
infra = {"folder": cache_path, "mode": "retry"}

TR = 1.5
freq = 1.0 / TR

# This is only slow the first time you run it, then the
# dataframe is cached.
study = StudyLoader(
    name="Nastase2020",
    path=data_path,
    # cache=cache_path,
    n_timelines=1,
)

events = study.build()
events = events.loc[events.type != "Phoneme"]

# Extract data
sel = events.type == "Sound"
sounds = events.loc[sel]
assert len(sounds) == 1
sound = sounds.iloc[0]
segments = list_segments(events, idx=sel, start=0.0, duration=sound.duration)

wordfreq = WordFrequency(frequency=freq, aggregation="sum")
fmri = Fmri(frequency=freq)
features = {"WordFrequency": wordfreq, "Fmri": fmri}
ds = SegmentDataset(features, segments)
batch = ds.as_one_batch()

# Encoding model
X = scale(batch.data["WordFrequency"][0].t())
Y = scale(batch.data["Fmri"][0].t())


def correlate(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    SX2 = (X**2).sum(0) ** 0.5
    SY2 = (Y**2).sum(0) ** 0.5
    SXY = (X * Y).sum(0)
    R = SXY / np.maximum(SX2 * SY2, 1e-12)
    return R


model = Ridge()
cv = KFold(3, shuffle=False)
R = []
for train, test in cv.split(X):
    model.fit(X[train], Y[train])
    Y_pred = model.predict(X[test])
    R.append(correlate(Y[test], Y_pred))
R = np.mean(R, 0)

# Plot
fsaverage = datasets.fetch_surf_fsaverage("fsaverage6")
n_vertices = len(R) // 2

r = R[:n_vertices]
fig = plotting.plot_surf_stat_map(
    surf_mesh=str(fsaverage["infl_left"]),
    stat_map=r,
    hemi="left",
    view="lateral",
    cmap="cold_hot",
    symmetric_cbar=True,
    threshold=0.01,
    bg_map=str(fsaverage["sulc_left"]),
    colorbar=True,
    title="",
)
