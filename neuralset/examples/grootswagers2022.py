# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from tqdm import trange
from wordfreq import zipf_frequency

import neuralset as ns

# load study
data_path = "/large_experiments/brainai/shared/studies/"
events_ = ns.data.StudyLoader(
    name="Grootswagers2022",
    path=data_path,
    cache=ns.CACHE_FOLDER,
    download=False,
    install=False,
    n_timelines=3,
).build()

all_R = []
for tl, events in events_.groupby("timeline"):
    # Preprocess EEG
    eeg = events.loc[events.type == "Eeg"]
    raw = ns.events.Eeg.from_dict(eeg.loc[0]).read()

    raw.load_data()
    raw = raw.filter(0.5, 20.0, n_jobs=-1)

    # Segment data
    images = events.loc[events.type == "Image"]
    mne_events = np.ones((len(images), 3), dtype=int)
    mne_events[:, 0] = images.start * raw.info["sfreq"]

    epochs = mne.Epochs(
        raw, mne_events, metadata=images, tmin=-0.2, tmax=1, decim=10, preload=True
    )

    # Post-process
    X = epochs.get_data()
    X -= X.mean(0)
    X /= X.std(0)

    # get word-frequency (rare words ~ surprising images?)
    y = np.asarray([zipf_frequency(w, "en") for w in epochs.metadata.category])

    # Decode at each time point
    model = RidgeCV(np.logspace(-2, 4, 10))

    n_times = len(epochs.times)
    cv = GroupKFold(5)
    R = []
    for train, test in cv.split(X, groups=epochs.metadata.category):
        r = np.zeros(n_times)
        for t in trange(n_times):
            model.fit(X[train, :, t], y[train])
            y_pred = model.predict(X[test, :, t])
            r[t], _ = pearsonr(y[test], y_pred)
        R.append(r)
    R = np.mean(R, 0)

    all_R.append(R)

# plot results
fig = plt.fill_between(epochs.times, np.mean(all_R, 0))


def plot_in_terminal(fig):
    """show matplotlib figure"""
    # -- terminal plot
    in_notebook = False
    try:
        from IPython import get_ipython

        if "IPKernelApp" in get_ipython().config:
            in_notebook = True
    except ImportError:
        pass
    if not in_notebook:
        from pathlib import Path

        import plotext

        file = Path("tmp.jpg").absolute()
        fig.savefig(file, format="jpg")
        plotext.image_plot(file)
        plotext.show()
        file.unlink()
    else:
        fig.show()


plot_in_terminal(fig)
