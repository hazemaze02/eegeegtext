# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import neuralset as ns

from .pipelines import NeuroLoader, TimeDecoding


@pytest.mark.parametrize("frequency", (100, 200))
def test_evoked(frequency: float) -> None:
    meg = ns.features.Meg(
        frequency=frequency,
        filter=(0.05, 20.0),
        baseline=(0.0, 0.5),
    )
    study = ns.data.StudyLoader(name="MneSample2013", path=ns.CACHE_FOLDER)
    loader = NeuroLoader(
        study=study, event_type="Stimulus", start=-0.1, duration=1, neuro=meg
    )
    batch = loader.load_neuro().numpy()
    evoked = np.median(batch, 0)
    assert evoked.shape[1] == frequency
    index = np.argmax(abs(evoked).mean(0))  # largest peak
    assert index / frequency == pytest.approx(0.2, abs=0.01)


@pytest.mark.parametrize("model", ["RidgeCV", "RidgeClassifierCV"])
def test_decoding(model) -> None:
    study = ns.data.StudyLoader(name="MneSample2013", path=ns.CACHE_FOLDER)
    loader = TimeDecoding(
        study=study,
        model=model,
        event_type="Stimulus",
        start=-0.1,
        duration=1,
        target=ns.features.Stimulus(aggregation="trigger"),
    )
    scores = loader.decode()
    assert scores.shape == (1, 50)  # 1 subject, 1s @50Hz
