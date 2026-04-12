# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from pathlib import Path

import numpy as np
import pytest

import neuralset as ns
from neuralset import events as ev
from neuralset.data import STUDIES, _get_study
from neuralset.infra import MapInfra

logging.getLogger("neuralset").setLevel(logging.DEBUG)


STUDY_FOLDER = Path("/large_experiments/brainai/shared/studies/")
CLUSTER = "FAIR"
# AWS cluster
if not STUDY_FOLDER.exists():
    CLUSTER = "AWS"
    STUDY_FOLDER = Path("/storage/datasets01/shared/studies")
if not STUDY_FOLDER.exists():
    pytest.skip("Skipping as we are not on cluster", allow_module_level=True)


def test_allen_beta() -> None:
    events = ns.data.StudyLoader(
        name="Allen2022Beta",
        path=STUDY_FOLDER,
        # cache=cache,
        download=False,
        install=False,
        n_timelines=1,
    ).build()
    assert set(events["type"].unique()) == {"Fmri", "Image"}
    assert len(events) == 5

    event = ev.Fmri.from_dict(events.loc[events.type == "Fmri"].iloc[0])
    assert event.frequency == 1
    nii = event.read()
    n_voxels, n_times = nii.shape
    assert (n_voxels, n_times) == (15724, 1)


def test_allen_bold() -> None:
    name = "Allen2022Bold"
    events = ns.data.StudyLoader(
        name=name,
        path=STUDY_FOLDER,
        n_timelines=1,
    ).build()
    dset = ns.segments.iter_segments(events, idx=events.type == "Image", duration=4)
    event = ev.Fmri.from_dict(events.loc[events.type == "Fmri"].iloc[0])
    assert event.frequency == 0.625
    segment = next(iter(dset))
    for mesh, shape in [(None, (84, 105, 88, 2)), ("fsaverage5", (20484, 2))]:
        feature = ns.features.Fmri(mesh=mesh)
        data = feature(**segment.asdict())
        assert data.shape == shape


def test_contier(tmp_path: Path) -> None:
    # FIXME download failed
    name = "Contier2022"
    for cache in (tmp_path, tmp_path):  # no initial cache then cached
        events = ns.data.StudyLoader(
            name=name,
            path=STUDY_FOLDER / name.lower(),  # full path should work as well
            cache=cache,
            download=False,
            install=False,
            n_timelines=1,
        ).build()
        assert set(events["type"].unique()) == {"Meg", "Image"}
        assert len(events) == 207


def test_wen(tmp_path: Path) -> None:
    name = "Wen2017"
    events = ns.data.StudyLoader(
        name=name,
        path=STUDY_FOLDER,
        cache=tmp_path,
        download=False,
        install=False,
        n_timelines=1,
    ).build()
    assert set(events["type"].unique()) == {"Fmri", "Video"}
    assert len(events) == 2

    feature = ns.features.Fmri(frequency="native")
    dset = ns.segments.iter_segments(events, stride=4.0, duration=4.0)
    segment = next(iter(dset))
    data = feature(**segment.asdict())
    n_x, ny, nz, n_times = data.shape
    assert all(n > 0 for n in (n_x, ny, nz, n_times))


@pytest.mark.skip(reason="currently failing")
def test_zhou() -> None:
    name = "Zhou2023"
    events = ns.data.StudyLoader(
        name=name,
        path=STUDY_FOLDER,
        download=False,
        install=False,
        n_timelines=1,
    ).build()
    assert set(events["type"].unique()) == {"Fmri", "Video"}
    assert len(events) == 61

    feature = ns.features.Fmri(frequency=0.5)
    dset = ns.segments.iter_segments(events, stride=4.0, duration=4.0)
    segment = next(iter(dset))
    feature(**segment.asdict())


def test_gwilliams2022(tmp_path: Path) -> None:
    name = "Gwilliams2022"
    for _ in range(2):  # create cache and reoload it
        events = ns.data.StudyLoader(
            name=name,
            path=STUDY_FOLDER,
            cache=tmp_path,
            download=False,
            install=False,
            n_timelines=1,
        ).build()
        assert len(events)
        assert set(events["type"].unique()) == {
            "Meg",
            "Sound",
            "Phoneme",
            "Word",
            "Text",
            "Sentence",
        }
        event = ev.Meg.from_dict(events.loc[events.type == "Meg"].iloc[0])
        meg = event.read()
        assert (len(meg.ch_names), meg.n_times) == (256, 396000)


def test_grootswagers2022(tmp_path: Path) -> None:
    events = ns.data.StudyLoader(
        name="Grootswagers2022",
        path=STUDY_FOLDER,
        cache=tmp_path,
        download=False,
        install=False,
        n_timelines=1,
    ).build()
    assert set(events["type"].unique()) == {"Eeg", "Image"}
    assert len(events) == 22248 + 1

    event = ev.Eeg.from_dict(events.loc[events.type == "Eeg"].iloc[0])
    eeg = event.read()
    assert (len(eeg.ch_names), eeg.n_times) == (63, 3035740)


def test_li2022() -> None:
    events = ns.data.StudyLoader(
        name="Li2022",
        path=STUDY_FOLDER,
        query="timeline_index < 1",
        # query="subject_index == 35",  # english
        # infra={"cluster": None},
    ).build()
    assert set(events["type"].unique()) == {"Fmri", "Word", "Sound", "Text"}
    assert len(events) == 1617
    event = ev.Fmri.from_dict(events.loc[events.type == "Fmri"].iloc[0])
    nii = event.read()
    assert nii.shape == (73, 90, 74, 309)


def test_babayan2019(tmp_path: Path) -> None:
    cache = tmp_path
    events = ns.data.StudyLoader(
        name="Babayan2019",
        path=STUDY_FOLDER,
        cache=cache,
        download=False,
        install=False,
        n_timelines=1,
    ).build()

    assert set(events["type"].unique()) == {"Eeg", "EyeState"}
    assert len(events) == 17, "16 eye status events + 1 EEG."
    assert (
        (events.type == "EyeState") & (events.state == "open")
    ).sum() == 8, "8 mins eye open."
    assert (
        (events.type == "EyeState") & (events.state == "closed")
    ).sum() == 8, "8 mins eye closed."

    feature = ns.features.Eeg(frequency=250.0, filter=(None, 40.0), infra={"folder": cache})  # type: ignore
    dset = ns.segments.list_segments(
        events, idx=events.type == "EyeState", start=0.0, duration=30
    )
    segment = next(iter(dset))

    data = feature(**segment.asdict())
    assert data.shape == (62, 7500)


def test_armeni(tmp_path: Path) -> None:
    name = "Armeni2022"
    events = ns.data.StudyLoader(
        name=name,
        path=STUDY_FOLDER,
        cache=tmp_path,
        download=False,
        install=False,
        n_timelines=1,
    ).build()
    assert set(events["type"].unique()) == {
        "Meg",
        "Phoneme",
        "Sound",
        "Word",
        "Text",
        "Sentence",
    }


def test_nastase2020() -> None:
    name = "Nastase2020"
    loader = ns.data.StudyLoader(name=name, path=STUDY_FOLDER, query="timeline_index < 1")
    # summary
    summary = loader.study_summary(apply_query=False)
    assert summary.subject.nunique() == 321
    # non_excluded = summary.query("not excluded")  # already excluded
    # print(non_excluded.loc[:, ["subject", "story", "session"]].to_string())
    # assert non_excluded.subject.nunique() == 305
    # events
    events = loader.build()
    expected = {"Fmri", "Sound", "Word", "Text", "Phoneme", "Sentence"}
    assert set(events["type"].unique()) == expected


def test_gifford2021(tmp_path: Path) -> None:
    events = ns.data.StudyLoader(
        name="Gifford2021",
        path=STUDY_FOLDER,
        cache=None,
        download=False,
        install=False,
        n_timelines=2,  # At least one train and one test files
    ).build()
    assert set(events["type"].unique()) == {"Eeg", "Image"}
    # No overlap between train and test recordings
    train_descs = events.loc[events.split == "train", "description"]
    test_descs = events.loc[events.split == "test", "description"]
    assert len(np.intersect1d(train_descs, test_descs)) == 0

    assert len(events) == 20768  # There are some undocumented stimulus repetitions...
    dset = ns.segments.list_segments(
        events, idx=events.type == "Image", start=0.0, duration=0.299
    )

    segment = next(iter(dset))

    # Test EEG features
    eeg_feature = ns.features.Eeg(
        frequency=100.0, filter=(0.1, 40.0), infra={"folder": None}  # type: ignore
    )  # type: ignore
    data = eeg_feature(**segment.asdict())
    assert data.shape == (63, 30)

    # Test image features
    img_feature = ns.features.Image(frequency=0, device="cpu", aggregation="average")
    data = img_feature(**segment.asdict())
    assert data.shape == (768,)


def test_hebart2023_roi_configuration() -> None:

    from . import hebart2023

    loader = ns.data.StudyLoader(
        name="Hebart2023",
        path=STUDY_FOLDER,
        download=False,
        n_timelines=2,
    )

    events = loader.build()
    assert set(events["type"].unique()) == {"Fmri", "Image"}

    # Check that we obtain a number of voxels corresponding to selected ROIs
    # All available ROIs are selected by default
    events_no_rois = hebart2023.Hebart2023.get_events_with_roi_union(events)
    one_nifti = ns.events.Fmri.from_dict(
        events_no_rois[events_no_rois.type == "Fmri"].iloc[0]
    ).read()
    assert one_nifti.shape == (211339, 1)

    events_union_all_rois = hebart2023.Hebart2023.get_events_with_roi_union(
        events, hebart2023.Hebart2023.available_rois
    )
    one_nifti = ns.events.Fmri.from_dict(
        events_union_all_rois[events_union_all_rois.type == "Fmri"].iloc[0]
    ).read()
    assert one_nifti.shape == (10799, 1)


def test_lebel2023bold() -> None:
    name = "Lebel2023Bold"
    loader = ns.data.StudyLoader(
        name=name,
        path=STUDY_FOLDER,
        query="timeline_index < 1",
    )
    # summary
    summary = loader.study_summary(apply_query=False)
    assert len(summary) == 432  # timelines

    # event types
    events = loader.build()
    assert set(events["type"].unique()) == {"Fmri", "Word", "Phoneme", "Sound"}

    # Extract fMRI around first word
    event = ev.Fmri.from_dict(events.loc[events.type == "Fmri"].iloc[0])
    assert event.frequency == 0.5
    nii = event.read()
    assert nii.shape == (84, 84, 54, 363)


def test_lebelprocessed2023bold(tmp_path: Path) -> None:
    cache = np.random.choice([tmp_path, None])  # type: ignore
    print(f"Testing with infra.folder={cache!r}")
    name = "LebelProcessed2023Bold"
    loader = ns.data.StudyLoader(
        name=name,
        path=STUDY_FOLDER / "lebel2023bold",
        query="timeline_index < 1",
        infra={"folder": cache},  # type: ignore
    )
    # summary
    summary = loader.study_summary(apply_query=False)
    assert len(summary) == 386  # timelines
    # events
    events = loader.build()
    assert set(events["type"].unique()) == {"Fmri", "Word", "Phoneme", "Sound"}

    # Extract fMRI around first word
    event = ev.Fmri.from_dict(events.loc[events.type == "Fmri"].iloc[0])
    assert event.frequency == 0.5
    nii = event.read()
    assert nii.shape == (81126, 241)


def test_allen2022bold() -> None:
    loader = ns.data.StudyLoader(
        name="Allen2022Bold",
        path=STUDY_FOLDER,
        download=False,
        n_timelines=1,
    )

    events = loader.build()
    assert set(events["type"].unique()) == {"Fmri", "Image"}

    # Check that we obtain a number of voxels corresponding to selected ROIs
    # No ROI is selected by default
    one_nifti = ns.events.Fmri.from_dict(events[events.type == "Fmri"].iloc[0]).read()
    assert one_nifti.shape == (84, 105, 88, 188)


def test_hebart2023bold() -> None:
    events = ns.data.StudyLoader(
        name="Hebart2023Bold",
        path=STUDY_FOLDER,
        download=False,
        n_timelines=1,
        max_workers=20,
    ).build()
    assert events[events.type == "Fmri"].duplicated().sum() == 0

    assert set(events["type"].unique()) == {"Fmri", "Image"}
    assert events.shape[0] == 83
    one_nifti = ns.events.Fmri.from_dict(events[events.type == "Fmri"].iloc[0]).read()
    assert one_nifti.shape == (77, 94, 80, 284)


def test_pinetmeg2024(tmp_path: Path) -> None:
    name = "Pinet2024Meg"
    for cache in (tmp_path, tmp_path):
        events = ns.data.StudyLoader(
            name=name,
            path=STUDY_FOLDER,
            cache=cache,
            download=False,
            install=False,
            n_timelines=1,
        ).build()
        assert len(events) == 1851
        assert set(events["type"].unique()) == {"Meg", "Button", "Sentence", "Word"}

        feature = ns.features.Meg(frequency=100.0, infra=MapInfra(folder=cache))
        dset = ns.segments.list_segments(
            events,
            idx=events.type == "Button",
            start=0.0,
            duration=0.5,
        )
        segment = next(iter(dset))
        data = feature(**segment.asdict())
        n_channels, n_times = data.shape
        assert n_channels == 306
        assert n_times == 50


def test_pallier_listen() -> None:
    if CLUSTER != "AWS":
        pytest.skip("Pallier is not available")
    # FIXME download failed
    name = "PallierListen2023"
    events = ns.data.StudyLoader(
        name=name,
        install=True,
        path=STUDY_FOLDER,
        n_timelines=1,
    ).build()
    assert set(events["type"].unique()) == {"Meg", "Sentence", "Word", "Sound", "Text"}


def test_brennan() -> None:
    name = "Brennan2019"
    events = ns.data.StudyLoader(
        name=name,
        install=True,
        path=STUDY_FOLDER / "brennan2019",
        n_timelines=1,
    ).build()
    assert set(events["type"].unique()) == {"Eeg", "Word", "Sound", "Sentence"}


def test_broderick() -> None:
    name = "Broderick2019"
    events = ns.data.StudyLoader(
        name=name,
        install=True,
        path=STUDY_FOLDER,
        n_timelines=1,
    ).build()
    assert set(events["type"].unique()) == {"Eeg", "Word", "Phoneme", "Sound", "Sentence"}


def _iter_mne_studies() -> tp.Iterator[str]:
    try:
        _get_study("")  # trick into loading all existing studies
    except ValueError:
        pass
    return (n for n, c in STUDIES.items() if c.device in ("Meg", "Eeg"))


def test_iter_mne_studies() -> None:
    assert "Contier2022" in list(_iter_mne_studies()), "Something is wrong!"


@pytest.mark.parametrize("name", list(_iter_mne_studies()))
def test_first_samp(name: str) -> None:
    already_checked = [
        "Gwilliams2022",
        "Grootswagers2022",
        "Gifford2021",
        "Contier2022",
        "Pinet2024Meg",
        "MneSample2013",
    ]
    if name in already_checked:
        return  # avoid spending unnecessary time since they have no first_samp
    cls = STUDIES[name]
    loader = ns.data.StudyLoader(
        name=name, path=STUDY_FOLDER, download=False, n_timelines=1
    )
    try:
        events = loader.build()
    except Exception:  # pylint:disable=broad-except
        pytest.skip(f"No data or failed loading for {name}")
    events = events.loc[events.type == cls.device]
    raw = ev.Event.from_dict(events.iloc[0]).read()  # type: ignore
    if bool(raw.first_samp):
        RuntimeError(
            "first_samp > 0 requires specific checks as 0 of raw is not 0 of the timeline"
        )


def test_xu2024(tmp_path: Path) -> None:
    events = ns.data.StudyLoader(
        name="Xu2024",
        path=STUDY_FOLDER,
        cache=tmp_path,
        download=False,
        install=False,
        n_timelines=2,
        max_workers=0,
    ).build()
    assert set(events["type"].unique()) == {"Eeg", "Image"}
    assert len(events) == 7315

    event = ev.Eeg.from_dict(events.loc[events.type == "Eeg"].iloc[0])
    eeg = event.read()
    assert (len(eeg.ch_names), eeg.n_times) == (65, 1778688)


def test_chang2019() -> None:
    events = ns.data.StudyLoader(
        name="Chang2019",
        path=STUDY_FOLDER,
        download=False,
        n_timelines=1,
        max_workers=20,
    ).build()
    assert events[events.type == "Fmri"].duplicated().sum() == 0

    assert set(events["type"].unique()) == {"Fmri", "Image"}
    assert events.shape[0] == 38
    one_nifti = ns.events.Fmri.from_dict(events[events.type == "Fmri"].iloc[0]).read()
    assert one_nifti.shape == (77, 94, 80, 194)


def test_shen2020() -> None:
    events = ns.data.StudyLoader(
        name="Shen2020",
        path=STUDY_FOLDER,
        download=False,
        n_timelines=1,
    ).build()
    assert events[events.type == "Fmri"].duplicated().sum() == 0

    assert set(events["type"].unique()) == {"Fmri", "Image"}
    assert events.shape[0] == 56
    one_nifti = ns.events.Fmri.from_dict(events[events.type == "Fmri"].iloc[0]).read()
    assert one_nifti.shape == (77, 94, 80, 239)


def test_nieuwland2018() -> None:
    events = ns.data.StudyLoader(
        name="Nieuwland2018",
        path=STUDY_FOLDER,
        download=False,
        n_timelines=1,
    ).build()
    assert set(events.type.unique()) == {"Eeg", "Word", "Sentence"}


def test_schoffelen2019() -> None:
    events = ns.data.StudyLoader(
        name="Schoffelen2019",
        path=STUDY_FOLDER,
        n_timelines=1,
    ).build()
    assert set(events["type"].unique()) == {"Meg", "Word", "Sentence"}


def test_luke2021() -> None:
    events = ns.data.StudyLoader(
        name="Luke2021",
        path=STUDY_FOLDER,
        download=False,
        n_timelines="all",
    ).build()

    assert events[events.type == "Fnirs"].duplicated().sum() == 0
    assert set(events["type"].unique()) == {"Fnirs", "Stimulus"}
    assert events.shape[0] == 1037

    one_snirf = ns.events.Fnirs.from_dict(events[events.type == "Fnirs"].iloc[0]).read()
    assert one_snirf.get_data().shape == (66, 10443)


def test_accou2023() -> None:
    events = ns.data.StudyLoader(
        name="Accou2023",
        path=STUDY_FOLDER / "accou2023",
        download=False,
        n_timelines=1,
    ).build()
    assert set(events.type.unique()) == {"Eeg", "Word", "Sound", "Sentence"}


def test_khan2022() -> None:
    events = ns.data.StudyLoader(
        name="Khan2022",
        path=STUDY_FOLDER,
        download=False,
    ).build()

    assert set(events.type.unique()) == {"Eeg"}
    assert set(events.label.unique()) == {"normal", "abnormal"}
    assert events.shape[0] == 2417
    event = ev.Eeg.from_dict(events.loc[events.type == "Eeg"].iloc[0])
    eeg = event.read()
    assert len(eeg.ch_names) == 21


def test_vandijk2022() -> None:
    events = ns.data.StudyLoader(
        name="Vandijk2022",
        path=STUDY_FOLDER,
        download=False,
    ).build()

    assert len(events.loc[events.type == "Eeg"]) == 2692
    assert set(events.type.unique()) == {"Eeg", "EyeState"}
    assert set(events.state.dropna().unique()) == {"open", "closed"}
    event = ev.Eeg.from_dict(events.loc[events.type == "Eeg"].iloc[0])
    eeg = event.read()
    assert len(eeg.ch_names) == 28


def test_schalk2009() -> None:
    events = ns.data.StudyLoader(
        name="Schalk2009",
        path=STUDY_FOLDER,
        download=False,
        n_timelines="all",
    ).build()

    assert events.shape[0] == 41091
    assert len(set(events.subject)) == 109
    assert set(events.type) == {"Eeg", "Stimulus", "EyeState"}
    assert set(events.task.dropna()) == {"Rest", "Motor", "Imagery"}
    assert set(events.code.dropna()) == set(range(9))
    assert set(events.description.dropna()) == {
        "rest",
        "motor_left_fist",
        "motor_right_fist",
        "motor_bilateral_fist",
        "motor_bilateral_feet",
        "imagery_left_fist",
        "imagery_right_fist",
        "imagery_bilateral_fist",
        "imagery_bilateral_feet",
    }
    assert set(events[events.type == "EyeState"].state) == {"open", "closed"}
    event = ev.Eeg.from_dict(events.loc[events.type == "Eeg"].iloc[0])
    eeg = event.read()
    assert len(eeg.ch_names) == 64
    assert eeg.info["sfreq"] == 160.0


def test_ghassemi2018() -> None:
    events = ns.data.StudyLoader(
        name="Ghassemi2018",
        path=STUDY_FOLDER,
        download=False,
        n_timelines="all",
    ).build()

    assert set(events.type.unique()) == {"Eeg"}
    assert events.shape[0] == 1983
    assert events[events["split"] == "train"].shape[0] == 994
    assert events[events["split"] == "test"].shape[0] == 989
    event = ev.Eeg.from_dict(events.loc[events.type == "Eeg"].iloc[0])
    eeg = event.read()
    assert len(eeg.ch_names) == 13
    assert eeg.info["sfreq"] == 200.0


def test_lopez2017(tmp_path: Path) -> None:
    cache = tmp_path
    events = ns.data.StudyLoader(
        name="Lopez2017",
        path=STUDY_FOLDER,
        cache=cache,
        download=False,
        install=False,
        n_timelines="all",
    ).build()

    assert set(events.label) == {"abnormal", "normal"}

    # Verify with /large_experiments/brainai/shared/studies/lopez2017/tuh_eeg/tuab/AAREADME.txt
    assert len(events) == 2993
    assert ((events.split == "train") & (events.label == "normal")).sum() == 1371
    assert ((events.split == "train") & (events.label == "abnormal")).sum() == 1346
    assert ((events.split == "eval") & (events.label == "normal")).sum() == 150
    assert ((events.split == "eval") & (events.label == "abnormal")).sum() == 126

    eeg = ev.Eeg.from_dict(events.loc[events.type == "Eeg"].iloc[0]).read()
    assert (len(eeg.ch_names), eeg.n_times) == (26, 304250)


def test_obeid2016(tmp_path: Path) -> None:
    events = ns.data.StudyLoader(
        name="Obeid2016",
        path=STUDY_FOLDER,
        query=None,
        infra={"folder": tmp_path},  # type: ignore
    ).build()

    assert len(events) == 69652
    assert events.subject.nunique() == 14987
    assert set(events.frequency) == {250.0, 256.0, 400.0, 512.0, 1000.0}
    event = ev.Eeg.from_dict(events.loc[events.type == "Eeg"].iloc[0])
    eeg = event.read()
    assert len(eeg.ch_names) > 0
    assert eeg.n_times > 0


def test_hamid2020(tmp_path: Path) -> None:
    cache = tmp_path
    events = ns.data.StudyLoader(
        name="Hamid2020",
        path=STUDY_FOLDER,
        cache=cache,
        download=False,
        install=False,
        n_timelines="all",
    ).build()
    assert len(events.subject.unique()) == 213
    assert set(events.type) == {"Eeg", "Artifact", "Seizure"}
    assert len(events[events.type == "Artifact"]) == 190128
    assert len(events[events.type == "Seizure"]) == 6222
    assert len(events[events.type == "Eeg"]) == 310
    assert set(events[events.type == "Eeg"].frequency) == {
        250.0,
        256.0,
        400.0,
        512.0,
        1000.0,
    }
    eeg = ev.Eeg.from_dict(events.loc[events.type == "Eeg"].iloc[0]).read()
    assert len(eeg.ch_names) > 0
    assert eeg.n_times > 0


def test_veloso2017(tmp_path: Path) -> None:
    cache = tmp_path
    events = ns.data.StudyLoader(
        name="Veloso2017",
        path=STUDY_FOLDER,
        cache=cache,
        download=False,
        install=False,
        n_timelines="all",
    ).build()
    assert len(events) == 2298
    assert len(events.subject.unique()) == 200
    assert set(events.frequency) == {250.0, 256.0, 400.0, 512.0, 1000.0}
    assert set(events.type) == {"Eeg"}
    assert set(events.label) == {"epilepsy", "no_epilepsy"}
    eeg = ev.Eeg.from_dict(events.loc[events.type == "Eeg"].iloc[0]).read()
    assert len(eeg.ch_names) > 0
    assert eeg.n_times > 0


def test_harati2015(tmp_path: Path) -> None:
    cache = tmp_path
    events = ns.data.StudyLoader(
        name="Harati2015",
        path=STUDY_FOLDER,
        cache=cache,
        download=False,
        install=False,
        n_timelines="all",
    ).build()
    assert set(events.frequency.dropna()) == {250.0}
    assert set(events.type.dropna()) == {"Eeg", "EpileptiformActivity", "Artifact"}
    assert len(set(events.subject)) == 370
    assert len(events[events.type == "Eeg"]) == 518
    assert len(events[events.type == "EpileptiformActivity"]) == 22552
    assert len(events[events.type == "Artifact"]) == 4600
    assert set(events.channel.dropna()) == set(range(22))
    assert set(events.state.dropna()) == {"artf", "bckg", "eyem", "gped", "pled", "spsw"}
    eeg = ev.Eeg.from_dict(events.loc[events.type == "Eeg"].iloc[0]).read()
    assert len(eeg.ch_names) == 24
    assert eeg.n_times > 0


def test_vonweltin2017(tmp_path: Path) -> None:
    cache = tmp_path
    events = ns.data.StudyLoader(
        name="VonWeltin2017",
        path=STUDY_FOLDER,
        cache=cache,
        download=False,
        install=False,
        n_timelines="all",
    ).build()
    assert len(events) == 112
    assert len(events.subject.unique()) == 38
    assert set(events.frequency) == {250.0, 256.0, 400.0, 512.0}
    assert set(events.type) == {"Eeg"}
    eeg = ev.Eeg.from_dict(events.loc[events.type == "Eeg"].iloc[0]).read()
    assert len(eeg.ch_names) > 0
    assert eeg.n_times > 0


def test_shah2018(tmp_path: Path) -> None:
    cache = tmp_path
    events = ns.data.StudyLoader(
        name="Shah2018",
        path=STUDY_FOLDER,
        cache=cache,
        download=False,
        install=False,
        n_timelines="all",
    ).build()
    assert len(events) == 7361
    assert len(events.subject.unique()) == 675
    assert set(events.type) == {"Eeg"}
    assert set(events.frequency) == {250.0, 256.0, 400.0, 512.0, 1000.0}
    eeg = ev.Eeg.from_dict(events.loc[events.type == "Eeg"].iloc[0]).read()
    assert len(eeg.ch_names) > 0
    assert eeg.n_times > 0


def test_ctrllabs2024() -> None:
    name = "Ctrllabs2024"
    events = ns.data.StudyLoader(
        name=name,
        install=False,
        path="/private/home/agramfort/work/src/emg2qwerty/bids_data",
        n_timelines=2,
    ).build()
    assert len(events) == 9322
    assert set(events["type"].unique()) == {"Button", "Emg", "Sentence"}

    feature = ns.features.Emg(frequency=100.0)
    dset = ns.segments.list_segments(
        events,
        idx=events.type == "Button",
        start=0.0,
        duration=0.5,
    )
    segment = next(iter(dset))
    data = feature(**segment.asdict())
    assert data.shape == (32, 50)


def test_singer2023bold() -> None:
    events = ns.data.StudyLoader(
        name="Singer2023Bold",
        path=STUDY_FOLDER,
        download=False,
    ).build()
    assert events[events.type == "Fmri"].duplicated().sum() == 0

    assert set(events["type"]) == {"Fmri"}  # {"Fmri", "Image"}
    assert len(events[events["type"] == "Fmri"]) == 348
    one_nifti = ns.events.Fmri.from_dict(events[events.type == "Fmri"].iloc[0]).read()
    assert one_nifti.shape == (197, 233, 189, 251)  # May need to update after preproc


def test_singer2023meg() -> None:
    events = ns.data.StudyLoader(
        name="Singer2023Meg",
        path=STUDY_FOLDER,
        download=False,
    ).build()
    assert events[events.type == "Meg"].duplicated().sum() == 0
    assert len(events.subject.unique()) == 30

    assert set(events["type"]) == {"Meg"}  # {"Meg", "Image"}
    assert len(events[events["type"] == "Meg"]) == 270
    assert set(events.frequency) == {1000.0}
    meg = ev.Meg.from_dict(events.loc[events.type == "Meg"].iloc[0]).read()
    assert (len(meg.ch_names), meg.n_times) == (310, 483000)


def test_grootswagers2024(tmp_path: Path) -> None:
    cache = tmp_path
    events = ns.data.StudyLoader(
        name="Grootswagers2024",
        path=STUDY_FOLDER,
        cache=cache,
        download=False,
        install=False,
        n_timelines="all",
    ).build()
    assert len(events.subject.unique()) == 16
    assert len(events[events.type == "Eeg"]) == 16
    assert len(events[events.type == "Image"]) == 322524
    assert set(events.type) == {"Eeg", "Image", "Stimulus"}
    assert set(events.description.dropna()) == {
        "contrast",
        "orientation",
        "rgb_color",
        "spatial_frequency",
    }
    assert set(events.code.dropna()) == {0, 1, 2, 3}
    assert set(events[events.type == "Eeg"].frequency) == {1000.0}
    eeg = ev.Eeg.from_dict(events.loc[events.type == "Eeg"].iloc[0]).read()
    assert len(eeg.ch_names) == 125  # study says 128 expected


def test_kemp2000(tmp_path: Path) -> None:
    events = ns.data.StudyLoader(
        name="Kemp2000",
        path=STUDY_FOLDER,
        query=None,
        infra={"folder": tmp_path},  # type: ignore
    ).build()

    assert events.subject.nunique() == 78
    assert len(events[events.type == "Eeg"]) == 153
    assert set(events.type) == {"Eeg", "SleepStage", "Artifact"}
    assert set(events[events.type == "SleepStage"]["stage"]) == {
        "W",
        "N1",
        "N2",
        "N3",
        "R",
    }
    assert set(events[events.type == "Artifact"]["state"]) == {"musc"}
    assert set(events[events.type == "Eeg"].frequency) == {100.0}
    eeg = ev.Eeg.from_dict(events.loc[events.type == "Eeg"].iloc[0]).read()
    assert len(eeg.ch_names) == 6

    # Extract sliding windows from a single timeline
    tl_events = events[events.timeline == events.timeline[0]]
    duration = 30.0
    dset = tl_events.ns.iter_segments(
        tl_events.type == "SleepStage",
        duration=duration,
        stride=duration,
    )
    starts = np.array([seg.start for seg in dset])
    assert len(starts) == np.round(tl_events.iloc[0].stop / duration)
    strides = np.diff(starts)
    assert (strides == duration).all()
