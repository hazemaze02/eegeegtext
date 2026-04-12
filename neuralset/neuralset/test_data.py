# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import concurrent.futures
import multiprocessing as mp
import os
import typing as tp
from pathlib import Path

import cloudpickle
import pandas as pd
import pydantic
import pytest

import neuralset as ns
import neuralset.enhancers as enh


def _check_loaded(loader: ns.data.StudyLoader) -> bool:
    if "fork" not in mp.get_start_method():
        msg = "In a non-forked subprocess, instances should not be registered yet"
        assert not ns.data.TIMELINES, msg
    _ = loader.build()
    return bool(ns.data.TIMELINES)


class DoNothing(enh.BaseEnhancer):
    name: tp.Literal["DoNothing"] = "DoNothing"
    param: str
    is_default: int = 12

    def __call__(self, events: pd.DataFrame) -> pd.DataFrame:
        return events


def test_loader(tmp_path: Path) -> None:
    loader = ns.data.StudyLoader(
        name="MneSample2013",
        path=ns.CACHE_FOLDER,
        infra={"folder": tmp_path},  # type: ignore
        enhancers=[{"name": "DoNothing", "param": "blublu-param"}],  # type: ignore
    )
    df = loader.build()
    folder = loader.infra.uid_folder()
    assert folder is not None
    # it was loaded from job
    assert isinstance(df, pd.DataFrame)
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as ex:
        job = ex.submit(_check_loaded, loader)
    assert job.result(), "Instances should have been registered at load time"
    # check signature on build infra (since infra ignores enhancers)
    cfg = loader._build_infra.config(uid=True, exclude_defaults=True)
    assert "DoNothing" in str(cfg)
    assert "blublu-param" in str(cfg)
    assert "is_default" not in str(cfg)


def test_loader_v2(tmp_path: Path) -> None:
    loader = ns.data.StudyLoader(
        name="MneSample2013",
        path=ns.CACHE_FOLDER,
        infra={"folder": tmp_path},  # type: ignore
        query="index==0",
    )
    df = loader.build()
    folder = loader.infra.uid_folder()
    assert folder is not None
    if not any(f.suffix == ".csv" for f in folder.iterdir()):
        names = [f.name for f in folder.iterdir()]
        raise RuntimeError(f"Missing csv file in folder with {names}")
    # timelines can be loaded from job (timelines are registered)
    assert isinstance(df, pd.DataFrame)
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as ex:
        job = ex.submit(_check_loaded, loader)  # type: ignore
    assert job.result(), "Instances should have been registered at load time"
    # check only one folder
    names = [f.name for f in tmp_path.iterdir()]
    if len(names) > 1:
        raise RuntimeError(f"Only one cache folder should have been created, got {names}")
    assert names[0] == "StudyLoader,0-v3,MneSample2013"
    names = [f.name for f in (tmp_path / names[0]).iterdir()]
    if len(names) != 2:
        raise RuntimeError(f"Only 2 infra folders should have been created, got {names}")
    # test version
    clone = loader.infra.clone_obj()
    assert clone.infra.version == loader.infra.version
    # test pickling
    string = cloudpickle.dumps(loader)
    unpickled = cloudpickle.loads(string)
    assert unpickled._build_infra.version == loader.infra.version
    # subjects
    subjects = {"MneSample2013/sample"}
    assert set(loader.study_summary().subject.unique()) == subjects
    assert set(df.subject.unique()) == subjects


@pytest.mark.parametrize(
    "name",
    (
        "Stuff123",
        "stuff1234",
        "xStuff1234",
        "Stuff12345",
        "Stuff1234Bolds",
        "Stuff1234Bol",
        "Stuff1234Mold",
    ),
)
def test_validation_study_name_error(name: str) -> None:
    with pytest.raises(ValueError):
        ns.data._validate_study_name(name)


@pytest.mark.parametrize("name", ("Name1234", "Name1234Bold", "Name1234Meg"))
def test_validation_study_name_correct(name: str) -> None:
    ns.data._validate_study_name(name)


class FakeData2222(ns.data.BaseData):
    # study/class level
    device: tp.ClassVar[str] = "Meg"
    run: int

    @classmethod
    def _download(cls, path: Path) -> None:
        for subject in [12, 13]:
            for run in range(2):
                ss_dir = path / f"sub-{subject}" / f"run-{run}"
                ss_dir.mkdir(parents=True, exist_ok=True)
                tmp_file = ss_dir / "tmp_fakedata2222.txt"
                tmp_file.touch()

    @classmethod
    def _iter_timelines(cls, path: str | Path) -> tp.Iterator["FakeData2222"]:
        for run in range(2):
            for subject in [12, 13]:
                yield FakeData2222(subject=str(subject), run=run, path=path)

    def _load_events(self) -> pd.DataFrame:
        return pd.DataFrame([])


def test_loader_on_external_study(tmp_path: Path) -> None:
    ns.data.StudyLoader(name="FakeData2222", path=tmp_path)
    with pytest.raises(pydantic.ValidationError):  # (does not exist)
        ns.data.StudyLoader(name="FakeData2223", path=tmp_path)


def test_loader_download(tmp_path: Path) -> None:
    FakeData2222.download(path=tmp_path)
    for subject in [12, 13]:
        for run in range(2):
            ss_dir = tmp_path / f"sub-{subject}" / f"run-{run}"
            tmp_file = ss_dir / "tmp_fakedata2222.txt"
            assert tmp_file.is_file()
            assert ss_dir.is_dir()
            assert oct(os.stat(ss_dir).st_mode & 0o777) == "0o777"
    assert oct(os.stat(tmp_path).st_mode & 0o777) == "0o777"


def test_loader_export() -> None:
    loader = ns.data.StudyLoader(name="MneSample2013", path=ns.CACHE_FOLDER)
    # fails with inf cast as int
    ns.data.StudyLoader(**loader.model_dump())


def test_study_loader_summary(tmp_path: Path) -> None:
    loader = ns.data.StudyLoader(
        name="FakeData2222", query="subject_index < 1", path=tmp_path
    )
    summary = loader.study_summary()  # filtered summary
    assert tuple(summary.subject_index) == (0, 0)
    summary = loader.study_summary(apply_query=False)
    assert tuple(summary.subject_index) == (0, 1, 0, 1)
    assert tuple(summary.timeline_index) == (0, 1, 2, 3)
    assert tuple(summary.subject_timeline_index) == (0, 0, 1, 1)


class Xp(pydantic.BaseModel):
    study: ns.data.StudyLoader
    infra: ns.infra.TaskInfra = ns.infra.TaskInfra()

    @infra.apply
    def build(self) -> pd.DataFrame:
        # assert self.study.infra.folder == self.study._build_infra.folder
        return self.study.build()


def test_xp(tmp_path: Path) -> None:
    xp = Xp(
        infra={"folder": tmp_path / "xp", "cluster": "local"},  # type: ignore
        study=dict(  # type: ignore
            name="MneSample2013",
            path=ns.CACHE_FOLDER,
            infra={"folder": tmp_path / "study"},
        ),
    )
    # test pickling
    string = cloudpickle.dumps(xp)
    unpickled = cloudpickle.loads(string)
    assert unpickled.study._build_infra.version == xp.study.infra.version
    assert unpickled.study._build_infra._infra_method.infra_name
    # run it
    xp.build()
    subs = list((tmp_path / "study").iterdir())
    assert len([x.name for x in subs]) == 1
    folder = subs[0]
    subnames = [x.name for x in folder.iterdir()]
    assert len(subnames) == 2, "A cache is missing (hidden infra has lost its params?)"


@pytest.mark.parametrize("n_timelines", ["all", 1])
def test_study_loader_compat(tmp_path: Path, n_timelines: tp.Any) -> None:
    params: tp.Any = dict(
        infra={"folder": tmp_path / "xp"},
        study=dict(
            name="MneSample2013",
            n_timelines=n_timelines,
            path=ns.CACHE_FOLDER,
            infra={"folder": tmp_path / "study"},
        ),
    )

    class CompatXp(pydantic.BaseModel):
        study: ns.data.StudyLoader
        infra: ns.infra.TaskInfra = ns.infra.TaskInfra()

        @infra.apply
        def build(self) -> pd.DataFrame:
            return self.study.build()

    xp = CompatXp(**params)
    expected = "study={name=MneSample2013,ntimelines=1}-42298278"
    if n_timelines == "all":
        expected = "study={name=MneSample2013,infra.version=0-v3}-08a6cbe1"
    assert xp.infra.uid().split("/")[-1] == expected
