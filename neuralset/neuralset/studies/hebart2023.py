# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import typing as tp
from functools import lru_cache
from pathlib import Path

import nibabel
import numpy as np
import pandas as pd

import neuralset as ns
from neuralset.data import BaseData


def _get_category(stimulus_filename: str):
    return "_".join(stimulus_filename.split("_")[:-1])


@lru_cache
def _get_stim_df(data_path: str, subject: str):
    betas_csv_dir = os.path.join(data_path, "betas_csv")
    if subject not in ["1", "2", "3"]:
        raise ValueError(f"Subject should be 1,2, or 3 but is {subject}")
    stim_f = os.path.join(betas_csv_dir, f"sub-{int(subject):02}_StimulusMetadata.csv")
    stimulus_data = pd.read_csv(stim_f)

    # add category column
    stimulus_data["category"] = stimulus_data.stimulus.map(_get_category)

    # add zero_shot_split column
    stimulus_data["zero_shot_split"] = stimulus_data.trial_type

    test_cats = stimulus_data[stimulus_data.trial_type == "test"].category.to_list()
    stimulus_data.loc[
        (stimulus_data.trial_type == "train") & (stimulus_data.category.isin(test_cats)),
        "zero_shot_split",
    ] = "trash"
    return stimulus_data


@lru_cache
def _get_voxel_dfs(data_path: str, subject: str):
    betas_csv_dir = os.path.join(data_path, "betas_csv")
    if subject not in ["1", "2", "3"]:
        raise ValueError(f"Subject should be 1,2, or 3 but is {subject}")

    data_file = os.path.join(betas_csv_dir, f"sub-{int(subject):02}_ResponseData.h5")
    betas = pd.read_hdf(data_file)  # this may take a minute
    betas = betas.drop(columns=["voxel_id"]).values.T  # type: ignore

    vox_f = os.path.join(betas_csv_dir, f"sub-{int(subject):02}_VoxelMetadata.csv")
    voxel_metadata = pd.read_csv(vox_f)

    return betas, voxel_metadata


class Hebart2023(BaseData):
    device: tp.ClassVar[str] = "Fmri"
    url: tp.ClassVar[str] = "https://plus.figshare.com/ndownloader/files/36789690"
    bibtex: tp.ClassVar[
        str
    ] = """
    @article{hebart2023things,
          title={THINGS-data, a multimodal collection of large-scale datasets for
          investigating object representations in human brain and behavior},
          author={Hebart, Martin N and Contier, Oliver and Teichmann, Lina and Rockter,
           Adam H and Zheng, Charles Y and Kidder, Alexis and Corriveau, Anna
           and Vaziri-Pashkam, Maryam and Baker, Chris I},
          journal={Elife},
          volume={12},
          pages={e82580},
          year={2023},
          publisher={eLife Sciences Publications Limited}
    }
    """
    doi: tp.ClassVar[str] = "doi:10.25452/figshare.plus.c.6161151.v1"
    licence: tp.ClassVar[str] = "CC0"
    description: tp.ClassVar[str] = (
        "Single-subject trials (betas) for 3 subjects watching still images in 3T fMRI"
    )
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("tables>=3.9.2",)

    N_SUBJECTS: tp.ClassVar[int] = 3
    path_to_stimulus: tp.ClassVar[str] = (
        "/checkpoint/jeanremi/babblebrain/contier_2022/prepare/"
    )

    available_rois: tp.ClassVar[str] = (
        "V1,V2,V3,hV4,VO1,VO2,LO1 (prf),LO2"
        " (prf),TO1,TO2,V3b,V3a,lEBA,rEBA,lFFA,rFFA,lOFA,rOFA,lPPA,rPPA,"
        "lRSC,rRSC,lTOS,rTOS,lLOC,rLOC"
    )

    trial: int
    original_split: str
    zero_shot_split: str
    image: str
    category: str
    session: int
    run: int

    @classmethod
    def _download(cls, path: Path) -> None:
        path.mkdir(exist_ok=True, parents=True)

        subprocess.run(["wget", "-P", path, cls.url])
        print(f'Running tar -xzvf {os.path.join(path, "36789690")}')
        subprocess.run(["tar", "-xzvf", os.path.join(path, "36789690"), "-C", path])

    @classmethod
    def get_events_with_roi_union(
        cls, events: pd.DataFrame, rois: str = "none"
    ) -> pd.DataFrame:
        """Returns a copy of `events` where each Fmri event has been modified to load
        data restricted to specified `rois` only

        Args:
            events (pd.DataFrame): A DataFrame containing event data.
            rois (str): A comma-separated string of ROIs, to be used
            for updating the filepath column. Defaults to 'none' (= no ROI applied).

        Raises:
            ValueError: 'rois' should be a None or a sublist of Hebart2023.available_rois

        Returns:
            _type_: The input DataFrame of events, where 'Fmri' events will now load
            only the corresponding ROI union
        """
        if rois == "none":
            return events

        rois_set = set(rois.split(","))
        all_rois_set = set(cls.available_rois.split(","))
        if not rois_set.issubset(all_rois_set):
            raise ValueError(
                f"ROIs {rois} should be selected among available ROIs:"
                f" {cls.available_rois}"
            )
        events.loc[events.type == "Fmri", "filepath"] = events.filepath.map(
            lambda fp: fp + f"&rois={rois}"
        )
        return events

    @classmethod
    def _iter_timelines(cls, path: str | Path):
        for subject in range(1, cls.N_SUBJECTS + 1):
            stimulus_data = _get_stim_df(path, str(subject))
            for row in stimulus_data.itertuples():
                yield cls(
                    subject=str(subject),
                    path=Path(path),
                    trial=row.trial_id,
                    original_split=row.trial_type,
                    zero_shot_split=row.zero_shot_split,
                    image=row.stimulus,
                    session=row.session,
                    run=row.run,
                    category=row.category,
                )

    def _load_raw(self, timeline: str, rois: str = "none") -> nibabel.Nifti2Image:
        betas, voxel_metadata = _get_voxel_dfs(self.path, self.subject)
        betas = betas[..., None]
        if rois == "none":
            return nibabel.Nifti2Image(betas[self.trial, :], np.eye(4))

        rois_list = [roi.strip() for roi in rois.split(",")]
        union_roi = voxel_metadata[rois_list].any(axis=1)
        return nibabel.Nifti2Image(betas[self.trial, union_roi], np.eye(4))

    def _load_events(self) -> pd.DataFrame:
        filepath = self._get_path_to_image()
        im = dict(
            type="Image",
            filepath=filepath,
            original_split=self.original_split,
            zero_shot_split=self.zero_shot_split,
            category=self.category,
            start=0,
            duration=1,
        )
        raw = dict(
            type="Fmri",
            filepath=f"method:_load_raw?timeline={self.timeline}",
            original_split=self.original_split,
            zero_shot_split=self.zero_shot_split,
            category=self.category,
            start=0,
            duration=1,
            frequency=1,
        )
        return pd.DataFrame([im, raw])

    def _get_path_to_image(self) -> str:
        return os.path.join(self.path_to_stimulus, self.category, self.image)  # type: ignore


SUBJ_TO_VOXEL_SHAPE: tp.Dict[str, tp.Tuple[int]] = {
    "1": (211339,),
    "2": (226950,),
    "3": (189164,),
}


def validate_dataset_assumptions() -> None:
    """Checks several assumptions on THINGS-fMRI-single-trial betas dataset
    derived from the companion-paper: https://elifesciences.org/articles/82580
    Takes more 10 mins to run
    """
    path = "/large_experiments/brainai/shared/studies/"
    loader = ns.data.StudyLoader(
        name="Hebart2023",
        path=path,
        download=False,
        n_timelines="all",
    )
    events = loader.build()
    for subj in ["1", "2", "3"]:
        _validate_subject_dataset(events[events.subject == subj], subj)


def _validate_subject_dataset(events_df: pd.DataFrame, subj: str) -> None:
    events = events_df[events_df.subject == subj]
    events_fmri = events[events.type == "Fmri"]
    events_image = events[events.type == "Image"]

    # There are two types of events
    allowed_types = {"Fmri", "Image"}
    assert (
        set(events.type.unique()) == allowed_types
    ), f"Allowed types are {allowed_types} but found {set(events.type.unique())}"

    # There are 9840 trials per subject (one Image event & one Fmri event for each trial)
    assert events_fmri.shape[0] == 9840
    assert events_image.shape[0] == 9840

    # Checking that the total number of recorded voxels is correct
    one_nifti = next(iter(ns.segments.read_events(events_fmri)))
    assert SUBJ_TO_VOXEL_SHAPE[subj] == one_nifti.shape, (
        f"Beta for subject {subj} has shape {one_nifti.shape} but should be"
        f" {SUBJ_TO_VOXEL_SHAPE[subj]}"
    )

    # There are 8640 train trials and 1200 test trials in the original split
    trials_by_type = events_image.original_split.value_counts().to_dict()
    assert trials_by_type["train"] == 8640
    assert trials_by_type["test"] == 1200

    # There are 7440 train trials and 1200 test trials in the zero-shot split
    # The zero-shot split is obtained from the original split by removing from
    # the training-split any image whose category appears in the original test-split
    # (there are 100 * 12 = 1200 such images)
    # The original test split is left untouched
    trials_by_type = events_image.zero_shot_split.value_counts().to_dict()
    assert trials_by_type["train"] == 7440
    assert trials_by_type["test"] == 1200

    events_image_train_original = events_image[events_image.original_split == "train"]
    events_image_test_original = events_image[events_image.original_split == "test"]

    # Each train stimulus is repeated only once
    events_image_train_original.filepath.nunique() == len(events_image_train_original)

    # Each of the 100 unique test stimulus is repeated 12 times
    # (once across each of the 12 sessions)
    stim_counts = (
        events_image_test_original.groupby("filepath").agg("count")["type"].unique()
    )
    assert list(stim_counts) == [12]

    # Each test category has exactly one image in the original test split
    test_cat_count = (
        events_image_test_original.groupby("category").filepath.nunique().unique()
    )
    assert list(test_cat_count) == [1]

    # Each test category has exactly 12 images in the original test split
    train_cat_count = (
        events_image_train_original.groupby("category").filepath.nunique().unique()
    )
    assert list(train_cat_count) == [12]

    # All test categories are training categories in the original split
    set(events_image_test_original.category).issubset(
        events_image_train_original.category
    )

    # Training set has 720 categories (12 stimulus per category)
    assert events_image_train_original.category.nunique() == 720

    # The zero_shot_split has disjoint categories between train and test
    set(
        events_image[events_image.zero_shot_split == "test"].category.to_list()
    ).isdisjoint(events_image[events_image.zero_shot_split == "train"].category.to_list())


if __name__ == "__main__":
    validate_dataset_assumptions()
