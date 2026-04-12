# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functools
import subprocess
import typing as tp
from pathlib import Path

import nibabel
import numpy as np
import pandas as pd

from neuralset.studies.allen2022bold import (
    Allen2022Bold,
    _loadmat,
    get_allen2022_common_path,
)


@functools.lru_cache
def get_expdesign_mat(path):
    return _loadmat(get_allen2022_common_path(path) / "nsd_expdesign.mat")


class Allen2022Beta(Allen2022Bold):
    description: tp.ClassVar[str] = (
        "Pre-processed fMRI betas for 8 subjects watching still images in 7T fMRI"
        "masked with subject-wise 'nsdgeneral' ROI"
    )

    BETA_TYPES: tp.ClassVar[tp.Tuple[str, ...]] = (
        "betas_fithrf",
        "betas_fithrf_GLMdenoise_RR",
    )

    spaces: tp.ClassVar[tp.Tuple[str, ...]] = ("volume_nsd_native", "fsaverage5")

    N_TRIALS_PER_SESSION: tp.ClassVar[int] = 750
    session_wise_trial_idx: int  # 750, 0-based
    subject_wise_trial_idx: int  # 30k, 0-based

    @classmethod
    def _download(cls, path: Path, s3_profile: str = "saml") -> None:

        s3_bucket = "s3://natural-scenes-dataset"
        hemis = ["lh", "rh"]

        to_download = []
        for beta_type in cls.BETA_TYPES:
            for subj in range(1, 9):
                to_download.append(
                    f"nsddata/ppdata/subj{subj:02}/func1pt8mm/roi/nsdgeneral.nii.gz"
                )
                for hemi in hemis:
                    to_download.append(
                        f"nsddata_betas/ppdata/subj{subj:02}/fsaverage/{beta_type}/"
                        f"{hemi}.ncsnr.mgh"
                    )

                for session in range(1, cls.SESSIONS_PER_SUBJECT[subj] + 1):
                    to_download.append(
                        f"nsddata_betas/ppdata/subj{subj:02}/func1pt8mm/{beta_type}/"
                        f"betas_session{session:02}.nii.gz"
                    )

                    for hemi in hemis:
                        to_download.append(
                            f"nsddata_betas/ppdata/subj{subj:02}/fsaverage/{beta_type}/"
                            f"{hemi}.betas_session{session:02}.mgh"
                        )

        for filepath in to_download:
            local_path = path / filepath
            if not local_path.parent.exists():
                local_path.parent.mkdir(parents=True, exist_ok=True)

            command = [
                "aws",
                "s3",
                "cp",
                f"{s3_bucket}/{filepath}",
                str(local_path),
                "--profile",
                s3_profile,
            ]
            if not local_path.exists():
                subprocess.run(" ".join(command), shell=True)

    @classmethod
    def _iter_timelines(cls, path: str | Path):
        path = Path(path)
        for subject in cls.SESSIONS_PER_SUBJECT:
            for session in range(1, cls.SESSIONS_PER_SUBJECT[subject] + 1):
                for session_wise_trial_idx in range(cls.N_TRIALS_PER_SESSION):
                    subject_wise_trial_idx = (session - 1) * 750 + session_wise_trial_idx
                    yield cls(
                        subject=str(subject),
                        session=session,
                        session_wise_trial_idx=session_wise_trial_idx,
                        subject_wise_trial_idx=subject_wise_trial_idx,
                        run=None,
                        path=path,
                    )

    def _load_events(self) -> pd.DataFrame:
        fmri_events = []
        for space in self.spaces:
            for beta_type in self.BETA_TYPES:
                fmri_events.append(
                    {
                        "filepath": f"method:_load_raw?timeline={self.timeline}"
                        f"&beta_type={beta_type}"
                        f"&space={space}",
                        "type": "Fmri",
                        # convention for neuralset betas studies
                        "start": 0.0,
                        "frequency": 1.0,
                        "duration": 1.0,
                        "beta_type": beta_type,
                        "space": space,
                    }
                )

        expdesign_mat = get_expdesign_mat(self.path)
        # 1-based
        stim_idx_10k = expdesign_mat["masterordering"][self.subject_wise_trial_idx]
        # 0-based
        stim_idx_73k = (
            expdesign_mat["subjectim"][int(self.subject) - 1, stim_idx_10k - 1] - 1
        )

        path_to_stimuli = get_allen2022_common_path(self.path) / "nsd_stimuli/"

        stimulus = dict(
            type="Image",
            start=0.0,
            duration=1.0,
            filepath=str(Path(path_to_stimuli) / f"{stim_idx_73k}.png"),
            split=("test" if stim_idx_10k - 1 < 1000 else "train"),
            caption=self._get_captions(stim_idx_73k),
        )

        return pd.concat(
            [
                pd.DataFrame(fmri_events),
                pd.DataFrame([stimulus]),
            ],
            axis=0,
        )

    def _load_raw(self, timeline: str, beta_type: str, space: str) -> nibabel.Nifti1Image:  # type: ignore
        path = Path(self.path)
        # load the beta
        if space == "volume_nsd_native":
            path_to_betas = (
                path
                / f"nsddata_betas/ppdata/subj{int(self.subject):02}/func1pt8mm/{beta_type}/"
                f"betas_session{self.session:02}.nii.gz"
            )
            betas = nibabel.load(path_to_betas, mmap=True).get_fdata().astype(np.float32)  # type: ignore
            fmri = betas[..., self.session_wise_trial_idx]
            # load the nsdgeneral mask for the subject
            path_to_subj_roi = (
                path / f"nsddata/ppdata/subj{int(self.subject):02}/func1pt8mm/roi/"
                "nsdgeneral.nii.gz"
            )
            mask = nibabel.load(path_to_subj_roi, mmap=True).get_fdata()  # type: ignore
            fmri = fmri[mask > 0.0]
            return nibabel.Nifti1Image(fmri[..., None], np.eye(4))
        elif space == "fsaverage5":
            # Load the hemispheres
            path_to_betas = (
                path
                / f"nsddata_betas/ppdata/subj{int(self.subject):02}/fsaverage/{beta_type}/"
            )

            fs_hemis = []
            for hemi in ["lh", "rh"]:
                betas = nibabel.load(
                    path_to_betas / f"{hemi}.betas_session{self.session:02}.mgh",
                    mmap=True,
                )
                assert betas.shape == (163842, 1, 1, 750)
                fs5_beta = betas.slicer[:10242, :1, :1, self.session_wise_trial_idx]  # type: ignore
                fs_hemis.append(fs5_beta.get_fdata())

            fsaverage5 = np.concatenate(fs_hemis)
            return nibabel.Nifti1Image(fsaverage5.squeeze((1, 2))[..., None], np.eye(4))
        else:
            raise ValueError(f"Unknown fMRI space: {space}")
