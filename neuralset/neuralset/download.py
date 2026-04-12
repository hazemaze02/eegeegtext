# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import shutil
import subprocess
import typing as tp
from abc import abstractmethod
from glob import glob
from pathlib import Path

import pydantic
from tqdm import tqdm

from .base import _Module


class BaseDownload(_Module):
    study: str
    dset_dir: str | Path
    folder: str = "download"

    _dl_dir: Path = pydantic.PrivateAttr()

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        # check that parent folder exist and create download sub-folder
        dset_dir = Path(self.dset_dir).resolve()
        if not dset_dir.parent.exists():
            raise ValueError(f"Parent folder must exist for {dset_dir}")
        dset_dir.mkdir(exist_ok=True)
        self._dl_dir = dset_dir / self.folder
        self._dl_dir.mkdir(exist_ok=True, parents=True)

    def get_success_file(self) -> Path:
        cls_name = self.__class__.__name__.lower()
        return self._dl_dir / f"{cls_name}_{self.study}_success_download.txt"

    @tp.final
    def download(self, overwrite: bool = False) -> None:
        if self.get_success_file().exists() and not overwrite:
            return
        print(f"Downloading {self.study} to {self._dl_dir}...")
        self._download()
        self.get_success_file().write_text("success")
        print("Done! Consider running giving read/write permissions to everyone:")
        print(f"chmod 777 -R {self._dl_dir}")  # we should do this on FAIR cluster

    @abstractmethod
    def _download(self):
        raise NotImplementedError


class Osf(BaseDownload):
    storage_inds: list[int] = [0]  # In case of multiple storages, storages to download
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("osfclient>=0.0.5",)

    def _download(self) -> None:
        import osfclient  # noqa

        project = osfclient.OSF().project(self.study)
        store = list(project.storages)

        pbar = tqdm()
        for ind in self.storage_inds:
            for source in store[ind].files:
                path = source.path
                if path.startswith("/"):
                    path = path[1:]

                file_ = self._dl_dir / path

                if file_.exists():
                    continue

                pbar.set_description(file_.name)
                file_.parent.mkdir(parents=True, exist_ok=True)
                with file_.open("wb") as fb:
                    source.write_to(fb)


class Donders(BaseDownload):
    parent: str = "dccn"
    study_id: str
    user: str = ""
    password: str = ""

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if not self.user or not self.password:
            print(
                "Get the user and password from "
                "https://data.donders.ru.nl/collections/"
                "di/dccn/DSC_3011020.09_236?0"
            )
            self.user = input("user:")
            self.password = input("password:")

    def _download(self) -> None:
        command = "wget -r -nH -np --cut-dirs=1"
        command += " --no-check-certificate -U Mozilla"
        command += f" --user={self.user} --password={self.password}"
        command += " https://webdav.data.donders.ru.nl/"
        command += f"{self.parent}/{self.study_id}/ -P {self.dset_dir}"
        command += ' -R "index.html*" -e robots=off'
        print("Running command : ", command)
        result = subprocess.run(command.split(), capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        if "Authentication Failed" in result.stderr:
            raise ValueError("Authentication Failed.")
        # donders download in the authorYEAR/study_code/
        # we want the content to be in authorYEAR/download/
        shutil.move(Path(self.dset_dir) / self.study, self._dl_dir)


class Openneuro(BaseDownload):
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("boto3",)

    def _download(self) -> None:
        import boto3  # noqa
        from botocore import UNSIGNED
        from botocore.config import Config

        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

        # List all objects in the bucket
        bucket_name = "openneuro.org"
        content: tp.List[tp.Dict[str, tp.Any]] = []
        continuation = ""
        while continuation or not content:
            kwargs = {} if not continuation else {"ContinuationToken": continuation}
            response = s3.list_objects_v2(Bucket=bucket_name, Prefix=self.study, **kwargs)
            content.extend(response["Contents"])
            continuation = response.get("NextContinuationToken", "")
        desc = f"Downloading {self.study}"
        for obj in tqdm(content, desc=desc, mininterval=10, ncols=20):
            key = obj["Key"]
            # Remove prefix from the filename
            filename = self._dl_dir / Path(key).relative_to(self.study)
            # Create directory structure if it doesn't exist
            filename.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket_name, key, str(filename))
            print(f"Downloaded: {filename}")


class Wildcard(pydantic.BaseModel):
    folder: str


class Datalad(BaseDownload):
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("datalad-installer",)
    # url of the datalad repo to clone
    repo_url: str
    # number of threads used for the datalad operations
    threads: int = 1
    # list of folders (or wildcards) to actually clone
    folders: list[str | Wildcard] = []

    @classmethod
    def install_requirements(cls) -> None:
        # install datalad-installer
        super().install_requirements()
        # use datalad-installer to install datalad and git-annex;
        # this is datalab's recommended installation: https://www.datalad.org/#install
        subprocess.run(
            [
                "datalad-installer",
                "datalad",
                "git-annex",  # "-m datalad/git-annex:release" requires root
            ]
        )

    @pydantic.computed_field  # type: ignore
    @property
    def repo_name(self) -> str:
        # retrieve name of the repo
        repo_name = Path(self.repo_url).name
        if Path(repo_name).suffix == ".git":
            repo_name = repo_name[:-4]
        return repo_name

    def _datalad(self, cmd: str, path: Path | str) -> None:
        # TODO: do not use subprocess
        proc = subprocess.run(
            cmd, cwd=str(path), capture_output=True, text=True, shell=True
        )
        if "install(error)" in proc.stdout:
            logging.warning("Potential error in datalad clone:\n> %s", proc.stdout)
        if proc.stderr:
            # NOTE: stderr might be populated even in success case
            logging.warning("Potential error in datalad clone:\n> %s", proc.stderr)
            # raise RuntimeError(f"Clone Failed: {proc.stderr}")

    def _dl_item(self, cur_path: Path | str) -> None:
        threads_ = "" if self.threads > 1 else f" -J {self.threads}"
        cmd = f'datalad get "{cur_path}"{threads_}'
        self._datalad(cmd, self._dl_dir / self.repo_name)

    def _download(self) -> None:  # weird stuff happening, had to deactivate typing
        """Downloads data from datalab

        Since this requires a git connection to handle, make sure that
        git ssh-key is password free

        Parameters
            path: Path to store the dataset (will clone repo to that folder)
            url: Url of the datalad repository
            threads: Number of threads to parallize dataset download
            folders: List of folders to clone explicitly (otherwise everything is cloned).
                Contains a tuple of str and bool. Bool defines if str is a glob
        """
        # clone repo
        self._datalad(f"datalad clone {self.repo_url}", self._dl_dir)

        # expand folders
        folders = self.folders if self.folders else [Wildcard(folder="*")]

        all_folders: list[Path] = []
        for folder in folders:
            if isinstance(folder, Wildcard):
                all_folders += [
                    Path(str(p))
                    for p in glob(str(self._dl_dir / self.repo_name / folder.folder))
                ]
            else:
                all_folders += [self._dl_dir / self.repo_name / folder]  # type: ignore
        print(f"Loading {len(all_folders)} folders: ", all_folders)

        # download
        for item in tqdm(all_folders, desc=f"Downloading {self.study}", ncols=100):
            if not item.is_dir():
                continue
            self._dl_item(item)

        print("\nDownloaded Dataset")
