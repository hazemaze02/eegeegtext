# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import concurrent.futures
import os
import typing as tp
from pathlib import Path

import numpy as np
import pydantic
import torch
from sentence_decoding.callbacks import TestRetrieval
from sentence_decoding.main import Data
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from tqdm import trange

from neuralset.infra.task import TaskInfra
from neuraltrain.metrics import TopkAcc


class Mapper(pydantic.BaseModel):

    model_config = pydantic.ConfigDict(extra="allow")

    dynamic: bool = True
    alphas_per_target: bool = True
    stride: int = 1
    _model: tp.Any = pydantic.PrivateAttr()

    def apply(self, neuro, feature, words) -> np.ndarray:
        ridge = RidgeCV(np.logspace(-2, 8, 7), alpha_per_target=self.alphas_per_target)
        # scaler = StandardScaler()
        # model = make_pipeline(scaler, ridge)
        model = ridge

        _, n_channels, n_times = neuro["train"].shape
        n_dims = feature["train"].shape[1]
        Y = feature["train"]
        Y_test = feature["test"]

        if self.dynamic:
            R = torch.zeros(n_times // self.stride, n_dims)
            coefs = torch.zeros(n_times // self.stride, n_dims, n_channels)
            acc = torch.zeros(n_times // self.stride)
            agg_acc = torch.zeros(n_times // self.stride)
            for t in trange(0, n_times, self.stride, desc="Decoding"):
                X = neuro["train"][:, :, t]
                X_test = neuro["test"][:, :, t]
                model.fit(X, Y)
                Y_pred = model.predict(X_test)
                for d in range(Y.shape[1]):
                    R[t // self.stride, d], _ = pearsonr(Y_test[:, d], Y_pred[:, d])
                coefs[t // self.stride] = torch.from_numpy(ridge.coef_)

                Y_pred = torch.from_numpy(Y_pred).float()
                metrics = {
                    "acc10": TopkAcc(topk=10),
                    "acc10_instance-agg": TopkAcc(topk=10),
                }
                out = TestRetrieval._get_retrieval_metrics(
                    Y_pred,
                    Y_test,
                    words["test"],
                    subjects_pred=[0] * len(Y_pred),
                    metrics=metrics,
                    retrieval_set_size=250,
                )
                acc[t // self.stride] = out["acc10_size=250_macro"]
                agg_acc[t // self.stride] = out["acc10_instance-agg_size=250"]

        else:
            acc = None
            R = torch.zeros(n_dims)
            X = neuro["train"].mean(-1)
            X_test = neuro["test"].mean(-1)
            model.fit(X, Y)
            Y_pred = model.predict(X_test)
            for d in range(Y.shape[1]):
                R[d], _ = pearsonr(Y_test[:, d], Y_pred[:, d])
        #            coefs = torch.zeros(n_times, n_dims, n_channels)
        #           coefs = torch.from_numpy(ridge.coef_)

        return R, acc, agg_acc


class DecodingExperiment(pydantic.BaseModel):
    """ """

    model_config = pydantic.ConfigDict(extra="allow")

    data: Data
    mapper: Mapper = Mapper()
    task: tp.Literal["decoding", "evoked"] = "decoding"

    # Others
    infra: TaskInfra = TaskInfra(version="1")

    def model_post_init(self, __context: tp.Any) -> None:
        self.data.batch_size = 10000000

    def setup_run(self):
        import yaml

        os.makedirs(self.infra.folder, exist_ok=True)
        config_path = Path(self.infra.folder) / "config.yaml"
        if not config_path.exists():
            with open(config_path, "w") as outfile:
                yaml.dump(self.dict(), outfile, indent=4, default_flow_style=False)

        if self.infra.cluster:
            stdout_path = self.infra.job().paths.stdout
            stderr_path = self.infra.job().paths.stderr
            if os.path.exists(Path(self.infra.folder) / "log.out"):
                os.remove(Path(self.infra.folder) / "log.out")
            if os.path.exists(Path(self.infra.folder) / "log.err"):
                os.remove(Path(self.infra.folder) / "log.err")
            os.symlink(stdout_path, Path(self.infra.folder) / "log.out")
            os.symlink(stderr_path, Path(self.infra.folder) / "log.err")

    @infra.apply
    def run(self):
        self.setup_run()
        if self.task == "decoding":
            return self.decoding()
        elif self.task == "evoked":
            return self.plot_evoked()

    def decoding(self):
        futures = {}
        out = {}

        events = self.data.get_events()
        subjects = events.subject.unique()

        def run_subject(subject):
            print(f"Loading data for subject {subject}")
            subject_events = events.loc[events.subject == subject]
            loaders = self.data.get_loaders(subject_events)
            neuro, feature, words = {}, {}, {}
            for split in ["train", "test", "val"]:
                loader = loaders[split]
                if isinstance(loader, list):
                    loader = loader[0]
                batch = next(iter(loader))
                neuro[split] = batch.data["neuro"]
                feature[split] = batch.data["feature"]
                words[split] = [segment._trigger["text"] for segment in batch.segments]

            # concat val and test
            neuro["test"] = torch.cat([neuro["test"], neuro["val"]], dim=0)
            feature["test"] = torch.cat([feature["test"], feature["val"]], dim=0)
            words["test"] = words["test"] + words["val"]

            print(f"Mapping for subject {subject}")
            R, accs, agg_accs = self.mapper.apply(neuro, feature, words)
            return subject, R, accs, agg_accs

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=1  # self.infra.cpus_per_task
        ) as ex:
            futures = [ex.submit(run_subject, subject) for subject in subjects]
            for future in concurrent.futures.as_completed(futures):
                subject, R, accs, agg_accs = future.result()
                import pickle

                with open(Path(self.infra.folder) / "out.pkl", "ab") as f:
                    pickle.dump({subject: (R, accs, agg_accs)}, f)
