# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from abc import abstractmethod, abstractproperty
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pydantic
import torch
from tqdm import tqdm, trange

from neuralset.infra import MapInfra

from .. import events
from .base import BaseDynamic
from .image import Image, ImageTransformer

logger = logging.getLogger(__name__)
# activate with:
# logging.getLogger("neuralset").setLevel(logging.DEBUG)


class _VideoImage(events.Image):
    """Image event based on a video
    This is used to process a video as a list of images
    """

    start: float = 0.0
    timeline: str = "fake"
    duration: float = 1.0
    video: tp.Any
    time: float = 0.0
    filepath: str = ""

    def model_post_init(self, log__: tp.Any) -> None:
        if self.filepath:
            raise ValueError("Filepath is automatically filled")
        # create a custom filepath for caching
        self.filepath = f"{self.video.filename}:{self.time:.3f}"
        super().model_post_init(log__)

    def _read(self) -> tp.Any:
        import PIL  # noqa

        # may require: pip install moviepy==2.0.0.dev2
        img = self.video.get_frame(self.time)
        return PIL.Image.fromarray(img.astype("uint8"))


class BaseVideo(BaseDynamic):
    event_type: tp.ClassVar[tp.Type[events.Event]] = events.Video
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = (
        "julius>=0.2.7",
        "pillow>=9.2.0",
    )
    # feature attributes
    device: str = "cuda"
    frequency: tp.Literal["native"] | float = "native"  # will be video freq by default
    infra: MapInfra = MapInfra(
        timeout_min=25,
        gpus_per_node=1,
        cpus_per_task=8,
        min_samples_per_job=4096,
        version="1",
    )

    @classmethod
    def _exclude_from_cls_uid(cls) -> tp.List[str]:
        return super()._exclude_from_cls_uid() + ["device"]

    @abstractproperty
    def model(self) -> torch.nn.Module:
        raise NotImplementedError()

    def _init_preprocess(self) -> None:
        """Helper Function that can be used to init class attributes"""

    @abstractmethod
    def _preprocess_frame(self, img: np.ndarray) -> torch.Tensor:
        raise NotImplementedError()

    def _resample_frames(
        self, frames: torch.Tensor, n_frames: int, event: tp.Any
    ) -> torch.Tensor:
        # resample if frequency is provided and different from event frequency
        if self.frequency and self.frequency != event.frequency:
            logger.debug(
                "Resampling video embedding from %s to %s",
                event.frequency,
                self.frequency,
            )
            import julius  # noqa

            # config resampling
            resample = julius.resample.ResampleFrac(
                old_sr=int(event.frequency / self.frequency),
                new_sr=1,
            ).to(self.device)
            # proceede by latent dimension
            dims = []
            for dim in tqdm(frames.reshape(n_frames, -1).T):
                dim_ = dim.float().to(
                    self.device
                )  # sub-optim since back to cpus afterwards?
                dims.append(resample(dim_).t().cpu())
            # TODO: stack an extra frame here?
            frames = torch.stack(dims).T.reshape(-1, *frames.shape[1:])

        # time must be last dimension (not first as here)
        frames = frames.permute((1, 2, 3, 0))
        return frames

    def _postprocess_frames(self, frames: torch.Tensor) -> torch.Tensor:
        return frames

    def prepare(self, events: pd.DataFrame) -> None:
        events_ = self._events_from_dataframe(events)
        self._get_latents(events_)

    @infra.apply(item_uid=lambda event: str(event.filepath))
    def _get_latents(self, events: tp.List[events.Video]) -> tp.Iterator[torch.Tensor]:
        # read all videos of the events
        for event in events:
            # FIXME simplify acccessor api to read directly from each row
            video = event.read()
            n_frames = int(video.duration * video.fps)

            logger.debug(
                "Load Video (duration %ss at %sfps, %s frames of shape %s)",
                video.duration,
                video.fps,
                n_frames,
                tuple(video.size),
            )
            self._init_preprocess()

            for frame in trange(n_frames):
                # read video & preproc
                img = video.get_frame(frame / video.fps)
                img = self._preprocess_frame(img)

                # initialize on first call to get dimensionality
                if frame == 0:
                    # note: this can be huge for high freqs and may not fit in GPUs
                    frames = torch.zeros(n_frames, *img.shape, dtype=torch.float32)
                    logger.debug("Created Tensor with size %s", frames.shape)
                frames[frame] = img

            # execute resampling
            frames = self._resample_frames(frames, n_frames, event)

            # execute postprocess
            frames = self._postprocess_frames(frames)

            yield frames

    def _get(self, event: events.Video, start: float, duration: float) -> torch.Tensor:
        latents = next(self._get_latents([event]))
        return self._fill_slice(latents, event, start, duration)


def resamp_first_dim(data: torch.Tensor, new_first_dim: int) -> torch.Tensor:
    if data.shape[0] == new_first_dim:
        return data
    import julius

    logger.debug(
        "Resampling video embedding from %s samples to %s", data.shape[0], new_first_dim
    )
    resample = julius.resample.ResampleFrac(
        old_sr=data.shape[0],
        new_sr=new_first_dim,
    ).to(data.device)
    dims = []
    for dim in tqdm(data.reshape(data.shape[0], -1).T):
        dims.append(resample(dim.float()))
    # TODO: stack an extra frame here?
    output = torch.stack(dims).reshape(-1, *data.shape[1:])
    return output


class Video(BaseDynamic):
    event_type: tp.ClassVar[tp.Type[events.Event]] = events.Video
    name: tp.Literal["Video"] = "Video"
    # class attributes
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = (
        "torchvision>=0.15.2",
        "julius>=0.2.7",
    )
    image: Image | ImageTransformer = pydantic.Field(
        ImageTransformer(infra={"keep_in_ram": False}, imsize=None), discriminator="name"  # type: ignore
    )
    decimated: bool = False  # decimates video to the embdedding frequency
    infra: MapInfra = MapInfra(
        timeout_min=120,
        gpus_per_node=1,
        cpus_per_task=8,
        min_samples_per_job=128,
        version="3",
    )

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if self.image.infra.keep_in_ram:
            msg = "video.image.infra.keep_in_ram must be False to avoid overload"
            raise ValueError(msg)
        for name in ["folder", "cluster"]:
            val = getattr(self.image.infra, name)
            if val is not None:
                raise ValueError(f"image.infra.{name} must be None, (got {val!r})")

    def _exclude_from_cache_uid(self) -> tp.List[str]:
        im_ex = self.image._exclude_from_cache_uid()
        return [f"image.{n}" for n in im_ex]

    @infra.apply(
        item_uid=lambda event: str(event.filepath),
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
    )
    def _get_latents(self, events: tp.List[events.Video]) -> tp.Iterator[torch.Tensor]:
        # read all videos of the events
        logging.getLogger("neuralset").setLevel(logging.DEBUG)
        for event in events:
            video = event.read()
            n_frames = int(video.duration * video.fps)
            freq = self._output_frequency(event)  # deals with native/specified frequency
            # [0, 2] at freq = 1 -> 3 samples: (0, 1, 2)
            expect_frames = int(round(event.duration * freq + 1))
            logger.debug(
                "Loaded Video (duration %ss at %sfps, %s frames of shape %s):\n %s",
                video.duration,
                video.fps,
                n_frames,
                tuple(video.size),
                event.filepath,
            )
            times = np.linspace(
                0, video.duration, expect_frames if self.decimated else n_frames
            )
            # TODO warn about aspect ratio? resize leads to aspect ratio 1:1
            ims = [_VideoImage(video=video, time=t) for t in times]
            output = torch.Tensor([])
            # pylint: disable=protected-access
            k = -1
            for k, embd in enumerate(
                tqdm(self.image._get_latents(ims), total=len(times))
            ):
                if not k:
                    output = torch.zeros(len(times), *embd.shape)
                    logger.debug("Created Tensor with size %s", output.shape)
                output[k] = embd
            logger.debug("Finished encoding video at video frame rate")
            assert k == len(times) - 1  # security
            # resample full output
            if abs(output.shape[0] - expect_frames) > 1:  # some flexibility allowed
                output = output.to(self.image.device)
                output = resamp_first_dim(output, expect_frames).cpu()
                logger.debug("Resampled video embeddings at frequency %s", self.frequency)
            # set first (time) dim to last
            output = output.permute(list(range(1, output.dim())) + [0])
            yield output

    def prepare(self, events: pd.DataFrame) -> None:
        events_ = self._events_from_dataframe(events)
        self._get_latents(events_)

    def _get(self, event: events.Video, start: float, duration: float) -> torch.Tensor:
        latents = next(self._get_latents([event]))
        latents = self.image._select_token(latents)
        return self._fill_slice(latents, event, start, duration)


def _get_raft_model(base_path: Path, name: str, device: str) -> torch.nn.Module:
    # validate path
    model_path = base_path / f"{name}.pth"
    assert model_path.exists(), f"RAFT model {name} does not exist"

    # define args
    args = Namespace(
        model=str(model_path),
        small=False,
        mixed_precision=False,
        alternate_corr=False,
    )

    from raft.raft import RAFT  # noqa

    # load model
    raft_model = torch.nn.DataParallel(RAFT(args))
    raft_model.load_state_dict(torch.load(model_path, map_location=device))
    model = raft_model.module
    model.to(device)
    model.eval()

    return model


class OpticalFlow(BaseVideo):
    name: tp.Literal["OpticalFlow"] = "OpticalFlow"
    # update this
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("raft",)

    # feature attributes
    model_name: str = "raft/raft-kitti"
    model_path: Path = Path("/large_experiments/brainai/pretrained")
    device: str = "cuda"  # does not fit on either CPU or GPU currently
    iters: int = 10  # iterations for the flow refinement
    _model: torch.nn.Module
    _padder: tp.Any = None

    @classmethod
    def _exclude_from_cls_uid(cls) -> tp.List[str]:
        return super()._exclude_from_cls_uid() + ["model_path"]

    @property
    def model(self) -> torch.nn.Module:
        if not hasattr(self, "_model"):
            if self.model_name.startswith("raft"):
                model = _get_raft_model(self.model_path, self.model_name, self.device)
            else:
                raise ValueError(f"Unkown model type: {self.model_name}")
            self._model = model
            self._model.to(self.device)
        return self._model

    def _init_preprocess(self):
        self._padder = None

    def _preprocess_frame(self, img: np.ndarray) -> torch.Tensor:
        from raft.utils.utils import InputPadder  # noqa

        # execute raft preproc
        timg = (
            torch.from_numpy(img.astype("uint8")).permute(2, 0, 1).float().to(self.device)
        )
        if self._padder is None:
            # TODO: might provide fixed size
            self._padder = InputPadder(timg.shape)
        # TODO: check if that can be added to batch
        timg = self._padder.pad(timg)[0]

        return timg

    def _postprocess_frames(self, frames: torch.Tensor) -> torch.Tensor:
        # compute optical flow as batch
        print("Flow embedding...")
        frames = frames.to(self.device)
        assert len(frames) >= 2, "at least 2 frames required for flow compute"
        with torch.no_grad():
            latents, _ = self.model(
                frames[:-1, ...], frames[1:, ...], iters=self.iters, test_mode=True
            )

        return latents.cpu()
