# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp

import numpy as np
import torch
from torchvision import transforms

import neuralset as ns
from neuralset.features import video as _vid
from neuralset.infra import MapInfra

from . import image as _im

logger = logging.getLogger(__name__)


class DecimVideo(_vid.Video):
    event_type: tp.ClassVar[tp.Type[ns.events.Event]] = ns.events.Video
    name: tp.Literal["DecimVideo"] = "DecimVideo"  # type: ignore
    # class attributes
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = (
        "torchvision>=0.15.2",
        "julius>=0.2.7",
    )
    decimated: tp.Literal[True] = True
    image: _im.HuggingFaceImage = _im.HuggingFaceImage(
        model_name="MCG-NJU/videomae-base", infra={"keep_in_ram": False}  # type: ignore
    )
    infra: MapInfra = MapInfra(
        timeout_min=120,
        gpus_per_node=1,
        cpus_per_task=8,
        min_samples_per_job=128,
        version="3",
    )

    def model_post_init(self, log__: tp.Any) -> None:
        model = self.image.model_name
        if "video" in model and "videomae" not in model:
            msg = "Currently unclear if this supports any video model but videomae model"
            raise NotImplementedError(msg)
        super().model_post_init(log__)

    @infra.apply(
        item_uid=lambda event: str(event.filepath),
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
    )
    def _get_latents(self, events: tp.List[ns.events.Video]) -> tp.Iterator[torch.Tensor]:
        # read all videos of the events
        logging.getLogger("neuralset").setLevel(logging.DEBUG)
        if "videomae" not in self.image.model_name:
            yield from super()._get_latents(events)
            return
        for event in events:
            video = event.read()
            n_frames = int(video.duration * video.fps)
            # [0, 2] at freq = 1 -> 3 samples: (0, 1, 2)
            freq = self._output_frequency(event)
            expect_frames = int(round(event.duration * freq + 1))
            logger.debug(
                "Loaded Video (duration %ss at %sfps, %s frames of shape %s):\n %s",
                video.duration,
                video.fps,
                n_frames,
                tuple(video.size),
                event.filepath,
            )
            times = np.linspace(0, video.duration, expect_frames)
            num_frames = self.image.model.model.config.num_frames
            T = 1.0 / freq
            # samples the frames in-between the main frequency
            subtimes = list(k / num_frames * T for k in reversed(range(num_frames)))
            transf = transforms.ToTensor()
            output = torch.Tensor([])
            # pylint: disable=protected-access
            for k, t in enumerate(times):
                ims = [
                    _vid._VideoImage(video=video, time=max(0, t - t2)) for t2 in subtimes
                ]
                data = torch.stack([transf(i.read()) for i in ims]).unsqueeze(0)
                embd = self.image._extract_batched_latents(data)[0]
                if not output.numel():
                    output = torch.zeros(len(times), *embd.shape)
                    logger.debug("Created Tensor with size %s", output.shape)
                output[k] = embd
            # set first (time) dim to last
            output = output.permute(list(range(1, output.dim())) + [0])
            yield output
