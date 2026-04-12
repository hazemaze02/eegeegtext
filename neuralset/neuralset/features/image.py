# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp

import numpy as np
import pandas as pd
import pydantic
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader, Dataset

import neuralset as ns
from neuralset.infra import MapInfra

from .base import BaseStatic

logger = logging.getLogger(__name__)
CLUSTER_DEFAULTS: tp.Dict[str, tp.Any] = dict(
    timeout_min=25,
    gpus_per_node=1,
    cpus_per_task=8,
    min_samples_per_job=4096,
)


class _DummyModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, ::10, ::10]


class _Dummy(nn.Module):
    def __init__(self, repo: str, model_name: str) -> None:
        # pylint: disable=unused-argument
        super().__init__()
        self.model = _DummyModel()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)


class _TorchHub(nn.Module):  # FIXME should be merge with base.Model
    """wrapper to get class and patch tokens with the same api as clip"""

    def __init__(self, repo, model_name) -> None:
        super().__init__()
        self.model = torch.hub.load(repo, model_name, pretrained=True)

    def forward(self, images) -> torch.Tensor:
        out = self.model.forward(images)
        return out[:, None]  # n_image, n_token, n_dims


class _HuggingFace(nn.Module):
    """wrapper to get hidden state from vit"""

    def __init__(
        self,
        repo: str,
        model_name: str,
        output_hidden_states: bool = False,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        assert repo == "hf"
        from transformers import AutoModel as Model
        from transformers import AutoProcessor as Processor

        if model_name == "facebook/dpt-dinov2-base-kitti":
            from transformers import DPTForDepthEstimation as Model
        try:
            self.model = Model.from_pretrained(
                model_name, output_hidden_states=output_hidden_states
            )
        except ValueError as e:
            # handle specific cases
            if "VisionEncoderDecoderConfig" in str(e):
                from transformers import VisionEncoderDecoderModel as Model
                from transformers import ViTImageProcessor as Processor
            elif "vit-hybrid" in str(e):
                from transformers import ViTHybridForImageClassification as Model
                from transformers import ViTHybridImageProcessor as Processor
            elif "UperNetConfig" in str(e):
                from transformers import UperNetForSemanticSegmentation as Model
            self.model = Model.from_pretrained(
                model_name, output_hidden_states=output_hidden_states
            )
        if not pretrained:
            self.model = Model.from_config(self.model.config)
        self.model.eval()
        # do_rescale=False because ToTensor does the rescaling
        self.processor = Processor.from_pretrained(model_name, do_rescale=False)

    def _full_predict(  # return the raw output, used in tests
        self, images: torch.Tensor, text: str | tp.List[str] = ""
    ) -> tp.Any:
        is_video = False
        if images[0].ndim == 4:  # video: B, T, C, H, W
            # videos produce only one batch at a time, so the batch dim must be removed
            assert images.shape[0] == 1, "Only batch=1 allowed for videos"
            images = images[0]
            is_video = True
        inputs = self.processor(
            images=[i.float() for i in images], text=text, return_tensors="pt"
        )
        if is_video and inputs["pixel_values"].shape[0] != 1:
            # confirm that the video model processed the images as a single batch
            raise RuntimeError("Failed to preprocess video batch")
        # prevent nans (happening for uniform images)
        if "pixel_values" in inputs:
            nans = inputs["pixel_values"].isnan()
            if nans.any():
                inputs["pixel_values"][nans] = 0
                inputs["pixel_values"] = inputs["pixel_values"].float()
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            pred = self.model(**inputs)
        return pred

    def forward(self, images) -> torch.Tensor:
        pred = self._full_predict(images)
        pred = getattr(pred, "vision_model_output", pred)  # for clip
        outputs = pred.last_hidden_state
        return outputs


def _get_image_model(repo, model_name, device):
    if repo == "dummy":
        cls = _Dummy
    elif repo == "hf":
        cls = _HuggingFace
    elif repo == "pytorch/vision":
        cls = _TorchHub
    else:
        raise ValueError(f"Not-implemented repo {repo}")
    model = cls(repo, model_name).to(device)
    return model.eval()


class _ImageDataset(Dataset):
    """Used For batch preprocessing"""

    def __init__(self, events: tp.Sequence[ns.events.Image], transform=None):
        self.events = events
        self.transform = transform

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, idx: int):
        # FIXME: simplify api with pd.DataFrame accessor at single
        try:
            image = self.events[idx].read()
            if self.transform:
                image = self.transform(image)
        except:
            logger.warning("Failed to process image event %s", self.events[idx])
            raise
        return image

    @staticmethod
    def collate_fn(images: tp.List[torch.Tensor]) -> tp.Any:
        # we can't concatenate if the outputs have different sizes
        # for huggingface -> transform is applied later
        if all(i.shape == images[0].shape for i in images):
            return torch.stack(images)
        return images


class BaseImage(BaseStatic):
    # class attributes
    event_type: tp.ClassVar[tp.Type[ns.events.Event]] = ns.events.Image
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = (
        "torchvision>=0.15.2",
        "transformers>=4.29.2",
        "pillow>=9.2.0",
    )

    # feature attributes
    model_name: str
    device: str = "cuda"
    batch_size: int = 32
    imsize: int | None = None
    _model: nn.Module = pydantic.PrivateAttr()  # initialized later
    # for precomputing/caching
    infra: MapInfra = MapInfra(version="1", **CLUSTER_DEFAULTS)

    @classmethod
    def _exclude_from_cls_uid(cls) -> tp.List[str]:
        return super()._exclude_from_cls_uid() + ["device", "batch_size"]

    def _make_transform(self) -> tp.Any:
        from torchvision import transforms

        transfs = [transforms.ToTensor()]
        if self.imsize is not None:
            transfs = [transforms.Resize(self.imsize)] + transfs
        return transforms.Compose(transfs)

    def prepare(self, events: pd.DataFrame) -> None:
        events_ = self._events_from_dataframe(events)
        self._get_latents(events_)

    @infra.apply(
        item_uid=lambda e: str(e.filepath),
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
    )
    def _get_latents(
        self, events: tp.Sequence[ns.events.Image]
    ) -> tp.Iterator[torch.Tensor]:
        logger.info(f"Computing {len(events)} image latents")
        dset = _ImageDataset(events, transform=self._make_transform())
        dloader = DataLoader(
            dset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=_ImageDataset.collate_fn,
        )
        if len(events) > 1:
            dloader = tqdm.tqdm(dloader, desc="Computing image embeddings")  # type: ignore
        # Embed the images in batches
        with torch.no_grad():
            for batch_images in dloader:
                if isinstance(batch_images, torch.Tensor):
                    batch_images = batch_images.to(self.device)
                else:  # should be list of different sizes
                    batch_images = [i.to(self.device) for i in batch_images]
                with torch.no_grad():
                    latents = self._extract_batched_latents(batch_images)
                for latent in latents:
                    yield latent.cpu()

    def get_static(self, event: ns.events.Image) -> torch.Tensor:
        raise NotImplementedError

    def _extract_batched_latents(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Image(BaseImage):
    # class attributes
    # feature attributes
    name: tp.Literal["Image"] = "Image"
    model_name: str = "facebook/dinov2-base"
    repo: str = "hf"
    token: tp.Literal["cls", "avg", "all"] = "cls"

    def _exclude_from_cache_uid(self) -> tp.List[str]:
        prev = super()._exclude_from_cache_uid()
        return prev + ["duration", "frequency", "token"]

    def _make_transform(self) -> tp.Any:
        if self.repo == "hf":
            return super()._make_transform()

        if self.repo not in ("dummy", "pytorch/vision"):
            raise NotImplementedError(
                f"Transforms need to be implemented for repo {self.repo}"
            )

        import torchvision.transforms as transforms  # noqa

        # this is what is used in (most?) pytorch/vision models, but may not to be cross-checked
        transfs = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        if self.imsize is not None:
            # previously: transforms.Resize(256), transforms.CenterCrop(224),
            transfs = [transforms.Resize(self.imsize)] + transfs
        return transforms.Compose(transfs)

    @property
    def model(self) -> nn.Module:
        if not hasattr(self, "_model"):
            self._model = _get_image_model(self.repo, self.model_name, self.device)
        return self._model

    def _extract_batched_latents(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def get_static(self, event: ns.events.Image) -> torch.Tensor:
        latent = torch.Tensor(next(self._get_latents([event])))
        return self._select_token(latent)

    def _select_token(self, latent: torch.Tensor) -> torch.Tensor:
        # this method is extracted for use in Video
        if self.token == "cls":
            # n_tokens, n_dims = latent.shape
            latent = latent[0, :]
        elif self.token == "avg":
            latent = latent.mean(0, keepdim=False)
        elif self.token != "all":
            raise ValueError(f"Unknown token choice {self.token}")
        return latent


class ImageTransformer(BaseImage):
    """Image transformers embeddings, through huggingface API.

    All layers are dumped at once so they only need computing once. Average token ("avg") and class
    token ("cls") are cached together to limit the dimensionality of what needs to be loaded.

    Additional parameters
    ---------------------
    model_name:
        name of the model to use
    token: str
        token to be extracted
    layer: int
        layer to be used for the embedding
    rel_layer: int
        layer to be used for the embedding (in relative position)

    Note
    ----
    either layer or rel_layer can be used (default to last layer if both are None)
    """

    # class attributes
    name: tp.Literal["ImageTransformer"] = "ImageTransformer"
    # feature attributes
    token: tp.Literal["avg", "cls"] = "avg"
    model_name: str = "facebook/dinov2-base"
    layer: int | None = None
    rel_layer: float | None = None
    pretrained: bool = True
    # for precomputing/caching
    infra: MapInfra = MapInfra(version="2", **CLUSTER_DEFAULTS)

    @infra.apply(
        item_uid=lambda e: str(e.filepath),
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
    )
    def _get_latents(
        self, events: tp.Sequence[ns.events.Image]
    ) -> tp.Iterator[torch.Tensor]:
        # override to have version 2 (version=1 in BaseImage)
        return super()._get_latents(events)

    def model_post_init(self, log__: tp.Any) -> None:
        if not any(v is None for v in [self.layer, self.rel_layer]):
            raise ValueError('Either "layer" or "rel_layer" must be specified, not both')
        if self.rel_layer is not None:
            if not 0 <= self.rel_layer <= 1.0:
                raise ValueError(
                    f'"rel_layer" must be a float between 0 and 1 (got {self.rel_layer})'
                )
        if self.imsize is not None:
            logger.warning(
                'The effect of "imsize"=%s might be cancelled by the '
                "HuggingFace processor.",
                self.imsize,
            )
        super().model_post_init(log__)

    @property
    def model(self) -> nn.Module:
        if not hasattr(self, "_model") or self._model is None:
            self._model = _HuggingFace(
                "hf",
                model_name=self.model_name,
                output_hidden_states=True,
                pretrained=self.pretrained,
            )
            self._model.to(self.device)
        return self._model

    def _get_hidden_states(self, images: torch.Tensor) -> tp.List[torch.Tensor]:
        """Extract hidden_states as n_layers n_layers x (batch, tokens,  features)"""
        # this method is overriden in experimental features for more hugging face models
        out = self.model._full_predict(images)
        out = getattr(out, "vision_model_output", out)  # for clip
        return out.hidden_states  # type: ignore

    def _extract_batched_latents(self, images: torch.Tensor) -> torch.Tensor:
        states = self._get_hidden_states(images)
        out = torch.cat([x.unsqueeze(1) for x in states], axis=1)  # type: ignore
        # (batch, n_layers, tokens, n_features)
        if out.shape[2] > 1:  # if we have several tokens, extract average and first/cls
            avg_token = out.mean(dim=2, keepdim=True)
            out = torch.cat([avg_token, out[:, :, [0]]], axis=2)  # type: ignore
        return out  # type: ignore

    def get_static(self, event: ns.events.Image) -> torch.Tensor:
        # layer * patches * size
        latent = torch.Tensor(next(self._get_latents([event])))
        return self._select_token(latent)

    def _select_token(self, latent: torch.Tensor) -> torch.Tensor:
        layer_ind = self.layer
        if layer_ind is None:
            layer_ind = latent.shape[0] - 1  # default to last
            if self.rel_layer is not None:
                layer_ind = int(round(self.rel_layer * layer_ind))
        latent = latent[layer_ind]
        if self.token == "cls" and latent.shape[0] == 1:
            msg = f"No class token for model with unaligned layers: {self.model_name}"
            raise ValueError(msg)
        token_ind = 0 if self.token == "avg" else 1
        return latent[token_ind]  # type: ignore


class BaseClassicImageFeature(BaseStatic):
    """Base class for classic image features, e.g. based on numpy, skimage, OpenCV, etc."""

    name: str
    # class attributes
    event_type: tp.ClassVar[tp.Type[ns.events.Event]] = ns.events.Image
    imsize: int | None = None
    infra: MapInfra = MapInfra(version="1", **CLUSTER_DEFAULTS)

    def prepare(self, events: pd.DataFrame) -> None:
        events_ = self._events_from_dataframe(events)
        self._get_features(events_)

    @infra.apply(item_uid=lambda event: str(event.filepath))
    def _get_features(
        self, events: tp.List[ns.events.Image]
    ) -> tp.Generator[np.ndarray, None, None]:
        logger.info("Computing %s for %s images.", self.name, len(events))

        for event in events:
            image = event.read()
            if self.imsize is not None:
                image = image.resize((self.imsize, self.imsize))

            yield self._get_image_features(np.array(image))

    def _get_image_features(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_static(self, event: ns.events.Image) -> torch.Tensor:
        return torch.Tensor(next(self._get_features([event])))


class RFFT2D(BaseClassicImageFeature):
    """(Cropped) 2D Fourier spectrum of an image of real values.

    Parameters
    ----------
    n_components_to_keep :
        Number of components of the FFT to keep, starting from low frequencies and
        moving towards higher frequencies. If None, use all components.
    average_channels :
        If True, average RGB channels before taking the FFT (to reduce dimensionality).
    return_log_psd :
        If True, return the flattened log PSD instead of the "viewed-as-real" complex FFT.
    return_angle :
        If True, return the flattened angle. Can be combined with the log PSD.
    """

    name: tp.Literal["RFFT2D"] = "RFFT2D"
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("torchvision>=0.15.2",)

    n_components_to_keep: int | None = None
    average_channels: bool = True
    return_log_psd: bool = False
    return_angle: bool = False

    _eps: tp.ClassVar[float] = 1e-12

    def _fft(self, image: torch.Tensor) -> torch.Tensor:
        fft = torch.fft.rfft2(image)
        if self.average_channels:
            fft = fft.mean(axis=0, keepdims=True)
        fft = torch.fft.fftshift(fft, dim=1)

        if self.n_components_to_keep is not None:  # Crop FFT by keeping lower frequencies
            mid_point_x = fft.shape[1] // 2
            fft = fft[
                :,
                mid_point_x
                - self.n_components_to_keep : mid_point_x
                + self.n_components_to_keep,
                : self.n_components_to_keep + 1,
            ]

        return fft

    @staticmethod
    def _ifft(
        fft: torch.Tensor, average_channels: bool, width: int, height: int
    ) -> torch.Tensor:
        """Convenience function to return in image-space after an FFT.

        Only supports "viewed as real" FFT.
        """
        if fft.ndim == 1:
            fft = fft.reshape(  # Unflatten and convert back to complex
                1 if average_channels else 3,
                width,
                height // 2 + 1,
                2,
            )
            fft = torch.view_as_complex(fft)

        fft = torch.fft.ifftshift(fft, dim=1)
        inv_fft = torch.fft.irfft2(fft).real
        inv_fft = inv_fft / inv_fft.max()
        return inv_fft

    def _get_image_features(self, image: np.ndarray) -> torch.Tensor:
        import torchvision.transforms.functional as TF  # noqa

        fft = self._fft(TF.to_tensor(image))

        out = []
        if self.return_log_psd:
            out.append((fft.abs() ** 2 + self._eps).log())
        if self.return_angle:
            out.append(fft.angle())
        if not (self.return_log_psd or self.return_angle):
            out.append(torch.view_as_real(fft))  # Complex tensor -> Real vector
        features = torch.cat(out, dim=-1).flatten()

        return features


class HOG(BaseClassicImageFeature):
    """Histogram of oriented gradients.

    See https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_hog.html
    """

    name: tp.Literal["HOG"] = "HOG"
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("scikit-image>=0.22.0",)

    _orientations: tp.ClassVar[int] = 8
    _pixels_per_cell: tp.ClassVar[tuple[int, int]] = (8, 8)
    _cells_per_block: tp.ClassVar[tuple[int, int]] = (2, 2)
    _channel_axis: tp.ClassVar[int] = -1

    def _get_image_features(self, image: np.ndarray) -> np.ndarray:
        from skimage.feature import hog  # noqa

        features = hog(
            image,
            orientations=self._orientations,
            pixels_per_cell=self._pixels_per_cell,
            cells_per_block=self._cells_per_block,
            channel_axis=self._channel_axis,
            visualize=False,
        )
        return features


class LBP(BaseClassicImageFeature):
    """Local Binary Pattern (LBP).

    See https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_local_binary_pattern.html
    """

    name: tp.Literal["LBP"] = "LBP"
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = (
        "opencv-python>=4.8.1",
        "scikit-image>=0.22.0",
    )
    _P: tp.ClassVar[int] = 8
    _R: tp.ClassVar[int] = 1
    _method: tp.ClassVar[str] = "uniform"
    _n_bins: tp.ClassVar[int] = 10
    _bin_range: tp.ClassVar[tuple[int, int]] = (0, 10)

    def _get_image_features(self, image: np.ndarray) -> np.ndarray:
        import cv2  # noqa
        from skimage.feature import local_binary_pattern  # noqa

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # requires grayscale
        lbp = local_binary_pattern(gray, P=self._P, R=self._R, method=self._method)
        hist, _ = np.histogram(lbp.ravel(), bins=self._n_bins, range=self._bin_range)
        hist = hist.astype("float")
        hist /= hist.sum() + 1e-7

        return hist


class ColorHistogram(BaseClassicImageFeature):
    """Color histogram."""

    name: tp.Literal["ColorHistogram"] = "ColorHistogram"
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("opencv-python>=4.8.1",)

    _channels: tp.ClassVar[tuple[int, ...]] = (0, 1, 2)
    _hist_size: tp.ClassVar[tuple[int, ...]] = (8, 8, 8)
    _ranges: tp.ClassVar[tuple[int, ...]] = (0, 256, 0, 256, 0, 256)

    def _get_image_features(self, image: np.ndarray) -> np.ndarray:
        import cv2  # noqa

        hist = cv2.calcHist([image], self._channels, None, self._hist_size, self._ranges)
        hist = cv2.normalize(hist, hist).flatten()

        return hist
