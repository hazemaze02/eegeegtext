# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp

import pydantic
import torch
import torch.nn.functional as F

from neuralset.features import image as _im

from . import torchhub

logger = logging.getLogger(__name__)


class HuggingFaceImage(_im.ImageTransformer):
    """Extension of ImageTransformer for more models"""

    name: tp.Literal["HuggingFaceImage"] = "HuggingFaceImage"  # type: ignore
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("timm>=0.9.16",)
    _transforms: tp.Any = pydantic.PrivateAttr()  # initialized later

    @property
    def model(self) -> torch.nn.Module:
        # use cached model if already loaded
        if hasattr(self, "_model"):
            return self._model
        # load from torchhub
        if self.model_name.startswith("torchhub/"):
            model = torchhub.TorchHubModel(
                self.model_name.split("/")[1], pretrained=self.pretrained
            )
            model.to(self.device)
            model.eval()
            self._transforms = model.transforms
            self._model = model
            return self._model
        # standard hugging face loading
        if not self.model_name.startswith("timm/"):
            self._transforms = super()._make_transform()
            return super().model
        # load from timm interface
        if not hasattr(self, "_model"):
            import timm  # pylint: disable=import-outside-toplevel

            self._model = timm.create_model(
                "hf-hub:" + self.model_name,
                pretrained=self.pretrained,
                features_only=True,
            )
            self._model.eval()
            self._model.to(self.device)
            data_config = timm.data.resolve_model_data_config(self._model)
            self._transforms = timm.data.create_transform(
                **data_config, is_training=False
            )
        return self._model

    def _make_transform(self) -> tp.Any:
        _ = self.model
        return self._transforms

    def _get_hidden_states(self, images: torch.Tensor) -> tp.List[torch.Tensor]:
        """Extract hidden_states as n_layers x (batch, tokens, features)"""
        # timm case (exports around 5 layers)
        if self.model_name.startswith("timm/"):
            with torch.inference_mode():
                out = self.model(images)
            return _normalize_states(out, name=self.model_name)
        # torchhub case (manually defined layers to extract, see torchhub.py)
        elif isinstance(self.model, torchhub.TorchHubModel):
            with torch.inference_mode():
                out = self.model.hidden_states(images)
            out = [x.unsqueeze(1) if x.ndim == 2 else x for x in out]
            return _normalize_states(out, name=self.model_name)
        # general hugging face case
        # extract states for each layer
        # n_layers x (batch, tokens, features) or (batch, features, tokens_x, tokens_y)
        out = self.model._full_predict(images)
        out = getattr(out, "vision_model_output", out)  # for clip
        if hasattr(out, "hidden_states"):
            states = out.hidden_states
        elif "sam-vit" in self.model_name:  # (eg: facebook/sam-vit-base)
            # looks like the features were non-conventionally placed (1, 64, 64, 768)
            states = [x.transpose(1, 3) for x in out.vision_hidden_states]
        elif hasattr(out, "encoder_hidden_states"):
            # (eg: facebook/detr-resnet-50, mask2former)
            states = out.encoder_hidden_states
            if hasattr(out, "decoder_hidden_states"):  # deter
                states += out.decoder_hidden_states
            if hasattr(out, "pixel_decoder_hidden_states"):  # mask2former
                states += out.pixel_decoder_hidden_states
            # if hasattr(out, "transformer_decoder_hidden_states"):  # mask2former -> wrong batch size?
            #     states += out.transformer_decoder_hidden_states
        else:
            avail = [x for x in dir(out) if not x.startswith("_")]
            msg = f"Unsupported model {self.model_name} output with fields {avail})"
            raise NotImplementedError(msg)
        if not all(len(images) == x.shape[0] for x in states):
            shs = [x.shape for x in states]
            msg = f"Failed in keeping batch dimension: {shs} Vs {images.shape}"
            raise RuntimeError(msg)
        # renormalize to n_layers x (batch, tokens, features)
        return _normalize_states(states, name=self.model_name)


def _normalize_states(states: tp.List[torch.Tensor], name: str) -> tp.List[torch.Tensor]:
    """Reformat sequence of states to n_layers x (batch, tokens, features)
    with tokens and features constant

    Note: if number of tokens is not constant, they will be averaged to a unique token
    ImageTransformer.get_static will raise if we try to access non-avg (class) token
    """
    in_shapes = [x.shape for x in states]
    if any(s.ndim == 4 for s in states):
        # assume varying number of features and x/y dims (batch, features, tokens_x, tokens_y)
        # Eg for nvidia/mit-b0:
        # [(B, 32, 128, 128), (B, 64, 64, 64), (B, 160, 32, 32), (B, 256, 16, 16)]
        states = [x.mean(dim=(-2, -1)).unsqueeze(1) if x.ndim == 4 else x for x in states]
        # transformed to (B, 1, n_features) (with variable n_fearures)
    assert all(s.ndim == 3 for s in states), [s.shape for s in states]
    if not all(x.shape[2] == states[0].shape[2] for x in states):
        # n_features is not constant -> pad to larger one (last)
        x1_x2 = zip(states[:-1], states[1:])
        decreasing = (
            "efficientnet",
            "kakaobrain/align-",
            "mask2former",
            "torchhub/alexnet",
            "torchhub/mobilenet",
        )
        if not any(x in name for x in decreasing):  # decreasing in the beggining
            if not all(x2.shape[2] >= x1.shape[2] for x1, x2 in x1_x2):
                raise ValueError(
                    f"Not increasing {[x.shape for x in states]}\n"
                    f"(input shapes were {in_shapes})"
                )
        states = [x.mean(1, keepdim=True) for x in states]  # average on tokens
        max_feats = max(x.shape[-1] for x in states)
        states = [F.pad(x, (0, max_feats - x.shape[-1])) for x in states]
    if not all(x.shape[1] == states[0].shape[1] for x in states):
        states = [x.mean(1, keepdim=True) for x in states]  # non matching token sizes
    return states
