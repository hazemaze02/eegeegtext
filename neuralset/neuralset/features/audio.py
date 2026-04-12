# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""All the supported audio features."""

import typing as tp
import warnings
from abc import abstractmethod
from typing import List

import numpy as np
import pandas as pd
import pydantic
import torch
from torch import nn
from torch.nn import functional as F

import neuralset as ns
from neuralset.infra import MapInfra

from .base import BaseDynamic

# pylint: disable=import-outside-toplevel


class BaseAudio(BaseDynamic):
    """Audio feature

    Note
    ----
    Default frequency is dervived from event duration and computed latent dimension
    after the first call. Note that this can be slightly off due to sampling, so you
    should provide the frequency yourself if you want consistency.
    """

    event_type: tp.ClassVar[tp.Type[ns.events.Event]] = ns.events.Sound
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = (
        "julius>=0.2.7",
        "pillow>=9.2.0",
    )
    # feature attributes
    device: str = "cuda"
    # frequency derived from sampling rate of the produced feature
    frequency: tp.Literal["native"] | float = "native"
    norm_audio: bool = True

    infra: MapInfra = MapInfra(
        timeout_min=25,
        gpus_per_node=1,
        cpus_per_task=8,
        min_samples_per_job=4096,
        version="2",
    )

    def _exclude_from_cache_uid(self) -> List[str]:
        return super()._exclude_from_cache_uid() + ["device"]

    @abstractmethod
    def _process_wav(self, event: ns.events.Sound) -> torch.Tensor:
        raise NotImplementedError

    def _preprocess_wav(self, wav: torch.Tensor) -> torch.Tensor:
        wav = torch.mean(wav, dim=0)  # stereo to mono
        if self.norm_audio:
            wav = (wav - wav.mean()) / (1e-8 + wav.std())
        return wav

    def _resample_wav(
        self, wav: torch.Tensor, old_frequency: float, new_frequency: float
    ) -> torch.Tensor:
        for freq in (old_frequency, new_frequency):
            if not float(freq).is_integer():
                raise ValueError(f"Frequencies need to be integers, got {freq}")
        old_frequency, new_frequency = int(old_frequency), int(new_frequency)
        import julius  # noqa

        wav = julius.resample.ResampleFrac(
            old_sr=old_frequency, new_sr=new_frequency  # type: ignore
        )(wav)
        return wav

    @infra.apply(
        item_uid=lambda event: f"{event.filepath}_{event.offset:.2f}_{event.duration:.2f}",
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
        cache_type="NumpyMemmapArray",
    )
    def _get_latents(self, events: tp.List[ns.events.Sound]) -> tp.Iterator[np.ndarray]:
        if len(events) > 1:
            from tqdm import tqdm

            events = tqdm(events, desc="Computing audio embeddings")  # type: ignore
        for event in events:
            latents = self._process_wav(event)
            if self.frequency == "native" and self._frequency_override is None:
                self._frequency_override = latents.shape[-1] / event.duration
            feature_samples = self._output_frequency(event).to_ind(event.duration)
            if abs(feature_samples - latents.shape[-1]) > 0:  # allow some variability
                if len(latents.shape) == 2:  # d, t
                    latents = F.interpolate(latents[None], feature_samples)[0]
                else:  # n_layers, d, t
                    latents = F.interpolate(latents, feature_samples)
            yield latents.cpu().numpy()

    @abstractmethod
    def _get(self, event: ns.events.Sound, start: float, duration: float) -> torch.Tensor:
        raise NotImplementedError

    def prepare(self, events: pd.DataFrame) -> None:
        events_ = self._events_from_dataframe(events)
        self._get_latents(events_)


class MelSpectrum(BaseAudio):
    """Outputs the sound waves with the features frequency"""

    name: tp.Literal["MelSpectrum"] = "MelSpectrum"
    event_type: tp.ClassVar[tp.Type[ns.events.Event]] = ns.events.Sound
    n_mels: int = 40
    n_fft: int = 512
    hop_length: int | None = None  # defaults to n_fft // 4
    in_sampling: int = 16_000
    normalized: bool = True
    use_log_scale: bool = True
    log_scale_eps: float = 1e-5
    # internal
    _transform: tp.Any = pydantic.PrivateAttr()
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("torchaudio",)

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        import torchaudio

        hop_length = self.n_fft // 4 if self.hop_length is None else self.hop_length
        self._transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.in_sampling,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=hop_length,
            normalized=self.normalized,
        )

    def _get(self, event: ns.events.Sound, start: float, duration: float) -> torch.Tensor:
        latents = next(self._get_latents([event]))
        if self.frequency == "native" and self._frequency_override is None:
            # get_latent is bypassed by infra, so we need to duplicate logic here
            self._frequency_override = latents.shape[-1] / event.duration
        return self._fill_slice(latents, event, start, duration)

    def _process_wav(self, event: ns.events.Sound) -> torch.Tensor:
        """Returns the wav at the processing frequency (default wav frequency)"""
        wav = event.read()
        wav = self._preprocess_wav(wav)

        if event.frequency != self.in_sampling:
            wav = self._resample_wav(wav, event.frequency, self.in_sampling)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            melspec = self._transform(wav)
        if self.use_log_scale:
            melspec = torch.log10(melspec + self.log_scale_eps)
        return melspec  # type: ignore


class BaseHuggingFaceAudio(BaseAudio):
    """
    Base class for HuggingFace audio models.

    Parameters
    ----------
    model_name : str
        Name of the model to use.
    normalized : bool
        Whether to normalize the input.
    device : str
        Device to use for the model. Can be "cpu", "cuda" or "auto".
        If "auto", will use "cuda" if available, "cpu" otherwise.
    layer_type : str
        Type of layer to use. Can be "transformer" or "convolution".
    layers : list
        List of layers to use.
        It can either be :
        - Each element should be a float between 0 and 1,
          where 0 stands for the first layer and 1 for the last layer.
        - Each element should be an integer between -n and n - 1, with n
          the number of layers. A negative index refers to the layers in
          a backward order (i.e. -1 refers to the last layer).
        The output will be the average of the activations of the selected layers.
        Note: all layers are cached, but only the selected ones are used for the output.
    cache_all_layers : bool
        If True, the output of all layers is cached. If False, only cache the output of the layers
        specified by `layers`. This is useful in case a very specific set of layers is required
        (e.g. [-1, -2, -3, -4]) but we want to avoid reloading all activations of all layers.
    """

    event_type: tp.ClassVar[tp.Type[ns.events.Event]] = ns.events.Sound
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("transformers>=4.29.2",)

    model_name: str = "facebook/wav2vec2-large-xlsr-53"
    normalized: bool = True
    device: tp.Literal["cpu", "cuda", "auto"] = "auto"
    layer_type: tp.Literal["transformer", "convolution"] = "transformer"
    layers: int | float | list[int] | list[float] = 0.5
    cache_all_layers: bool = True

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return super()._exclude_from_cls_uid() + [
            "device",
            "cache_all_layers",
        ]

    def _exclude_from_cache_uid(self) -> list[str]:
        prev = super()._exclude_from_cache_uid()
        if self.cache_all_layers:
            prev += ["layers"]
        # return prev + ["frequency", "duration", "device"]  # XXX Need frequency and duration?
        return prev + ["device"]

    # internal
    _model: nn.Module
    _feature_extractor: nn.Module

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def feature_extractor(self) -> nn.Module:
        if not hasattr(self, "_feature_extractor"):
            self._feature_extractor = self._get_feature_extractor(self.model_name)
        return self._feature_extractor

    @property
    def model(self) -> nn.Module:
        if not hasattr(self, "_model"):
            self._model = self._get_sound_model(self.model_name)
        return self._model

    @abstractmethod
    def _get_feature_extractor(self, model_name: str) -> torch.nn.Module:
        raise NotImplementedError

    @abstractmethod
    def _get_sound_model(self, model_name: str) -> torch.nn.Module:
        raise NotImplementedError

    @abstractmethod
    def _get_features(self, wav: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _get(self, event: ns.events.Sound, start: float, duration: float) -> torch.Tensor:
        nplatents = next(self._get_latents([event]))
        if self.frequency == "native" and self._frequency_override is None:
            # get_latent is bypassed by infra, so we need to duplicate logic here
            self._frequency_override = event.duration / nplatents.shape[-1]
        latents = self._fill_slice(nplatents, event, start, duration)
        if self.cache_all_layers:
            latents = self._aggregate_layers(latents)
        return latents

    def _aggregate_layers(self, latents: torch.Tensor) -> torch.Tensor:
        layers = self.layers if isinstance(self.layers, list) else [self.layers]
        n_layers = latents.shape[0]
        if any([isinstance(element, float) for element in layers]):
            assert all([0 <= l <= 1 for l in layers]), "Layers must be between 0 and 1"
            layer_indices = np.unique(
                [int(i * n_layers - 1e-6) for i in layers]
            ).tolist()  # 1e-6 to avoid taking index n_layers
        else:  # Pick layers according to their indices
            duplicates = np.array(layers) % n_layers
            assert len(duplicates) == len(set(duplicates)), "The list contains duplicates"
            assert all(
                [-n_layers <= l < n_layers for l in layers]
            ), f"The list contains duplicate layer indices: {duplicates}"
            layer_indices = layers

        return latents[layer_indices].mean(0)

    def _process_wav(self, event: ns.events.Sound) -> torch.Tensor:
        wav = event.read()
        wav = self._preprocess_wav(wav)
        wav = self._resample_wav(wav, event.frequency, self.feature_extractor.sampling_rate)  # type: ignore

        features = self._get_features(wav)

        with torch.no_grad():
            outputs = self.model(features.to(self.device), output_hidden_states=True)
        if self.layer_type == "transformer":
            out: tp.Any = outputs.get("hidden_states")
        elif self.layer_type == "convolution":
            out = outputs.get("extract_features")
        else:
            raise ValueError(f"Unknown layer type: {self.layer_type}")
        if isinstance(out, tuple):
            out = torch.stack(out)

        out = out.squeeze(1).detach().cpu().clone().transpose(-1, -2)  # type: ignore
        if not self.cache_all_layers:
            out = self._aggregate_layers(out)

        return out  # (n_layers, n_features, n_times)


class Wav2Vec(BaseHuggingFaceAudio):
    """
    Pretrained Wav2Vec model from BaseHuggingFaceAudio.
    """

    name: tp.Literal["Wav2Vec"] = "Wav2Vec"
    model_name: str = "facebook/wav2vec2-large-xlsr-53"

    def _get_sound_model(self, model_name: str) -> torch.nn.Module:
        from transformers import Wav2Vec2Model

        _model = Wav2Vec2Model.from_pretrained(model_name)
        _model.to(self.device)
        _model.eval()
        return _model

    def _get_feature_extractor(self, model_name: str) -> torch.nn.Module:
        from transformers import Wav2Vec2FeatureExtractor

        return Wav2Vec2FeatureExtractor.from_pretrained(model_name)

    def _get_features(self, wav):
        return self._feature_extractor(
            wav,
            return_tensors="pt",
            sampling_rate=self.feature_extractor.sampling_rate,
            do_normalize=self.normalized,
        ).input_values


class SeamlessM4T(BaseHuggingFaceAudio):
    """
    Pretrained Seamless M4T model from BaseHuggingFaceAudio.
    """

    name: tp.Literal["SeamlessM4T"] = "SeamlessM4T"
    model_name: str = "facebook/hf-seamless-m4t-medium"
    layers: float | tp.List[float] = [0.5]

    def _get_feature_extractor(self, model_name: str) -> torch.nn.Module:
        from transformers import SeamlessM4TFeatureExtractor

        return SeamlessM4TFeatureExtractor.from_pretrained(model_name)

    def _get_sound_model(self, model_name: str) -> torch.nn.Module:
        from transformers import SeamlessM4TModel

        model = SeamlessM4TModel.from_pretrained(model_name).speech_encoder.to(
            self.device
        )
        return model

    def _get_features(self, wav):
        return self._feature_extractor(
            wav,
            return_tensors="pt",
            sampling_rate=self.feature_extractor.sampling_rate,
            do_normalize=self.normalized,
        ).input_features


class SonarAudio(BaseAudio):
    name: tp.Literal["SonarAudio"] = "SonarAudio"
    event_type: tp.ClassVar[tp.Type[ns.events.Event]] = ns.events.Sound
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("sonar-space", "fairseq2")

    device: str = "cuda"
    sampling_rate: int = 16_000
    # use hidden_states for transformer layers and extract_features for convolutional layers
    layer: float = 0.5

    # internal
    _model: nn.Module
    _feature_extractor: nn.Module

    @property
    def model(self) -> nn.Module:
        if not hasattr(self, "_model"):
            self._model = self._get_sound_model()
        return self._model

    def _get_sound_model(self) -> nn.Module:
        from sonar.inference_pipelines.speech import (  # type: ignore
            SpeechToEmbeddingModelPipeline,
        )

        pipeline = SpeechToEmbeddingModelPipeline(encoder="sonar_speech_encoder_eng")
        model = pipeline.model
        n_layers = len(model.encoder.layers)
        layer_idx = int(self.layer * n_layers)
        model.encoder.layers = model.encoder.layers[:layer_idx]
        model.forward = lambda x: model.encoder(
            model.encoder_frontend(x.seqs, None)[0], None
        )
        return pipeline

    def _get(self, event: ns.events.Sound, start: float, duration: float) -> torch.Tensor:
        latents = next(self._get_latents([event]))
        if self.frequency == "native" and self._frequency_override is None:
            # get_latent is bypassed by infra, so we need to duplicate logic here
            self._frequency_override = event.duration / latents.shape[-1]
        return self._fill_slice(latents, event, start, duration)

    def _process_wav(self, event: ns.events.Sound) -> torch.Tensor:
        wav = event.read()
        wav = self._resample_wav(wav, event.frequency, self.sampling_rate)

        with torch.no_grad():
            out = self.model.predict([wav])

        return out.squeeze(1).detach().cpu().clone().transpose(-1, -2)  # type: ignore
