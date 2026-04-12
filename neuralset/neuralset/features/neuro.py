# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from itertools import compress

import mne
import numpy as np
import pandas as pd
import pydantic
import sklearn.preprocessing
import torch

import neuralset as ns
from neuralset.infra import MapInfra

from .base import BaseDynamic, BaseFeature

logger = logging.getLogger(__name__)


class Meg(BaseDynamic):
    """If frequency is set to "native", the frequency used will be the one provided by the Meg event
    filter and resample preprocessing steps can be cached.

    Parameters
    ----------
    baseline :
        If provided as a tuple (start, end), corresponds to the start and end times (in seconds)
        relative to the **beginning of a window** (i.e. NOT relative to the epoch onset as opposed
        to MNE's convention) of the segment to use for baselining.
    """

    name: tp.Literal["Meg"] = "Meg"
    event_type: tp.ClassVar[tp.Type[ns.events.Event]] = ns.events.Meg

    frequency: tp.Literal["native"] | float = "native"
    offset: float = 0.0
    baseline: tp.Tuple[float, float] | None = None
    pick_types: tp.Tuple[str, ...] = pydantic.Field(("meg",), min_length=1)
    apply_proj: bool = False
    filter: tp.Tuple[float | None, float | None] | None = None
    apply_hilbert: bool = False
    mne_cpus: int = -1
    infra: MapInfra = MapInfra(
        timeout_min=120,
        gpus_per_node=0,
        cpus_per_task=10,
        version="2",
    )
    scaler: None | tp.Literal["RobustScaler", "StandardScaler"] = None
    clamp: float | None = None

    _channels: tp.Dict[str, int] = {}

    @classmethod
    def _exclude_from_cls_uid(cls) -> tp.List[str]:
        return super()._exclude_from_cls_uid() + ["mne_cpus"]

    def _exclude_from_cache_uid(self) -> tp.List[str]:
        prev = super()._exclude_from_cache_uid()
        return prev + ["baseline", "offset", "clamp"]

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        # Update channel mapping to be robust to mne.Raw
        self._channels = {}
        # check baseline
        if self.baseline is not None:
            message = f"baseline must be None or 2 floats, got {self.baseline}"
            assert len(self.baseline) == 2, message
            assert isinstance(self.baseline[0], float), message
            assert isinstance(self.baseline[1], float), message
            assert self.baseline[1] > self.baseline[0], message

    def prepare(self, events: pd.DataFrame) -> None:
        """specify how to load and preprocess the event.
        Can be overriden by user.
        """
        events_ = self._events_from_dataframe(events)
        self._get_preprocessed_data(events_)
        self._prepare_channels(events_)

    @infra.apply(
        item_uid=lambda e: str(e.filepath),
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
    )
    def _get_preprocessed_data(
        self, events: tp.List[ns.events.Meg]
    ) -> tp.Iterator[mne.io.Raw]:
        for event in events:
            raw = event.read()
            raw = raw.pick(self.pick_types, verbose=False)
            if self.filter is not None:
                raw.load_data()
                raw.filter(
                    self.filter[0], self.filter[1], n_jobs=self.mne_cpus, verbose=False
                )
            if self.apply_hilbert:
                raw.load_data()
                raw = raw.apply_hilbert(envelope=True)

            freq = self._output_frequency(event)  # deals with native/specified frequency
            if freq != event.frequency:
                raw.load_data()
                raw = raw.resample(freq, n_jobs=self.mne_cpus, verbose=False)
            if self.scaler is not None:
                raw.load_data()
                scaler = getattr(sklearn.preprocessing, self.scaler)()
                raw._data = scaler.fit_transform(raw._data.T).T
            if self.apply_proj:
                raw.apply_proj()
            yield raw

    def _get(self, event: ns.events.Meg, start: float, duration: float) -> torch.Tensor:
        start += self.offset

        # Extend window in case of disjoint baseline
        window_start, window_duration = start, duration
        if self.baseline is not None and (
            self.baseline[0] < 0.0 or self.baseline[1] > duration
        ):
            start_offset = min(self.baseline[0], 0.0)
            start += start_offset
            duration = max(duration, self.baseline[1]) - start_offset
            baseline = (max(0.0, self.baseline[0]), self.baseline[1] - start_offset)
        else:
            start_offset = 0.0
            baseline = self.baseline  # type: ignore

        # cached_preprocessing
        # (copy to avoid corrupting cache)
        raw = next(self._get_preprocessed_data([event]))
        assert isinstance(raw, mne.io.BaseRaw)  # for typing

        # safeguard for first_samp
        if raw.first_samp and not event.start:
            raise RuntimeError(
                "event.start should be raw.first_samp / freq for consistency"
            )
        overlap_start, overlap_duration = self._get_overlap(
            event.start, event.duration, start, duration
        )
        data_start = overlap_start - event.start  # time in the M/EEG referential
        freq = self._output_frequency(event)  # deals with native/specified frequency
        # times in time_as_index are assumed to be relative to first_samp (cf doc)
        # time_as_index is slow, so let's do it manually
        # start_idx, stop_idx = raw.time_as_index([meg_start, meg_start + overlap_duration])
        start_idx = freq.to_ind(data_start)
        if start_idx == raw.n_times:
            start_idx -= 1
        # apply freq on overlap to keep always the same size, and minimum to 1
        stop_idx = start_idx + max(1, freq.to_ind(overlap_duration))

        try:
            npdata, _ = raw[:, start_idx:stop_idx]
        except ValueError:
            msg = (
                "Failed to read event %r (start=%s duration=%s)\n"
                "(start_idx=%s stop_idx=%s in %s)"
            )
            logger.warning(msg, event, start, duration, start_idx, stop_idx, raw)
            raise
        data = torch.from_numpy(npdata).float()

        # Apply baseline to the data
        if self.baseline is not None:
            msg = f"unexpected baseline:{baseline}"
            tmin, tmax = [freq.to_ind(t) for t in baseline]
            assert (tmax - tmin) > 0, msg
            data -= data[:, tmin:tmax].mean(1, keepdim=True)

            # Crop larger window in case of disjoint baseline
            if window_start != start or window_duration != duration:
                start, duration = window_start, window_duration
                tmin = freq.to_ind(-start_offset)
                tmax = freq.to_ind(-start_offset + duration)
                data = data[:, tmin:tmax]

                # Recompute start_idx for cropped window
                overlap_start, overlap_duration = self._get_overlap(
                    event.start, event.duration, start, duration
                )
                data_start = overlap_start - event.start
                start_idx = freq.to_ind(data_start)

        # initialize output
        channel_idx = self._get_channels(raw.ch_names)
        n_samples = max(1, freq.to_ind(duration))
        out: torch.Tensor = torch.zeros(
            (len(self._channels), n_samples), dtype=torch.float32
        )
        # get overlap times between output and meg
        out_slice, event_slice = self._get_overlap_slice(
            freq,  # equal to raw.info["sfreq"]
            event.start,
            event.duration,
            start,
            duration,
        )
        event_slice = slice(
            event_slice.start - start_idx,
            event_slice.stop - start_idx,
            event_slice.step,
        )
        if event_slice.stop == data.shape[-1] + 1:
            # rounding for last sample of data, let's repeat the final sample
            es = event_slice
            event_slice = list(range(es.start, es.stop, es.step))  # type: ignore
            if event_slice:  # for empty case
                event_slice[-1] -= 1  # type: ignore
        # set to output
        try:
            out[channel_idx, out_slice] = data[:, event_slice]
        except:
            print(f"Failure with {out_slice=} and {event_slice=}")
            print(f"{start=}, {duration=}, {freq=}")
            print(f"{start_idx=} {stop_idx=}")
            print(f"{data.shape=}\n{event=}\n{raw=}")
            raise

        if self.clamp is not None:
            out = out.clamp(min=-self.clamp, max=self.clamp)

        return out

    def _update_channels(self, ch_names: tp.List[str]) -> None:
        channels = self._channels  # avoid calling pydantic attr too many times
        for ch in ch_names:
            if ch not in self._channels:
                channels[ch] = len(channels)

    def _prepare_channels(self, events: tp.List[ns.events.Meg]) -> None:
        for raw in self._get_preprocessed_data(events):
            raw = raw.pick(self.pick_types, verbose=False)
            self._update_channels(raw.ch_names)

    def _get_channels(self, ch_names: tp.List[str]) -> tp.List[int]:
        if not self._channels:
            self._update_channels(ch_names)
        try:
            channel_idx = [self._channels[ch] for ch in ch_names]
        except KeyError as e:
            raise KeyError(
                f"Channel {e} not found in the channel mapping, likely because this dataset contains recordings with different sets of channel names. Try calling self.prepare on the whole events dataframe."
            ) from e
        return channel_idx


class Eeg(Meg):
    name: tp.Literal["Eeg"] = "Eeg"  # type: ignore
    event_type: tp.ClassVar[tp.Type[ns.events.Event]] = ns.events.Eeg
    pick_types: tp.Tuple[str, ...] = pydantic.Field(("eeg",), min_length=1)


class Emg(Meg):
    name: tp.Literal["Emg"] = "Emg"  # type: ignore
    event_type: tp.ClassVar[tp.Type[ns.events.Event]] = ns.events.Emg
    pick_types: tp.Tuple[str, ...] = pydantic.Field(("emg",), min_length=1)


class Fnirs(Meg):
    requirements: tp.ClassVar[tp.Any] = ("mne-nirs",)
    name: tp.Literal["Fnirs"] = "Fnirs"  # type: ignore
    event_type: tp.ClassVar[tp.Type[ns.events.Event]] = ns.events.Fnirs
    pick_types: tp.Tuple[str, ...] = pydantic.Field(("fnirs",), min_length=1)
    # Preprocessing
    distance_threshold: float | None = None
    compute_optical_density: bool = False
    scalp_coupling_index_threshold: float | None = None
    apply_tddr: bool = False  # Apply temporal derivative distribution repair
    compute_heamo_response: bool = False
    partial_pathlength_factor: float = 0.1
    enhance_negative_correlation: bool = False
    #
    infra: MapInfra = MapInfra(
        timeout_min=120,
        gpus_per_node=0,
        cpus_per_task=10,
        version="1",
    )

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)

        # Ensure preprocessing steps are consistent with one another
        if self.compute_heamo_response and not self.compute_optical_density:
            msg = "Computing haemodynamic response requires computing optical density first."
            raise ValueError(msg)
        if self.scalp_coupling_index_threshold is not None:
            if not self.compute_optical_density:
                raise ValueError(
                    "Thresholding with the SCI requires computing optical density first."
                )

        if self.enhance_negative_correlation and not self.compute_heamo_response:
            msg = "Applying negative correlation enhancement requires haemodynamic responses."
            raise ValueError(msg)

    @infra.apply(
        item_uid=lambda e: str(e.filepath),
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
    )
    def _get_preprocessed_data(
        self, events: tp.List[ns.events.Fnirs]
    ) -> tp.Iterator[mne.io.Raw]:
        for event in events:
            raw = event.read()
            raw = raw.pick(self.pick_types, verbose=False)

            if self.distance_threshold is not None:
                dists = mne.preprocessing.nirs.source_detector_distances(raw.info)
                if np.isnan(dists).any():
                    msg = "Some or all distances are nan, please fix montage information."
                    raise ValueError(msg)
                picks = compress(raw.ch_names, dists > self.distance_threshold)
                raw = raw.pick(list(picks))

            if self.compute_optical_density:
                raw = mne.preprocessing.nirs.optical_density(raw)

            if self.scalp_coupling_index_threshold is not None:
                sci = mne.preprocessing.nirs.scalp_coupling_index(raw)
                picks = compress(raw.ch_names, sci > self.scalp_coupling_index_threshold)
                raw = raw.pick(list(picks))

            if self.apply_tddr:
                raw = mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw)

            if self.compute_heamo_response:
                raw = mne.preprocessing.nirs.beer_lambert_law(
                    raw, ppf=self.partial_pathlength_factor
                )

            if self.filter is not None:
                raw.load_data()
                raw.filter(
                    self.filter[0], self.filter[1], n_jobs=self.mne_cpus, verbose=False
                )

            if self.enhance_negative_correlation:
                import mne_nirs

                raw = mne_nirs.signal_enhancement.enhance_negative_correlation(raw)

            freq = self._output_frequency(event)  # deals with native/specified frequency
            if freq != event.frequency:
                raw.load_data()
                raw = raw.resample(freq, n_jobs=self.mne_cpus, verbose=False)
            if self.scaler is not None:
                raw.load_data()
                scaler = getattr(sklearn.preprocessing, self.scaler)()
                raw._data = scaler.fit_transform(raw._data.T).T

            yield raw


class Fmri(BaseDynamic):
    """If frequency is not specified (or equal to 0), frequency is assigned from
    the first nifti object read.
    """

    requirements: tp.ClassVar[tp.Any] = ("nilearn",)
    name: tp.Literal["Fmri"] = "Fmri"
    offset: float = 0.0
    event_type: tp.ClassVar[tp.Type[ns.events.Event]] = ns.events.Fmri
    mesh: str | None = None  # fsaverage5
    standardize: tp.Literal["zscore_sample", "zscore", "psc"] | bool = "zscore_sample"
    detrend: bool = False
    high_pass: float | None = None
    frequency: tp.Literal["native"] | float = "native"
    infra: MapInfra = MapInfra(
        timeout_min=120,
        gpus_per_node=0,
        cpus_per_task=10,
        version="2",
    )

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if self.frequency != "native":
            cls = self.__class__.__name__
            msg = f"{cls} only support frequency='native' (resampling not implemented)"
            raise ValueError(msg)

    def _exclude_from_cache_uid(self) -> tp.List[str]:
        return super()._exclude_from_cache_uid() + ["offset"]

    def prepare(self, events: pd.DataFrame) -> None:
        """specify how to load and preprocess the event.
        Can be overriden by user.
        """
        events_ = self._events_from_dataframe(events)
        self._events_to_data(events_)

    def _preprocess_event(self, event: ns.events.Fmri) -> np.ndarray:
        rec = event.read()
        if self.mesh is not None:
            data_space = "volumetric" if len(rec.shape) == 4 else "surface"
            if data_space == "volumetric":
                from nilearn import datasets, surface

                fsaverage = datasets.fetch_surf_fsaverage(self.mesh)
                hemis = [
                    surface.vol_to_surf(
                        rec,
                        surf_mesh=fsaverage[f"pial_{hemi}"],
                        inner_mesh=fsaverage[f"white_{hemi}"],
                    )
                    for hemi in ("left", "right")
                ]
                data = np.vstack(hemis)
                data[np.isnan(data)] = 0
            else:
                if len(rec.shape) != 2:
                    raise ValueError(f"Unexpected shape for volumetric data {rec.shape}")
                voxels = rec.shape[0] // 2
                sizes = {"fsaverage6": 40962, "fsaverage5": 10242}
                if self.mesh not in sizes:
                    raise NotImplementedError(f"Can only 2d project to {sizes} currently")
                if voxels not in list(sizes.values()) or rec.shape[0] % 2:
                    msg = f"Could not detect current 2d format from {sizes} with {rec.shape[0]} voxels"
                    raise NotImplementedError(msg)
                data = rec.get_fdata()
                if voxels < sizes[self.mesh]:
                    raise ValueError(
                        f"Cannot project from smaller {voxels} voxels to {self.mesh}"
                    )
                if voxels > sizes[self.mesh]:
                    left = data[: sizes[self.mesh], :]
                    right = data[voxels : voxels + sizes[self.mesh], :]
                    data = np.concatenate([left, right], axis=0)
        else:
            data = rec.get_fdata()
        # data shape = featdim1 x [featdim2 x ...] x time

        if self.detrend or self.standardize or self.high_pass is not None:
            import nilearn.signal

            data = data.T  # set time as first dim
            shape = data.shape
            data = nilearn.signal.clean(
                # required shape: (instant number, features number)
                data.reshape(shape[0], -1),
                detrend=self.detrend,
                high_pass=self.high_pass,
                t_r=1 / event.frequency,
                standardize=self.standardize,
            )
            data = data.reshape(shape).T
        return data.astype(np.float32)  # no need to keep float64 precision

    @infra.apply(
        item_uid=lambda e: str(e.filepath),
        exclude_from_cache_uid=_exclude_from_cache_uid,
        cache_type="NumpyMemmapArray",
    )
    def _events_to_data(self, events: tp.List[ns.events.Fmri]) -> tp.Iterable[np.ndarray]:
        for event in events:
            yield self._preprocess_event(event)

    def _get(self, event: ns.events.Fmri, start: float, duration: float) -> torch.Tensor:
        start += self.offset
        data = next(iter(self._events_to_data([event])))
        return self._fill_slice(data, event, start, duration)


class ChannelPositions(BaseFeature):
    """Channel positions in 2D, extracted from a Raw object's mne.Info.

    Parameters
    ----------
    meg :
        Feature ns.features.neuro.Meg that defines the preprocessing steps applied to the Raw
        objects.
    """

    event_type: tp.ClassVar[tp.Type[ns.events.Event]] = ns.events.Meg
    name: tp.Literal["ChannelPositions"] = "ChannelPositions"
    meg: Meg
    n_spatial_dims: int = 2

    # Value to use for channels that are not found in the layout
    INVALID_VALUE: tp.ClassVar[float] = -0.1

    infra: MapInfra = MapInfra()

    def prepare(self, events: pd.DataFrame) -> None:
        self.meg.prepare(events)  # Ensure the Raw objects have been precomputed
        events_ = self._events_from_dataframe(events)
        indices = events.loc[events.type.isin(["Meg", "Eeg"])].index
        to_remove = []
        for i, out in enumerate(self._get_channel_positions(events_)):
            if out is None:
                to_remove.append(i)
        # remove inplace from dataframe
        if to_remove:
            import warnings

            warnings.warn(
                f"ChannelPositions: {len(to_remove)} events have no channel positions"
                " and will be removed from the dataset."
            )
            events.drop(indices[to_remove], inplace=True)
        return events

    @infra.apply(item_uid=lambda e: str(e.filepath))
    def _get_channel_positions(
        self, events: tp.List[ns.events.Meg]
    ) -> tp.Generator[torch.Tensor, None, None]:
        for meg in self.meg._get_preprocessed_data(events):
            try:
                layout = mne.find_layout(meg.info)
            except RuntimeError as err:
                # Synthetic or cached Raw may lack sensor positions; standard montage
                # lets MNE build a layout for typical 10–20 channel names.
                if "No digitization points found" not in str(err):
                    raise
                montage = mne.channels.make_standard_montage("standard_1020")
                meg.set_montage(montage, match_case=False, on_missing="ignore")
                layout = mne.find_layout(meg.info)
            inds: list[int] = []
            valid_inds: list[int] = []
            for meg_index, name in enumerate(meg.info.ch_names):
                name = name.rsplit("-", 1)[0]
                try:
                    inds.append(layout.names.index(name))
                except ValueError:
                    pass
                else:
                    valid_inds.append(meg_index)

            positions = torch.full((len(meg.info.ch_names), 2), self.INVALID_VALUE)
            coords = layout.pos[inds, : self.n_spatial_dims]
            coords = (coords - coords.min(axis=0, keepdims=True)) / (
                coords.max(axis=0, keepdims=True) - coords.min(axis=0, keepdims=True)
            )
            positions[valid_inds] = torch.from_numpy(coords).float()

            channel_idx = self.meg._get_channels(meg.ch_names)
            out = torch.full(
                (len(self.meg._channels), self.n_spatial_dims), self.INVALID_VALUE
            )
            out[channel_idx, :] = positions[:, :]
            yield out

    def _get(self, event: ns.events.Meg, start: float, duration: float) -> torch.Tensor:
        return next(self._get_channel_positions([event]))
