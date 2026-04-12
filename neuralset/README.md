# neuralset
Neuro AI made easy.

## Install
To install the dependencies, make sure your current directory is brainai/neuralset (you can use pwd to check your current location)

(neuralset-install)=
```bash
export ENVNAME=neuralset
conda create -n $ENVNAME python=3.10 ipython -y
conda activate $ENVNAME
# - make sure pip is in the env (check "which pip")
# - for cuda enabled torch (on FAIR cluster, AWS, etc.):
pip install -U torch==2.3.1 torchvision==0.18.1

# - for main neuralset dependencies:
# If not already the case, change your current working directory so that it finishes in "brainai/neuralset"
pip install --config-settings editable_mode=strict -e '.[dev]'
```

To install additional without dependencies, you can simply run `pip install --config-settings editable_mode=strict -e .` instead. With `dev` requirements, you can install pre-commit hooks through `pre-commit install`, they will fix `black`, `isort` and a few more things automatically (if need be, deactivate for a commit with `-n`)


*Note*: `editable_mode=strict` is only useful for `mypy` to pick up `neuralset` typing from within `neuraltrain` or other external packages when installed as editable (`-e`). Also, as this creates symlinks for each file, you may need to rerun the command when new files are added.

## Step by step

### From scratch to dataloader
```python
import pandas as pd
from torch.utils.data import DataLoader
import neuralset as ns

# A study is just a dataframe of events
timeline = 'subject-1_recording-99-session-foo'
events = pd.DataFrame([
    dict(type='Word', start=10., duration=.5, text='Hello', timeline=timeline),
    dict(type='Word', start=12., duration=.5, text='world', timeline=timeline),
])
events = ns.segments.validate_events(events)

# For each event, we need to specify how these discrete events
# can be converted into a dense time series.
feature = ns.features.SpacyEmbedding(language='english', aggregation='sum')
data = feature(events, start=10., duration=3.)
n_dims, = data.shape

# We may want to get a dynamic event
feature = ns.features.SpacyEmbedding(frequency=100., language='english', aggregation='sum')
data = feature(events, start=10., duration=3.)
n_dims, n_times = data.shape

# To make a DataLoader we need to build a dataset (a list of segments).
# For this, we can use the `ns` accessor
segments = ns.segments.list_segments(events, idx=events.type=="Word", start=-.3, duration=2.)
segment1, segment2 = segments

# We then need to specify our collate function:
collate_fn = ns.CollateSegments(features={"embedding": feature})
for batch in DataLoader(segments, collate_fn=collate_fn, batch_size=2):
    break
batch_size, n_dims, n_times = batch.data['embedding'].shape
```

Alternatively, one can use a pytorch dataset instead of the `CollateSegments` class.
This is not simpler as a `collate_fn` still needs to be specified, but it will improve
type checking and possibly robustness as it is a more standard approach:

```python continuation
ds = ns.SegmentDataset({"embedding": feature}, segments)
dataloader = DataLoader(ds, collate_fn=ds.collate_fn, batch_size=20)
batch = next(iter(dataloader))
print(batch.data["embedding"].shape)
```

### Make dataset from a rolling window
```python continuation
segments = ns.segments.list_segments(events, stride=1.5, duration=3.)
```

### Make dataset from rolling windows within specific events
```python continuation
segments = ns.segments.list_segments(events, idx=events.type=="Word", stride=0.5, duration=1.)
```

### I don't get it, there are many different recordings in a study
```python continuation
# Yes, the core idea is that each event is associated with a 'timeline'.
for tl, df in events.groupby('timeline', sort=False):
    print(tl, len(ns.segments.list_segments(events, stride=1.5, duration=3.)))
```

### Neuroimaging data is a feature too.
```python
import neuralset as ns
feature = ns.features.Meg(frequency=100.)
loader = ns.data.StudyLoader(name='TestMeg2023', path='./')
events = loader.build()
dset = ns.segments.list_segments(events, idx=events.type == "Image", start=0., duration=.5)
segment = dset[0]
data = feature(segment.events, segment.start, segment.duration)
n_channels, n_times = data.shape
```

### Implement a standard cross-validation
```python
import neuralset as ns
from sklearn.model_selection import GroupKFold
loader = ns.data.StudyLoader(name='TestMeg2023', path='./')
events = loader.build()
images = events.query('type=="Image"')
dset = ns.segments.list_segments(events, idx=images.index, duration=.5)

cv = GroupKFold(3)
for train, test in cv.split(dset, groups=images.filepath):
    pass
```

### Use the study cache
```python
import neuralset as ns
from pathlib import Path
cache = ns.CACHE_FOLDER  # defaults to ~/.cache/neuralset
loader = ns.data.StudyLoader(name='TestMeg2023', path='./', infra={"folder": cache})
events = loader.build()
dset = ns.segments.list_segments(events, idx=events.type == "Image", start=-.3, duration=.5)
```


### Advanced use-cases

#### Scenario 1: I want to split by an ad-hoc concept, such as 'sentence'

```python
import neuralset as ns
from sklearn.model_selection import GroupKFold
import numpy as np

loader = ns.data.StudyLoader(name='TestFmri2023', path='./')
events = loader.build()

# Ensure words are in order
events = events.sort_values(['timeline', 'start'])
words = events.query('type=="Word"')

# Parse sentences
sent_start = words.text.str.endswith('.').shift(1).fillna(False)
events.loc[words.index, 'sentence_id'] = np.cumsum(sent_start)

# update field
words = events.loc[words.index]

# We could stop here:
cv = GroupKFold(3)
for train, test in cv.split(words, groups=words.sentence_id):
    pass
```

#### Scenario 2: I need sentence-level information to each sub-units (e.g. words)

```python continuation
for _, sent in words.groupby('sentence_id', sort=False):
    # Find all events within the sentence
    duration = sent.stop.max() - sent.start.min() + 1e-8

    # Select all events within it
    sel = ns.segments.find_enclosed(sent, start=sent.start.min(), duration=duration)
    events.loc[sel, 'sentence'] = ' '.join(sent.text)

# update field
words = events.loc[words.index]

cv = GroupKFold(3)
for train, test in cv.split(words, groups=words.sentence):
    pass

```

#### Scenario 3: I want to merge short sentences
```python continuation
min_duration = 3.

for sentence_id, sent in words.groupby('sentence_id', sort=False):
    # update as this may be changing within the loop
    sent = events.loc[events.sentence_id == sentence_id]

    # if the sentence is too short
    if (sent.stop.max() - sent.start.min()) < min_duration:

        # find the next sentence
        next_sentence = events.query(f"sentence_id == {sentence_id + 1}")

        # and given them the same sentence_id and
        sel = sent.index.union(next_sentence.index)
        sentence = ' '.join(events.loc[sel].sentence.unique())
        events.loc[sel, 'sentence'] = sentence
        events.loc[sel, 'sentence_id'] = sentence_id

# update field
words = events.loc[words.index]

cv = GroupKFold(2)
for train, test in cv.split(words, groups=words.sentence):
    pass
```

#### Scenario 4: I want to assign the split in a deterministic fashion.
```python continuation
from neuralset.splitting import DeterministicSplitter

# This object assigns a split from the hash of a string
# It is fully deterministic, and thus dataset independent.
random_split = dict(train=.5, val=.25, test=.25)
splitter = DeterministicSplitter(random_split, seed=0)
valid = ~events.sentence.isna()
events.loc[valid, 'split'] = events.loc[valid].sentence.apply(splitter)

# Finally proceede as usual
sel = (events.type == "Word") & (events.split == "train")
dset = ns.segments.list_segments(events, idx=sel, duration=.5)
```

### Onboard a new feature
```python
import typing as tp
import neuralset as ns
import torch

class MyEmbedding(ns.features.BaseStatic):
    name: tp.Literal['MyEmbedding'] = 'MyEmbedding'
    event_type: tp.ClassVar[tp.Any] = ns.events.Text
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("spacy>=3.5.4",)

    def get_static(self, event: dict) -> torch.Tensor:
        import spacy

        if not hasattr(self, 'nlp'):
            self.nlp = spacy.load('en_core_web_md')

        vector = self.nlp(text).vector
        return torch.from_numpy(vector).float()

```

The "name" field is mandatory when creating a new feature (an explicit error is raised if you forget about it)
so as to allow for pydantic
[discriminated unions](https://docs.pydantic.dev/2.0/usage/types/unions/#discriminated-unions-aka-tagged-unions)
based on the name field.
This lets you use `ns.features.FeatureConfig` to instantiate any of the feature classes based on the provided name:

```python continuation
import pydantic
import neuralset as ns
ns.features.update_config_feature()  # call this if you defined a feature externally


class Model(pydantic.BaseModel):
    features: tp.Sequence[ns.features.FeatureConfig] = ()


model = Model(features=[{"name": "Pulse"},  {"name": "MyEmbedding"}])
```


### Onboard a new study
```python
import typing as tp
import mne
import numpy as np
import pandas as pd
from neuralset import BaseData


class MyStudy2023(BaseData):
    subject: str
    device: tp.ClassVar[str] = 'Fmri'

    @classmethod
    def _download(cls, path):
        cls.path = path

    @classmethod
    def _iter_timelines(cls) -> BaseData:
        """
        Iterate over the different recording timelines:
        e.g. subjects x runs x sessions
        """
        n_subjects = 3
        for subject in range(n_subjects):
            yield cls(subject=subject, path=cls.path)

    def _load_events(self) -> pd.DataFrame:
        """Reads the events of a given timeline
        """
        events = pd.DataFrame([
            dict(type='Image', filepath='image.png', start=10., duration=1.),
            dict(type='Fmri', filepath='foo.nii.gz'),
        ])

```

### Onboard a new study: Advanced case, when specific data reader needs to be given
```python
import typing as tp
import mne
import numpy as np
import pandas as pd
from neuralset import BaseData

class MyStudy2023(BaseData):
    subject: str
    device: tp.ClassVar[str] = 'Meg'

    @classmethod
    def _download(cls, path):
        cls.path = path

    @classmethod
    def _iter_timelines(cls) -> BaseData:
        """
        Iterate over the different recording timelines:
        e.g. subjects x runs x sessions
        """
        n_subjects = 3
        for subject in range(n_subjects):
            yield cls(subject=subject, path=cls.path)

    def _load_events(self) -> pd.DataFrame:
        """Reads the events of a given timeline
        """
        im = dict(type='Image', filepath='image.png', start=10., duration=1.)
        uri = f"method:_load_raw?timeline={self.timeline}"  # states to use _load_raw
        meg = {"filepath": uri, "type": "Meg", "start": 0}
        events = pd.DataFrame([im, meg])

    def _load_raw(self, timeline: str):
        """Reads the raw header of a given timeline
        """
        n_chans = 20
        sfreq = 100.
        n_times = 10_000
        info = mne.create_info(n_chans, sfreq=sfreq, ch_types='mag')
        data = np.zeros((n_chans, n_times))
        raw = mne.io.RawArray(data, info)
        return raw

```
