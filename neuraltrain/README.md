# neuraltrain

`neuraltrain` is a lightweight library of useful classes and functions for training deep learning models with neuralset.

## Install

If you have already installed `neuralset` following its installation instructions, you can just do:

```
conda activate <your_neuralset_env>
pip install --config-settings editable_mode=strict -e .
```
*Note*: `editable_mode=strict` is required for `mypy` to pick up `neuralset` typing from within `neuraltrain` when installed as editable (`-e`).

If starting from scratch:

```
conda create -n neuraltrain python=3.11 ipython -y
conda activate neuraltrain
pip install -e .
```

To install additional `dev` or `lightning` dependencies, you can instead run: 
```
pip install -e .'[dev,lightning]'
```

## Project example

See [`project_example`](project_example/README.md) for an example project built with `neuraltrain` and `pytorch-lightning`.
