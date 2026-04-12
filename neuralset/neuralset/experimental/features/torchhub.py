# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import torch
from torch import nn


def _struct_str(m: nn.Module) -> str:
    """Return a simplified string representation of a model (for debugging /
    identifying where to put hooks)
    """
    parts = []
    for name, m2 in m._modules.items():
        if m2 is None:
            continue
        cls = m2.__class__.__name__
        parts.append(f"({name}): {cls}")
        sub = _struct_str(m2)
        if sub:
            sub = "\n".join("    " + s for s in sub.splitlines())
            parts.append(sub)
    return "\n".join(parts)


HookRegistration = tp.Callable[["TorchHubModel", nn.Module], None]


class TorchHubModel(nn.Module):
    """Selected torch hub models with layer features extraction

    Parameters
    ----------
    name: str
        name of the model in torchhub
    pretrained: bool
        whether the pretrained weights should be loaded

    Notes
    -----
    - This is only classification models so far, segmentation models may require
      different transforms
    - Each model needs to be registered through a function named as the model and
      decoratated with @TorchHubModel.register.
    """

    REGISTER: tp.Dict[str, HookRegistration] = {}

    def __init__(self, name: str, pretrained: bool = True) -> None:
        super().__init__()
        from torchvision import transforms

        weights: tp.Any = None
        if pretrained:
            # Grab the weights to pass it into an enum
            weights = True  # TODO: figure it out at some point
        m = torch.hub.load("pytorch/vision:v0.10.0", name, weights=weights)
        m.eval()
        self.model = m
        # info: https://pytorch.org/vision/0.10/models.html#
        self.resize, self.crop = 256, 224
        self._activations: tp.List[torch.Tensor] = []
        self.info = {"name": name, "imsize": self.crop, "pipeline_tag": "classification"}
        if name == "inception_v3":
            self.resize, self.crop = 299, 299
        self.transforms = transforms.Compose(
            [
                transforms.Resize(self.resize),
                transforms.CenterCrop(self.crop),  # for classification models!
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.REGISTER[name](self, m)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def hidden_states(self, images: torch.Tensor) -> tp.List[torch.Tensor]:
        """Returns the sequence of registered hidden states"""
        self._activations.clear()
        _ = self(images)
        return list(self._activations)

    @classmethod
    def register(cls, func: HookRegistration) -> HookRegistration:
        """Register a function for setting up hooks on a model type
        The function name must match the model name
        """
        cls.REGISTER[func.__name__] = func
        return func

    def _hook(self, model: tp.Any, input_: tp.Any, output: tp.Any) -> None:
        """Hook used for registering output activations"""
        self._activations.append(output.detach())


@TorchHubModel.register
def alexnet(thm: TorchHubModel, m: nn.Module) -> None:
    thm.info.update(year=2012, arxiv="https://arxiv.org/abs/1404.5997")
    layers = [m.features[k] for k in [3, 6, 8, 10]]
    layers += [m.classifier[k] for k in [1, 4]]  # 6 is output / classes
    for layer in layers:
        assert isinstance(layer, (nn.Conv2d, nn.Linear))
        layer.register_forward_hook(thm._hook)


@TorchHubModel.register
def resnet18(thm: TorchHubModel, m: nn.Module) -> None:
    thm.info.update(year=2015, arxiv="https://arxiv.org/abs/1512.03385")
    layers = [m.layer1, m.layer2, m.layer3, m.layer4]
    for layer in layers:
        for block in layer:
            block.register_forward_hook(thm._hook)


@TorchHubModel.register
def vgg16(thm: TorchHubModel, m: nn.Module) -> None:
    thm.info.update(year=2014, arxiv="https://arxiv.org/abs/1409.1556")
    layers = [m.features[k] for k in (5, 10, 17, 24, 28)]
    layers += [m.classifier[k] for k in (0, 3)]  # 6 is output
    for layer in layers:
        assert isinstance(layer, (nn.Conv2d, nn.Linear))
        layer.register_forward_hook(thm._hook)


@TorchHubModel.register
def squeezenet1_0(thm: TorchHubModel, m: nn.Module) -> None:
    thm.info.update(year=2016, arxiv="https://arxiv.org/abs/1602.07360")
    registered = 0
    for layer in m.features:
        if not isinstance(layer, (nn.MaxPool2d, nn.Conv2d, nn.ReLU)):
            assert layer.__class__.__name__ == "Fire"
            registered += 1
            layer.register_forward_hook(thm._hook)
    assert registered == 8, f"Expected 8 Fire layers, got {registered}"


@TorchHubModel.register
def densenet161(thm: TorchHubModel, m: nn.Module) -> None:
    thm.info.update(year=2016, arxiv="https://arxiv.org/abs/1608.06993")
    registered = 0
    for layer in m.features:
        if layer.__class__.__name__ in "_DenseBlock":
            registered += 1
            layer.register_forward_hook(thm._hook)
    assert registered == 4, f"Expected 4 dense blocks, got {registered}"


@TorchHubModel.register
def inception_v3(thm: TorchHubModel, m: nn.Module) -> None:
    thm.info.update(year=2015, arxiv="https://arxiv.org/abs/1512.00567")
    registered = 0
    for name, m2 in m._modules.items():
        assert m2 is not None
        if name.startswith("Mixed"):
            m2.register_forward_hook(thm._hook)
            registered += 1
    assert registered == 11, f"Expected 4 mixed blocks, got {registered}"


@TorchHubModel.register
def googlenet(thm: TorchHubModel, m: nn.Module) -> None:
    thm.info.update(year=2014, arxiv="https://arxiv.org/abs/1409.4842")
    registered = 0
    for name, m2 in m._modules.items():
        if m2 is None:
            continue
        if name.startswith("inception"):
            m2.register_forward_hook(thm._hook)
            registered += 1
    assert registered == 9, f"Expected 9 inception blocks, got {registered}"


@TorchHubModel.register
def shufflenet_v2_x1_0(thm: TorchHubModel, m: nn.Module) -> None:
    thm.info.update(year=2018, arxiv="https://arxiv.org/abs/1807.11164")
    registered = 0
    for name, m2 in m._modules.items():
        if m2 is None:
            continue
        if name.startswith("stage") and isinstance(m2, nn.Sequential):
            for layer in m2:
                layer.register_forward_hook(thm._hook)
                registered += 1
    assert registered == 16, f"Expected 16 sub stage blocks, got {registered}"


@TorchHubModel.register
def mobilenet_v2(thm: TorchHubModel, m: nn.Module) -> None:
    thm.info.update(year=2018, arxiv="https://arxiv.org/abs/1801.04381")
    for layer in m.features:
        layer.register_forward_hook(thm._hook)


@TorchHubModel.register
def mobilenet_v3_large(thm: TorchHubModel, m: nn.Module) -> None:
    thm.info.update(year=2019, arxiv="https://arxiv.org/abs/1905.02244")
    mobilenet_v2(thm, m)


@TorchHubModel.register
def mobilenet_v3_small(thm: TorchHubModel, m: nn.Module) -> None:
    thm.info.update(year=2019, arxiv="https://arxiv.org/abs/1905.02244")
    mobilenet_v2(thm, m)


@TorchHubModel.register
def resnext50_32x4d(thm: TorchHubModel, m: nn.Module) -> None:
    thm.info.update(year=2016, arxiv="https://arxiv.org/abs/1611.05431")
    registered = 0
    for name, m2 in m._modules.items():
        if m2 is None:
            continue
        if name.startswith("layer"):
            for layer in m2:  # type: ignore
                layer.register_forward_hook(thm._hook)
                registered += 1
    assert registered == 16, f"Expected 16  bottleneck blocks, got {registered}"


@TorchHubModel.register
def wide_resnet50_2(thm: TorchHubModel, m: nn.Module) -> None:
    thm.info.update(year=2016, arxiv="https://arxiv.org/abs/1605.07146")
    resnext50_32x4d(thm, m)


@TorchHubModel.register
def mnasnet1_0(thm: TorchHubModel, m: nn.Module) -> None:
    thm.info.update(year=2018, arxiv="https://arxiv.org/abs/1807.11626")
    registered = 0
    for layer in m.layers:
        if isinstance(layer, nn.Sequential):
            layer.register_forward_hook(thm._hook)
            registered += 1
    assert registered == 6, f"Expected 6 blocks, got {registered}"
