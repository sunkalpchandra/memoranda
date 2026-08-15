"""Registry of vision models used for stimulus characterization.

Each entry knows how to build the network, which preprocessing to use and
which internal modules to hook for layer-wise features. Everything is lazy so
importing this module never downloads weights.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class ModelSpec:
    name: str
    family: str  # cnn | vit | clip | dino
    source: str  # torchvision | timm | open_clip
    layers: dict[str, str]  # label -> dotted module path (in forward order)
    build: Callable[[], tuple[nn.Module, Callable]] = field(repr=False)
    input_size: int = 224
    notes: str = ""


REGISTRY: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> ModelSpec:
    REGISTRY[spec.name] = spec
    return spec


def get_spec(name: str) -> ModelSpec:
    try:
        return REGISTRY[name]
    except KeyError as e:
        raise KeyError(f"unknown model {name!r}; known: {sorted(REGISTRY)}") from e


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ------------------------------------------------------------------ torchvision
def _tv(name: str, weights_attr: str):
    def build():
        import torchvision.models as tvm

        weights = getattr(tvm, weights_attr).DEFAULT
        model = getattr(tvm, name)(weights=weights).eval()
        return model, weights.transforms()

    return build


register(
    ModelSpec(
        name="alexnet",
        family="cnn",
        source="torchvision",
        layers={
            "conv1": "features.0",
            "conv2": "features.3",
            "conv3": "features.6",
            "conv4": "features.8",
            "conv5": "features.10",
            "fc6": "classifier.1",
            "fc7": "classifier.4",
            "logits": "classifier.6",
        },
        build=_tv("alexnet", "AlexNet_Weights"),
        notes="Krizhevsky 2012; classic in neuro-DNN comparisons",
    )
)
register(
    ModelSpec(
        name="vgg16",
        family="cnn",
        source="torchvision",
        layers={
            "conv1_2": "features.2",
            "conv2_2": "features.7",
            "conv3_3": "features.14",
            "conv4_3": "features.21",
            "conv5_3": "features.28",
            "fc6": "classifier.0",
            "fc7": "classifier.3",
            "logits": "classifier.6",
        },
        build=_tv("vgg16", "VGG16_Weights"),
    )
)
register(
    ModelSpec(
        name="resnet18",
        family="cnn",
        source="torchvision",
        layers={
            "stem": "maxpool",
            "layer1": "layer1",
            "layer2": "layer2",
            "layer3": "layer3",
            "layer4": "layer4",
            "avgpool": "avgpool",
            "logits": "fc",
        },
        build=_tv("resnet18", "ResNet18_Weights"),
    )
)
register(
    ModelSpec(
        name="resnet50",
        family="cnn",
        source="torchvision",
        layers={
            "stem": "maxpool",
            "layer1": "layer1",
            "layer2": "layer2",
            "layer3": "layer3",
            "layer4": "layer4",
            "avgpool": "avgpool",
            "logits": "fc",
        },
        build=_tv("resnet50", "ResNet50_Weights"),
    )
)
register(
    ModelSpec(
        name="convnext_tiny",
        family="cnn",
        source="torchvision",
        layers={
            "stage1": "features.1",
            "stage2": "features.3",
            "stage3": "features.5",
            "stage4": "features.7",
            "avgpool": "avgpool",
            "logits": "classifier.2",
        },
        build=_tv("convnext_tiny", "ConvNeXt_Tiny_Weights"),
    )
)
register(
    ModelSpec(
        name="vit_b_16",
        family="vit",
        source="torchvision",
        layers={
            "block2": "encoder.layers.encoder_layer_2",
            "block5": "encoder.layers.encoder_layer_5",
            "block8": "encoder.layers.encoder_layer_8",
            "block11": "encoder.layers.encoder_layer_11",
            "ln": "encoder.ln",
            "logits": "heads.head",
        },
        build=_tv("vit_b_16", "ViT_B_16_Weights"),
    )
)


# ------------------------------------------------------------------------- timm
def _timm(arch: str):
    def build():
        import timm

        model = timm.create_model(arch, pretrained=True).eval()
        cfg = timm.data.resolve_data_config({}, model=model)
        transform = timm.data.create_transform(**cfg)
        return model, transform

    return build


register(
    ModelSpec(
        name="dinov2_small",
        family="dino",
        source="timm",
        layers={
            "block2": "blocks.2",
            "block5": "blocks.5",
            "block8": "blocks.8",
            "block11": "blocks.11",
            "norm": "norm",
        },
        build=_timm("vit_small_patch14_dinov2.lvd142m"),
        notes="self-supervised; no ImageNet labels",
    )
)


# --------------------------------------------------------------------- open_clip
def _clip(arch: str, pretrained: str):
    def build():
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        model = model.eval()
        return model, preprocess

    return build


register(
    ModelSpec(
        name="clip_vitb32",
        family="clip",
        source="open_clip",
        layers={
            "block2": "visual.transformer.resblocks.2",
            "block5": "visual.transformer.resblocks.5",
            "block8": "visual.transformer.resblocks.8",
            "block11": "visual.transformer.resblocks.11",
            "ln_post": "visual.ln_post",
            "embed": "visual",
        },
        build=_clip("ViT-B-32", "openai"),
        notes="language-aligned embedding; used for zero-shot labels too",
    )
)

DEFAULT_MODELS = ["alexnet", "vgg16", "resnet18", "resnet50", "convnext_tiny", "vit_b_16", "dinov2_small", "clip_vitb32"]
