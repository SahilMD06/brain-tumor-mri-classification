"""Image loading and preprocessing shared by training and inference.

Contract used everywhere in the project
---------------------------------------
1. Decode the file and convert it to 3-channel RGB.
2. Resize to ``IMG_SIZE`` with bilinear interpolation.
3. Scale pixel values from [0, 255] to [0, 1] (float32).

Every saved model expects exactly this [0, 1] RGB tensor. Any
architecture-specific rescaling (e.g. MobileNetV2's [-1, 1] range) lives
*inside* the model graph, so the notebook, the Streamlit app and any other
client can never apply mismatched preprocessing.
"""
from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Union

import numpy as np
from PIL import Image, UnidentifiedImageError

IMG_SIZE: tuple[int, int] = (224, 224)  # (height, width) expected by the models
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")
MIN_IMAGE_SIDE: int = 32  # anything smaller carries too little anatomy to classify

ImageSource = Union[str, Path, BinaryIO, Image.Image]


class InvalidImageError(ValueError):
    """Raised when a file cannot be decoded into a usable image."""


def open_image(source: ImageSource) -> Image.Image:
    """Safely decode ``source`` into an RGB PIL image.

    ``Image.open`` is lazy, so ``load()`` is called to force full decoding and
    surface truncated / corrupted files here rather than deep inside training.
    """
    try:
        image = source if isinstance(source, Image.Image) else Image.open(source)
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(f"Could not decode image: {exc}") from exc

    if min(image.size) < MIN_IMAGE_SIDE:
        raise InvalidImageError(
            f"Image is too small ({image.size[0]}x{image.size[1]} px); "
            f"minimum side is {MIN_IMAGE_SIDE} px."
        )
    # MRI slices are grayscale, but ImageNet backbones need 3 channels; RGB
    # conversion also normalises palette / RGBA / 16-bit inputs to one format.
    return image.convert("RGB")


def load_and_resize(source: ImageSource, img_size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Return a uint8 array of shape (H, W, 3) resized to ``img_size``."""
    image = open_image(source)
    height, width = img_size
    image = image.resize((width, height), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def normalize(pixels: np.ndarray) -> np.ndarray:
    """Scale uint8 pixels to float32 in [0, 1]."""
    return pixels.astype(np.float32) / 255.0


def preprocess_image(source: ImageSource, img_size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Full inference preprocessing: decode -> RGB -> resize -> [0, 1] float32."""
    return normalize(load_and_resize(source, img_size))
