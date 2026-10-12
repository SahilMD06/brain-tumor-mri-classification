"""Prediction and Grad-CAM utilities shared by the notebook and the Streamlit app."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .preprocessing import IMG_SIZE, ImageSource, preprocess_image
from .utils import load_json


def load_trained_model(model_path: Path):
    """Load a saved Keras model for inference only (no optimizer state needed)."""
    import tensorflow as tf

    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return tf.keras.models.load_model(model_path, compile=False)


def load_class_names(path: Path) -> list[str]:
    """Load the index -> class-name mapping saved at training time."""
    class_names = load_json(path)
    if not isinstance(class_names, list) or not class_names:
        raise ValueError(f"{path} must contain a non-empty JSON list of class names.")
    return class_names


def predict_image(model, source: ImageSource, class_names: list[str],
                  img_size: tuple[int, int] = IMG_SIZE) -> dict:
    """Classify one image and return the label, confidence and all probabilities.

    ``confidence`` is the softmax probability of the predicted class. It measures
    how strongly the model prefers that class - it is not clinical certainty.
    """
    pixels = preprocess_image(source, img_size)
    probabilities = model.predict(pixels[np.newaxis, ...], verbose=0)[0]
    if probabilities.shape[0] != len(class_names):
        raise ValueError(
            f"Model outputs {probabilities.shape[0]} classes but "
            f"{len(class_names)} class names were supplied."
        )
    top_index = int(np.argmax(probabilities))
    return {
        "predicted_class": class_names[top_index],
        "confidence": float(probabilities[top_index]),
        "probabilities": {name: float(p) for name, p in zip(class_names, probabilities)},
    }


def find_last_conv_layer(model) -> str:
    """Name of the deepest layer that still outputs a spatial (4-D) feature map."""
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except AttributeError:
            continue
    raise ValueError("Model has no 4-D feature map; Grad-CAM is not applicable.")


def make_gradcam_heatmap(model, pixels: np.ndarray, class_index: int | None = None,
                         layer_name: str | None = None) -> np.ndarray:
    """Compute a Grad-CAM heatmap in [0, 1] for one preprocessed image.

    Grad-CAM weights each channel of the last convolutional feature map by the
    gradient of the class score with respect to that channel, then keeps only
    the positive evidence (ReLU). Bright areas influenced the prediction most.
    """
    import tensorflow as tf

    layer_name = layer_name or find_last_conv_layer(model)
    grad_model = tf.keras.Model(
        model.inputs, [model.get_layer(layer_name).output, model.output]
    )
    batch = tf.convert_to_tensor(pixels[np.newaxis, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        feature_maps, predictions = grad_model(batch, training=False)
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, feature_maps)
    channel_weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.nn.relu(tf.reduce_sum(feature_maps[0] * channel_weights, axis=-1))
    max_value = tf.reduce_max(heatmap)
    if max_value > 0:
        heatmap = heatmap / max_value
    return heatmap.numpy()


def overlay_heatmap(pixels: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a Grad-CAM heatmap over the image; returns float RGB in [0, 1]."""
    import matplotlib
    from PIL import Image

    height, width = pixels.shape[:2]
    resized = Image.fromarray(np.uint8(heatmap * 255)).resize((width, height), Image.BILINEAR)
    coloured = matplotlib.colormaps["jet"](np.asarray(resized) / 255.0)[..., :3]
    return np.clip((1 - alpha) * pixels + alpha * coloured, 0.0, 1.0)
