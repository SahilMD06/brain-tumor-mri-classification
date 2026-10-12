"""Streamlit application: Brain Tumor MRI Image Classification.

Run from the repository root:

    streamlit run app/app.py

The app only performs inference. It loads the model selected in the notebook
(``models/final_model.h5``) once, and preprocesses uploads with the same
function that was used for training and evaluation.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import (load_class_names, load_trained_model,  # noqa: E402
                           make_gradcam_heatmap, overlay_heatmap)
from src.preprocessing import (IMG_SIZE, SUPPORTED_EXTENSIONS,  # noqa: E402
                               InvalidImageError, normalize, open_image)
from src.utils import load_json  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_CANDIDATES = (MODELS_DIR / "final_model.h5", MODELS_DIR / "final_model.keras")
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
CONFIG_PATH = MODELS_DIR / "preprocessing_config.json"
SAMPLES_DIR = PROJECT_ROOT / "assets" / "sample_images"

MAX_UPLOAD_MB = 10
DEFAULT_LOW_CONFIDENCE = 0.70
COLOUR_DIFFERENCE_LIMIT = 40   # channel difference above which an image is clearly a colour picture
UPLOAD_TYPES = [extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS]

DISCLAIMER = (
    "**Medical disclaimer.** This is an academic prototype for education and research. "
    "It is not a medical device, has not been clinically validated, and must not be used "
    "to diagnose, treat or rule out any condition. Always consult a qualified medical professional."
)

logger = logging.getLogger("brain_mri_app")


class ResourceError(RuntimeError):
    """A file the app needs is missing or unusable; the message is safe to show to users."""


@st.cache_resource(show_spinner="Loading the model...")
def load_resources():
    """Load the model, class names and settings once per server process."""
    model_path = next((path for path in MODEL_CANDIDATES if path.is_file()), None)
    if model_path is None:
        raise ResourceError(
            "No trained model was found in the `models/` folder. Run the notebook "
            "`notebook/Brain_Tumor_MRI_Classification.ipynb` to train and save `final_model.h5`."
        )
    if not CLASS_NAMES_PATH.is_file():
        raise ResourceError(
            "`models/class_names.json` is missing. Run the notebook to regenerate it; "
            "without it the app cannot name the predicted class."
        )
    try:
        model = load_trained_model(model_path)
        class_names = load_class_names(CLASS_NAMES_PATH)
    except Exception as exc:  # details go to the server log, not to the user
        logger.exception("Failed to load model resources")
        raise ResourceError(
            "The model files exist but could not be loaded. They may be corrupted or were saved "
            "with an incompatible TensorFlow version. Re-run the notebook to recreate them."
        ) from exc

    if model.output_shape[-1] != len(class_names):
        raise ResourceError(
            f"The model predicts {model.output_shape[-1]} classes but `class_names.json` lists "
            f"{len(class_names)}. Re-run the notebook so both files come from the same training run."
        )
    config = load_json(CONFIG_PATH) if CONFIG_PATH.is_file() else {}
    return model, class_names, config, model_path.name


def pretty(class_name: str) -> str:
    return class_name.replace("_", " ").title()


def read_image(source) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return (display RGB array, model input in [0, 1], looks_like_colour_photo)."""
    image = open_image(source)
    original = np.asarray(image)
    channel_gap = np.abs(original[..., 0].astype(np.int16) - original[..., 1].astype(np.int16)).max()
    height, width = IMG_SIZE
    resized = np.asarray(image.resize((width, height)), dtype=np.uint8)   # bilinear, as in training
    return original, normalize(resized), bool(channel_gap > COLOUR_DIFFERENCE_LIMIT)


def render_sidebar(class_names: list[str], config: dict, model_file: str) -> tuple[bool, float]:
    with st.sidebar:
        st.header("About")
        st.write(
            "Classifies a single 2-D brain MRI slice with a convolutional neural network "
            "trained on a public dataset."
        )
        st.markdown(
            f"- **Model:** {config.get('selected_model', 'trained CNN')} (`{model_file}`)\n"
            f"- **Classes:** {', '.join(pretty(name) for name in class_names)}\n"
            f"- **Input:** {IMG_SIZE[0]} x {IMG_SIZE[1]} px, RGB, pixel values scaled to 0-1"
        )
        st.header("Options")
        show_gradcam = st.checkbox("Show Grad-CAM heatmap", value=True,
                                   help="Highlights the regions that most influenced the prediction.")
        threshold = float(config.get("low_confidence_threshold", DEFAULT_LOW_CONFIDENCE))
        st.caption(f"Predictions below {threshold:.0%} confidence are flagged for human review.")
        st.divider()
        st.caption(DISCLAIMER)
    return show_gradcam, threshold


def choose_image():
    """Upload widget plus optional bundled sample images. Returns (source, display name)."""
    uploaded = st.file_uploader("Upload a brain MRI image", type=UPLOAD_TYPES,
                                help=f"Supported formats: {', '.join(UPLOAD_TYPES)}. Maximum {MAX_UPLOAD_MB} MB.")
    if uploaded is not None:
        if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"The file is larger than {MAX_UPLOAD_MB} MB. Please upload a smaller image.")
            return None, None
        return uploaded, uploaded.name

    samples = sorted(path for path in SAMPLES_DIR.glob("*") if path.suffix.lower() in SUPPORTED_EXTENSIONS)
    if samples:
        choice = st.selectbox("...or try a sample image from the test set",
                              ["(none)"] + [path.name for path in samples])
        if choice != "(none)":
            return SAMPLES_DIR / choice, choice
    return None, None


def render_prediction(model, class_names, pixels, threshold: float) -> int:
    probabilities = model.predict(pixels[np.newaxis, ...], verbose=0)[0]
    top_index = int(np.argmax(probabilities))
    confidence = float(probabilities[top_index])

    st.subheader("Prediction")
    left, right = st.columns(2)
    left.metric("Predicted class", pretty(class_names[top_index]))
    right.metric("Confidence", f"{confidence:.1%}")

    if confidence < threshold:
        st.warning(
            f"Low confidence (below {threshold:.0%}). The model is unsure about this image; "
            "the result should be reviewed by a qualified professional."
        )

    table = (pd.DataFrame({"Class": [pretty(name) for name in class_names], "Probability": probabilities})
             .sort_values("Probability", ascending=False).reset_index(drop=True))
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={"Probability": st.column_config.ProgressColumn(
            "Probability", format="percent", min_value=0.0, max_value=1.0)},
    )
    st.caption(
        "Confidence is the model's softmax probability for the predicted class. "
        "It is not a measure of medical certainty."
    )
    return top_index


def main() -> None:
    st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="wide")
    st.title("Brain Tumor MRI Image Classification")
    st.write("Upload a brain MRI slice to get the predicted category and the model's confidence.")
    st.info(DISCLAIMER)

    try:
        model, class_names, config, model_file = load_resources()
    except ResourceError as exc:
        st.error(str(exc))
        st.stop()

    show_gradcam, threshold = render_sidebar(class_names, config, model_file)
    source, name = choose_image()
    if source is None:
        st.caption("Waiting for an image.")
        return

    try:
        original, pixels, looks_colour = read_image(source)
    except InvalidImageError:
        st.error("This file could not be read as an image. It may be corrupted or in an unsupported "
                 f"format. Please upload a valid {', '.join(UPLOAD_TYPES)} image.")
        return
    except Exception:  # unexpected failure: log the details, show a neutral message
        logger.exception("Unexpected error while reading an upload")
        st.error("Something went wrong while reading the image. Please try a different file.")
        return

    if looks_colour:
        st.warning("This image contains colour. Brain MRI slices are grayscale, so this may not be an MRI. "
                   "The model cannot recognise unrelated images and will still output one of its classes.")

    image_column, result_column = st.columns([1, 1.2])
    with image_column:
        st.subheader("Uploaded image")
        st.image(original, caption=f"{name} ({original.shape[1]} x {original.shape[0]} px)",
                 width="stretch")

    try:
        with result_column:
            with st.spinner("Analysing..."):
                top_index = render_prediction(model, class_names, pixels, threshold)
            if show_gradcam:
                heatmap = make_gradcam_heatmap(model, pixels, class_index=top_index)
                st.subheader("Grad-CAM")
                st.image(overlay_heatmap(pixels, heatmap), width="stretch",
                         caption="Warm colours mark regions that most influenced this prediction. "
                                 "This is an indication, not a tumor outline.")
    except Exception:
        logger.exception("Unexpected error during prediction")
        st.error("The prediction could not be completed. Please try again or use a different image.")


if __name__ == "__main__":
    main()
