from pathlib import Path
import hashlib
import importlib
import sys
from html import escape

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# The two existing projects remain the sources for matcher logic and model weights.
APP_ROOT = Path(__file__).resolve().parent
MATCHER_ROOT = APP_ROOT
INSPECTOR_ROOT = APP_ROOT
sys.path.insert(0, str(MATCHER_ROOT))
import matcher as matcher_module
matcher_module = importlib.reload(matcher_module)
HybridImageMatcher = matcher_module.HybridImageMatcher

MODEL_PATHS = {
    "YOLO26n": INSPECTOR_ROOT / "models" / "yolo26n_cls_best.pt",
    "YOLO11n": INSPECTOR_ROOT / "models" / "yolo11n_cls_best.pt",
}
MATCHER_VERSION = "qr-candidate-then-background-v18"

CLASS_TO_RULE = {
    "Shop ": ("Shop", "Manually verify that the plate location and mounting surface meet the approved requirements."),
    "GAP BETWEEN PLATE AND WALL": ("Gap between plate and wall", "Make sure the plate sits flush against the wall."),
    "Nails Not Proper": ("Only one nail is holding the QR code plate", "Reject the image; secure the plate with at least two properly fixed nails."),
    "Rescan Scanify Image": ("Rescan properly Scanify image", "Retake a clear Scanify image."),
    "Stick plate properly": ("Put plate properly on the wall", "Fix the plate securely and properly on the wall."),
    "Write O for owner and R1, R2 for Rent": ("In property image there is two plates so mention R1 and R2 in Plates", "Mark owner plates as O and rental plates as R1 or R2, as applicable."),
}

st.set_page_config(page_title="QR Plate Matching + Inspector", layout="wide")
st.title("QR Plate Matching + Inspector")
st.write("Match the property photo against the reference plate. After an ACCEPT, the inspector checks the reference plate image.")


@st.cache_resource
def load_matcher(matcher_version):
    return HybridImageMatcher()


@st.cache_resource
def load_model(model_path, model_modified_time):
    return YOLO(str(model_path))


def normalize(text):
    return " ".join(str(text).strip().split()).casefold()


def manual_review(class_name):
    normalized = normalize(class_name)
    return normalized == "stick plate properly" or any(x in normalized for x in ("shop", "store", "merchant", "metal"))


def predict_one_model(model, image):
    result = model.predict(source=image, device="cpu", verbose=False)[0]
    probabilities = result.probs.data.detach().cpu().numpy().astype(float)
    return {str(result.names[i]): float(probabilities[i]) for i in range(len(probabilities))}


def inspect_image(image, models):
    predictions = {}
    for name, path in models:
        model = load_model(path, path.stat().st_mtime_ns)
        predictions[name] = predict_one_model(model, image)
    classes = sorted(set().union(*(scores.keys() for scores in predictions.values())))
    averages = {c: float(np.mean([scores.get(c, 0.0) for scores in predictions.values()])) for c in classes}
    ranked = sorted(averages.items(), key=lambda item: item[1], reverse=True)
    winners = [max(scores.items(), key=lambda item: item[1])[0] for scores in predictions.values()]
    highest = ranked[0][0]
    nail_review = normalize(highest) == "accept+" and any(normalize(c) == "nails not proper" for c in winners)
    return {"predictions": predictions, "ranked_scores": ranked, "highest_class": highest,
            "highest_score": ranked[0][1], "agreement": len(set(winners)) == 1,
            "nail_manual_review": nail_review}


def show_inspection(image, result):
    st.subheader("Inspector result — reference plate image")
    st.image(image, caption="Reference plate image inspected", use_container_width=True)
    chart = pd.DataFrame(result["ranked_scores"], columns=["Predicted class", "Score"]).set_index("Predicted class")
    st.markdown("### Rule prediction scores")
    st.bar_chart(chart, y="Score", height=300)
    st.markdown("### Individual model predictions")
    for name, scores in result["predictions"].items():
        predicted, confidence = max(scores.items(), key=lambda item: item[1])
        st.write(f"**{name}:** {predicted} ({confidence:.1%})")
    class_name = result["highest_class"]
    is_manual = result["nail_manual_review"] or manual_review(class_name)
    if result["nail_manual_review"]:
        st.markdown('<p style="color:#b45309;font-size:1.4rem;font-weight:800">REQUIRED MANUAL REVIEW</p>', unsafe_allow_html=True)
        st.write("**Manual review item:** Confirm the plate is held by at least two properly fixed nails.")
    elif normalize(class_name) == "accept+":
        st.markdown('<p style="color:#15803d;font-size:1.4rem;font-weight:800">ACCEPT</p>', unsafe_allow_html=True)
    elif is_manual:
        st.markdown('<p style="color:#b45309;font-size:1.4rem;font-weight:800">REQUIRED MANUAL REVIEW</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<p style="color:#b91c1c;font-size:1.4rem;font-weight:800">REJECT: {escape(class_name)}</p>', unsafe_allow_html=True)
    st.write(f"**Highest score:** {result['highest_score']:.1%} | **Model agreement:** {result['agreement']}")
    for dataset_class, (reason, suggestion) in CLASS_TO_RULE.items():
        if normalize(dataset_class) == normalize(class_name):
            st.write(f"**{'Manual review item' if is_manual else 'Rejection reason'}:** {reason}")
            st.write(f"**Suggestion:** {suggestion}")
            break


st.header("1. QR plate matching")
reference_file = st.file_uploader("Reference plate image", type=["jpg", "jpeg", "png", "webp"], key="reference")
property_file = st.file_uploader("Property image (used for matching only)", type=["jpg", "jpeg", "png", "webp", "bmp"], key="property")
prompt = st.text_input("Optional region hint", "plate . sign . house number .")

if reference_file and property_file:
    reference_image = Image.open(reference_file).convert("RGB")
    property_image = Image.open(property_file).convert("RGB")
    left, right = st.columns(2)
    left.image(reference_image, caption="Reference plate", use_container_width=True)
    right.image(property_image, caption="Property image", use_container_width=True)
    # Cache the decision per uploaded image pair and region hint. Streamlit reruns
    # the script on UI changes; this avoids repeating the matcher or inspector.
    run_key = (
        MATCHER_VERSION,
        hashlib.sha256(reference_file.getvalue()).hexdigest(),
        hashlib.sha256(property_file.getvalue()).hexdigest(),
        prompt,
    )
    if st.session_state.get("last_run_key") != run_key:
        st.session_state.pop("last_run_key", None)
        st.session_state.pop("match_result", None)
        st.session_state.pop("inspection_result", None)
        with st.spinner("Checking whether the reference plate is present…"):
            try:
                st.session_state.match_result = load_matcher(MATCHER_VERSION).verify_presence(
                    reference_image, property_image, target_prompt=prompt
                )
                st.session_state.last_run_key = run_key
            except Exception as exc:
                st.error(f"Could not analyze these images: {exc}")
                st.stop()

    match_result = st.session_state.get("match_result", {})
    if not match_result.get("contains_reference"):
        if match_result.get("needs_manual_review"):
            st.warning(f"REQUIRED MANUAL REVIEW — {match_result.get('manual_review_reason', 'The images are unclear or the match is uncertain.')} The inspector was not run.")
        elif match_result.get("detection_method") == "QR background context mismatch":
            similarity = match_result.get("background_similarity")
            detail = f" (background similarity {similarity:.0%})" if similarity is not None else ""
            st.error(f"REJECT — The QR plate's surrounding background does not match the property image{detail}. The inspector was not run.")
        elif match_result.get("detection_method") == "QR symbol not localized; use clearer original images":
            st.warning("REJECT — The QR plate could not be localized well enough to check its surrounding background. Upload the original, higher-resolution images; the inspector was not run.")
        elif match_result.get("best_template_score") is not None:
            st.error(f"REJECT — The QR plate and surrounding background match was too weak ({match_result['best_template_score']:.0%}). The inspector was not run.")
        else:
            st.error("REJECT — Reference plate and its background were not confirmed in the property image. The inspector was not run.")
        st.stop()

    st.success("ACCEPT — Reference plate is present. Continuing to the inspector with the reference plate image.")
    if match_result.get("background_similarity") is not None:
        st.caption(f"QR plate and surrounding background similarity: {match_result['background_similarity']:.0%}")
    available_models = [(name, path) for name, path in MODEL_PATHS.items() if path.exists()]
    if not available_models:
        st.error("Plate match passed, but no inspector model weights were found. Check the model files in the inspector project.")
        st.stop()
    if "inspection_result" not in st.session_state:
        with st.spinner("Inspecting the reference plate image…"):
            try:
                st.session_state.inspection_result = inspect_image(reference_image, available_models)
            except Exception as exc:
                st.error(f"The plate match passed, but the inspector could not analyze the reference image: {exc}")
                st.stop()
    st.divider()
    st.header("2. QR plate inspector")
    show_inspection(reference_image, st.session_state.inspection_result)
else:
    st.info("Upload the reference plate and property image. Matching starts automatically; the reference plate is sent to the inspector only after the matcher accepts.")
