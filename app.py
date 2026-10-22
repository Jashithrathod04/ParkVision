
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO


# =========================================================
# CONFIGURATION
# =========================================================
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "best.pt"

st.set_page_config(
    page_title="ParkVision AI | Smart Parking",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# PREMIUM DARK UI
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    .stApp {
        background:
            radial-gradient(ellipse at 10% 0%, #132a3b 0%, transparent 42%),
            #08111b;
        color: #e8f0f7;
        font-family: 'DM Sans', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #f4f8fc !important;
    }

    [data-testid="stSidebar"] {
        background: #0d1925;
        border-right: 1px solid #233746;
    }

    .hero {
        padding: 26px 28px;
        border: 1px solid #294355;
        border-radius: 20px;
        background: linear-gradient(120deg, #142c3c, #101c2a 70%, #122b2b);
        margin-bottom: 24px;
    }

    .eyebrow, .section-label {
        color: #53e0bd;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.7px;
        text-transform: uppercase;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(28px, 4vw, 42px);
        line-height: 1.15;
        font-weight: 700;
        margin: 10px 0;
        color: #f4f8fc;
    }

    .hero-sub {
        color: #a9bdcb;
        font-size: 15px;
        line-height: 1.7;
        max-width: 760px;
    }

    .metric-card {
        background: linear-gradient(145deg, #142636, #101d29);
        border: 1px solid #294252;
        border-radius: 16px;
        padding: 19px;
        min-height: 126px;
    }

    .metric-label {
        color: #9fb4c3;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    .metric-value {
        color: #f5f9fc;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 32px;
        font-weight: 700;
        margin-top: 8px;
    }

    .metric-note {
        color: #8ba3b4;
        font-size: 12px;
        margin-top: 4px;
    }

    .insight-card {
        background: #11212e;
        border: 1px solid #294252;
        border-radius: 14px;
        padding: 18px;
        margin: 8px 0;
        color: #d9e5ee;
        line-height: 1.8;
    }

    .status-pill {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 12px;
        letter-spacing: .4px;
        background: #173d32;
        color: #72edbd;
        border: 1px solid #28634f;
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 42px;
    }

    div[data-testid="stMetric"] {
        background: #122331;
        border: 1px solid #294252;
        border-radius: 14px;
        padding: 15px;
    }

    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# MODEL LOADING
# =========================================================
@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)


# =========================================================
# CLASS MAPPING
# =========================================================
def get_class_type(class_name, class_id):
    name = str(class_name).lower().replace("_", "-").replace(" ", "-")

    if "occupied" in name:
        return "occupied"

    if "empty" in name:
        return "empty"

    # Fallback for this project's known model class IDs.
    if class_id == 0:
        return "empty"

    if class_id == 1:
        return "occupied"

    return "unknown"


# =========================================================
# IMAGE ANALYSIS
# =========================================================
def analyze_image(image_rgb, model, confidence, enhance, image_size):
    """
    Run YOLO on an RGB image and return:
    annotated image, counts, total, occupancy, detection rows.
    """
    input_rgb = image_rgb.copy()

    if enhance:
        lab = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

        lightness = clahe.apply(lightness)

        input_rgb = cv2.cvtColor(
            cv2.merge((lightness, channel_a, channel_b)),
            cv2.COLOR_LAB2RGB,
        )

    # Higher resolution may help with small parking spaces.
    result = model.predict(
        source=input_rgb,
        imgsz=image_size,
        conf=confidence,
        iou=0.7,
        max_det=1000,
        verbose=False,
    )[0]

    annotated = input_rgb.copy()

    counts = {
        "empty": 0,
        "occupied": 0,
        "unknown": 0,
    }

    rows = []

    height, width = annotated.shape[:2]

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            score = float(box.conf.item())
            class_name = result.names[class_id]

            category = get_class_type(class_name, class_id)
            counts[category] += 1

            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords.tolist()

            # Keep coordinates within the original image.
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width - 1))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height - 1))

            if category == "empty":
                color = (35, 220, 125)  # Green in RGB
                label = "EMPTY"
            elif category == "occupied":
                color = (255, 75, 85)   # Red in RGB
                label = "OCCUPIED"
            else:
                color = (255, 190, 70)
                label = "UNKNOWN"

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            text = f"{label} {score:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.42
            thickness = 1

            (text_w, text_h), baseline = cv2.getTextSize(
                text, font, scale, thickness
            )

            label_h = text_h + baseline + 8
            label_w = text_w + 8

            # Put the label inside the image if there isn't
            # enough space above the detection.
            if y1 - label_h >= 0:
                label_top = y1 - label_h
            else:
                label_top = min(y1 + 2, height - label_h)

            label_top = max(0, label_top)
            label_bottom = min(height - 1, label_top + label_h)
            label_right = min(width - 1, x1 + label_w)

            cv2.rectangle(
                annotated,
                (x1, label_top),
                (label_right, label_bottom),
                color,
                -1,
            )

            text_y = min(
                height - 2,
                label_top + text_h + 4,
            )

            cv2.putText(
                annotated,
                text,
                (x1 + 3, text_y),
                font,
                scale,
                (8, 18, 25),
                thickness,
                cv2.LINE_AA,
            )

            rows.append({
                "class": category,
                "model_class": str(class_name),
                "confidence": round(score, 4),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "box_width": max(0, x2 - x1),
                "box_height": max(0, y2 - y1),
            })

    # Unknown detections are reported separately rather than
    # included in the classified-space occupancy calculation.
    total = counts["empty"] + counts["occupied"]

    occupancy = (
        counts["occupied"] / total * 100
        if total > 0
        else 0.0
    )

    return annotated, counts, total, occupancy, rows


# =========================================================
# CONGESTION INSIGHTS
# =========================================================
def get_congestion(occupancy):
    if occupancy > 75:
        return (
            "HIGH",
            "Limited availability. Consider directing drivers "
            "toward another parking area.",
        )

    if occupancy >= 40:
        return (
            "MODERATE",
            "Some spaces remain available. Guide incoming drivers "
            "toward detected empty spaces.",
        )

    return (
        "LOW",
        "Availability is relatively good based on the spaces detected.",
    )


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 🅿️ ParkVision AI")
    st.caption("SMART PARKING INTELLIGENCE")
    st.divider()

    st.markdown("### Detection settings")

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.90,
        value=0.15,
        step=0.05,
        help=(
            "Lower values retain weaker predictions but may "
            "increase false positives."
        ),
    )

    image_size = st.select_slider(
        "Inference resolution",
        options=[640, 960, 1280],
        value=1280,
        help=(
            "Higher resolutions may improve detection of small "
            "spaces but use more memory and processing time."
        ),
    )

    enhance = st.toggle(
        "Enhance image contrast",
        value=False,
        help=(
            "Apply CLAHE contrast enhancement. It can help some "
            "images and reduce accuracy on others."
        ),
    )

    st.divider()
    st.markdown("### Box legend")
    st.markdown("🟢 **Empty space**")
    st.markdown("🔴 **Occupied space**")
    st.markdown("🟡 **Unrecognized class**")

    st.divider()
    st.caption("Detector: YOLO")
    st.caption("Classes: space-empty / space-occupied")
    st.caption("Inference: uploaded image")


# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">COMPUTER VISION · SMART CITIES</div>
        <div class="hero-title">
            Parking intelligence,<br>at a glance.
        </div>
        <div class="hero-sub">
            Detect parking spaces, estimate occupancy and turn
            parking-lot images into actionable availability insights.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# CHECK MODEL
# =========================================================
if not MODEL_PATH.is_file():
    st.error(
        f"Model file not found: {MODEL_PATH}\n\n"
        "Place your trained best.pt file inside the models folder."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))

except Exception as exc:
    st.error("The YOLO model could not be loaded.")
    st.exception(exc)
    st.stop()

model_names = model.names

with st.expander("Model diagnostics", expanded=False):
    st.write("**Model task:**", model.task)
    st.write("**Model classes:**", model_names)
    st.write("**Inference resolution:**", image_size)
    st.write("**Confidence threshold:**", f"{confidence:.0%}")


# =========================================================
# IMAGE INPUT
# =========================================================
left, right = st.columns([1.5, 1], gap="large")

with left:
    st.markdown(
        '<div class="section-label">01 / Input image</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a parking-lot image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Use a clear image showing visible parking spaces.",
    )

    if uploaded_file is None:
        st.info("Upload an image to start the parking analysis.")

        st.markdown(
            """
            **For best results**
            - Use a clear, high-resolution image.
            - Keep parking spaces visible.
            - Avoid extreme blur and very distant views.
            - Test with an image similar to the training dataset.
            """
        )

with right:
    st.markdown(
        '<div class="section-label">02 / Analysis settings</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="insight-card">
            <b>Confidence threshold</b><br>
            {confidence:.0%}<br><br>
            <b>Inference resolution</b><br>
            {image_size} × {image_size}<br><br>
            <b>Image enhancement</b><br>
            {"Enabled" if enhance else "Disabled"}<br><br>
            <b>Detection output</b><br>
            Empty and occupied parking spaces
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# RUN ANALYSIS
# =========================================================
if uploaded_file is not None:
    try:
        original_image = Image.open(uploaded_file)
        original_image = ImageOps.exif_transpose(
            original_image
        ).convert("RGB")

        image_rgb = np.array(original_image)

        with st.spinner(
            f"Scanning parking spaces at {image_size}px resolution..."
        ):
            (
                annotated,
                counts,
                total,
                occupancy,
                rows,
            ) = analyze_image(
                image_rgb=image_rgb,
                model=model,
                confidence=confidence,
                enhance=enhance,
                image_size=image_size,
            )

        # ---------------------------------------------
        # OVERVIEW METRICS
        # ---------------------------------------------
        st.divider()

        st.markdown(
            '<div class="section-label">03 / Parking overview</div>',
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)

        metrics = [
            (
                m1,
                "TOTAL DETECTED",
                total,
                "Classified spaces",
            ),
            (
                m2,
                "EMPTY SPACES",
                counts["empty"],
                "Green boxes",
            ),
            (
                m3,
                "OCCUPIED SPACES",
                counts["occupied"],
                "Red boxes",
            ),
            (
                m4,
                "OCCUPANCY",
                f"{occupancy:.1f}%",
                "Of classified spaces",
            ),
        ]

        for column, label, value, note in metrics:
            with column:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value}</div>
                        <div class="metric-note">{note}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ---------------------------------------------
        # DIAGNOSTIC MESSAGES
        # ---------------------------------------------
        if total == 0:
            st.warning(
                "No classified parking spaces were detected. "
                "Try another image, check the camera viewpoint, "
                "or test a lower confidence threshold."
            )

            st.info(
                "If the model continues to detect zero or very few "
                "spaces at low confidence and high resolution, the "
                "problem may be the model's training or compatibility "
                "with this image. Lowering the threshold further "
                "cannot guarantee a fix."
            )

        elif total == 1:
            st.warning(
                "Only one classified space was detected. If the image "
                "contains many spaces, verify whether the model was "
                "trained on similar parking images."
            )

        if counts["unknown"] > 0:
            st.warning(
                f"{counts['unknown']} additional detections have "
                "unrecognized classes and are excluded from occupancy."
            )

        # ---------------------------------------------
        # OCCUPANCY AND CONGESTION
        # ---------------------------------------------
        if total > 0:
            congestion, recommendation = get_congestion(occupancy)

            st.markdown("")

            overview_left, overview_right = st.columns(
                [1.15, 1]
            )

            with overview_left:
                st.markdown(
                    '<div class="section-label">04 / Occupancy level</div>',
                    unsafe_allow_html=True,
                )

                st.progress(
                    min(1.0, max(0.0, occupancy / 100))
                )

                empty_percentage = (
                    counts["empty"] / total * 100
                )

                st.markdown(
                    f"**{occupancy:.1f}% occupied** · "
                    f"{empty_percentage:.1f}% empty"
                )

            with overview_right:
                st.markdown(
                    '<div class="section-label">05 / Congestion status</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f'<span class="status-pill">'
                    f'{congestion} CONGESTION</span>',
                    unsafe_allow_html=True,
                )

                st.write(recommendation)

        # ---------------------------------------------
        # SIDE-BY-SIDE VISION RESULTS
        # ---------------------------------------------
        st.divider()

        st.markdown(
            '<div class="section-label">06 / Vision results</div>',
            unsafe_allow_html=True,
        )

        view_left, view_right = st.columns(
            2,
            gap="large",
        )

        with view_left:
            st.markdown("**Original image**")

            st.image(
                original_image,
                use_container_width=True,
            )

        with view_right:
            st.markdown("**ParkVision AI detections**")

            st.image(
                annotated,
                use_container_width=True,
            )

        # ---------------------------------------------
        # DETECTION REPORT
        # ---------------------------------------------
        st.divider()

        st.markdown(
            '<div class="section-label">07 / Detection report</div>',
            unsafe_allow_html=True,
        )

        if rows:
            report_df = pd.DataFrame(rows)

            st.dataframe(
                report_df,
                use_container_width=True,
                hide_index=True,
            )

            csv_bytes = report_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download detection report (CSV)",
                data=csv_bytes,
                file_name=(
                    f"parkvision_report_"
                    f"{datetime.now():%Y%m%d_%H%M%S}.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )

        else:
            st.info(
                "No detections to include in the report."
            )

        st.caption(
            "Predictions are estimates, not guaranteed real-time "
            "parking availability. Image quality, model training, "
            "viewpoint and confidence threshold affect results."
        )

    except Exception as exc:
        st.error(
            "The uploaded image could not be processed. "
            "Try another image or check the error below."
        )
        st.exception(exc)

else:
    st.divider()

    st.markdown("### How it works")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### 01 · Upload")
        st.write("Choose a parking-lot photograph.")

    with c2:
        st.markdown("#### 02 · Detect")
        st.write(
            "YOLO identifies visible empty and occupied spaces."
        )

    with c3:
        st.markdown("#### 03 · Decide")
        st.write(
            "Review occupancy estimates and export the report."
        )


# =========================================================
# FOOTER
# =========================================================
st.divider()

st.caption(
    "PARKVISION AI · AI-ASSISTED PARKING OCCUPANCY ANALYSIS"
)
