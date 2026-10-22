
from pathlib import Path
from datetime import datetime
import io

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "best.pt"

st.set_page_config(
    page_title="ParkVision AI | Smart Parking",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# STYLING
# ---------------------------------------------------------
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
        padding: 25px 28px;
        border: 1px solid #294355;
        border-radius: 20px;
        background: linear-gradient(120deg, #142c3c, #101c2a 70%, #122b2b);
        margin-bottom: 24px;
    }

    .eyebrow {
        color: #53e0bd;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
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
        font-size: 34px;
        font-weight: 700;
        margin-top: 8px;
    }

    .metric-note {
        color: #8ba3b4;
        font-size: 12px;
        margin-top: 2px;
    }

    .section-label {
        color: #52dfbd;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.7px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .insight-card {
        background: #11212e;
        border: 1px solid #294252;
        border-radius: 14px;
        padding: 18px;
        margin: 8px 0;
        color: #d9e5ee;
        line-height: 1.65;
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

    div[data-testid="stMetric"] {
        background: #122331;
        border: 1px solid #294252;
        border-radius: 14px;
        padding: 15px;
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 42px;
    }

    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------
@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def get_class_type(class_name: str, class_id: int):
    """Map model output names to our two parking-space classes."""
    name = class_name.lower().replace("_", "-").replace(" ", "-")

    if "occupied" in name:
        return "occupied"

    if "empty" in name:
        return "empty"

    # Fallback for the known dataset class IDs.
    if class_id == 0:
        return "empty"
    if class_id == 1:
        return "occupied"

    return "unknown"


def analyze_image(image_rgb, model, confidence, enhance):
    """Run YOLO and create annotated image, counts and detection rows."""
    input_rgb = image_rgb.copy()

    if enhance:
        # Optional contrast enhancement for difficult lighting.
        lab = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )
        lightness = clahe.apply(lightness)
        input_rgb = cv2.cvtColor(
            cv2.merge((lightness, channel_a, channel_b)),
            cv2.COLOR_LAB2RGB
        )

    result = model.predict(
        source=input_rgb,
        imgsz=640,
        conf=confidence,
        iou=0.7,
        verbose=False
    )[0]

    annotated = input_rgb.copy()
    counts = {"empty": 0, "occupied": 0, "unknown": 0}
    rows = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            score = float(box.conf.item())
            class_name = result.names[class_id]
            category = get_class_type(class_name, class_id)

            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [int(v) for v in xyxy]

            counts[category] += 1

            if category == "empty":
                color = (35, 220, 125)       # RGB green
                label = "EMPTY"
            elif category == "occupied":
                color = (255, 75, 85)        # RGB red
                label = "OCCUPIED"
            else:
                color = (255, 190, 70)
                label = "UNKNOWN"

            height, width = annotated.shape[:2]
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width - 1))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height - 1))

            cv2.rectangle(
                annotated, (x1, y1), (x2, y2), color, 2
            )

            # Compact label; avoid oversized text on small boxes.
            text = f"{label} {score:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.42
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(
                text, font, scale, thickness
            )

            label_top = y1 - text_h - baseline - 6
            if label_top < 0:
                label_top = min(y1 + 2, max(0, height - text_h - baseline - 5))

            label_bottom = min(height - 1, label_top + text_h + baseline + 6)
            label_right = min(width - 1, x1 + text_w + 8)

            cv2.rectangle(
                annotated,
                (x1, label_top),
                (label_right, label_bottom),
                color,
                -1
            )

            text_y = min(height - 2, label_top + text_h + 2)
            cv2.putText(
                annotated,
                text,
                (x1 + 3, text_y),
                font,
                scale,
                (8, 18, 25),
                thickness,
                cv2.LINE_AA
            )

            rows.append({
                "class": category,
                "model_class": class_name,
                "confidence": round(score, 4),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "box_width": x2 - x1,
                "box_height": y2 - y1,
            })

    total = counts["empty"] + counts["occupied"]
    occupancy = counts["occupied"] / total * 100 if total else 0.0

    return annotated, counts, total, occupancy, rows


def get_congestion(occupancy):
    if occupancy > 75:
        return "HIGH", "Limited availability. Consider directing drivers to another parking area."
    elif occupancy >= 40:
        return "MODERATE", "Some spaces remain. Guide incoming drivers toward available spaces."
    return "LOW", "Availability is relatively good. Direct drivers toward empty spaces."


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 🅿️ ParkVision AI")
    st.caption("SMART PARKING INTELLIGENCE")
    st.divider()

    st.markdown("### Detection settings")
    confidence = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Higher values keep only more confident predictions."
    )

    enhance = st.toggle(
        "Enhance image contrast",
        value=False,
        help="Apply CLAHE contrast enhancement before inference. Compare results carefully."
    )

    st.divider()
    st.markdown("### Box legend")
    st.markdown("🟢 **Empty space**")
    st.markdown("🔴 **Occupied space**")
    st.markdown("🟡 **Unrecognized class**")

    st.divider()
    st.caption("Model: YOLOv8 Nano")
    st.caption("Dataset: PKLot")
    st.caption("Inference: uploaded image")


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">COMPUTER VISION · SMART CITIES</div>
        <div class="hero-title">Parking intelligence,<br>at a glance.</div>
        <div class="hero-sub">
            Upload a parking-lot image to detect individual parking spaces,
            estimate occupancy, and generate actionable availability insights.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# CHECK MODEL
# ---------------------------------------------------------
if not MODEL_PATH.exists():
    st.error(
        f"Model not found at `{MODEL_PATH}`. "
        "Place your trained `best.pt` file inside the `models` folder."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error("The model could not be loaded. Check the model file and dependencies.")
    st.exception(exc)
    st.stop()


# ---------------------------------------------------------
# IMAGE INPUT
# ---------------------------------------------------------
left, right = st.columns([1.5, 1], gap="large")

with left:
    st.markdown('<div class="section-label">01 / Input image</div>',
                unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a parking-lot image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Use a clear image showing parking spaces from a visible viewpoint."
    )

    if uploaded_file is None:
        st.info("Upload an image to start the parking analysis.")
        st.markdown(
            """
            **For best results**
            - Use a clear, reasonably high-resolution image.
            - Keep parking spaces visible and avoid extreme blur.
            - Try both empty and occupied parking areas.
            """
        )

with right:
    st.markdown('<div class="section-label">02 / Model settings</div>',
                unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="insight-card">
            <b>Detection threshold</b><br>
            {confidence:.0%} confidence<br><br>
            <b>Image enhancement</b><br>
            {"Enabled" if enhance else "Disabled"}<br><br>
            <b>Output</b><br>
            Slot detections and occupancy insights
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# INFERENCE AND RESULTS
# ---------------------------------------------------------
if uploaded_file is not None:
    try:
        original_image = Image.open(uploaded_file)
        original_image = ImageOps.exif_transpose(original_image).convert("RGB")
        image_rgb = np.array(original_image)

        with st.spinner("Scanning parking spaces..."):
            annotated, counts, total, occupancy, rows = analyze_image(
                image_rgb, model, confidence, enhance
            )

        st.divider()
        st.markdown('<div class="section-label">03 / Parking overview</div>',
                    unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)

        metrics = [
            (m1, "TOTAL DETECTED", total, "Recognized parking spaces"),
            (m2, "EMPTY SPACES", counts["empty"], "Green boxes"),
            (m3, "OCCUPIED SPACES", counts["occupied"], "Red boxes"),
            (m4, "OCCUPANCY", f"{occupancy:.1f}%", "Occupied / classified spaces"),
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
                    unsafe_allow_html=True
                )

        if total == 0:
            st.warning(
                "No parking spaces were detected at this confidence threshold. "
                "Try lowering the threshold or uploading a clearer image."
            )
        else:
            congestion, recommendation = get_congestion(occupancy)

            st.markdown("")
            overview_left, overview_right = st.columns([1.15, 1])

            with overview_left:
                st.markdown('<div class="section-label">04 / Occupancy level</div>',
                            unsafe_allow_html=True)
                st.progress(min(1.0, max(0.0, occupancy / 100)))
                st.markdown(
                    f"**{occupancy:.1f}% occupied** · "
                    f"{100 - occupancy:.1f}% of detected spaces empty"
                )

            with overview_right:
                st.markdown('<div class="section-label">05 / Congestion status</div>',
                            unsafe_allow_html=True)
                st.markdown(
                    f'<span class="status-pill">{congestion} CONGESTION</span>',
                    unsafe_allow_html=True
                )
                st.markdown("")
                st.write(recommendation)

        st.divider()
        st.markdown('<div class="section-label">06 / Vision results</div>',
                    unsafe_allow_html=True)

        view_left, view_right = st.columns(2, gap="large")

        with view_left:
            st.markdown("**Original image**")
            st.image(original_image, use_container_width=True)

        with view_right:
            st.markdown("**ParkVision AI detections**")
            st.image(annotated, use_container_width=True)

        # Optional detection table and export.
        st.divider()
        st.markdown('<div class="section-label">07 / Detection report</div>',
                    unsafe_allow_html=True)

        if rows:
            report_df = pd.DataFrame(rows)
            st.dataframe(report_df, use_container_width=True, hide_index=True)

            csv_bytes = report_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download detection report (CSV)",
                data=csv_bytes,
                file_name=f"parkvision_report_{datetime.now():%Y%m%d_%H%M%S}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("There are no detections to include in the report.")

        st.caption(
            "Results are model predictions, not guaranteed real-time availability. "
            "Image quality, camera angle and confidence threshold can affect counts."
        )

    except Exception as exc:
        st.error("The image could not be processed. Try another image format or file.")
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
        st.write("YOLO identifies and classifies visible parking spaces.")
    with c3:
        st.markdown("#### 03 · Decide")
        st.write("Review occupancy estimates and availability insights.")

st.divider()
st.caption("PARKVISION AI · AI-assisted parking occupancy analysis")
