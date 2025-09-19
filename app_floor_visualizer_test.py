# app_floor_visualizer.py
import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import torch

from floorseg.segmentation import SAMRefiner
from floorseg.segmentation import (
    heuristic_floor_mask_bgr,
    load_binary_mask,
    get_floor_mask,
    FloorSegModel,
)
from floorseg.visualize import (
    apply_colour_overlay,
    apply_texture_overlay,
    overlay_mask_outline,
)
from floorseg.warp_and_blend import warp_and_blend_texture  # <-- NEW import

sam_refiner = SAMRefiner(checkpoint="sam_vit_b.pth")

st.set_page_config(page_title="Floor Color Visualizer", page_icon="🧰", layout="wide")
st.title("🧰 Floor Color Visualizer")
st.caption("Upload a room photo, detect the floor, and preview new colours or textures.")

col_input, col_opts = st.columns([1, 1])

# --- Image uploader ---
with col_input:
    up = st.file_uploader("Upload room image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    mask_up = st.file_uploader(
        "(Optional) Upload floor mask (white = floor, black = not-floor)",
        type=["png", "jpg", "jpeg"],
    )
    texture_up = st.file_uploader(
        "(Optional) Upload texture image (for texture mode)",
        type=["png", "jpg", "jpeg"],
    )

# --- Options ---
with col_opts:
    mode = st.radio("Overlay mode", ["Colour", "Texture", "Warp+Blend"], horizontal=True)
    if mode == "Colour":
        colour = st.color_picker("Pick a colour", value="#a37b4b")
        strength = st.slider("Colour strength (saturation blend)", 0.1, 1.0, 0.7, 0.05)
    elif mode == "Texture":
        st.info("Tip: use a seamless wood/tile texture for best results")
    else:
        st.info("Warp+Blend projects the texture realistically onto the detected floor.")

    seg_method = st.radio("Segmentation method", ["Heuristic", "Model"], horizontal=True)
    use_gpu = st.checkbox("Use GPU (CUDA)", value=False, disabled=not torch.cuda.is_available())
    show_outline = st.checkbox("Show detected floor outline", value=True)
    resize_preview = st.checkbox("Resize preview if image is large (>1280px)", value=True)

if up is None:
    st.info("Upload a room image to start.")
    st.stop()

# --- Read uploaded image ---
file_bytes = np.frombuffer(up.read(), np.uint8)
img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
if img_bgr is None:
    st.error("Could not decode the uploaded image. Try a different file.")
    st.stop()

# --- Compute / load floor mask ---
if mask_up is not None:
    try:
        mask_bytes = np.frombuffer(mask_up.read(), np.uint8)
        mask_img = cv2.imdecode(mask_bytes, cv2.IMREAD_UNCHANGED)
        floor_mask = load_binary_mask(mask_img)
    except Exception as e:
        st.warning(f"Failed to read custom mask, falling back to heuristic. ({e})")
        floor_mask = heuristic_floor_mask_bgr(img_bgr)
else:
    if seg_method == "Heuristic":
        with st.spinner("Detecting floor (heuristic) …"):
            floor_mask = get_floor_mask(img_bgr, method="heuristic")
    else:
        with st.spinner("Detecting floor (deep model) … this may take a while"):
            device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
            if "floor_model" not in st.session_state or st.session_state.floor_model.device != device:
                st.session_state.floor_model = FloorSegModel(device=device)
            floor_mask = get_floor_mask(
                img_bgr, method="model", model=st.session_state.floor_model
            )

# --- Refine with SAM if available ---
if floor_mask is not None and np.count_nonzero(floor_mask) > 0:
    try:
        sam_refiner = SAMRefiner(checkpoint="sam_vit_b.pth")
        floor_mask = sam_refiner.refine(img_bgr, floor_mask)
    except Exception as e:
        st.warning(f"SAM refinement skipped: {e}")

# --- Empty mask check ---
if floor_mask is None or np.count_nonzero(floor_mask) == 0:
    st.warning("Floor mask is empty — detection failed. Try uploading a manual mask or a clearer photo.")

# --- Prepare display versions (resize) ---
h, w = img_bgr.shape[:2]
max_w = 1280
scale = 1.0
if resize_preview and w > max_w:
    scale = max_w / w
    img_bgr_disp = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    mask_disp = cv2.resize(
        floor_mask,
        (img_bgr_disp.shape[1], img_bgr_disp.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
else:
    img_bgr_disp = img_bgr.copy()
    mask_disp = floor_mask.copy()

# --- Apply overlay based on mode ---
result_bgr = img_bgr.copy()
if mode == "Colour":
    # Convert hex to BGR tuple
    hex_val = colour.lstrip("#")
    r = int(hex_val[0:2], 16)
    g = int(hex_val[2:4], 16)
    b = int(hex_val[4:6], 16)
    colour_bgr = (b, g, r)
    result_bgr = apply_colour_overlay(img_bgr, floor_mask, colour_bgr, strength=strength)

elif mode == "Texture":
    if texture_up is None:
        st.warning("Upload a texture image to use Texture mode.")
        result_bgr = img_bgr.copy()
    else:
        tex_bytes = np.frombuffer(texture_up.read(), np.uint8)
        texture_bgr = cv2.imdecode(tex_bytes, cv2.IMREAD_COLOR)
        if texture_bgr is None:
            st.warning("Could not read texture file. Showing original image.")
            result_bgr = img_bgr.copy()
        else:
            result_bgr = apply_texture_overlay(img_bgr, floor_mask, texture_bgr)

elif mode == "Warp+Blend":
    if texture_up is None:
        st.warning("Upload a texture image to use Warp+Blend mode.")
        result_bgr = img_bgr.copy()
    else:
        tex_bytes = np.frombuffer(texture_up.read(), np.uint8)
        texture_bgr = cv2.imdecode(tex_bytes, cv2.IMREAD_COLOR)
        if texture_bgr is None:
            st.warning("Could not read texture file. Showing original image.")
            result_bgr = img_bgr.copy()
        else:
            result_bgr = warp_and_blend_texture(img_bgr, floor_mask, texture_bgr)  # NEW

# --- Display UI ---
left, right = st.columns(2)
with left:
    st.subheader("Original")
    if show_outline:
        out_orig = overlay_mask_outline(img_bgr_disp, mask_disp, colour=(0, 255, 255))
        st.image(
            cv2.cvtColor(out_orig, cv2.COLOR_BGR2RGB),
            caption="Original + Detected Floor",
            use_container_width=True,
        )
    else:
        st.image(
            cv2.cvtColor(img_bgr_disp, cv2.COLOR_BGR2RGB),
            caption="Original",
            use_container_width=True,
        )

with right:
    st.subheader("Result")
    if scale != 1.0:
        res_disp = cv2.resize(
            result_bgr,
            (img_bgr_disp.shape[1], img_bgr_disp.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    else:
        res_disp = result_bgr.copy()
    st.image(
        cv2.cvtColor(res_disp, cv2.COLOR_BGR2RGB),
        caption=f"{mode} preview",
        use_container_width=True,
    )

# --- Download final PNG ---
is_success, buffer = cv2.imencode(".png", result_bgr)
if is_success:
    st.download_button(
        "⬇️ Download PNG",
        data=buffer.tobytes(),
        file_name="floor_visualized.png",
        mime="image/png",
    )
else:
    st.error("Failed to encode result image for download.")
