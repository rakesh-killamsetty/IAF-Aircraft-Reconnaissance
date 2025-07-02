import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import torch
import cv2
import os
from PIL import ImageDraw
import matplotlib.pyplot as plt
from skimage.transform import resize
from streamlit import columns
from segment_anything import SamAutomaticMaskGenerator

# --- Streamlit page config and title ---
st.set_page_config(page_title="Aerial Pattern Detection Prototype", layout="wide")
st.title("Pattern-Based Object Detection Prototype")

# --- Device selection for GPU/CPU ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.write(f"Using device: {device}")

# --- Step 1: Upload Images ---
# Upload and display pattern and query images for annotation and detection.
st.sidebar.header("Step 1: Upload Images")
pattern_file = st.sidebar.file_uploader("Upload Pattern Image (Support)", type=["png", "jpg", "jpeg"])
query_file = st.sidebar.file_uploader("Upload Query Image (Scene)", type=["png", "jpg", "jpeg"])

col1, col2 = st.columns(2)
pattern_img = None
query_img = None

with col1:
    st.subheader("Pattern Image (Support)")
    if pattern_file:
        pattern_img = Image.open(pattern_file).convert("RGB")
        st.image(pattern_img, caption="Pattern Image", use_column_width=True)
    else:
        st.info("Please upload a pattern image.")

with col2:
    st.subheader("Query Image (Scene)")
    if query_file:
        query_img = Image.open(query_file).convert("RGB")
        st.image(query_img, caption="Query Image", use_column_width=True)
    else:
        st.info("Please upload a query image.")

st.markdown("---")
st.header("Step 2: Annotate Pattern Image")

# --- SAM Integration ---
@st.cache_resource(show_spinner=True)
def load_sam_model(checkpoint_path="./sam_vit_h.pth"):
    # Load the SAM model for segmentation
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
    sam.to(device)
    predictor = SamPredictor(sam)
    return predictor

def run_sam_on_bboxes(image_pil, bboxes, predictor):
    # Run SAM on bounding boxes to get masks
    image = np.array(image_pil)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    predictor.set_image(image_bgr)
    masks = []
    for bbox in bboxes:
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        input_box = np.array([x, y, x + w, y + h])
        mask, _, _ = predictor.predict(box=input_box[None, :], multimask_output=True)
        mask_areas = [np.sum(m) for m in mask]
        idx = int(np.argmax(mask_areas))
        masks.append(mask[idx])
    return masks

def overlay_mask_on_image(image_pil, mask, color=(255, 0, 0), alpha=0.4):
    # Overlay a mask on the image for visualization
    import numpy as np
    from PIL import Image
    image = image_pil.convert("RGBA")
    mask=(mask>0).astype(np.uint8)
    mask_img=Image.fromarray(mask*255).resize(image.size)
    color_img=Image.new("RGBA",image.size,color+(0,))
    mask_alpha=Image.fromarray((mask*int(255*alpha)).astype(np.uint8)).resize(image.size)
    color_img.putalpha(mask_alpha)
    blended=Image.alpha_composite(image,color_img)
    return blended

# --- DINOv2 Integration ---
@st.cache_resource(show_spinner=True)
def load_dinov2_model():
    # Load the DINOv2 model for embedding extraction
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    model = AutoModel.from_pretrained("facebook/dinov2-large")
    model.to(device)
    return model, processor

def extract_object_region(image_pil, mask):
    # Crop the segmented object region from the image using the mask
    import numpy as np
    mask = (mask > 0).astype(np.uint8)
    image_np = np.array(image_pil)
    object_pixels = image_np * mask[:, :, None]
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = object_pixels[y0:y1, x0:x1]
    if cropped.shape[2] == 4:
        cropped = cropped[:, :, :3]
    return Image.fromarray(cropped)

def get_dinov2_embedding(image_pil, model, processor):
    # Get the DINOv2 embedding for a given image region
    import torch
    inputs = processor(images=image_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy().flatten()
    return embedding

def cosine_similarity(a, b):
    # Compute cosine similarity between two vectors
    import numpy as np
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))

def l2_distance(a, b):
    # Compute L2 (Euclidean) distance between two vectors
    import numpy as np
    return float(np.linalg.norm(a - b))

def sliding_window_similarity(query_img, pattern_embedding, model, processor, window_size=64, stride=32):
    # Slide a window over the query image, compute DINOv2 embedding for each patch, and build a similarity map
    import numpy as np
    from PIL import Image, ImageDraw
    W, H = query_img.size
    min_dist = float('inf')
    best_box = None
    similarity_map = np.zeros(((H - window_size) // stride + 1, (W - window_size) // stride + 1))
    for i, y in enumerate(range(0, H - window_size + 1, stride)):
        for j, x in enumerate(range(0, W - window_size + 1, stride)):
            patch = query_img.crop((x, y, x + window_size, y + window_size))
            patch_emb = get_dinov2_embedding(patch, model, processor)
            dist = 1 - cosine_similarity(pattern_embedding, patch_emb)
            similarity_map[i, j] = dist
            if dist < min_dist:
                min_dist = dist
                best_box = (x, y, x + window_size, y + window_size)
    return similarity_map, best_box, min_dist

def nms(boxes, scores, iou_threshold=0.3):
    # Non-Maximum Suppression to filter overlapping boxes
    # boxes: list of (x1, y1, x2, y2), scores: list of float (lower is better)
    import numpy as np
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()  # ascending (lower distance is better)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

# --- Pattern Image Annotation and SAM Segmentation ---
if pattern_img:
    st.write("Draw shapes around objects of interest in the pattern image.")
    drawing_mode = st.selectbox(
        "Select annotation shape",
        ["rect", "circle", "polygon", "freedraw", "line", "transform"],
        format_func=lambda x: {
            "rect": "Rectangle",
            "circle": "Circle/Ellipse",
            "polygon": "Polygon (e.g., rhombus)",
            "freedraw": "Free Draw",
            "line": "Line",
            "transform": "Move/Resize"
        }[x],
        index=0
    )
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",  # Red with alpha
        stroke_width=2,
        background_image=pattern_img,
        update_streamlit=True,
        height=pattern_img.height,
        width=pattern_img.width,
        drawing_mode=drawing_mode,
        key="canvas",
    )
    if canvas_result.json_data:
        st.subheader("Annotation Data (Raw JSON)")
        st.json(canvas_result.json_data)

        # --- Extract and display shape data in a user-friendly way ---
        shapes = canvas_result.json_data.get("objects", [])
        shape_list = []
        for obj in shapes:
            shape_type = obj.get("type")
            if shape_type == "rect":
                shape_list.append({
                    "type": "rect",
                    "x": int(obj["left"]),
                    "y": int(obj["top"]),
                    "width": int(obj["width"]),
                    "height": int(obj["height"])
                })
            elif shape_type == "circle":
                rx = obj.get("rx")
                ry = obj.get("ry")
                if rx is not None and ry is not None:
                    shape_list.append({
                        "type": "circle",
                        "center_x": int(obj["left"] + rx),
                        "center_y": int(obj["top"] + ry),
                        "radius_x": int(rx),
                        "radius_y": int(ry)
                    })
                else:
                    shape_list.append({"type": "circle", **obj})
            elif shape_type == "polygon":
                shape_list.append({
                    "type": "polygon",
                    "points": obj.get("path", [])
                })
            elif shape_type == "line":
                shape_list.append({
                    "type": "line",
                    "start": obj.get("x1", None),
                    "end": obj.get("x2", None)
                })
            elif shape_type == "path":  # Free draw
                shape_list.append({
                    "type": "freedraw",
                    "path": obj.get("path", [])
                })
            else:
                shape_list.append({"type": shape_type, **obj})

        st.markdown("**Extracted Shapes:**")
        st.write(shape_list)

        # --- Extract rectangles for SAM ---
        rect_bboxes = [s for s in shape_list if s["type"] == "rect"]
        if rect_bboxes:
            st.markdown("---")
            st.subheader("Find Major Object Region with SAM")
            if st.button("Run SAM Segmentation on Bounding Boxes"):
                # Run SAM on the annotated bounding boxes
                with st.spinner("Loading SAM model and segmenting..."):
                    predictor = load_sam_model()
                    masks = run_sam_on_bboxes(pattern_img, rect_bboxes, predictor)
                    mask_areas = [np.sum(m) for m in masks]
                    if mask_areas:
                        major_idx = int(np.argmax(mask_areas))
                        major_mask = masks[major_idx]
                        st.success(f"Major object region is in bounding box #{major_idx+1} (area: {mask_areas[major_idx]})")
                        overlay_img = overlay_mask_on_image(pattern_img, major_mask)
                        st.image(overlay_img, caption="Major Object Region (SAM Mask)", use_column_width=True)
                        # Extract and store the segmented region in session state
                        object_img = extract_object_region(pattern_img, major_mask)
                        if object_img is not None:
                            st.session_state["object_img"] = object_img
                            st.image(object_img, caption="Segmented Object Passed to DINOv2")
                        else:
                            st.warning("Could not extract object region from mask.")
                    else:
                        st.warning("No masks found.")
            # Always display the segmented region if available
            if "object_img" in st.session_state:
                st.image(st.session_state["object_img"], caption="Segmented Object Passed to DINOv2")
                if st.button("Extract Object Embedding (DINOv2)"):
                    # Extract feature embedding for the segmented object using DINOv2
                    model, processor = load_dinov2_model()
                    pattern_embedding = get_dinov2_embedding(st.session_state["object_img"], model, processor)
                    st.session_state["pattern_embedding"] = pattern_embedding
                    st.write("Object Embedding (DINOv2):")
                    st.write(f"Embedding shape: {pattern_embedding.shape}")
                    st.text_area("Pattern Embedding (all dimensions)", ', '.join([f'{v:.4f}' for v in pattern_embedding]), height=80)
                    st.success("Pattern object embedding extracted and saved.")
# --- Step 3: Detection and Visualization ---
if query_img is not None and "pattern_embedding" in st.session_state:
    st.markdown("---")
    st.header("Step 4: Region Proposal + Pattern Matching")
    # Option to select Selective Search mode
    ss_mode = st.selectbox("Selective Search Mode", options=["fast", "quality"], index=0, help="'quality' is slower but may give better proposals.")
    # Option to select proposal method
    proposal_method = st.selectbox("Region Proposal Method", options=["Selective Search", "Sliding Window", "Both"], index=2)
    # Slider for max proposals
    max_proposals = st.slider("Max Region Proposals to Process", min_value=100, max_value=2000, value=500, step=50)
    if st.button("Run Region Proposal + Pattern Matching"):
        # 1. Get region proposals from the query image
        st.info(f"Generating region proposals with {proposal_method}...")
        rects = []
        if proposal_method in ["Selective Search", "Both"]:
            rects_ss = get_multiscale_region_proposals(query_img, mode=ss_mode, max_regions=2000, scales=[1.0, 0.85, 0.7, 0.5, 0.35])
            rects += rects_ss
        if proposal_method in ["Sliding Window", "Both"]:
            rects_sw = sliding_window_supplement(query_img, window_sizes=[64, 96, 128, 160, 192], stride_ratio=0.5)
            rects += rects_sw
        st.write(f"Total region proposals before deduplication: {len(rects)}")
        # Deduplicate proposals
        dedup_boxes = deduplicate_boxes([(x, y, x + w, y + h) for (x, y, w, h) in rects], iou_thresh=0.7)
        st.write(f"Region proposals after deduplication: {len(dedup_boxes)}")
        # Limit to max_proposals
        dedup_boxes = dedup_boxes[:max_proposals]
        st.write(f"Region proposals to process: {len(dedup_boxes)}")
        # 2. Extract DINOv2 embedding for each region
        model, processor = load_dinov2_model()
        pattern_embedding = st.session_state["pattern_embedding"]
        region_boxes = []
        region_distances = []
        region_confidences = []
        for (x0, y0, x1, y1) in dedup_boxes:
            w = x1 - x0
            h = y1 - y0
            if w < 10 or h < 10:
                continue  # Lowered minimum size
            patch = query_img.crop((x0, y0, x1, y1))
            region_emb = get_dinov2_embedding(patch, model, processor)
            dist = l2_distance(pattern_embedding, region_emb)
            conf = 1 / (1 + dist)
            region_boxes.append((x0, y0, x1, y1))
            region_distances.append(dist)
            region_confidences.append(conf)
        # Store results in session state for interactive thresholding
        st.session_state["region_boxes"] = region_boxes
        st.session_state["region_distances"] = region_distances
        st.session_state["region_confidences"] = region_confidences
        # Optional: Visualize all proposals (for debugging)
        if st.checkbox("Show all region proposals (debug)"):
            all_props_img = query_img.copy()
            draw_props = ImageDraw.Draw(all_props_img)
            for box in region_boxes:
                draw_props.rectangle(box, outline="gray", width=1)
            st.image(all_props_img, caption=f"All Region Proposals (Total: {len(region_boxes)})", use_column_width=True)
    # --- Interactive thresholding and NMS on stored region proposals ---
    if "region_boxes" in st.session_state and "region_distances" in st.session_state and "region_confidences" in st.session_state:
        region_boxes = st.session_state["region_boxes"]
        region_distances = st.session_state["region_distances"]
        region_confidences = st.session_state["region_confidences"]
        st.markdown("**Set Similarity Threshold for Region Matching**")
        if region_distances:
            min_dist = min(region_distances)
            max_dist = max(region_distances)
        else:
            min_dist = 0
            max_dist = 1
        slider_val = st.slider(
            "Threshold (relative to best match, region proposals)",
            min_value=0.01, max_value=0.5, value=0.1, step=0.01,
            help="Lower values = fewer, more confident detections."
        )
        threshold = min_dist + slider_val * (max_dist - min_dist)
        st.write(f"Current threshold: {threshold:.4f}")
        filtered_boxes = []
        filtered_distances = []
        filtered_confidences = []
        for box, dist, conf in zip(region_boxes, region_distances, region_confidences):
            if dist <= threshold:
                filtered_boxes.append(box)
                filtered_distances.append(dist)
                filtered_confidences.append(conf)
        # Visualize all filtered boxes before NMS
        all_filtered_img = query_img.copy()
        draw_filtered = ImageDraw.Draw(all_filtered_img)
        for box, dist, conf in zip(filtered_boxes, filtered_distances, filtered_confidences):
            draw_filtered.rectangle(box, outline="blue", width=2)
            draw_filtered.text((box[0], box[1]), f"L2: {dist:.2f}\nConf: {conf:.2f}", fill="cyan")
        st.image(all_filtered_img, caption=f"All Filtered Boxes Before NMS (Total: {len(filtered_boxes)})", use_column_width=True)
        # 4. Apply NMS
        nms_iou = st.slider("NMS IoU Threshold (region proposals)", min_value=0.1, max_value=0.9, value=0.3, step=0.02)
        keep_indices = nms(filtered_boxes, filtered_distances, iou_threshold=nms_iou)
        nms_boxes = [filtered_boxes[k] for k in keep_indices]
        nms_distances = [filtered_distances[k] for k in keep_indices]
        nms_confidences = [filtered_confidences[k] for k in keep_indices]
        # 5. Draw results after NMS (expand boxes)
        all_boxes_img = query_img.copy()
        draw_all = ImageDraw.Draw(all_boxes_img)
        img_w, img_h = all_boxes_img.size
        for box, dist, conf in zip(nms_boxes, nms_distances, nms_confidences):
            exp_box = expand_box(box, 0.1, img_w, img_h)  # Expand by 10%
            draw_all.rectangle(exp_box, outline="orange", width=2)
            draw_all.text((exp_box[0], exp_box[1]), f"L2: {dist:.2f}\nConf: {conf:.2f}", fill="yellow")
        st.image(all_boxes_img, caption=f"Query Image with Region Proposal Matches After NMS (Total: {len(nms_boxes)})", use_column_width=True)
else:
    st.info("Upload a pattern image to enable annotation.") 