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
            dist = l2_distance(pattern_embedding, patch_emb)
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
    st.header("Step 3: Query Image Segmentation and Comparison Table")
    if st.button("Segment Query Image with SAM and Compare Embeddings") or (
        "query_segmentation_results" not in st.session_state and query_img is not None and "pattern_embedding" in st.session_state
    ):
        with st.spinner("Segmenting query image and extracting embeddings..."):
            # Load SAM model and DINOv2 model
            predictor = load_sam_model()
            model, processor = load_dinov2_model()
            # Use SamAutomaticMaskGenerator for full image segmentation
            image_np = np.array(query_img)
            if image_np.shape[2] == 4:
                image_np = image_np[:, :, :3]
            sam = predictor.model
            mask_generator = SamAutomaticMaskGenerator(
                sam,
                points_per_side=32,  # higher for more segments
                pred_iou_thresh=0.88,  # higher threshold to avoid background
                stability_score_thresh=0.92,  # higher threshold to avoid unstable/background masks
                min_mask_region_area=500,  # filter out large background regions
                # These settings help exclude the background from segmentation
            )
            masks = mask_generator.generate(image_np)
            st.write(f"Found {len(masks)} segments in query image.")
            # --- Display the segmented image with all masks overlaid ---
            import random
            seg_vis = image_np.copy()
            overlay = np.zeros_like(seg_vis, dtype=np.uint8)
            for mask_dict in masks:
                mask = mask_dict["segmentation"]
                color = [random.randint(0,255) for _ in range(3)]
                overlay[mask > 0] = color
            alpha = 0.5
            seg_vis = (seg_vis * (1 - alpha) + overlay * alpha).astype(np.uint8)
            st.image(seg_vis, caption="Segmented Query Image (All Segments Overlaid)", use_column_width=True)
            # For each mask, extract region and embedding
            pattern_embedding = st.session_state["pattern_embedding"]
            results = []
            for idx, mask_dict in enumerate(masks):
                mask = mask_dict["segmentation"]
                region = extract_object_region(query_img, mask)
                if region is None:
                    continue
                emb = get_dinov2_embedding(region, model, processor)
                cos_sim = cosine_similarity(pattern_embedding, emb)
                l2 = l2_distance(pattern_embedding, emb)
                results.append({
                    "idx": idx,
                    "cosine_similarity": cos_sim,
                    "1-cosine_similarity": 1-cos_sim,
                    "l2_distance": l2,
                    "region_img": region,
                    "embedding": emb,
                })
            # Sort by L2 distance (best match first)
            results = sorted(results, key=lambda x: x["l2_distance"])
            # Store results in session_state for later use
            st.session_state["query_segmentation_results"] = {
                "image_np": image_np,
                "masks": masks,
                "results": results,
                "seg_vis": seg_vis,
            }
    # Use cached results if available
    if "query_segmentation_results" in st.session_state:
        image_np = st.session_state["query_segmentation_results"]["image_np"]
        masks = st.session_state["query_segmentation_results"]["masks"]
        results = st.session_state["query_segmentation_results"]["results"]
        seg_vis = st.session_state["query_segmentation_results"]["seg_vis"]
        st.image(seg_vis, caption="Segmented Query Image (All Segments Overlaid)", use_column_width=True)
        # Display as a table
        import pandas as pd
        table_data = [{
            "Segment #": r["idx"],
            "Cosine Similarity": f"{r['cosine_similarity']:.4f}",
            "1-Cosine Similarity": f"{r['1-cosine_similarity']:.4f}",
            "L2 Distance": f"{r['l2_distance']:.4f}"
        } for r in results]
        st.subheader("Comparison Table (Pattern vs. Query Segments)")
        st.dataframe(pd.DataFrame(table_data))
        # Optionally, show the top-N segment images
        st.markdown("**Top 5 Matching Segments (by L2 distance):**")
        for r in results[:5]:
            st.image(r["region_img"], caption=f"Segment #{r['idx']} | L2: {r['l2_distance']:.4f}", use_column_width=True)
        # --- Detect all similar objects in the query image ---
        st.markdown("**Detect All Similar Objects in Query Image**")
        l2s = [r['l2_distance'] for r in results]
        if l2s:
            min_l2, max_l2 = float(min(l2s)), float(max(l2s))
            default_thresh = min_l2 + 0.2 * (max_l2 - min_l2)
            l2_thresh = st.slider("L2 Distance Threshold for Detection", min_value=float(min_l2), max_value=float(max_l2), value=float(default_thresh), step=0.01)
            detected = [r for r in results if r['l2_distance'] <= l2_thresh]
            st.write(f"Detected {len(detected)} similar objects (L2 ≤ {l2_thresh:.4f})")
            # Overlay detected segments on the query image
            detected_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
            for r in detected:
                mask = masks[r['idx']]['segmentation']
                detected_mask[mask > 0] = 255
            overlay_img = image_np.copy()
            overlay_img[detected_mask > 0] = [255, 0, 0]  # Red overlay for detected
            st.image(overlay_img, caption="Query Image with Detected Similar Objects Highlighted", use_column_width=True)
            # Show each detected segment with confidence score
            for r in detected:
                confidence = 1 / (1 + r['l2_distance'])
                st.image(r["region_img"], caption=f"Detected Segment #{r['idx']} | L2: {r['l2_distance']:.4f} | Confidence: {confidence:.2f}", use_column_width=False, width=128)
else:
    st.info("Upload a pattern image to enable annotation.") 