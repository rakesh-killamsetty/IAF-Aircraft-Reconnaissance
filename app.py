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

# --- Streamlit page config and title ---
st.set_page_config(page_title="Aerial Pattern Detection Prototype", layout="wide")
st.title("Pattern-Based Object Detection Prototype")

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
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base")
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
    inputs = processor(images=image_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
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
                    st.write(pattern_embedding)
                    st.success("Pattern object embedding extracted and saved.")
# --- Step 3: Detection and Visualization ---
if query_img is not None and "pattern_embedding" in st.session_state:
    # --- User controls for detection parameters ---
    st.markdown("**Detection Parameters**")
    window_size = st.slider("Sliding Window Size (pixels)", min_value=32, max_value=256, value=64, step=16)  # Window size for sliding window search
    stride = window_size // 2  # Stride for sliding window
    nms_iou = st.slider("NMS IoU Threshold", min_value=0.1, max_value=0.9, value=0.3, step=0.05)  # IoU threshold for NMS
    if st.button("Run Lightweight Decoder (Sliding Window Search)"):
        # Slide a window over the query image, compute similarity to the pattern embedding, and build a similarity map
        model, processor = load_dinov2_model()
        similarity_map, best_box, min_dist = sliding_window_similarity(query_img, st.session_state["pattern_embedding"], model, processor, window_size=window_size, stride=stride)
        # Store results in session state for interactive visualization
        st.session_state["similarity_map"] = similarity_map
        st.session_state["min_dist"] = min_dist
        st.session_state["window_size"] = window_size
        st.session_state["stride"] = stride
        st.session_state["nms_iou"] = nms_iou
        st.success("Sliding window similarity computed. Adjust the threshold below.")
    # --- If similarity_map is available, show slider and visualizations ---
    if "similarity_map" in st.session_state and "min_dist" in st.session_state:
        similarity_map = st.session_state["similarity_map"]
        min_dist = st.session_state["min_dist"]
        window_size = st.session_state.get("window_size", 64)
        stride = st.session_state.get("stride", 32)
        nms_iou = st.session_state.get("nms_iou", 0.3)
        # --- User-adjustable slider to set the similarity threshold for detection ---
        st.markdown("**Adjust Similarity Threshold for Detection**")
        slider_val = st.slider(
            "Threshold (relative to best match)",
            min_value=0.01, max_value=0.5, value=0.1, step=0.01,
            help="Lower values = fewer, more confident detections."
        )
        threshold = min_dist + slider_val * (similarity_map.max() - min_dist)
        st.write(f"Current threshold: {threshold:.4f}")
        # --- Display the bounding box of the best match (lowest L2 distance) on the query image ---
        import numpy as np
        min_idx = np.unravel_index(np.argmin(similarity_map), similarity_map.shape)
        best_box = (min_idx[1] * stride, min_idx[0] * stride, min_idx[1] * stride + window_size, min_idx[0] * stride + window_size)
        query_img_boxed = query_img.copy()
        draw = ImageDraw.Draw(query_img_boxed)
        draw.rectangle(best_box, outline="red", width=3)
        st.image(query_img_boxed, caption="Query Image with Best Match Bounding Box", use_column_width=True)
        # --- Show a heatmap of the similarity map to visualize where the pattern is most/least similar ---
        fig, ax = plt.subplots()
        heatmap = ax.imshow(similarity_map, cmap='viridis', origin='upper')
        plt.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title('Similarity Map (L2 Distance)')
        st.pyplot(fig)
        # --- Overlay a binary mask on the query image to highlight all regions above the similarity threshold ---
        mask = (similarity_map <= threshold).astype(np.uint8)
        mask_img_resized = resize(mask, (query_img.height, query_img.width), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
        query_np = np.array(query_img.convert('RGBA'))
        overlay = query_np.copy()
        overlay[mask_img_resized > 0, :3] = [255, 0, 0]  # Red overlay
        overlay[mask_img_resized > 0, 3] = 120  # Alpha
        overlay_img = Image.fromarray(overlay)
        st.image(overlay_img, caption="Query Image with Mask Overlay", use_column_width=True)
        # --- Use NMS to merge overlapping detections and display final bounding boxes with confidence scores ---
        boxes = []
        distances = []
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] > 0:
                    x = j * stride
                    y = i * stride
                    box = (x, y, x + window_size, y + window_size)
                    boxes.append(box)
                    distances.append(similarity_map[i, j])
        # Apply NMS
        keep_indices = nms(boxes, distances, iou_threshold=nms_iou)
        nms_boxes = [boxes[k] for k in keep_indices]
        nms_distances = [distances[k] for k in keep_indices]
        nms_confidences = [1 / (1 + d) for d in nms_distances]
        all_boxes_img = query_img.copy()
        draw_all = ImageDraw.Draw(all_boxes_img)
        for box, dist, conf in zip(nms_boxes, nms_distances, nms_confidences):
            draw_all.rectangle(box, outline="lime", width=2)
            draw_all.text((box[0], box[1]), f"L2: {dist:.2f}\nConf: {conf:.2f}", fill="yellow")
        st.image(all_boxes_img, caption=f"Query Image with NMS Boxes (Total: {len(nms_boxes)})", use_column_width=True)
    elif query_img is not None:
        st.info("Extract the pattern object embedding first.")
else:
    st.info("Upload a pattern image to enable annotation.") 