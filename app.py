import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np

st.set_page_config(page_title="Aerial Pattern Detection Prototype", layout="wide")
st.title("Pattern-Based Object Detection Prototype")

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

        # Extract and display shape data in a user-friendly way
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
    else:
        st.info("Draw shapes to annotate objects.")
else:
    st.info("Upload a pattern image to enable annotation.") 