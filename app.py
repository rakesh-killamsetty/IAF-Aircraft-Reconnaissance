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
    st.write("Draw bounding boxes around objects of interest in the pattern image.")
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",  # Red with alpha
        stroke_width=2,
        background_image=pattern_img,
        update_streamlit=True,
        height=pattern_img.height,
        width=pattern_img.width,
        drawing_mode="rect",
        key="canvas",
    )
    if canvas_result.json_data:
        st.subheader("Annotation Data (Bounding Boxes)")
        st.json(canvas_result.json_data)

        # # Extract bounding boxes in two formats
        # shapes = canvas_result.json_data.get("objects", [])
        # bbox_dicts = []
        # bbox_tuples = []
        # for obj in shapes:
        #     if obj.get("type") == "rect":
        #         left = int(obj["left"])
        #         top = int(obj["top"])
        #         width = int(obj["width"])
        #         height = int(obj["height"])
        #         bbox_dicts.append({"x": left, "y": top, "width": width, "height": height})
        #         bbox_tuples.append((left, top, width, height))

        # st.markdown("**Bounding Boxes (as list of dicts):**")
        # st.write(bbox_dicts)
        # st.markdown("**Bounding Boxes (as list of tuples):**")
        # st.write(bbox_tuples)
    else:
        st.info("Draw bounding boxes to annotate objects.")
else:
    st.info("Upload a pattern image to enable annotation.") 