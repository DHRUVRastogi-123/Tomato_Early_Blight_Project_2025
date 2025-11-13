# app/streamlit.py

import streamlit as st
import os
import tempfile
import sys
import time
import torch

# --- Path Setup ---
# Add the project root to the Python path to find the 'scripts' module
# This assumes you run `streamlit run app/streamlit.py` from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    # --- Import Your Project Scripts ---
    from scripts.infer_model import (
        load_classifier, 
        process_image, 
        process_video, 
        get_val_transforms
    )
    from scripts.utils.segmenter import LeafSegmenter
except ImportError as e:
    st.error(f"""
        **Error: Failed to import project scripts.**
        
        Make sure you are running Streamlit from your project's root directory,
        and that the following files exist:
        - `scripts/infer_model.py`
        - `scripts/utils/segmenter.py`
        - `scripts/utils/augmentations.py`
        
        **Details:** {e}
    """)
    st.stop()

# --- Model & Config Paths ---
SEG_MODEL_PATH = "yolov8n-seg.pt"
CLS_MODEL_WEIGHTS = "models/best_cls_weights_final.pt" # Update this path if needed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLS_CLASSES = ['early_blight', 'healthy'] # Must match training order

# --- Model Loading (Cached) ---
# Use @st.cache_resource to load models only once, not on every re-run
@st.cache_resource
def load_all_models():
    """Loads and caches both segmentation and classification models."""
    try:
        segmenter = LeafSegmenter(model_path=SEG_MODEL_PATH)
        classifier = load_classifier(
            weights_path=CLS_MODEL_WEIGHTS, 
            device=DEVICE
        )
        val_transforms = get_val_transforms()
        return segmenter, classifier, val_transforms
    except Exception as e:
        st.error(f"**Error loading models:** {e}")
        st.error("Please make sure 'yolov8n-seg.pt' is available (it should auto-download) "
                 f"and '{CLS_MODEL_WEIGHTS}' exists.")
        return None, None, None

# --- Page Configuration ---
st.set_page_config(
    page_title="Tomato Early Blight Detection",
    page_icon="🍅",
    layout="wide"
)

st.title("🍅 Tomato Early Blight Detection System")
st.write("Upload a video or image of a tomato plant. The system will run the "
         "two-stage pipeline (YOLOv8-seg + EfficientNet) to detect and classify leaves.")

# --- Load Models ---
with st.spinner("Loading AI models... This may take a moment on first run."):
    seg_model, cls_model, cls_transforms = load_all_models()

if seg_model is None or cls_model is None:
    st.stop()

st.success("Models loaded successfully!")

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "Choose a video or image file...", 
    type=["mp4", "avi", "mov", "jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    # Get file type
    file_name = uploaded_file.name
    file_ext = os.path.splitext(file_name)[1].lower()
    video_types = ['.mp4', '.avi', '.mov']
    image_types = ['.jpg', '.png', '.jpeg']

    # 1. Save uploaded file to a temporary path
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tfile:
        tfile.write(uploaded_file.read())
        input_file_path = tfile.name
        
    output_file_path = input_file_path + "_output" + file_ext

    # 2. Process based on file type
    try:
        # --- IMAGE PROCESSING ---
        if file_ext in image_types:
            with st.spinner("Running detection on image..."):
                # Call your processing function from infer_model.py
                process_image(
                    image_path=input_file_path, 
                    output_path=output_file_path, 
                    seg_model=seg_model, 
                    cls_model=cls_model, 
                    cls_transforms=cls_transforms, 
                    device=DEVICE
                )
            
            st.success("Processing complete!")
            st.image(output_file_path, caption="Processed Image")
            
            with open(output_file_path, 'rb') as img_file:
                st.download_button(
                    label="Download Processed Image",
                    data=img_file.read(),
                    file_name=f"processed_{file_name}",
                    mime="image/jpeg"
                )

        # --- VIDEO PROCESSING ---
        elif file_ext in video_types:
            st.info("Processing video... This may take time. See progress below.")
            
            # Create a progress bar
            progress_bar = st.progress(0.0)
            start_time = time.time()
            
            # Use the generator from infer_model.py to get real-time progress
            try:
                # Get total frames first
                total_frames, frame_gen = process_video(
                    video_path=input_file_path, 
                    output_path=output_file_path, 
                    seg_model=seg_model, 
                    cls_model=cls_model, 
                    cls_transforms=cls_transforms, 
                    device=DEVICE
                )
                
                # Iterate through the frame generator
                for i, _ in enumerate(frame_gen):
                    progress = (i + 1) / total_frames
                    progress_bar.progress(progress)
                
                end_time = time.time()
                st.success(f"Processing complete in {end_time - start_time:.2f} seconds!")
                
                # Display the video
                with open(output_file_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                st.video(video_bytes)
                
                st.download_button(
                    label="Download Processed Video",
                    data=video_bytes,
                    file_name=f"processed_{file_name}",
                    mime="video/mp4"
                )
            
            except Exception as e:
                st.error(f"An error occurred during video processing: {e}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        import traceback
        st.code(traceback.format_exc())

    finally:
        # 3. Clean up temporary files
        if os.path.exists(input_file_path):
            os.remove(input_file_path)
        if os.path.exists(output_file_path):
            os.remove(output_file_path)