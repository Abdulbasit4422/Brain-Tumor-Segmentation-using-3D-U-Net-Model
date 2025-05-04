import streamlit as st
import numpy as np
import nibabel as nib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
import gdown
import zipfile
import tempfile
from tensorflow.keras.utils import to_categorical

# Title of the app
st.title("Brain Tumor Segmentation using 3D U-Net")

# Function to download the default model from Google Drive
def download_default_model():
    # Google Drive file ID for the default model
    file_id = "1lV1SgafomQKwgv1NW2cjlpyb4LwZXFwX"
    output_path = "default_model.keras"
    
    # Download the file if it doesn't exist
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    
    return output_path

# Load the default model
@st.cache_resource
def load_default_model():
    try:
        model_path = download_default_model()
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading default model: {e}")
        return None

default_model = load_default_model()

# Function to preprocess a NIfTI file
def preprocess_nifti(file_path):
    try:
        image = nib.load(file_path).get_fdata()
        scaler = MinMaxScaler()
        image = scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)
        return image
    except Exception as e:
        st.error(f"Error processing NIfTI file: {e}")
        return None

# Function to combine 4 channels
def combine_channels(t1n, t1c, t2f, t2w):
    try:
        combined_image = np.stack([t1n, t1c, t2f, t2w], axis=3)
        combined_image = combined_image[56:184, 56:184, 13:141]  # Crop to standard size
        return combined_image
    except Exception as e:
        st.error(f"Error combining channels: {e}")
        return None

# Function to run segmentation
def run_segmentation(model, input_image):
    try:
        input_image = np.expand_dims(input_image, axis=0)
        if len(input_image.shape) != 5:
            st.error(f"Invalid input shape: {input_image.shape}")
            return None
        
        prediction = model.predict(input_image)
        return np.argmax(prediction, axis=4)[0, :, :, :]
    except Exception as e:
        st.error(f"Segmentation failed: {e}")
        return None

# Sidebar for model upload
st.sidebar.header("Model Options")
uploaded_model = st.sidebar.file_uploader("Upload custom model (.keras)", type=["keras"])

# Model selection logic
model = None
if uploaded_model is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmp:
            tmp.write(uploaded_model.getvalue())
            model = load_model(tmp.name, compile=False)
        st.sidebar.success("Custom model loaded!")
    except Exception as e:
        st.sidebar.error(f"Error loading custom model: {e}")
        model = default_model
else:
    model = default_model

if model is None:
    st.error("No model available for segmentation")
    st.stop()

# Main file upload section
st.header("Upload MRI Scans")
uploaded_zip = st.file_uploader("Upload zip file containing T1n, T1c, T2f, T2w NIfTI files", type=["zip"])

if uploaded_zip:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save and extract zip
        zip_path = os.path.join(temp_dir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getbuffer())
        
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find required files
            file_types = {
                't1n': None,
                't1c': None,
                't2f': None,
                't2w': None,
                'seg': None
            }
            
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    lower_file = file.lower()
                    for key in file_types:
                        if key in lower_file and lower_file.endswith(('.nii', '.nii.gz')):
                            file_types[key] = os.path.join(root, file)
            
            # Check if we have the 4 required scans
            if not all(file_types[key] for key in ['t1n', 't1c', 't2f', 't2w']):
                st.error("Missing required scan files (need T1n, T1c, T2f, T2w)")
                st.stop()
            
            # Load and preprocess scans
            scans = {}
            for key in ['t1n', 't1c', 't2f', 't2w']:
                scans[key] = preprocess_nifti(file_types[key])
                if scans[key] is None:
                    st.error(f"Failed to process {key} scan")
                    st.stop()
            
            # Combine channels
            combined = combine_channels(
                scans['t1n'], scans['t1c'],
                scans['t2f'], scans['t2w']
            )
            
            if combined is None:
                st.error("Failed to combine channels")
                st.stop()
            
            # Run segmentation
            with st.spinner("Running segmentation..."):
                segmentation = run_segmentation(model, combined)
            
            if segmentation is None:
                st.error("Segmentation failed")
                st.stop()
            
            # Visualization
            st.success("Segmentation complete!")
            
            # Load ground truth if available
            gt_mask = None
            if file_types['seg']:
                try:
                    gt_mask = nib.load(file_types['seg']).get_fdata().astype(np.uint8)
                    gt_mask[gt_mask == 4] = 3
                    gt_mask = np.argmax(to_categorical(gt_mask, num_classes=4), axis=3)
                except Exception as e:
                    st.warning(f"Couldn't load ground truth: {e}")
            
            # Display results
            slices = [int(combined.shape[2] * i // 4 for i in range(1, 4))]  # 25%, 50%, 75%
            
            fig, axes = plt.subplots(3, 4, figsize=(18, 12))
            for i, slice_idx in enumerate(slices):
                # Get slices
                img_slice = np.rot90(combined[:, :, slice_idx, 0])  # Using T1n for display
                pred_slice = np.rot90(segmentation[:, :, slice_idx])
                
                # Plot
                axes[i, 0].imshow(img_slice, cmap='gray')
                axes[i, 0].set_title(f'MRI Slice {slice_idx}')
                axes[i, 0].axis('off')
                
                if gt_mask is not None:
                    gt_slice = np.rot90(gt_mask[:, :, slice_idx])
                    axes[i, 1].imshow(gt_slice)
                    axes[i, 1].set_title('Ground Truth')
                    axes[i, 1].axis('off')
                else:
                    axes[i, 1].axis('off')
                
                axes[i, 2].imshow(pred_slice)
                axes[i, 2].set_title('Prediction')
                axes[i, 2].axis('off')
                
                axes[i, 3].imshow(img_slice, cmap='gray')
                axes[i, 3].imshow(pred_slice, alpha=0.5)
                axes[i, 3].set_title('Overlay')
                axes[i, 3].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Save and offer download
            output_path = "segmentation_result.nii.gz"
            nib.save(nib.Nifti1Image(segmentation.astype(np.float32), np.eye(4)), output_path)
            
            with open(output_path, "rb") as f:
                st.download_button(
                    "Download Segmentation",
                    data=f,
                    file_name=output_path,
                    mime="application/octet-stream"
                )
            
            os.remove(output_path)
            
        except Exception as e:
            st.error(f"Error processing upload: {e}")