import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO  # Import BytesIO for binary data conversion
from cv2 import dnn

# Grayscale conversion
def convert_image_to_grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

# Noise reduction
def noise_reduction(image):
    blurred_img = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred_img

# Contrast enhancement using CLAHE
def contrast_enhancement(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(image)
    return enhanced_img

# Scratch mask creation
def create_scratch_mask(image, threshold=240):
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image

    _, mask = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
    return mask

# Scratch removal using OpenCV inpainting
def remove_scratches(image):
    mask = create_scratch_mask(image)
    inpainted_img = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return inpainted_img

# Colorize image
def colorize_image(img_array):
    proto_file = 'models/colorization_deploy_v2.prototxt'
    model_file = 'models/colorization_release_v2.caffemodel'
    hull_pts = 'models/pts_in_hull.npy'
    
    net = dnn.readNetFromCaffe(proto_file, model_file)
    kernel = np.load(hull_pts)

    scaled = img_array.astype("float32") / 255.0
    lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = kernel.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    resized = cv2.resize(lab_img, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_channel = cv2.resize(ab_channel, (img_array.shape[1], img_array.shape[0]))

    L = cv2.split(lab_img)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    colorized = (255 * colorized).astype("uint8")
    return colorized

# Function to process the image pipeline
def process_image(image):
    bw_image = convert_image_to_grayscale(image)
    noise_reduced_image = noise_reduction(bw_image)
    contrast_enhanced = contrast_enhancement(noise_reduced_image)
    enhanced_bw_image = remove_scratches(contrast_enhanced)
    enhanced_bw_image_bgr = cv2.cvtColor(enhanced_bw_image, cv2.COLOR_GRAY2BGR)
    color_image = colorize_image(enhanced_bw_image_bgr)
    return enhanced_bw_image, color_image

# Function to convert PIL image to binary
def convert_image_to_bytes(image_pil):
    buf = BytesIO()
    image_pil.save(buf, format="PNG")
    byte_data = buf.getvalue()
    return byte_data

# Streamlit app
st.title("Old Photo Enhancer")

# Sidebar for image uploader and original image display
with st.sidebar:
    st.header("Upload Your Old Photo")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)
        print(type(input_image))
        st.image(input_image, caption="Original Image", use_column_width=True)

# Main page for enhanced images and download buttons
if uploaded_file is not None:
    input_image_cv = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    bw_image, color_image = process_image(input_image_cv)

    if len(bw_image.shape) == 2:
        bw_image_pil = Image.fromarray(bw_image)
    else:
        bw_image_pil = Image.fromarray(cv2.cvtColor(bw_image, cv2.COLOR_BGR2RGB))

    color_image_pil = Image.fromarray(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

    # Convert the images to binary data
    bw_image_bytes = convert_image_to_bytes(bw_image_pil)
    color_image_bytes = convert_image_to_bytes(color_image_pil)

    # Download buttons
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button("Download Enhanced Black and White Image", bw_image_bytes, file_name="enhanced_bw_image.png", mime="image/png")
        st.image(bw_image_pil, caption="Enhanced Black and White Image", use_column_width=True)

    with col2:
        st.download_button("Download Enhanced Color Image", color_image_bytes, file_name="enhanced_color_image.png", mime="image/png")
        st.image(color_image_pil, caption="Enhanced Color Image", use_column_width=True)
