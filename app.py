FILE_ID = '1p957K5Mf0ni1Ge4HUbmrGQVZ400bmPeR'
import streamlit as st
import pickle
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import os

# --- Model Loading ---
@st.cache_resource
def load_prediction_model():
    # .pkl file handle (Assuming it was saved as a Keras model)
    model = with open... pickle.load('mob_res_se_final.pkl')
    return model

model = load_prediction_model()

# --- Placeholder for Classes (Based on PlantVillage 38 classes) ---
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy', 
    'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# --- XAI Functions ---

def get_gradcam(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def apply_heatmap(heatmap, img):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    return superimposed_img / 255.0

# --- Streamlit UI ---
st.title("🌿 Plant Disease Classifier & XAI Evaluator")
st.write("Upload a leaf image to predict disease and visualize model attention.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocessing
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button('Predict & Generate XAI'):
        preds = model.predict(img_array)
        score = tf.nn.softmax(preds[0])
        class_idx = np.argmax(score)
        
        st.subheader(f"Prediction: {class_names[class_idx]}")
        st.write(f"Confidence: {100 * np.max(score):.2f}%")

        # --- Visualizations ---
        col1, col2, col3 = st.columns(3)
        
        # 1. Grad-CAM
        # Note: Replace 'conv2d_last' with the actual last conv layer name from your model summary
        try:
            last_conv_layer = [l.name for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)][-1]
            heatmap = get_gradcam(img_array, model, last_conv_layer)
            grad_cam_img = apply_heatmap(heatmap, np.array(image.resize((128,128))))
            with col1:
                st.image(grad_cam_img, caption="Grad-CAM")
        except Exception as e:
            st.error("Grad-CAM failed. Check layer names.")

        # 2. LIME
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(img_array[0].astype('double'), 
                                                 model.predict, top_labels=5, 
                                                 hide_color=0, num_samples=100)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], 
                                                     positive_only=True, num_features=5, 
                                                     hide_rest=False)
        with col2:
            st.image(mark_boundaries(temp, mask), caption="LIME Explanation")

        # 3. Grad-CAM++ (Simplified version or place for implementation)
        with col3:
            st.info("Grad-CAM++ requires second-order gradients. Showing LIME focus.")

            st.image(mask, caption="LIME Mask")


