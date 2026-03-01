import streamlit as st
import pickle
import os
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries


st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="wide",
)


MODEL_PATH = "mob_res_se_final.pkl"
GDRIVE_FILE_ID = "1p957K5Mf0ni1Ge4HUbmrGQVZ400bmPeR"

@st.cache_resource
def load_prediction_model():
    import gdown
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

model = load_prediction_model()


class_names = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry___Powdery_mildew", "Cherry___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot", "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight", "Corn___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def find_gradcam_layers(model):
    """
    Returns (path1_layer, path2_layer).
    Path 1 = last activation/relu in the residual blocks
    Path 2 = 'out_relu' inside MobileNetV2
    """
    path1_layer, path2_layer = None, None
    for layer in model.layers:
        name = layer.name
        if "activation" in name or "re_lu" in name:
            path1_layer = name
        if "out_relu" in name:
            path2_layer = name
    return path1_layer, path2_layer



def get_gradcam(model, img_array, layer_name):
    """Standard Grad-CAM heatmap for a given layer."""
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_score = predictions[:, pred_index]

    grads = tape.gradient(class_score, conv_outputs)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_outputs[0] * weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()



def get_gradcampp(model, img_array, layer_name):
    """Grad-CAM++ with higher-order gradients for better localization."""
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output],
    )
    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape3:
                conv_out, preds = grad_model(img_array)
                pred_class = tf.argmax(preds[0])
                score = preds[:, pred_class]
            g1 = tape3.gradient(score, conv_out)
        g2 = tape2.gradient(g1, conv_out)
    g3 = tape1.gradient(g2, conv_out)

    global_sum = tf.reduce_sum(conv_out[0], axis=(0, 1), keepdims=True)
    denom = 2.0 * g2[0] + global_sum * g3[0] + 1e-8
    alpha = g2[0] / denom
    alpha = tf.where(g1[0] > 0, alpha, tf.zeros_like(alpha))
    weights = tf.reduce_sum(alpha * tf.maximum(g1[0], 0), axis=(0, 1))
    heatmap = tf.reduce_sum(conv_out[0] * weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()



def overlay_heatmap(heatmap, img, alpha=0.4):
    """Resize heatmap and overlay on the original image. Returns float [0,1]."""
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = np.float32(heatmap_colored) / 255.0 * alpha + np.float32(img) * (1 - alpha)
    return np.clip(overlay, 0, 1)



st.title("🌿 Plant Disease Classifier & XAI Evaluator")
st.write("Upload a leaf image to predict the disease and visualize model attention.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)  # FIX 3: deprecated API

    # --- Preprocessing (128x128, normalize to [0,1]) ---
    img_resized = image.resize((128, 128))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_input = np.expand_dims(img_array, axis=0)  # shape: (1, 128, 128, 3)

    if st.button("🔍 Predict & Generate XAI"):
        # --- Prediction ---
        # FIX 4: model already outputs softmax — do NOT apply tf.nn.softmax again
        preds = model.predict(img_input, verbose=0)
        class_idx = np.argmax(preds[0])
        confidence = preds[0][class_idx] * 100

        st.subheader(f"Prediction: {class_names[class_idx]}")
        st.write(f"Confidence: **{confidence:.2f}%**")

        # Show top-5 predictions
        top5_idx = np.argsort(preds[0])[::-1][:5]
        st.markdown("**Top-5 Predictions:**")
        for rank, idx in enumerate(top5_idx, 1):
            bar_val = float(preds[0][idx])
            st.write(f"{rank}. {class_names[idx]} — {bar_val * 100:.2f}%")
            st.progress(bar_val)

        st.divider()

        # --- Find layers for Grad-CAM ---
        path1_layer, path2_layer = find_gradcam_layers(model)
        # Fallback: use the last Conv2D if path-specific layers aren't found
        if path1_layer is None:
            conv_layers = [l.name for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
            path1_layer = conv_layers[-1] if conv_layers else None

        # --- XAI Visualizations ---
        col1, col2, col3 = st.columns(3)

        # 1. Grad-CAM
        with col1:
            st.markdown("### Grad-CAM")
            try:
                target_layer = path2_layer if path2_layer else path1_layer
                heatmap = get_gradcam(model, img_input, target_layer)
                gradcam_img = overlay_heatmap(heatmap, img_array)
                st.image(gradcam_img, caption=f"Layer: {target_layer}", clamp=True)
            except Exception as e:
                st.error(f"Grad-CAM failed: {e}")

        # 2. Grad-CAM++
        with col2:
            st.markdown("### Grad-CAM++")
            try:
                target_layer = path2_layer if path2_layer else path1_layer
                heatmap_pp = get_gradcampp(model, img_input, target_layer)
                gradcampp_img = overlay_heatmap(heatmap_pp, img_array)
                st.image(gradcampp_img, caption=f"Layer: {target_layer}", clamp=True)
            except Exception as e:
                st.error(f"Grad-CAM++ failed: {e}")

        # 3. LIME
        with col3:
            st.markdown("### LIME")
            try:
                explainer = lime_image.LimeImageExplainer()

                # FIX 5: LIME passes uint8 [0,255] images to predict_fn.
                # Our model expects float [0,1], so we normalize INSIDE predict_fn.
                def lime_predict_fn(images):
                    # images come from LIME as float64 in [0, 255] range
                    images = images.astype(np.float32) / 255.0
                    return model.predict(images, verbose=0)

                explanation = explainer.explain_instance(
                    (img_array * 255).astype(np.uint8),  # LIME expects uint8
                    lime_predict_fn,
                    top_labels=5,
                    hide_color=0,
                    num_samples=300,
                )
                temp, mask = explanation.get_image_and_mask(
                    explanation.top_labels[0],
                    positive_only=True,
                    num_features=5,
                    hide_rest=False,
                )
                lime_img = mark_boundaries(temp / 255.0, mask)
                st.image(lime_img, caption="Positive regions", clamp=True)
            except Exception as e:
                st.error(f"LIME failed: {e}")

        st.success("✅ XAI visualizations generated!")



