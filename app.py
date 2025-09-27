# # src/app.py
# """
# Streamlit app for EyeCareAI:
# - Upload a retinal image or choose sample
# - Predict diabetic retinopathy stage (0-4)
# - Show probabilities and Grad-CAM heatmap overlay
# """
# import streamlit as st
# import os
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.efficientnet import preprocess_input
# import matplotlib.pyplot as plt
# import cv2

# ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# MODEL_PATH = os.path.join(ROOT, "models", "eye_care_model.h5")
# IMG_SIZE = (224, 224)
# CLASS_NAMES = ['0 - No DR', '1 - Mild', '2 - Moderate', '3 - Severe', '4 - Proliferative DR']

# @st.cache_resource
# def load_eye_model():
#     if not os.path.exists(MODEL_PATH):
#         raise FileNotFoundError("Model not found. Run src/train.py first.")
#     model = load_model(MODEL_PATH)
#     return model

# def preprocess_img_pil(pil_img):
#     img = pil_img.resize(IMG_SIZE)
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     return x

# def get_gradcam_heatmap(model, img_array, last_conv_layer_name=None):
#     # img_array is preprocessed with batch dimension
#     grad_model = model
#     # find a conv layer
#     if last_conv_layer_name is None:
#         # automatically pick last conv layer
#         for layer in reversed(model.layers):
#             if len(layer.output_shape) == 4:
#                 last_conv_layer_name = layer.name
#                 break
#     last_conv_layer = model.get_layer(last_conv_layer_name)
#     # Create a model that maps the input image to the activations of the last conv layer
#     from tensorflow.keras import Model, backend as K
#     cam_model = Model(inputs=model.inputs, outputs=[last_conv_layer.output, model.output])
#     with np.errstate(all='ignore'):
#         conv_outputs, predictions = cam_model(img_array)
#         pred_index = np.argmax(predictions[0])
#         conv_outputs = conv_outputs[0]
#         grads = K.gradients(model.output[:, pred_index], last_conv_layer.output)[0]
#         pooled_grads = K.mean(grads, axis=(0, 1, 2))
#         iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
#         pooled_grads_value, conv_layer_output_value = iterate([img_array])
#     for i in range(len(pooled_grads_value)):
#         conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
#     heatmap = np.mean(conv_layer_output_value, axis=-1)
#     heatmap = np.maximum(heatmap, 0)
#     heatmap /= np.max(heatmap) + 1e-8
#     heatmap = cv2.resize(heatmap, IMG_SIZE)
#     return heatmap, pred_index

# def overlay_heatmap_on_image(orig_img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
#     img = np.array(orig_img.resize(IMG_SIZE))[:,:, :3]
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.applyColorMap(heatmap, colormap)
#     overlay = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)
#     return overlay

# st.set_page_config(page_title="EyeCareAI", layout="centered")
# st.title("👁️ EyeCareAI — Diabetic Retinopathy Detection")

# model = load_eye_model()

# uploaded = st.file_uploader("Upload a retinal image (png/jpg)", type=['png','jpg','jpeg'])
# if st.button("Use sample image"):
#     # try to pick one sample from train_images
#     sample_dir = os.path.join(ROOT, "data", "raw", "train_images")
#     if os.path.exists(sample_dir):
#         files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.png','.jpg'))]
#         if files:
#             sample_path = os.path.join(sample_dir, files[0])
#             st.image(sample_path, caption="Sample image", use_column_width=True)
#             uploaded = open(sample_path, "rb")

# if uploaded is not None:
#     img = image.load_img(uploaded)
#     st.image(img, caption="Uploaded image", use_column_width=True)
#     x = preprocess_img_pil(img)

#     preds = model.predict(x)[0]
#     top_idx = int(np.argmax(preds))
#     st.subheader(f"Prediction: {CLASS_NAMES[top_idx]}")
#     st.write("Probabilities:")
#     for i, p in enumerate(preds):
#         st.write(f"{CLASS_NAMES[i]}: {p*100:.2f}%")

#     # Grad-CAM
#     try:
#         heatmap, pred_idx = get_gradcam_heatmap(model, x)
#         overlay = overlay_heatmap_on_image(img, heatmap)
#         st.subheader("Grad-CAM heatmap (overlay)")
#         st.image(overlay, use_column_width=True)
#     except Exception as e:
#         st.warning("Grad-CAM failed: " + str(e))

# # src/app.py
# import streamlit as st
# from PIL import Image
# import numpy as np
# import os
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

# # Paths
# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# MODEL_PATH = os.path.join(ROOT_DIR, "models", "eye_care_model.h5")

# # Load model
# model = load_model(MODEL_PATH)

# # Helper function
# def predict(img):
#     img = img.resize((224, 224))
#     img_array = np.array(img)/255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     pred = model.predict(img_array)
#     return np.argmax(pred, axis=1)[0]

# # Streamlit UI
# st.title("👁️ EyeCareAI – Diabetic Retinopathy Detection")
# st.write("Upload a retinal fundus image to detect the stage of diabetic retinopathy.")

# uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

# if uploaded_file:
#     img = Image.open(uploaded_file)
#     st.image(img, caption="Uploaded Image", use_column_width=True)
#     st.write("Predicting...")
#     label = predict(img)
#     st.success(f"Predicted Diagnosis: {label} / 4")

# src/app.py
import streamlit as st
from PIL import Image
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Paths
# ----------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "eye_care_model.h5")

# ----------------------------
# Load Model
# ----------------------------
model = load_model(MODEL_PATH)

# ----------------------------
# DR Labels
# ----------------------------
DR_LABELS = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}

# ----------------------------
# Suggested Follow-up
# ----------------------------
FOLLOW_UP = {
    0: "No immediate action needed. Routine check-ups recommended.",
    1: "Monitor and maintain good blood sugar control.",
    2: "Consult ophthalmologist. May require treatment.",
    3: "Urgent consultation needed. Risk of vision loss.",
    4: "Immediate medical attention required. High risk of blindness."
}

# ----------------------------
# Helper function
# ----------------------------
def predict(img):
    img = img.resize((224, 224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred_probs = model.predict(img_array)[0]
    pred_class = np.argmax(pred_probs)
    return pred_class, pred_probs

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="EyeCareAI", page_icon="👁️", layout="centered")
st.title("👁️ EyeCareAI – Diabetic Retinopathy Detection")
st.write("Upload a retinal fundus image to detect the stage of diabetic retinopathy.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("Predicting...")

    pred_class, pred_probs = predict(img)
    pred_label = DR_LABELS[pred_class]
    confidence = pred_probs[pred_class] * 100
    follow_up = FOLLOW_UP[pred_class]

    # Display result
    st.subheader("🔹 Predicted Diagnosis")
    if pred_class >= 3:
        st.error(f"**{pred_label}** (Confidence: {confidence:.2f}%) – Urgent attention required!")
    else:
        st.success(f"**{pred_label}** (Confidence: {confidence:.2f}%)")

    # Suggested follow-up
    st.subheader("🔹 Suggested Follow-up Action")
    st.info(follow_up)

    # Probability table
    st.subheader("🔹 Probability Scores for Each Stage")
    prob_df = pd.DataFrame({
        "Stage": [DR_LABELS[i] for i in range(5)],
        "Probability (%)": [p*100 for p in pred_probs]
    })
    st.dataframe(prob_df)

    # Bar chart
    fig, ax = plt.subplots()
    ax.barh(prob_df['Stage'], prob_df['Probability (%)'], color='skyblue')
    ax.set_xlabel("Probability (%)")
    ax.set_title("Diabetic Retinopathy Stage Probabilities")
    for i, v in enumerate(prob_df['Probability (%)']):
        ax.text(v + 0.5, i, f"{v:.1f}%", color='blue', va='center')
    st.pyplot(fig)
