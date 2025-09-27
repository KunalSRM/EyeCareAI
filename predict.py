# # src/predict.py
# """
# Generate predictions for test.csv using saved model.
# Outputs submission.csv in project root: columns id_code, diagnosis (0-4)
# """
# import os
# import pandas as pd
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.efficientnet import preprocess_input
# from tensorflow.keras.preprocessing import image

# ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# RAW_DIR = os.path.join(ROOT, "data", "raw")
# TEST_IM_DIR = os.path.join(RAW_DIR, "test_images")
# MODEL_PATH = os.path.join(ROOT, "models", "eye_care_model.h5")
# PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
# IMG_SIZE = (224, 224)

# def load_img(path, target_size=IMG_SIZE):
#     img = image.load_img(path, target_size=target_size)
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     return x

# def predict():
#     if not os.path.exists(MODEL_PATH):
#         raise FileNotFoundError("Model not found. Run src/train.py first.")
#     model = load_model(MODEL_PATH)
#     test_csv = os.path.join(RAW_DIR, "test.csv")
#     df = pd.read_csv(test_csv)
#     preds = []
#     for idx, row in df.iterrows():
#         idc = row['id_code']
#         png = os.path.join(TEST_IM_DIR, f"{idc}.png")
#         jpg = os.path.join(TEST_IM_DIR, f"{idc}.jpg")
#         path = png if os.path.exists(png) else (jpg if os.path.exists(jpg) else None)
#         if path is None:
#             print("Missing file for", idc)
#             preds.append(0)
#             continue
#         x = load_img(path)
#         p = model.predict(x)
#         label = int(np.argmax(p, axis=1)[0])
#         preds.append(label)
#     df['diagnosis'] = preds
#     out = os.path.join(ROOT, "submission.csv")
#     df[['id_code','diagnosis']].to_csv(out, index=False)
#     print("Saved submission to:", out)

# if __name__ == "__main__":
#     predict()

# src/predict.py
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "eye_care_model.h5")
TEST_DIR = os.path.join(ROOT_DIR, "data", "raw", "test_images")
TEST_CSV = os.path.join(ROOT_DIR, "data", "raw", "test.csv")

# Load model
model = load_model(MODEL_PATH)

# Load test CSV
test_df = pd.read_csv(TEST_CSV)
test_df['id_code'] = test_df['id_code'].apply(lambda x: f"{x}.png")

# Prediction function
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    return np.argmax(pred, axis=1)[0]

# Predict on all test images
predictions = []
for idx, row in test_df.iterrows():
    img_path = os.path.join(TEST_DIR, row['id_code'])
    label = predict_image(img_path)
    predictions.append(label)

# Save predictions
test_df['diagnosis'] = predictions
test_df.to_csv(os.path.join(ROOT_DIR, "data", "processed", "test_predictions.csv"), index=False)
print("✅ Predictions saved to data/processed/test_predictions.csv")
