# 👁️ EyeCareAI – Diabetic Retinopathy Detection

EyeCareAI is a deep learning project that detects diabetic retinopathy from retinal fundus images.
It leverages state-of-the-art convolutional neural networks to classify the severity of diabetic retinopathy.

---

## 📂 Dataset

* **Source:** [APTOS 2019 Blindness Detection – Kaggle](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data)
* **Files:**

  * `train_images/` – Training retinal images
  * `test_images/` – Test retinal images
  * `train.csv` – Contains `id_code` and `diagnosis` (0–4 severity levels)
  * `test.csv` – Contains `id_code` for predictions

### Diagnosis Levels:

* **0** – No DR
* **1** – Mild
* **2** – Moderate
* **3** – Severe
* **4** – Proliferative DR

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/EyeCareAI.git
cd EyeCareAI
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Train Model

```bash
python src/train.py
```

### 2. Evaluate Model

```bash
python src/evaluate.py
```

### 3. Run Predictions

```bash
python src/predict.py --image_path sample.jpg
```

### 4. Streamlit Demo

```bash
streamlit run app.py
```

---

## 📊 Model Pipeline

1. **Preprocessing:**

   * Image resizing
   * CLAHE (Contrast Limited Adaptive Histogram Equalization)
   * Normalization

2. **Data Augmentation:**

   * Random rotations, flips
   * Brightness/contrast adjustments
   * Zoom & crop

3. **Architecture:**

   * EfficientNetB0 / ResNet50 backbone
   * Fully connected layers for classification

4. **Training:**

   * Loss: Cross-Entropy
   * Optimizer: Adam / SGD
   * Metrics: Accuracy, F1-score, Quadratic Weighted Kappa

---

## 📊 Results (Example)

| Metric   | Value |
| -------- | ----- |
| Accuracy | 82%   |
| F1-Score | 0.78  |
| Kappa    | 0.81  |

---

## 🖥️ Demo Screenshot



---

## 📜 License

This project is licensed under the MIT License – you are free to use and modify it for research and educational purposes.

---

## 🙌 Acknowledgments

* Dataset by [APTOS 2019 Kaggle](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
* Pretrained models from **PyTorch / TensorFlow**
* Streamlit for UI deployment
