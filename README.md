# An Intelligent Optimized Multimodal Learning Framework for Early Prediction of Lung Cancer Recurrence

## 📌 Project Overview

An Intelligent Optimized Multimodal Learning Framework for Early Prediction of Lung Cancer Recurrence is an AI-driven healthcare project aimed at predicting lung cancer recurrence by integrating **clinical data** and **CT scan images**. The project leverages both **machine learning** and **deep learning** models to improve prediction accuracy and support early clinical decision-making.

This project is designed as a **Bachelor of Engineering (B.E.) major project** and focuses on building an end-to-end, research-oriented, real-world medical AI system.

---

## 🎯 Objectives

* Predict the likelihood of lung cancer recurrence at an early stage
* Fuse heterogeneous data sources (clinical + imaging)
* Improve prediction performance over unimodal approaches
* Provide an interpretable and deployable AI solution for healthcare

---

## 🧠 System Architecture

The system follows a **multimodal fusion pipeline**:

1. **Clinical Data Processing**

   * Data cleaning and preprocessing
   * Feature engineering and normalization
   * Model: XGBoost 

2. **Image Data Processing**

   * CT scan preprocessing
   * Feature extraction using CNN (VGG16)
   * Deep feature representation

3. **Multimodal Fusion**

   * Fusion of clinical and imaging features
   * Optimized feature combination strategy using Late fusion

4. **Prediction Layer**

   * Final recurrence prediction (Yes / No)

5. **Web Interface (Optional Deployment)**

   * Streamlit-based UI for prediction

---

## 🛠️ Tech Stack

### Programming & Frameworks

* Python 3.x
* NumPy, Pandas
* Scikit-learn
* TensorFlow / Keras
* OpenCV

### Models Used

* **XGBoost** – Clinical data prediction
* **VGG16 (CNN)** – CT image feature extraction


### Tools

* Jupyter Notebook
* Streamlit (for web deployment)
* Git & GitHub

---



## ⚙️ Installation & Setup

1. Clone the repository:

```bash
git clone https://github.com/your-DishaRanganath/Intelligent-Optimized-Multimodal-Learning-Framework-for-Early-Prediction-of-Lung-Cancer-recurrence.git
cd Intelligent-optimized-Multimodal-Learning-framework-for-early-Prediction-of-Lung-Cancer-recurrence
```



3. Run the web application:

```bash
streamlit run app.py
```

---

## 📊 Results

* Improved accuracy and robustness
* Better generalization for recurrence prediction.

---

## 🚀 Future Enhancements

* Integration of survival analysis
* Explainable AI (SHAP / Grad-CAM)
* Real-time hospital data integration
* Cloud deployment (AWS / Azure)

---

## ⚠️ Disclaimer

This project is intended for **academic and research purposes only**. It is **not a certified medical diagnostic tool** and should not be used for real-world clinical decisions without professional validation.

---

## 👩‍💻 Author

**Disha R**
Department of Electronics and Communication Engineering
Bachelor of Engineering (B.E.) Major Project

---

## ⭐ Acknowledgements

* Open-source medical datasets
* TensorFlow & Scikit-learn communities
* Research papers on multimodal medical AI
