
🫁 Optimized Multi-Model Fusion for Accurate Lung Cancer Prediction (OMMRLC)

This project focuses on developing a hybrid deep learning and machine learning framework for accurate lung cancer prediction.
It fuses insights from clinical data and medical imaging to achieve high diagnostic accuracy using XGBoost and VGG16 models.

🧠 Project Overview

Early detection of lung cancer significantly improves treatment outcomes and survival rates.
However, relying on a single data source (like images or clinical records alone) often limits accuracy.

The OMMRLC system integrates:

-A clinical model trained on patient data using XGBoost.

-An image model trained on CT scan images using VGG16 (a convolutional neural network).

-By fusing both models, the system leverages complementary information from structured and visual data to provide robust and precise cancer predictions.

⚙️ Key Features

🩺 Clinical Data Model (XGBoost): Trained on structured patient data such as age, gender, smoking history, and biomarkers.

🧠 Image Model (VGG16): Utilizes transfer learning from VGG16 pretrained on ImageNet for lung CT image classification.

🔗 Model Fusion: Combines outputs from both models using weighted averaging or a meta-classifier for improved performance.

📈 Comprehensive Evaluation: Analyzes both individual and fused model metrics for accuracy, recall, precision, and ROC-AUC.

🧩 Project Workflow

1.Data Preparation

Collected and cleaned clinical data (CSV format).

Preprocessed CT scan images (resizing, normalization, augmentation).

2.Clinical Model Training (XGBoost)

Encoded categorical features and handled missing values.

Trained an XGBoost classifier for predicting cancer probability.

3.Image Model Training (VGG16)

Used transfer learning with pre-trained VGG16 architecture.

Fine-tuned layers for domain-specific learning on medical images.

4.Fusion Layer

Combined prediction probabilities from both models.

Used a simple average or a meta-learner for final classification.

5.Evaluation

Compared individual vs. fused model performance using metrics:

     Accuracy

    Precision

    Recall

    F1-score

    ROC-AUC

🧰 Tech Stack

-Language: Python

-Libraries: NumPy, Pandas, Scikit-learn, XGBoost, TensorFlow, Keras, Matplotlib, Seaborn

-Tools: Jupyter Notebook / Google Colab

📊 Results & Insights

-The fused model achieved significantly higher accuracy compared to standalone models.

-Clinical and imaging features provided complementary diagnostic insights.

-XGBoost + VGG16 fusion improved both sensitivity and specificity for early lung cancer detection.
