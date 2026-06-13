# 🎬 IMDB Sentiment GRU Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-FF6F00.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B.svg)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000.svg)


A production-grade Deep Learning text classification dashboard built using a **Gated Recurrent Unit (GRU)** architecture trained on the IMDB Movie Review Dataset. The web layout features professional UI integration, interactive validation analytics dashboards, real-time textual inference profiling, and batch parsing utility pipelines.

## ✨ Core Features
* 🔮 **Real-Time Classifier:** Type or paste any movie review into the UI to instantly see the neural network's sentiment prediction (Positive/Negative) alongside a confidence probability gauge.
* 📊 **Model Validation Analytics:** Explore the interactive confusion matrix and classification reports detailing the model's 88% accuracy across 9,917 test samples.
* 📁 **Batch Processing Pipeline:** Upload `.csv` or `.xlsx` files containing hundreds of reviews to process them concurrently and download the sentiment predictions as a new file.

## 📈 Model Performance
The serialized sequential neural network reaches a high performance accuracy score of **88%** across an independent test evaluation data partition:

* **Overall Accuracy:** 0.88
* **Negative Sentiment F1-Score:** 0.89 (0.87 Precision / 0.90 Recall)
* **Positive Sentiment F1-Score:** 0.88 (0.90 Precision / 0.87 Recall)

## 📁 Architecture Setup
Ensure your root project directory contains the following structural files before executing the application:


📦 IMDB-Sentiment-GRU-Engine

    ┣ 📜 app.py                  # Streamlit Interactive Dashboard Front-End 
    ┣ 📜 best_gru_model.keras    # Pre-trained Keras GRU Network Weights (Keras 3)
    ┣ 📜 tokenizer.pkl           # Pickled Keras TextTokenizer Instance
    ┣ 📜 model_config.pkl        # Pickled serialized hyperparameter constraints
    ┗ 📜 README.md               # Project documentation

🚀 Local Installation & Setup
⚠️ Important Notice for Windows Users: This project utilizes the CPU-only version of TensorFlow (tensorflow-cpu) running inside a strictly isolated Virtual Environment to prevent common pywrap_function_lib DLL load errors.

1. Clone the repository

        git clone [https://github.com/akshitgajera1013/IMDB-Sentiment-GRU-Engine.git](https://github.com/akshitgajera1013/IMDB-Sentiment-GRU-Engine.git)

        cd IMDB-Sentiment-GRU-Engine

2. Create a Virtual Environment

Isolate the project dependencies to ensure a clean installation.

    python -m venv env

3. Activate the Environment

Windows: 

    env\Scripts\activate

Mac/Linux:

    source env/bin/activate

4. Install Dependencies
   
Install the required packages directly via pip (ensure your virtual environment is active):

    pip install requirements.txt

5. Launch the Dashboard
   
Start the local Streamlit application server:

    streamlit run app.py
