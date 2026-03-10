# Hindi Character Recognition using CNN

A deep learning system that recognizes **handwritten Hindi (Devanagari) characters and digits** using a Convolutional Neural Network (CNN).

This project demonstrates a **complete machine learning pipeline** including model training, evaluation, API deployment, and a web interface for real-time predictions.

---

# Project Overview

This project builds a **CNN-based classifier** capable of recognizing **46 handwritten Devanagari characters and digits** from images.

The system provides an end-to-end ML workflow:

* CNN model built with **PyTorch**
* Model **training and evaluation pipeline**
* **FastAPI backend** for inference
* **Streamlit web interface** for interactive predictions
* **Performance evaluation** with confusion matrix

Users can upload a handwritten character image and receive the **predicted Hindi character with confidence score**.

---

# Model Performance

Dataset: **Devanagari Handwritten Character Dataset**

Number of classes: **46**

Test Accuracy:

**98.80%**

### Confusion Matrix

![Confusion Matrix](results/confusion_matrix.png)

---

# Demo

### Streamlit User Interface

![UI](results/ui.png)

### Prediction Example 1

![Prediction Example 1](results/prediction_example.png)

### Prediction Example 2

![Prediction Example 2](results/prediction_example_2.png)

### Prediction Example 3

![Prediction Example 3](results/prediction_example_3.png)

---

# Tech Stack

### Programming Language

* Python

### Machine Learning

* PyTorch
* Torchvision

### Backend API

* FastAPI
* Uvicorn

### Frontend Interface

* Streamlit

### Supporting Libraries

* Pillow
* Requests

---

# System Architecture

The project follows a **modular machine learning deployment pipeline**:

```
User Image Upload
        │
        ▼
Streamlit Web Interface
        │
        ▼
FastAPI Inference API
        │
        ▼
PyTorch CNN Model
        │
        ▼
Predicted Hindi Character
```

This mirrors **real-world machine learning deployment architecture** used in production systems.

---

# Project Structure

```
hindi_character_recognition
│
├── api
│   └── api.py                # FastAPI inference server
│
├── app
│   └── app.py                # Streamlit web interface
│
├── data
│   ├── raw                   # Original dataset
│   └── processed             # Processed dataset
│
├── models
│   └── hindi_cnn_best.pth    # Trained CNN model
│
├── results
│   ├── confusion_matrix.png
│   ├── evaluation_metrics.txt
│   ├── ui_demo.png
│   ├── prediction_example_1.png
│   ├── prediction_example_2.png
│   └── prediction_example_3.png
│
├── src
│   ├── dataset.py            # Dataset loading
│   ├── model.py              # CNN architecture
│   ├── train.py              # Model training
│   ├── evaluate_model.py     # Model evaluation
│   └── test_model.py         # Accuracy testing
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Installation

Clone the repository:

```
git clone https://github.com/Flash6699/hindi-character-recognition-cnn.git
cd hindi-character-recognition-cnn
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Running the Project

### Start the FastAPI Server

```
uvicorn api.api:app --reload
```

API documentation will be available at:

```
http://127.0.0.1:8000/docs
```

---

### Start the Streamlit Application

```
streamlit run app/app.py
```

Upload a handwritten Hindi character image to receive predictions.

---

# Example Output

Input: Handwritten character image

Output:

```
Predicted Character: क
Confidence: 98.4%
```

---

# Key Features

* CNN-based handwritten Hindi character recognition
* **High accuracy model (98.8%)**
* GPU-supported training
* FastAPI inference API
* Streamlit interactive UI
* Confusion matrix evaluation
* Modular machine learning pipeline

---

# Future Improvements

* Containerize application using **Docker**
* Add **real-time drawing canvas input**
* Deploy API on **cloud platforms (AWS / GCP / HuggingFace Spaces)**
* Extend system to **full Hindi word recognition**

---

# Author

Vedant

Machine Learning & AI Enthusiast

GitHub:
https://github.com/Flash6699

---

# License

This project is open-source and available under the **MIT License**.
