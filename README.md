# 🎙️ Keyword Spotting System (Audio Deep Learning – Dockerized)

This project implements a **Keyword Spotting (KWS) system** using deep learning, designed with **production-grade deployment practices** in mind.
The system performs **audio classification** by recognizing a predefined set of spoken keywords from short `.wav` audio files.

Although the task focuses on speech audio, the **architecture and deployment pipeline are generic** and applicable to other machine learning domains such as computer vision or signal processing.

---

## 🚀 Project Overview

### 🔍 Task

* **Problem type:** Audio classification (Keyword Spotting)
* **Input:** `.wav` audio file
* **Output:** One predicted keyword

### 🏷️ Supported Keywords (10 classes)

```text
cat, dog, down, happy, left,
right, stop, up, wow, yes
```

---

## 🧠 Model Architecture

* **Framework:** TensorFlow / Keras
* **Model Type:** Convolutional Neural Network (CNN)
* **Features:** MFCCs (Mel-Frequency Cepstral Coefficients)
* **Output Layer:** Softmax (10 classes)

The model is trained offline and loaded into the API at inference time.

---

## 🏗️ System Architecture

The application follows a **production-oriented ML serving architecture**.

```
Client
  │
  ▼
Nginx (Reverse Proxy)
  │
  ▼
uWSGI (WSGI Server)
  │
  ▼
Flask API
  │
  ▼
TensorFlow Model
```

### 🔧 Component Roles

* **Client:** Sends HTTP POST requests containing `.wav` audio files
* **Nginx:** Handles incoming HTTP traffic and forwards requests
* **uWSGI:** Bridges Nginx and the Python application via WSGI
* **Flask:** Exposes the prediction endpoint
* **TensorFlow Model:** Performs keyword classification

---

## 🐳 Containerization with Docker

To ensure **portability, reproducibility, and scalability**, the application is fully containerized.

### 📦 Multi-Container Design

The system is split into two containers:

1. **Nginx Container**

   * Acts as a reverse proxy
   * Handles client requests

2. **Application Container**

   * Flask API
   * uWSGI server
   * TensorFlow model

### 🔄 Orchestration

* **Docker Compose** is used to:

  * Build images
  * Create a shared virtual network
  * Run the full system as a single application

---

## ▶️ How to Run the Project

### ✅ Prerequisites

* Docker
* Docker Compose

### 🛠️ Build the containers

```bash
docker-compose build
```

### ▶️ Start the application

```bash
docker-compose up
```

Once running, the API is accessible through **Nginx**.

---

## 📡 API Usage

* **Endpoint:** `/predict`
* **Method:** `POST`
* **Input:** `.wav` audio file
* **Response:** Predicted keyword

### Example response

```json
{
  "prediction": "happy"
}
```

---

## 📚 Key Concepts Covered

* Audio preprocessing and **MFCC feature extraction**
* CNN-based audio classification
* Model inference with **TensorFlow**
* REST API development using **Flask**
* Production serving with **uWSGI + Nginx**
* Multi-container applications with **Docker Compose**

---

## ☁️ Deployment Note

The application is **cloud-ready** and designed to be deployed on platforms such as **AWS EC2**.

Due to account limitations, the deployment was completed **up to the Dockerized stage**, which already reflects **real-world production ML practices**.

---

## 🔗 Reference

* MFCC Feature Extraction
  [https://www.geeksforgeeks.org/nlp/mel-frequency-cepstral-coefficients-mfcc-for-speech-recognition/](https://www.geeksforgeeks.org/nlp/mel-frequency-cepstral-coefficients-mfcc-for-speech-recognition/)

---

