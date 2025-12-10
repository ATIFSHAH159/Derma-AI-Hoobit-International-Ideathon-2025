# 🚀 DermAI - Skin Cancer Detection and Recommendation System

**AI-powered web application for early skin lesion detection with medical explanation and dermatologist recommendations.**

---

## Badges
![Python](https://img.shields.io/badge/python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-1.1.2-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange)
![HTML5](https://img.shields.io/badge/HTML5-orange)
![CSS3](https://img.shields.io/badge/CSS3-blue)
![JavaScript](https://img.shields.io/badge/JavaScript-yellow)

---

## Inspiration
Skin cancer is one of the most common cancers worldwide, yet early detection is often delayed due to limited access to dermatologists, high costs, and lack of awareness. I wanted to create an accessible AI-powered tool that helps users quickly analyze skin lesions, understand the results, and get recommendations for dermatologists in Pakistan.

---
## ✨ Features

- **AI Classification:** Classifies 7 types of skin lesions using EfficientNetB0  
- **LLM Medical Analysis:** Provides AI-generated medical explanations using Google Gemini AI  
- **Dermatologist Recommendations:** Suggests relevant dermatologists in Pakistan  
- **Interactive Web Interface:** Drag-and-drop image upload, animated prediction results, and chatbot integration  
- **Report Generation:** Downloadable PDF or DOCX reports summarizing predictions, confidence scores, and recommendations  
- **Future Mobile Support:** Real-time camera capture for mobile users  
- **User History & Login:** Track previous scans, reports, and risk patterns (planned for next version)  
- **Cloud Deployment:** Compatible with AWS / Render / Railway for scalable hosting  

---
## What it does
**DermAI** allows users to:  
- Upload an image of a skin lesion  
- Automatically classify it into one of seven categories using **EfficientNetB0**  
- Receive a confidence score for the prediction  
- Get an AI-generated medical explanation using **Google Gemini AI**  
- Obtain dermatologist recommendations in Pakistan  
- Download a detailed PDF or DOCX report summarizing the results  

---

## How we built it
- **Frontend:** HTML, CSS, JavaScript for a responsive and interactive interface  
- **Backend:** Python Flask handles image uploads, preprocessing, and model inference  
- **AI Models:**  
  - **EfficientNetB0** for lesion classification  
  - **Google Gemini AI** for medical reasoning and explanation  
- **Dataset:** HAM10000 images (part 1 & part 2)  
- **Reports:** PDF/DOCX generation using `reportlab` and `python-docx`  
- **UI/UX Assistance:** Claude AI for design inspiration and coding guidance  

---

## Challenges we ran into
- Integrating **Google Gemini AI** with Flask  
- Processing large HAM10000 image datasets efficiently  
- Designing a clean and user-friendly UI for both desktop and mobile  
- Securing API keys while preparing the project for GitHub  

---

## Accomplishments
- Successfully combined **deep learning classification** with **LLM explanations**  
- Built a working **web interface** with report generation and dermatologist recommendations  
- Created a **reproducible MVP** demonstrating AI-assisted skin lesion analysis  
- Ensured a **secure workflow** by excluding `.env` and `.venv` from GitHub  

---

## What we learned
- How to integrate **computer vision models** with **large language models** for healthcare  
- Best practices for **handling sensitive API keys** and large datasets  
- Web deployment strategies and **downloadable medical reports**  
- Importance of **UI/UX design** in user adoption of AI tools  

---

## What's next for DermAI
- Improve model accuracy with larger datasets and advanced augmentation  
- Add **real-time camera capture** for mobile and web  
- Implement **user accounts & history tracking**  
- Deploy on **cloud hosting** with GPU support for faster inference  
- Develop a **mobile app version** for iOS and Android  
- Enhance UI/UX with interactive dashboards and risk visualizations  

---

## Setup Instructions

### Prerequisites
- Python 3.10+  
- Node.js (optional, for frontend enhancements)  
- Kaggle account to download HAM10000 dataset  

### Installation
1. Clone the repository:

```bash
git clone https://github.com/ATIFSHAH159/Derma-AI-Hoobit-International-Ideathon-2025
cd DermAI

