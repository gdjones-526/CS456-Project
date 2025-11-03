# CS456 Project: Machine Learning Web Platform

## Overview
This web application allows users to upload datasets, train machine learning models, and visualize model performance with metrics and plots. It supports both classification and regression tasks and provides an interactive dashboard for managing datasets and models.

---

## System Requirements
- **Python Version:** 3.10+
---

## Environment Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/gdjones-526/CS456-Project.git
   cd CS456-Project

2. **Setup Environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   
3. **Install requirements and migrate database**
   ```bash
   pip install -r requirements.txt
   python manage.py migrate

4. **Run Server**
   ```bash
   python manage.py runserver

This guide provides a step-by-step walkthrough to use the website.

---

## 1. Account Login / Sign Up

1. On first visit, you will see the **Account Login Page**.  
2. Enter your **username** and **password** if you already have an account.  
3. If you do not have an account, click **Sign Up Now**.  
4. On the **Sign Up Page**:  
   - Create a **username**.  
   - Create a **password** that is **unique from your username** and **at least 8 characters long**.  
   - Confirm your password.  
   - Click **Sign Up**.  
5. After signing up, you will be automatically logged in.

---

## 2. Dashboard Overview

After logging in, you will see the **Dashboard Screen**.

### A. Data Ingestion and Pre-Processing

1. Under the **Data Ingestion and Pre-Processing** tab:  
   - Upload your dataset (**CSV, Excel, or TXT** format).  
   - Optionally, add a **description** for your dataset.  
   - Click **Upload Dataset**.  

2. You will be redirected to the **Data Set Dashboard**:  
   - View dataset features: **total rows, columns, file size**, and **missing values**.  

---

### B. Training a Model

1. Scroll down to the **Actions** tab and click **Train Model**.  
2. In the **Train Machine Learning Model View**:  
   - **Select your target variable** from the dropdown.  
   - **Select features** that the model will use.  
   - Choose the **task type**: `Classification` or `Regression`.  
   - Based on the task type, select the **algorithm** for training.  
   - Under **Model Configuration**:  
     - Enter a **Model Name**.  
     - Optionally adjust:  
       - **Test Size**  
       - **Validation Size**  
       - **Missing Value Strategy**  
       - **Random State**  
   - Click **Start Training**.  
   - Training may take a few seconds.

---

### C. Viewing Model Results

1. After training, the **Model View** displays:  
   - **Classification Models**: Accuracy, Precision, Recall, F1 Score  
   - **Regression Models**: Error Metrics, Accuracy, MSE, RMSE  
2. Scroll down to see:  
   - **Training Configuration**  
   - **Model Configuration**  
   - **Features Used**  
   - **Visualizations**: ROC Curve, Confusion Matrix, Feature Importance  

---

## 3. Managing Data Sets

1. From the **Dashboard**, click **ML Platform** in the top-left corner.  
2. Scroll to **Your Data Sets**:  
   - View uploaded datasets, descriptions, upload date, and processing status.  
   - Delete a dataset if no longer needed.  

---

## 4. Model Evaluation and Comparison

1. Under **Model Evaluation and Metrics**:  
   - Models are categorized by type: `Classification` or `Regression`.  
   - Models are sorted by **accuracy** (highest to lowest).  

2. **Comparing Model Figures**:  
   - Select **Model 1** and **Model 2** from dropdowns.  
   - Choose the **Figure Type**: `ROC Curve`, `Confusion Matrix`, or `Feature Importance`.  
   - View side-by-side visualizations to compare models.

---


