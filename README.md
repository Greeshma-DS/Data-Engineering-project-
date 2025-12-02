

---
# Predicting Student Academic Performance
## **Team Members**
- **Bipin Puri** – Data Engineer  
- **Greeshma Chanduri** – Data Scientist  
- **Erla Venkata** – Data Engineer  

---

## **Introduction**
Traditional methods for analyzing student performance are manual and have limited scalability and real-time capabilities.  
Our solution leverages AWS cloud services to create a **cloud-based machine learning pipeline** that predicts student academic performance for early intervention.  

**Business Impact:**  
- Enables personalized learning  
- Supports proactive interventions to improve student outcomes  

---

## **Project Goals**
1. **Create Scalable Pipeline** – Implement end-to-end AWS architecture for data processing  
2. **Ensure Cost Efficiency** – Leverage serverless AWS services to minimize operational costs  
3. **Build Hybrid Ensemble Model** – Combine Random Forest, XGBoost, and Neural Networks  
4. **Develop User-Friendly Interface** – Deploy Streamlit web application for real-time predictions  

---

## **Dataset**
- **Source:** [UCI Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/student+performance)  
- **Records:** 395 students, 32 features  
- **Features:**  
  - Demographic: Gender, age (15–22)  
  - Socioeconomic: Parent education, occupation  
  - Behavioral: Study time, absences, alcohol use  
  - Academic: G1, G2, final grade (G3)  
- **Target Variable:** G3 binned into Low (0–9), Medium (10–14), High (15–20)  

---

## **Data Preprocessing**
- Label Encoding for categorical features  
- Normalization for numerical features  
- Train-Test Split (80% train / 20% test)  

---

## **Solution Architecture**
```

UCI Student Dataset (CSV, 395 records, 32 features)
│
▼
Data Ingestion & Storage (S3 / Local, Pandas / PySpark)
│
▼
Data Preprocessing (Label Encoding, Normalization, Train-Test Split)
│
▼
Hybrid Ensemble Model (Random Forest, XGBoost, Neural Network, Dynamic Weight Adjustment)
│
▼
Model Evaluation (Accuracy, F1, Precision, Recall, Feature Importance)
│
▼
Streamlit Web App (User Inputs, Prediction, Feature Importance & Probabilities)
│
▼
Data Storage / Monitoring (Save predictions, CloudWatch / Logs)

```

---

## **Hybrid Ensemble Model**
- Random Forest  
- XGBoost  
- Neural Network (MLP)  
- Dynamic weight adjustment based on G1 & G2 grades  

---

## **Model Evaluation**
- Metrics: Accuracy, F1-Score, Precision, Recall  
- Feature Importance: Highlights key predictors (e.g., G1, G2)  

---

## **Streamlit Web Application**
- User-friendly interface for entering 32 student features  
- Real-time prediction: Low / Medium / High performance  
- Displays feature importance and prediction probabilities  

---

## **Expected Deliverables**
1. Functional Web Application for student performance prediction  
2. Automated ML Pipeline handling data processing to model deployment  
3. Cost-Optimized AWS Solution leveraging free-tier services  

---

## **How to Use**
1. Clone this repository  
2. Access the dataset (`data/student_mat.csv`)  
3. Follow preprocessing steps in `notebooks/data_preprocessing.ipynb`  

## **References**

* [UCI Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/student+performance)
* [AWS Data Engineering Resources](https://github.com/PacktPublishing/Data-Engineering-with-AWS)
* LinkedIn Learning Course: [Data Science Methodologies: Making Business Sense](https://www.linkedin.com/learning/data-science-methodologies-making-business-sense)

```


