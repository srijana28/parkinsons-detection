# 🧠 Parkinson’s Disease Detection Using Machine Learning (Voice-Based)

## 📌 Project Overview
Parkinson’s Disease (PD) is a progressive neurological disorder that affects movement and speech. Early detection can help in better disease management and improve the quality of life of patients.

This project implements a **machine learning-based system to detect Parkinson’s Disease using voice features**. The model analyzes biomedical vocal measurements and predicts whether a person is affected by Parkinson’s Disease or not.

---

## 🎯 Objectives
- To analyze voice-based features related to Parkinson’s Disease  
- To train and evaluate machine learning models for disease detection  
- To select the best-performing model  
- To save the trained model for deployment in a web application  

---

## 🗂️ Dataset
**UCI Machine Learning Repository – Parkinson’s Disease Voice Dataset**

- Total Samples: 195  
- Subjects: 31 (23 Parkinson’s patients, 8 healthy individuals)  
- Features: 22 voice-related biomedical attributes  
- Target Column:
  - `1` → Parkinson’s Disease  
  - `0` → Healthy  

🔗 Dataset Link:  
https://archive.ics.uci.edu/ml/datasets/parkinsons

---

## ⚙️ Technologies Used

### Programming Language
- Python 3.x

### Libraries
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- Joblib  

### Tools
- VS Code  
- Jupyter Notebook (optional)

---

## 🧠 Machine Learning Models Used
- Support Vector Machine (SVM)  
- Random Forest Classifier  
- Logistic Regression  

📌 **SVM was chosen as the final model** because it achieved the highest accuracy and performs well on small, high-dimensional biomedical datasets.

---

## 🔁 Methodology
1. Load the Parkinson’s voice dataset  
2. Perform data preprocessing and feature scaling  
3. Split data into training and testing sets  
4. Train multiple ML models  
5. Evaluate models using performance metrics  
6. Select the best model  
7. Save the trained model and scaler  

---

## 📁 Project Structure
parkinsons-detection/
├── data/
│ └── parkinsons.data
├── model/
│ ├── parkinsons_model.pkl
│ └── scaler.pkl
├── train_model.py
├── README.md
└── requirements.txt


---

## 🚀 How to Run the Project

### 1️⃣ Install Required Libraries
```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib 
```


### How to Run
``` bash 
python train_model.py
```
### 3️⃣ Output

Model performance metrics will be displayed in the terminal

Trained model and scaler will be saved in the model/ directory

### 📊 Evaluation Metrics

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

### 🧪 Results

The Support Vector Machine (SVM) model achieved an accuracy of approximately 85–92%, making it suitable for Parkinson’s Disease prediction using voice features.

### 🌐 Future Scope

Real-time voice input and feature extraction

Deep learning models (CNN / RNN)

Web or mobile application deployment

Multimodal detection using voice, gait, and handwriting

### 📚 References

UCI Machine Learning Repository – Parkinson’s Dataset

Research papers on Parkinson’s Disease detection using ML

Scikit-learn Documentation

