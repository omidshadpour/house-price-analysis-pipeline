# 🏠 House Price Analysis & Prediction Pipeline  
A complete end-to-end Machine Learning pipeline for predicting house prices using Python, Scikit-Learn, and XGBoost.  
This project demonstrates a clean, modular, and production-ready ML workflow — ideal for real-world applications and portfolio use.

---

## 📌 Project Overview
This project builds a full ML pipeline to predict house prices based on the Ames Housing dataset.  
It includes:

- Data preprocessing  
- Feature engineering  
- Model training & hyperparameter tuning  
- Model comparison  
- Feature importance visualization  
- Saving the best model for deployment  

The entire workflow is modular, scalable, and follows industry-standard practices.

---

## 📁 Project Structure

house_price_project/
│
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   └── test.csv
│
├── models/
│   └── best_model.pkl
│
├── reports/
│   └── figures/
│       └── feature_importance_top20.png
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   └── feature_importance.py
│
├── main.py
└── README.md

---

## ⚙️ Pipeline Steps

### **1. Data Preprocessing**
Handled by `DataPreprocessor`:
- Load raw data  
- Drop ID columns  
- Remove high-missing features  
- Remove low-variance features  
- Fill missing values (median/mode)  

### **2. Feature Engineering**
Handled by `FeatureEngineer`:
- Label Encoding for categorical features  
- Standard Scaling for numerical features  

### **3. Model Training**
Handled by `ModelTrainer`:
- Linear Regression  
- Random Forest (with GridSearchCV)  
- XGBoost (with GridSearchCV)  
- Evaluation using RMSE & R²  
- Automatic best model selection  

### **4. Feature Importance**
Handled by `FeatureImportancePlotter`:
- Extract feature importances  
- Plot top 20 features  
- Save visualization  

---

## 🧠 Best Model Results

| Model             | RMSE        | R²       |
|------------------|-------------|----------|
| LinearRegression | 34281.66    | 0.8468   |
| RandomForest     | 28274.78    | 0.8957   |
| **XGBoost**      | **26917.13** | **0.9055** |

**XGBoost** achieved the best performance and is saved as the final model.

---

## 🔍 Feature Importance (Top 20)

The project generates a bar plot showing the most influential features.  
Example output:

OverallQual
FullBath
GarageCars
BsmtQual
GrLivArea
...

The plot is saved at:

reports/figures/feature_importance_top20.png

---

## ▶️ How to Run the Pipeline

### **1. Install dependencies**
pip install -r requirements.txt


### **2. Run the main pipeline**
python main.py


This will:

- Preprocess data  
- Engineer features  
- Train all models  
- Compare performance  
- Save the best model  
- Generate feature importance plot  

---

## 🧩 Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-Learn  
- XGBoost  
- Seaborn / Matplotlib  
- Joblib  

---

## 📌 Key Highlights
- Fully modular ML architecture  
- Clean and scalable codebase  
- Automated model comparison  
- Production-ready preprocessing pipeline  
- Professional project structure  
- Ideal for ML portfolios and freelance work  

---

## 📬 Contact
If you have questions or want to collaborate, feel free to reach out.

Author: Omid Shadpour
Project Type: Portfolio / ML Pipeline


---

⭐ If you like this project, consider giving it a star on GitHub!
