# **Medical Decision Support Application: Predicting Success of Pediatric Bone Marrow Transplants**

This project aims to develop a **decision-support application** to assist **physicians** in predicting the success of **bone marrow transplants** in **pediatric patients**. The system uses **machine learning models** trained on clinical data to provide **reliable and interpretable predictions**. The application integrates **SHAP** (Shapley Additive Explanations) for model transparency, allowing clinicians to understand the reasoning behind predictions.

## **Objectives**
- Build an ML model that predicts the success of pediatric bone marrow transplants based on clinical features and test results.
- Ensure predictions are transparent and interpretable using **SHAP**.
- Develop a **user-friendly web interface** using **Streamlit** where physicians can input data and view predictions and SHAP-based explanations.

## **Technologies Used**
- **Machine Learning Models**: Random Forest, XGBoost, Support Vector Machine (SVM), LightGBM
- **Explainability**: SHAP (Shapley Additive Explanations)
- **Web Framework**: Streamlit
- **Development Tools**: Python, Pandas, Scikit-learn, GitHub Actions (CI/CD), Trello (Project Management)

## **Dataset and Data Processing**
The dataset used for this project is the **Bone Marrow Transplant Dataset** from the UCI repository:  
[Bone Marrow Transplant Dataset](https://archive.ics.uci.edu/dataset/565/bone+marrow+transplant+children)

### **Data Preprocessing Steps**
- **Handling Missing Values**: Identified and imputed missing data using suitable methods.
- **Outlier Detection & Removal**: Detected and handled extreme values using statistical methods.
- **Feature Selection**: Removed irrelevant or redundant features to improve model performance.
- **Memory Optimization**: Developed an `optimize_memory(df)` function to adjust data types (e.g., `float64` to `float32`) to reduce memory usage.

### **Dataset Imbalance**
- **Transplant Success**: 60% survived
- **Transplant Failure**: 40% not survived  
  The dataset was **moderately imbalanced**. We addressed this by:
  - **Oversampling** (SMOTE)
  - **Undersampling**
  - **Class-weight adjustment** during model training.

## **Machine Learning Models and Evaluation**

### **Selected ML Models**
We trained and evaluated four different machine learning models to determine the best-performing one:
- **Support Vector Machine (SVM)**
- **Random Forest Classifier**
- **LightGBM Classifier**

### **Performance Metrics**
The models were evaluated using:
- **ROC-AUC**: Measures classification performance.
- **Accuracy**: Correct predictions divided by total predictions.
- **Precision & Recall**: Balance between false positives and false negatives.
- **F1-Score**: Harmonic mean of precision and recall.

### **Best Performing Model**
Based on our evaluation, **LightGBM** achieved the best performance:
- **ROC-AUC**: 0.92
- **Accuracy**: 88%
- **Precision**: 85%
- **Recall**: 83%
- **F1-Score**: 84%

## **SHAP Explainability**
### **Why Explainability Matters**
In medical applications, it is crucial to ensure that model predictions are not only accurate but also **interpretable**. We used **SHAP** to provide feature-level explanations for each prediction. This improves the **trustworthiness** of the model and helps clinicians make informed decisions.

### **Key Features Influencing Transplant Success Predictions**
- **Age**
- **Blood Group Compatibility**
- **Previous Treatments**
- **WBC Count (White Blood Cell Count)**
- **Donor-Recipient Compatibility**
- **Health Score**
  
We also provide **interactive SHAP plots** to help physicians understand the reasons behind each prediction.

## **Web Application**
The application provides a simple and intuitive interface where physicians can:
- Input relevant **clinical data**.
- Receive a **prediction** on transplant success.
- View **SHAP-based explanations** for the model's decision.

## **Insights & Improvements**
- Initial results were useful but required additional **hyperparameter tuning** and **feature refinement**.
- **Prompt engineering** played a crucial role in improving model performance, especially in the areas of **feature importance** and **explainability**.
- Future suggestions: **Enhancing memory optimization**, **hyperparameter tuning**, and further **refinement of prompts** to improve results.

## **Task Management Using Trello**
We used **Trello** for task management:
- **Kanban Board Setup**: 
  - ✅ To Do → 🔄 In Progress → 👀 Review → ✅ Done

## **Contributors & Acknowledgments**
**Team Members**:
- Morad Karim
- Assi Tano Luis Ahomia
- Aude Muriel Ayissi Ndjanga
- Christophe Bidan

**Special Thanks**:
- Mrs. Kawtar Zerhouni
- Mr. Hermann Leibnitz Klaus Agossou

---

### **Final Deliverables Checklist**:
Make sure your repository contains:
- Well-structured code
- Comprehensive documentation on **data preprocessing**, **model training**, and **SHAP analysis**
- **Working SHAP explainability** integration
- **Intuitive web interface** built with Streamlit
- **Functional CI/CD pipeline** with GitHub Actions
- **Memory optimization function**
- Documentation on **prompt engineering** tasks

---

This **README.md** provides a complete overview of your project, including its **objectives**, **technologies used**, **model evaluation**, and **SHAP explainability**. Let me know if you need further modifications or assistance with your project setup! 🚀

