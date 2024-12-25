# Heart Disease Prediction - Data Analysis & Machine Learning Project

## Description

This project explores a dataset on cardiovascular disease diagnosis and develops a machine learning model to predict its presence in individuals.

The dataset contains clinical parameters and medical results. The main goal is to use exploratory analysis and supervised learning techniques to understand risk factors and create a predictive classification model.

---

## Table of Contents

1. [Dataset](#dataset)
2. [Tools and Technologies](#tools-and-technologies)
3. [Approach](#approach)
4. [Prerequisites](#prerequisites)
5. [How to Run the Project](#how-to-run-the-project)

---

## Dataset

### Key Columns Description

Here are the most important variables in the dataset:

- **age**: Age of the patient (in years)
- **sex**: Gender (1 = male, 0 = female)
- **cp**: Type of chest pain (4 categories)
- **trestbps**: Resting blood pressure (in mm Hg)
- **chol**: Serum cholesterol level (mg/dl)
- **fbs**: Fasting blood sugar (> 120 mg/dl: 1 = true, 0 = false)
- **restecg**: Resting electrocardiographic results
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise-induced angina (1 = yes, 0 = no)
- **oldpeak**: ST depression induced by exercise (relative to rest)
- **target**: Predictive target (1 = presence of disease, 0 = absence)

For a full definition of all variables, refer to the `data_dictionary.json` file.

---

## Tools and Technologies

This project uses:

- **Python** (3.x)
- **Jupyter Notebook**
- **Key Libraries**:
  - Pandas: Data manipulation and analysis
  - Matplotlib and Seaborn: Data visualization
  - Scikit-learn: Machine learning modeling and evaluation
  - xgboost
  - numpy
- **Other Libraries**:
  - joblib   
  - seaborn

---

## Approach
1. **Problem Definition:**
2. **Data loading and splitting:**
3. **Exploratory Data Analysis (EDA):**
   - Visualization of the relationship between 2 or more variables
   - Visualization of correlations between variables
   - Check variables distributions

4. **Modelling:**
   - Models tested:
     - Logistic Regression
     - Random Forest Classifier
     - KNN
     - Gradient Boosting
     - XGBoost
5. **Model comparison**
     - Accuracy
5. **Hyperparameter tuning for most promising models:**
    - Randomized Search
    - Grid Search
6. **Model Evaluation**
    - ROC Curve and AUC score
    - Confusion Matrix
    - Classification Report
    - Precision
    - Recall
    - F1-score
7. **Feature Importance:**


## Prerequisites

- Python 3.8 or later
- A virtual Python environment(via `venv` or `conda`)
- Jupyter Notebook to run the `heart_disease.ipynb` file

---

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/Dan-Popescu/heart-disease-prediction.git
   cd heart-disease-prediction
   ```
2. Activate the virtual environment:
   ```bash
   source venv/bin/activate 
   On Windows, use 'venv\Scripts\activate'
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the notebook:
   ```bash
   jupyter notebook heart_disease.ipynb
   ```

4. Follow the steps in the notebook to execute the analysis and models.


