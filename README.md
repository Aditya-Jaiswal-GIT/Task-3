# 🚢 Titanic Survival Prediction (Task 3 - AI & ML Internship)

## 📌 Objective
Build a Logistic Regression model to predict whether a passenger survived the Titanic disaster based on attributes such as age, class, sex, and embarkation point.

---

## 📁 Dataset
The dataset used is the *Titanic dataset*, provided in the internship task (Document from Aditya). It contains details about passengers such as:

- Pclass – Passenger class (1st, 2nd, 3rd)
- Sex – Gender
- Age – Age in years
- SibSp – # of siblings/spouses aboard
- Parch – # of parents/children aboard
- Fare – Passenger fare
- Embarked – Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
- Survived – Target variable (0 = No, 1 = Yes)

---

## 🛠 Tools & Libraries
- *Python*
- pandas – data loading and preprocessing
- scikit-learn – modeling, evaluation
- matplotlib – visualization

---

## 🧪 Workflow

### 1. Data Preprocessing
- Dropped high-cardinality or irrelevant columns: Cabin, Ticket, Name, PassengerId
- Filled missing values in Age (median) and Embarked (mode)
- Converted categorical variables (Sex, Embarked) into numeric using one-hot encoding

### 2. Modeling
- Used *Logistic Regression* with class balancing
- Data was split into *80% training* and *20% testing*

### 3. Evaluation Metrics
- Accuracy
- Precision, Recall, F1-score
- ROC-AUC score
- Confusion Matrix
- ROC Curve

---

## 📊 Results
> (Add your actual results after running the script)

- *Accuracy*: e.g., 82.13%
- *ROC-AUC Score*: e.g., 0.87

Confusion matrix and ROC curve are visualized using matplotlib.

---

## 🔍 Key Learnings
- Logistic Regression can model binary outcomes efficiently
- Preprocessing (like handling missing values and encoding categoricals) significantly affects model performance
- Evaluation metrics provide deeper insight than accuracy alone (especially for imbalanced datasets)

---

## 💡 Possible Improvements
- Feature engineering (e.g., FamilySize = SibSp + Parch + 1)
- Try ensemble models (e.g., Random Forest, Gradient Boosting)
- Hyperparameter tuning with GridSearchCV

---

## 📎 How to Run
1. Install dependencies:
   ```bash
   pip install pandas scikit-learn matplotlib
