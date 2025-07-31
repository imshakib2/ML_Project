# 🎓 Student Performance Prediction (ML Project)

This project is a machine learning-based solution that predicts student performance based on study hours and attendance. It utilizes both Logistic Regression and Decision Tree models to classify whether a student is likely to **Pass** or **Fail**.

---

## 📁 Project Structure

```
ML_Project/
├── data/
│   └── student_data.csv          # Input dataset
├── model/
│   ├── log_reg_model.pkl         # Trained Logistic Regression model
│   ├── dt_model.pkl              # Trained Decision Tree model
│   └── scaler.pkl                # Trained StandardScaler object
└── main.py                       # Main training and evaluation script
```

---

## 🧠 Features Used

- **Hours** (study hours per day)
- **Attendance** (% attendance in class)

---

## ⚙️ How It Works

1. **Data Loading**:
   - Reads the CSV dataset containing student records.

2. **Preprocessing**:
   - Handles missing values in attendance.
   - Converts categorical labels in "Result" to binary values.

3. **Feature Scaling**:
   - Normalizes input features using `StandardScaler`.

4. **Model Training**:
   - Trains:
     - Logistic Regression
     - Decision Tree Classifier

5. **Model Evaluation**:
   - Evaluates both models using:
     - Accuracy Score
     - Confusion Matrix

6. **Model Saving**:
   - Saves models and scaler using `pickle` in the `model/` directory.

---

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3 and the following packages installed:

```bash
pip install pandas scikit-learn
```

### Running the Project

```bash
python main.py
```

This will train the models and output performance metrics to the console.

---

## 📊 Sample Output

```
Sample Data:
   Hours  Attendance Result
0    4.0        88.0   Pass
...

Training data: (80, 2), Test data: (20, 2)

Logistic Regression Accuracy: 85%
Decision Tree Accuracy: 80%
```

---

## 📦 Output Files

- `log_reg_model.pkl`: Trained Logistic Regression model.
- `dt_model.pkl`: Trained Decision Tree model.
- `scaler.pkl`: Fitted scaler for transforming input features.

---

## 📌 Author

**Md. Shakib Hossen** – *Machine Learning Enthusiast*

---

## 📜 License

This project is licensed under the MIT License.
