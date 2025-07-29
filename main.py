import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import os


os.makedirs('model', exist_ok=True)


df = pd.read_csv('data/student_data.csv')


print("Sample Data:")
print(df.head())


print("\nMissing Values Count:")
print(df.isnull().sum()) 


df['Attendance'] = df['Attendance'].fillna(df['Attendance'].mean())
print("\nMissing Values after handling:")
print(df.isnull().sum())


df['Result'] = df['Result'].apply(lambda x: 1 if x == 'Pass' else 0)


print("\nUpdated Data:")
print(df.head())


X = df[['Hours', 'Attendance']] 
y = df['Result']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining data: {X_train.shape}, Test data: {X_test.shape}")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


log_reg_model = LogisticRegression()
log_reg_model.fit(X_train_scaled, y_train)


dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_scaled, y_train)


y_pred_log_reg = log_reg_model.predict(X_test_scaled)
y_pred_dt = dt_model.predict(X_test_scaled)


accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
accuracy_dt = accuracy_score(y_test, y_pred_dt)


print(f"\nAccuracy of Logistic Regression: {accuracy_log_reg * 100:.2f}%")
print(f"Accuracy of Decision Tree: {accuracy_dt * 100:.2f}%")


conf_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)
print("\nConfusion Matrix for Logistic Regression:")
print(conf_matrix_log_reg)


conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
print("\nConfusion Matrix for Decision Tree:")
print(conf_matrix_dt)


with open('model/log_reg_model.pkl', 'wb') as f:
    pickle.dump(log_reg_model, f)

with open('model/dt_model.pkl', 'wb') as f:
    pickle.dump(dt_model, f)

print("\nModels saved successfully!")


student_name = input("\nEnter student name: ")
attendance = float(input("Enter attendance percentage: "))
hours_studied = float(input("Enter number of hours studied: "))


new_data = pd.DataFrame({
    'Hours': [hours_studied],
    'Attendance': [attendance]
})


with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


new_data_scaled = scaler.transform(new_data)


with open('model/log_reg_model.pkl', 'rb') as f:
    log_model = pickle.load(f)

with open('model/dt_model.pkl', 'rb') as f:
    dt_model = pickle.load(f)


log_prediction = log_model.predict(new_data_scaled)
dt_prediction = dt_model.predict(new_data_scaled)


print("\n========================================")
print(f"Student Name: {student_name}")
print("========================================")
print(f"Attendance Percentage: {attendance:.2f}%")
print(f"Hours Studied: {hours_studied:.2f} hours")
print("========================================")
print("Predictions:")
print("----------------------------------------")
print(f"Logistic Regression Prediction: {'Pass' if log_prediction[0] == 1 else 'Fail'}")
print(f"Decision Tree Prediction: {'Pass' if dt_prediction[0] == 1 else 'Fail'}")
print("========================================")
