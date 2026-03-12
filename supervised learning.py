# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score

# ----------------------------
# Example 1: Regression (Predict House Price)
# ----------------------------
# Create dataset
reg_data = {
    'Area': [1000,1500,1800,2400,3000,3500,4000],
    'Bedrooms': [2,3,3,4,4,5,5],
    'Age': [10,8,6,4,2,3,1],
    'Price': [200000,300000,350000,450000,550000,600000,650000]
}

df_reg = pd.DataFrame(reg_data)

# Features and target
X_reg = df_reg[['Area','Bedrooms','Age']]
y_reg = df_reg['Price']

# Split dataset
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Train Linear Regression
lin_model = LinearRegression()
lin_model.fit(X_train_reg, y_train_reg)

# Predict
y_pred_reg = lin_model.predict(X_test_reg)
print("Regression R^2 Score:", r2_score(y_test_reg, y_pred_reg))

# Predict new house price
new_house = [[2500,4,5]]
predicted_price = lin_model.predict(new_house)
print("Predicted House Price:", predicted_price[0])

# ----------------------------
# Example 2: Classification (Predict Pass/Fail)
# ----------------------------
# Create dataset
class_data = {
    'Hours_Studied': [1,2,3,4,5,6,7,8],
    'Pass': [0,0,0,0,1,1,1,1]
}

df_class = pd.DataFrame(class_data)

# Features and target
X_class = df_class[['Hours_Studied']]
y_class = df_class['Pass']

# Split dataset
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.25, random_state=42)

# Train Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train_class, y_train_class)

# Predict
y_pred_class = log_model.predict(X_test_class)
print("Classification Accuracy:", accuracy_score(y_test_class, y_pred_class))

# Predict new student result
new_student = [[5]]
prediction = log_model.predict(new_student)
print("Prediction (0=Fail, 1=Pass):", prediction[0])
