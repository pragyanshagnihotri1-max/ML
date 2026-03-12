# Import libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Create dataset
data = {
    'Experience': [1, 2, 3, 4, 5, 6, 7],
    'Salary': [30000, 35000, 40000, 45000, 50000, 55000, 60000]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features (X) and Target (y)
X = df[['Experience']]
y = df['Salary']

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict salary
predicted_salary = model.predict([[5]])

print("Predicted Salary for 5 years experience:", predicted_salary[0])
