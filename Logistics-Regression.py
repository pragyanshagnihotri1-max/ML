# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create dataset
data = {
    'Hours_Studied': [1,2,3,4,5,6,7,8],
    'Pass': [0,0,0,0,1,1,1,1]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features (X) and Target (y)
X = df[['Hours_Studied']]
y = df['Pass']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Logistic Regression model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Predict results
y_pred = model.predict(X_test)

# Model accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Predict new data
new_data = [[5]]
prediction = model.predict(new_data)

print("Prediction (0 = Fail, 1 = Pass):", prediction[0])
