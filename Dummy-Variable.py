import pandas as pd

# Create a sample DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'City': ['Delhi', 'Mumbai', 'Delhi', 'Chennai']
}

df = pd.DataFrame(data)

# Create dummy variables for the City column
dummy_df = pd.get_dummies(df['City'], prefix='City')

print(dummy_df)

