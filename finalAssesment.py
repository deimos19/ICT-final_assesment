import pandas as pd
import numpy as np

data=pd.read_csv('/content/train_LZdllcl.csv')

data['education']=data['education'].fillna(data['education'].mode()[0])
data['previous_year_rating']=data['previous_year_rating'].fillna(data['previous_year_rating'].mean())
#check for null
data.isnull().sum()

data=pd.get_dummies(data,columns=['department','gender','education','recruitment_channel'])
data.head()

X=data.drop(columns=['employee_id', 'is_promoted','region'])
y=data['is_promoted']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)

test_data=pd.read_csv('/content/test_2umaH9m.csv')
test_data['education']=test_data['education'].fillna(test_data['education'].mode()[0])
test_data['previous_year_rating']=test_data['previous_year_rating'].fillna(test_data['previous_year_rating'].mean())
#check for null
data.isnull().sum()

test_data=pd.get_dummies(test_data,columns=['department','gender','education','recruitment_channel'])
data.head()

X=test_data

X.head()

from sklearn.preprocessing import StandardScaler

# Select only numerical columns from test_data
numerical_cols = test_data.select_dtypes(include=['number']).columns
test_data_numerical = test_data[numerical_cols]

scaler = StandardScaler()
scaler.fit(test_data_numerical)  # Fit the scaler to numerical data only

# Transform numerical data
X_test_scaled = scaler.transform(test_data_numerical)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)

