# Ex.No: 6               HOLT WINTERS METHOD
### Date: 15/05/2026



### AIM:

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:

```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv('/content/Customer_Transactions.csv')

print(data.head())

print(data.info())

print(data.describe())

data['last_purchase_date'] = pd.to_datetime(data['last_purchase_date'])

label_encoder = LabelEncoder()

data['gender'] = label_encoder.fit_transform(data['gender'])
data['country'] = label_encoder.fit_transform(data['country'])
data['feedback_text'] = label_encoder.fit_transform(data['feedback_text'])

data['purchase_year'] = data['last_purchase_date'].dt.year
data['purchase_month'] = data['last_purchase_date'].dt.month

data = data.drop(['last_purchase_date', 'customer_id'], axis=1)

X = data.drop('churned', axis=1)
y = data['churned']

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

data['annual_income'].plot(kind='hist', bins=20)
plt.xlabel('Annual Income')
plt.title('Income Distribution')
plt.show()

data['spending_score'].plot(kind='hist', bins=20)
plt.xlabel('Spending Score')
plt.title('Spending Score Distribution')
plt.show()


```

### OUTPUT:


<img width="592" height="859" alt="image" src="https://github.com/user-attachments/assets/f4eabfa4-7073-4281-bcb9-cc5e231cb2c1" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
