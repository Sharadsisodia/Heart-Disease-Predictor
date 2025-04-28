import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#load dataset
# df= pd.read_csv('/Dataset/heart_disease_data.csv')    '/Dataset/heart_disease_data.csv'
df = pd.read_csv('Dataset/heart_disease_data.csv')
df.head()

#EDA
# print(df.shape)
sns.pairplot(df[['age', 'cp', 'thalach','target']])
# plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True , cmap="coolwarm", vmin=-1)
plt.title("featrure correlation heatmap ")
# plt.show()

#fearure and target variables
X = df[['age', 'cp', 'thalach']]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model= LogisticRegression()
model.fit(X_train, y_train)
y_pred= model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

print(y_proba)

print(classification_report(y_test, y_pred))

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(  f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")


def predict_heart_disease(age, cp, thalach):
  age = int(input("Enter age: "))
  cp = int(input("Enter cp: "))
  thalach = int(input("Enter thalach:"))
  user_data = pd.DataFrame([[age, cp, thalach]], columns=['age', 'cp', 'thalach'])
  prediction = model.predict(user_data)
  if (prediction[0]==1):
    print("The person has heart disease")
  else:
    print("The person does not have heart disease")


