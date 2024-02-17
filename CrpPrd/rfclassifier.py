import json
import pickle
import os

from fastapi import FastAPI

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)

from models import Values

df = pd.read_csv("./Crop_recommendation.csv")

class_labels = df['label'].unique().tolist()

le = LabelEncoder()

df['label'] = le.fit_transform(df['label'])

class_labels = le.classes_

x = df.drop('label', axis=1)
y = df['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, shuffle=True)

rf = RandomForestClassifier()
param_grid = {'n_estimators': np.arange(50, 200),
              'criterion': ['gini', 'entropy'],
              'max_depth': np.arange(2, 25),
              'min_samples_split': np.arange(2, 25),
              'min_samples_leaf': np.arange(2, 25)}

rscv_model = RandomizedSearchCV(rf, param_grid, cv=5)
rscv_model.fit(x_train, y_train)
rscv_model.best_estimator_

new_rf_model = rscv_model.best_estimator_

y_pred = new_rf_model.predict(x_test)

label_dict = {}
for index, label in enumerate(class_labels):
    label_dict[label] = index

features_data = {'columns': list(x.columns)}

with open('new_rf_model.pickle', 'wb') as file:
    pickle.dump(new_rf_model, file)

with open('features_data.json', 'w') as file:
    json.dump(features_data, file)

app = FastAPI()


@app.get('/')
async def hi():
    return "Hi"


@app.post('/predict')
async def predictCrop(values: Values):
    print(values)

    test_series = pd.Series(np.zeros(len(features_data['columns'])), index=features_data['columns'])
    test_series['N'] = int(values.N)
    test_series['P'] = int(values.P)
    test_series['K'] = int(values.K)
    test_series['temperature'] = values.temperature
    test_series['humidity'] = values.humidity
    test_series['ph'] = values.ph
    test_series['rainfall'] = values.rainfall
    probabilities = new_rf_model.predict_proba([test_series])[0]
    sorted_indices = np.argsort(probabilities)[::-1]
    top_classes = class_labels[sorted_indices[:3]]
    top_probabilities = probabilities[sorted_indices[:3]]

    result = [{"crop": crop, "probability": prob} for crop, prob in zip(top_classes, top_probabilities)]
    return {"crops": result}
