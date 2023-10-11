# SSRROO6
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
# Загрузим датасет "Вино"
wine = load_wine()
X = wine.data
y = wine.target
# Разделим датасет на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Создадим модель Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)  # Решающее дерево
# Обучим модель Decision Tree на тренировочных данных
dt_model.fit(X_train, y_train)
# Сделаем прогнозы на тестовых данных с Decision Tree
y_pred_dt = dt_model.predict(X_test)
# Оценим точность модели Decision Tree
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f'Точность модели Decision Tree: {accuracy_dt:.2f}')
# Создадим модель Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)  # Случайный лес
# Обучим модель Random Forest на тренировочных данных
rf_model.fit(X_train, y_train)
# Сделаем прогнозы на тестовых данных с Random Forest
y_pred_rf = rf_model.predict(X_test)
# Оценим точность модели Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Точность модели Random Forest: {accuracy_rf:.2f}')
# Выведем отчет по классификации для Random Forest
report_rf = classification_report(y_test, y_pred_rf, target_names=wine.target_names)
print('Отчет по классификации для Random Forest:')
print(report_rf)
