# SSRROO6
from sklearn.ensemble import RandomForestClassifier

# Создадим модель Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)  # Можно настроить гиперпараметры по вашим потребностям

# Обучим модель на тренировочных данных
rf_model.fit(X_train, y_train)

# Сделаем прогнозы на тестовых данных
y_pred_rf = rf_model.predict(X_test)

# Оценим точность модели
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Точность модели Random Forest: {accuracy_rf:.2f}')
