#Load libraries
import selectors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

#load dataset
housing = fetch_california_housing()

#Split dataset
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

#Preprocessamento dos dados
preprocessor = ColumnTransformer(
    [("scaler", StandardScaler(), list(range(X_train.shape[1])))]
)

#Arquitetura do MLP
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
    keras.layers.Dense(units=50, activation='relu'),
    keras.layers.Dense(units=25, activation='relu'),
    keras.layers.Dense(units=5, activation='relu'),
    keras.layers.Dense(1)
])

#Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Ajustar o pré-processador nos dados de treinamento
pipeline.named_steps['preprocessor'].fit(X_train)

# Compila e treina o modelo
pipeline.named_steps['model'].compile(optimizer='sgd', loss='mean_squared_error', metrics=['mse', 'mae'])

history  = pipeline.named_steps['model'].fit(
    pipeline.named_steps['preprocessor'].transform(X_train),
    y_train, 
    epochs=30, 
    batch_size=100,
    validation_data=(pipeline.named_steps['preprocessor'].transform(X_valid), y_valid))

#Gráficos
pd.DataFrame(history.history)[['loss', 'val_loss', 'mse', 'val_mse']].plot(figsize=(8,5))
plt.grid(True)
plt.show()

pd.DataFrame(history.history)[['mae', 'val_mae']].plot(figsize=(8,5))
plt.grid(True)
plt.show()

# Avaliar o modelo nos dados de teste
test_loss, test_mse, test_mae = pipeline.named_steps['model'].evaluate(
    pipeline.named_steps['preprocessor'].transform(X_test), y_test
)
print(f'Test Loss: {test_loss}, Test MSE: {test_mse}, Test MAE: {test_mae}')

# Fazer previsões nas três primeiras linhas dos dados de teste
predictions = pipeline.named_steps['model'].predict(pipeline.named_steps['preprocessor'].transform(X_test[:3]))
print("Predictions:")
print(predictions)
print("Valores da base de dados teste:")
print(y_test[:3])