#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

#%%
url = "https://raw.githubusercontent.com/chandanverma07/DataSets/master/Car_sales.csv"
dataset = pd.read_csv(url)
#%%
dataset
# %%
dataset.info()
#%%
dataset['4-year resale value'] = pd.to_numeric(dataset['4-year resale value'],errors='coerce')
dataset['Price in thousands'] = pd.to_numeric(dataset['Price in thousands'],errors='coerce')
dataset['Engine size'] = pd.to_numeric(dataset['Engine size'],errors='coerce')
dataset['Horsepower'] = pd.to_numeric(dataset['Horsepower'],errors='coerce')
dataset['Wheelbase'] = pd.to_numeric(dataset['Wheelbase'],errors='coerce')
dataset['Width'] = pd.to_numeric(dataset['Width'],errors='coerce')
dataset['Length'] = pd.to_numeric(dataset['Length'],errors='coerce')
dataset['Curb weight'] = pd.to_numeric(dataset['Curb weight'],errors='coerce')
dataset['Fuel capacity'] = pd.to_numeric(dataset['Fuel capacity'],errors='coerce')
dataset['Fuel efficiency'] = pd.to_numeric(dataset['Fuel efficiency'],errors='coerce')
# %%
dataset.info()
#%%
dataset.isnull().sum()
dataset.shape
#%%
dataset.dropna(axis=0, inplace=True)
#%%
dataset.isnull().sum()
dataset.shape
# %%
col = ['Model', 'Latest Launch']
dataset.drop(columns=col, inplace=True)
# %%
X = dataset.drop('Price in thousands', axis=1)
Y = dataset['Price in thousands'].copy()
# %%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
# %%
preprocessor = ColumnTransformer(
    transformers=[
        ("imputer", SimpleImputer(strategy='median'),list(X_train.select_dtypes(include="float64"))),
        ("scaler", StandardScaler(), list(X_train.select_dtypes(include="float64"))),
        ("onehot", OneHotEncoder(), list(X_train.select_dtypes(include="object")))
    ]
)
# %%
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(48,)),
    keras.layers.Dense(units=300, activation='relu'),
    keras.layers.Dense(units=50, activation='relu'),
    keras.layers.Dense(units=1, activation='linear')
])
# %%
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])
# %%
pipeline.named_steps['preprocessor'].fit(X_train)
# %%
# print(preprocessor.transform(X_train).shape[1])
# %%
pipeline.named_steps['model'].compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
pipeline.named_steps['model'].fit(pipeline.named_steps['preprocessor'].transform(X_train), y_train, epochs=600)
# %%
# Avalia o modelo no conjunto de teste
evaluation = pipeline.named_steps['model'].evaluate(pipeline.named_steps['preprocessor'].transform(X_test), y_test)
print(f'Mean Squared Error: {evaluation[0]}')
print(f'Mean Absolute Error: {evaluation[1]}')
# %%
# Previsões no conjunto de teste
predictions = pipeline.named_steps['model'].predict(pipeline.named_steps['preprocessor'].transform(X_test))
# Gráfico
plt.scatter(y_test, predictions, color='red', label='Previsões')
plt.scatter(y_test, y_test, color='blue', label='Valores Reais')
plt.xlabel('Valores Reais')
plt.ylabel('Previsões do Modelo')
plt.title('Comparação entre Valores Reais e Previsões do Modelo')
plt.legend()
plt.show()