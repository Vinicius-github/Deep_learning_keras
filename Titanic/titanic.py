#%% 
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

#Load dataset
url ="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
dataset = pd.read_csv(url)
# %%
# See the five rows of dataset
dataset.head()
# %%
# Remove rown with NA Embarked column
df = dataset.copy()
df.dropna(subset=['Embarked'], axis=0, inplace=True)
# %%
#Create X and Y dataset
X = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'])
y = df['Survived']
# %%
# Create train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# %%
X_train.info()
# %%
scale_fare = ["Fare"]
imputer_values = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
categorical = ["Sex", "Embarked"]

preprocessor = ColumnTransformer(
    transformers=[
         ("imputer", SimpleImputer(strategy='median'), imputer_values),
         ("scaler", StandardScaler(), scale_fare),
         ("cat", OneHotEncoder(), categorical)
     
])
#%%
# Crie um pipeline com o pré-processador e o modelo
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(11,)),  # Camada de entrada
    tf.keras.layers.Dense(units=100, activation='relu'),
    tf.keras.layers.Dense(units=72, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=8, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')  # Camada de saída para classificação binária
])
#%%
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])
# %%
# Ajustar o pré-processador nos dados de treinamento
pipeline.named_steps['preprocessor'].fit(X_train)
# %%
# Ajustar o modelo nos dados de treinamento
pipeline.named_steps['model'].compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
pipeline.named_steps['model'].fit(pipeline.named_steps['preprocessor'].transform(X_train), y_train, epochs=16, batch_size=32)
# %%
accuracy = pipeline.named_steps['model'].evaluate(pipeline.named_steps['preprocessor'].transform(X_test), y_test)
print(f'Acurácia do modelo nos dados de teste: {accuracy[1]:.4f}')
# %%
