import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

df = pd.read_csv(r"C:\Users\josep\Downloads\kc_house_data.csv")
print(df.head(5))

plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=50, kde=True)
plt.title('Distribution of House Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

#EJERCICIO 
'''
Entrena un modelo regresor de Perceptrón Multicapa (MLP) para predecir precios de viviendas, 
comenzando con la selección y el preprocesamiento de características. 
Luego divide los datos en conjuntos de entrenamiento (67%) y prueba (33%).

Estandariza tanto las características (features) como la variable objetivo,
 y optimiza el modelo MLP utilizando validación cruzada de 5 pliegues (5-fold cross-validation) con diferentes configuraciones de capas ocultas.

Finalmente, aplica la transformación inversa a las predicciones y a los valores reales, 
visualiza los precios reales vs. los predichos para ambos conjuntos (entrenamiento y prueba), y calcula y muestra los residuos.
'''


#Selección y preprocesamiento de características
#Subtarea
'''
Selecciona características significativas, maneja los valores faltantes y prepara los tipos de datos para el entrenamiento del modelo.

Razonamiento:
El primer paso es separar la variable objetivo de las características y 
luego verificar si hay valores faltantes tanto en las características como en la variable objetivo, 
según las instrucciones.
'''

y = df['price']
X = df.drop(columns=['id', 'date', 'price'])

print("Missing values in target variable (y):")
print(y.isnull().sum())

print("\nMissing values in feature DataFrame (X):")
print(X.isnull().sum())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")



#ESTANDARIZACIÓN DE LOS DATOS 
scaler_X = StandardScaler()
scaler_y = StandardScaler()




X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled shape: {X_test_scaled.shape}")
print(f"y_train_scaled shape: {y_train_scaled.shape}")
print(f"y_test_scaled shape: {y_test_scaled.shape}")




#Seleccion del modelo utilizado para la predicción, en este caso estamos utilizando MLPRegressor, el cual es un modelo de red neuronal para regresión.

mlp = MLPRegressor(max_iter=1000, random_state=42, solver='adam')
mlp_single_layer = MLPRegressor(hidden_layer_sizes=(100,), learning_rate_init=0.001, max_iter=1000, verbose=True, random_state=42)
mlp_single_layer.fit(X_train_scaled, y_train_scaled.ravel())
print("MLPRegressor with single hidden layer trained successfully.")


loss_curve = mlp_single_layer.loss_curve_

# Plot the training loss curve
plt.figure(figsize=(10, 6))
plt.plot(loss_curve)
plt.title('MLPRegressor Training Loss Curve')
plt.xlabel('Iteration')
plt.ylabel('Training Loss')
plt.grid(True)
plt.show()

#Actividad 
'''

Asegúrate de que GridSearchCV esté importado desde sklearn.model_selection; 
luego define un modelo MLPRegressor con hidden_layer_sizes=(100,), activation='relu', 
solver='sgd', max_iter=1000 y random_state=42, y crea un param_grid para ajustar 
el learning_rate_init con el fin de optimizar el modelo
'''


#subejercicio:
'''
Subtask:
Asegúrate de que MLPRegressor y GridSearchCV 
estén importados desde sklearn.neural_network y sklearn.model_selection respectivamente, 
para facilitar la definición del modelo y la optimización de hiperparámetros.
'''

try:
    from sklearn.model_selection import GridSearchCV

    print("GridSearchCV imported successfully.")
except: print("prueba")



#Subejercicio
'''
Define un modelo MLPRegressor con hidden_layer_sizes=(100,), 
activation='relu', solver='sgd', max_iter=1000 y random_state=42. 
Además, define un param_grid específicamente para ajustar learning_rate_init
con varios valores que se explorarán durante la validación cruzada.


Razonamiento: Voy a instanciar un MLPRegressor con los parámetros especificados y 
luego definir el param_grid para learning_rate_init, tal como se solicita en la subtarea.
'''
import torch 


