import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models

# Cargar el dataset de entrenamiento desde CSV
data = pd.read_csv('data/train.csv')

# Separar las características (imágenes) de las etiquetas
X = data.drop('label', axis=1).values  # Píxeles de la imagen, con axis=1 se elimina una columna.
y = data['label'].values  # Etiquetas (dígitos)

# Normalizar las imágenes (escalarlas a valores entre 0 y 1)
X = X / 255.0

# Reshape de las imágenes a 28x28 píxeles, ya que están aplanadas en el CSV
X = X.reshape(-1, 28, 28, 1)  # -1 es para que infiera el número de filas solo, 1 es el canal (grayscale)

# Convertir las etiquetas a one-hot encoding
y = to_categorical(y, num_classes=10)

# Dividir en conjuntos de entrenamiento y validación
# random_state tiene como objetivo hacer que la division de los datos siempre sea igual para debuguear
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

# Cargar el dataset de prueba desde CSV
test_data = pd.read_csv('data/test.csv')

# Separar las características (imágenes) de las etiquetas del conjunto de prueba
X_test = test_data.values  # Píxeles de la imagen

# Normalizar las imágenes de prueba
X_test = X_test / 255.0

# Reshape de las imágenes de prueba a 28x28 píxeles
X_test = X_test.reshape(-1, 28, 28, 1)

# Definir el modelo
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),  # Imágenes de 28x28 en escala de grises (1 canal)
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.MaxPool2D(pool_size=(2, 2), strides=2),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.MaxPool2D(pool_size=(2, 2), strides=2),
    layers.Flatten(),
    layers.Dense(units=128, activation='relu'),  # Capa densa adicional para más capacidad
    layers.Dense(units=10, activation='softmax')  # 10 clases para los dígitos (0-9)
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Mostrar el resumen del modelo
print(model.summary())

# Entrenar el modelo
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=10, batch_size=32, verbose=2)

# Realizar predicciones en el conjunto de prueba
predictions = model.predict(X_test)

# Mostrar las predicciones (la clase predicha para cada imagen)
predicted_labels = np.argmax(predictions, axis=1)
print(predicted_labels)

# Guardar las predicciones en un archivo CSV
submission = pd.DataFrame({'ImageId': np.arange(1, len(predicted_labels) + 1), 'Label': predicted_labels})
submission.to_csv('data/predictions.csv', index=False)

print("Predicciones guardadas en 'data/predictions.csv'")