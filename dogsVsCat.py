import numpy as np
import tensorflow
import keras as kr
from keras import layers
import os
import itertools
import random
import glob
import shutil
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

warnings.simplefilter(action="ignore",category=FutureWarning)

def plotImages(images_arr):
    fig, axes = plt.subplots(1,10,figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip (images_arr,axes):
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

#Funcion para plotear la confusion matrix
def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#<----Organizacion de los datos---->
os.chdir(os.path.dirname(os.path.realpath(__file__)))

if os.path.isdir("data/train/dog") is False:
    os.makedirs("data/train/dog")
    os.makedirs("data/train/cat")
    os.makedirs("data/valid/dog")
    os.makedirs("data/valid/cat")
    os.makedirs("data/test/dog")
    os.makedirs("data/test/cat")

    for c in random.sample(glob.glob('data/cat*'),500):
        shutil.move(c,"data/train/cat")
    for c in random.sample(glob.glob('data/dog*'),500):
        shutil.move(c,"data/train/dog")
    for c in random.sample(glob.glob('data/cat*'),100):
        shutil.move(c,"data/valid/cat")
    for c in random.sample(glob.glob('data/dog*'),100):
        shutil.move(c,"data/valid/dog")
    for c in random.sample(glob.glob('data/cat*'),50):
        shutil.move(c,"data/test/cat")
    for c in random.sample(glob.glob('data/dog*'),50):
        shutil.move(c,"data/test/dog")

#<----Procesamiento de los datos---->
train_path = "data/train"
valid_path = "data/valid"
test_path = "data/test"

# Crear generador de datos para el conjunto de entrenamiento
# - `preprocessing_function`: Preprocesa las imágenes según los requisitos del modelo VGG16.
# - `target_size`: Redimensiona las imágenes a 224x224 píxeles.
# - `batch_size`: Define el tamaño del lote (32 imágenes por lote).
# - `classes`: Lista de nombres de las clases (Se toma como referencia los subdirectorios de las carpetas).
train_batches = ImageDataGenerator(preprocessing_function=kr.applications.vgg16.preprocess_input).flow_from_directory(
    directory=train_path,          # Directorio de entrenamiento
    target_size=(224, 224),        # Redimensionar las imágenes a 224x224
    batch_size=32,                 # Tamaño del lote
    classes=['cat', 'dog']         # Clases de las imágenes
)

valid_batches = ImageDataGenerator(preprocessing_function=kr.applications.vgg16.preprocess_input).flow_from_directory(
    directory=valid_path,          # Directorio de validación
    target_size=(224, 224),        # Redimensionar las imágenes a 224x224
    batch_size=32,                 # Tamaño del lote
    classes=['cat', 'dog']         # Clases de las imágenes
)

test_batches = ImageDataGenerator(preprocessing_function=kr.applications.vgg16.preprocess_input).flow_from_directory(
    directory=test_path,           # Directorio de prueba
    target_size=(224, 224),        # Redimensionar las imágenes a 224x224
    batch_size=32,                 # Tamaño del lote
    classes=['cat', 'dog'],        # Clases de las imágenes
    shuffle=False                  # No mezclar el conjunto de prueba
)
#Comporacion de la correcion de los datos
assert train_batches.n == 1000
assert valid_batches.n == 200
assert test_batches.n == 100
assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == 2
#Muetras de imagenes y etiquetas
# imgs, labels = next(train_batches)
# print("<--Imagenes-->")
# plotImages(imgs)
# print("<--Etiquetas-->")
# print(labels)

#<----Definicion y entrenamiendo del Modelo---->
model = kr.Sequential([
    kr.Input(shape=(224,224,3)), #Definicion de las dimensiones de la capa de entrada
    layers.Conv2D(filters=32,kernel_size=(3,3),activation="relu",padding="same"),
    layers.MaxPool2D(pool_size=(2,2),strides=2),
    layers.Conv2D(filters=64,kernel_size=(3,3),activation="relu",padding="same"),
    layers.MaxPool2D(pool_size=(2,2),strides=2),
    layers.Flatten(),
    layers.Dense(units=2,activation="softmax")
])

print(model.summary())

model.compile(optimizer=kr.optimizers.Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=['accuracy'])
model.fit(x=train_batches, validation_data=valid_batches,epochs=10, verbose=2)

#<----Prediccion---->
test_imgs, test_labels = next(test_batches)
print(test_batches.classes)

predictions = model.predict(x=test_batches,verbose=0)

print(np.round(predictions))

cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))

print(test_batches.class_indices)

cm_plot_labels = ['cat','dog']

plot_confusion_matrix(cm=cm,classes=cm_plot_labels, title="Confusion Matrix")

#<----VGG16---->
#Importacion del modelo
vgg16_model = kr.applications.vgg16.VGG16()
print(vgg16_model.summary())

#Definimos un nuevo modelo receptor de la arquitectura de vgg16
model = kr.Sequential()
#Traspaso de la arquitectura de vgg16 a nuestro modelo secuencial
for layer in vgg16_model.layers[:-1]:
    model.add(layer)

print(model.summary())

#Evitamos el entrenamiento en las capas ocultas (VGG16 ya esta entrenado en perros y gatos)
for layer in model.layers:
    layer.trainable=False

#Agregamos una capa de output con dos salidas (Gato y Perro), capa a entrenar
model.add(layers.Dense(units=2,activation="softmax"))

print(model.summary())

#<----Entrenamiento de FineTunedVgg16---->
model.compile(optimizer=kr.optimizers.Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=['accuracy'])
model.fit(x=train_batches, validation_data=valid_batches,epochs=5, verbose=2)

#<----Prediccion de FineTunedVgg16---->
predictions=model.predict(x=test_batches,verbose=0)
print(test_batches.classes)

cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))

print(test_batches.class_indices)

cm_plot_labels = ['cat','dog']

plot_confusion_matrix(cm=cm,classes=cm_plot_labels, title="Confusion Matrix")