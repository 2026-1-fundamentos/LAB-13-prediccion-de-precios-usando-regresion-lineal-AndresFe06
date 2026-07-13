#
import os
import glob
import json
import gzip
import pickle

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)


def preparar_datos(ruta):
    datos = pd.read_csv(ruta, compression="zip", index_col=False)
    datos["Age"] = 2021 - datos["Year"]
    datos = datos.drop(["Year", "Car_Name"], axis=1)
    return datos.dropna()


def limpiar_directorio(ruta):
    if os.path.exists(ruta):
        for archivo in glob.glob(f"{ruta}/*"):
            os.remove(archivo)
    else:
        os.makedirs(ruta)


def entrenar_modelo(atributos, objetivo):

    columnas_categoricas = [
        "Fuel_Type",
        "Selling_type",
        "Transmission",
        "Owner",
    ]

    columnas_numericas = [
        columna
        for columna in atributos.columns
        if columna not in columnas_categoricas
    ]

    transformador = ColumnTransformer(
        transformers=[
            (
                "categorias",
                OneHotEncoder(handle_unknown="ignore"),
                columnas_categoricas,
            ),
            (
                "numeros",
                MinMaxScaler(),
                columnas_numericas,
            ),
        ],
        remainder="passthrough",
    )

    flujo = Pipeline(
        steps=[
            ("transformacion", transformador),
            ("seleccion", SelectKBest(score_func=f_regression)),
            ("regresion", LinearRegression()),
        ]
    )

    busqueda = GridSearchCV(
        estimator=flujo,
        param_grid={
            "seleccion__k": range(1, 20),
        },
        cv=10,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=2,
    )

    busqueda.fit(atributos, objetivo)

    limpiar_directorio("files/models")

    with gzip.open("files/models/model.pkl.gz", "wb") as archivo:
        pickle.dump(busqueda, archivo)


def generar_metricas(x_entrenamiento, x_prueba, y_entrenamiento, y_prueba, estimador):

    pred_train = estimador.predict(x_entrenamiento)
    pred_test = estimador.predict(x_prueba)

    resultados = [
        {
            "type": "metrics",
            "dataset": "train",
            "r2": r2_score(y_entrenamiento, pred_train),
            "mse": mean_squared_error(y_entrenamiento, pred_train),
            "mad": mean_absolute_error(y_entrenamiento, pred_train),
        },
        {
            "type": "metrics",
            "dataset": "test",
            "r2": r2_score(y_prueba, pred_test),
            "mse": mean_squared_error(y_prueba, pred_test),
            "mad": mean_absolute_error(y_prueba, pred_test),
        },
    ]

    limpiar_directorio("files/output")

    with open("files/output/metrics.json", "w") as archivo:
        for registro in resultados:
            archivo.write(json.dumps(registro) + "\n")


datos_entrenamiento = preparar_datos("files/input/train_data.csv.zip")
datos_prueba = preparar_datos("files/input/test_data.csv.zip")

x_train = datos_entrenamiento.drop(columns=["Present_Price"])
y_train = datos_entrenamiento["Present_Price"]

x_test = datos_prueba.drop(columns=["Present_Price"])
y_test = datos_prueba["Present_Price"]

entrenar_modelo(x_train, y_train)

with gzip.open("files/models/model.pkl.gz", "rb") as archivo:
    modelo_final = pickle.load(archivo)

generar_metricas(
    x_train,
    x_test,
    y_train,
    y_test,
    modelo_final,
)

# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:

# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
