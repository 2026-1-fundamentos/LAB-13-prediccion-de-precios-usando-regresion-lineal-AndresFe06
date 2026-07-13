#
import gzip
import glob
import json
import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def load_and_split_datasets():
    train = pd.read_csv("files/input/train_data.csv.zip", compression="zip", index_col=False)
    test = pd.read_csv("files/input/test_data.csv.zip", compression="zip", index_col=False)

    for df in (train, test):
        df["Age"] = 2021 - df["Year"].astype(int)
        df.drop(columns=["Year", "Car_Name"], inplace=True)
        df.dropna(inplace=True)

    x_train = train.drop(columns=["Present_Price"])
    y_train = train["Present_Price"]
    x_test = test.drop(columns=["Present_Price"])
    y_test = test["Present_Price"]
    return x_train, y_train, x_test, y_test


def build_pipeline(x_train):
    categorical_columns = ["Fuel_Type", "Selling_type", "Transmission", "Owner"]
    numerical_columns = [c for c in x_train.columns if c not in categorical_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
            ("scaler", MinMaxScaler(), numerical_columns),
        ],
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("select_k_best", SelectKBest(score_func=f_regression)),
            ("regressor", LinearRegression()),
        ]
    )


def train_model(x_train, y_train):
    pipeline = build_pipeline(x_train)
    param_grid = {"select_k_best__k": range(1, 20)}

    model = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return model


def save_model(model, path="files/models/model.pkl.gz"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for file in glob.glob(os.path.join(os.path.dirname(path), "*")):
        os.remove(file)
    with gzip.open(path, "wb") as file:
        pickle.dump(model, file)


def compute_metrics(y_true, y_pred, dataset_name):
    return {
        "type": "metrics",
        "dataset": dataset_name,
        "r2": r2_score(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "mad": mean_absolute_error(y_true, y_pred),
    }


def save_metrics(model, x_train, y_train, x_test, y_test, path="files/output/metrics.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for file in glob.glob(os.path.join(os.path.dirname(path), "*")):
        os.remove(file)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_metrics = compute_metrics(y_train, y_train_pred, "train")
    test_metrics = compute_metrics(y_test, y_test_pred, "test")

    with open(path, "w") as file:
        file.write(json.dumps(train_metrics) + "\n")
        file.write(json.dumps(test_metrics) + "\n")


def main():
    x_train, y_train, x_test, y_test = load_and_split_datasets()
    model = train_model(x_train, y_train)
    save_model(model)
    save_metrics(model, x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main()
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
