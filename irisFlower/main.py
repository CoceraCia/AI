# Manipulación de datos
import pandas as pd
import numpy as np
import Classifier

# Cargar datos y dividirlos
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch

# Para el algoritmo de clasificación (usaremos k-Nearest Neighbors, uno sencillo para empezar)
from sklearn.neighbors import KNeighborsClassifier

# Para evaluar el modelo
from sklearn.metrics import accuracy_score

# Global variables
iris_dataset = load_iris()


def main():
    knn_classifier = Classifier.Classifier(iris_dataset)
    while (True):
        printMenu()
        x = int(input("Select an option ->"))
        match x:
            case 0:
                print("GooodBye dummie!!")
                return
            case 1:
                showDataFrames()
            case 2:
                showPredictionScore(knn_classifier)
            case 3:
                predictNewFlower(knn_classifier)
            case 4:
                predictDataSet(knn_classifier)


def printMenu():
    print("-------------MENU--------------")
    print("0-EXIT")
    print("1-Show Data Frames")
    print("2-Show Prediction Score")
    print("3-Predict New Flower")
    print("4-Predict entire iris Dataset")
    print("--------------------------------")


def showPredictionScore(knn_classifier: Classifier):
    print(f"Prediction score: {knn_classifier.prediction_score() * 100:.2f}%")


def predictDataSet(knn_classifier: Classifier):
    prediction = knn_classifier.predictData()
    target = knn_classifier.dataset.target
    print(
        f"Dataset Prediction:\n{showPredictionData(prediction, target)}")


def predictNewFlower(knn_classifier: Classifier):
    sLength = mustBeFloat("set sepal length(cm): ")
    sWidth = mustBeFloat("set sepal width(cm):")
    pLength = mustBeFloat("set petal length(cm):")
    pWidth = mustBeFloat("set petal width(cm):")
    new_flower = [[sLength, sWidth, pLength, pWidth]]

    prediction = knn_classifier.predictData(new_flower)
    classes = ["setosa", "versicolor", "virginica"]
    print(f"Your flower it's a {classes[prediction[0]]}")


def mustBeFloat(prompt: str) -> float:
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("SET ONLY NUMBERS!!!!")


# Usamos pandas para mostrar visualmente los datos en forma de tabla


def showDataFrames():
    # Datos de las mediciones
    df_features = pd.DataFrame(
        iris_dataset.data, columns=iris_dataset.feature_names)
    # El target es la especie de la flor (0 para Setosa, 1 para vesicolor, 2 para Virginica)
    df_target = pd.DataFrame(iris_dataset.target, columns=["species"])
    print("-----------------------------------")
    print(
        f"Tenemos datos de {df_features.shape[0]} flores y {df_features.shape[1]} caracteristicas por flor")
    print("-----------------------------------")
    print("caracteristicas de features")
    print(df_features.head())
    print("-----------------------------------")
    print("caracteristicas de Target")
    print("Las especies en el target se muestran como: (0=setosa,1=Versicolor,2=virginica)")
    print(df_target)


def showPredictionData(prediccion: np.ndarray, target: np.ndarray):
    data = "Predictions\tTargets\n"
    clases = ["setosa", "versicolor", "virginica"]
    for i in range(len(prediccion)):
        pText = clases[prediccion[i]]
        tText = clases[target[i]]
        isTrue = (prediccion[i] == target[i])
        data += f"{prediccion[i]}({pText})\t{target[i]}({tText})-->{isTrue}\n"
    return data


if __name__ == "__main__":
    main()
