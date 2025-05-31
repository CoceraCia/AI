from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import sklearn
import numpy as np


class Classifier:

    def __init__(self, dataset: object, n_neighbors: int = 5):
        # publico
        self.dataset = dataset
        # privado
        self.__dataset = dataset
        self.__knn_classifier = KNeighborsClassifier(n_neighbors)
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = self.__trainTestSplit()
        # Trained model
        self.__knn_classifier.fit(self.__x_train, self.__y_train)

    def predictData(self, predict: tuple = None) -> np.ndarray:
        predict = self.__knn_classifier.predict(
            predict) if predict is not None else self.__knn_classifier.predict(self.__x_test)
        return predict

    def prediction_score(self, y_test=None, pred=None) -> float:
        y_test = y_test if y_test is not None else self.__y_test
        pred = pred if pred is not None else self.__knn_classifier.predict(
            self.__x_test)
        return metrics.accuracy_score(y_test, pred)

    # Private functions
    def __trainTestSplit(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = self.__dataset.data
        y = self.__dataset.target

        # spliteamos datos para entreno y testeo, 70% para entrenar y 30% para testear
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=42, stratify=y)

        return x_train, x_test, y_train, y_test
