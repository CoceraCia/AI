from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
import numpy as np


class Classifier:

    def __init__(self, dataset: object, n_neighbors: int = 5):
        self.dataset = dataset
        self.knn_classifier = KNeighborsClassifier(n_neighbors)
        self.x_train, self.x_test, self.y_train, self.y_test = self.__trainTestSplit()
        # Trained model
        self.knn_classifier.fit(self.x_train, self.y_train)

        # SCALED MODEL
        self.scaler = StandardScaler()
        self.x_train_scaled = self.scaler.fit_transform(self.x_train)
        self.x_test_scaled = self.scaler.transform(self.x_test)
        self.knn_classifierScaled = KNeighborsClassifier(n_neighbors)
        self.knn_classifierScaled.fit(self.x_train_scaled, self.y_train)

        # PIPELINE AND GRID
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif)),
            ("knn", KNeighborsClassifier())
        ])

        param_grid = {
            # cuantas features usar
            "select__k": list(range(1, self.x_train.shape[1]+1)),
            "knn__n_neighbors": [3, 5, 7, 9]  # cuantos vecinos usar
        }

        self.grid = GridSearchCV(
            pipeline, param_grid, cv=5, scoring="accuracy"
        )

        self.grid.fit(self.x_train, self.y_train)

    def predictData(self, predict: tuple = None) -> tuple[np.ndarray, np.ndarray]:
        predict = self.knn_classifier.predict(
            predict) if predict is not None else self.knn_classifier.predict(self.x_test)
        return predict, self.y_test

    def predictScaledData(self, predict: tuple = None) -> tuple[np.ndarray, np.ndarray]:
        predict = self.knn_classifierScaled.predict(
            predict) if predict is not None else self.knn_classifierScaled.predict(self.x_test_scaled)
        return predict, self.y_test

    def predictGridData(self, predict: tuple = None) -> tuple[np.ndarray, np.ndarray]:
        predict = self.grid.predict(
            predict) if predict is not None else self.grid.predict(self.x_test)
        return predict, self.y_test

    def prediction_score(self, pred=None) -> float:
        y_test = self.y_test
        pred = pred if pred is not None else self.knn_classifier.predict(
            self.x_test)
        return metrics.accuracy_score(y_test, pred)

    def prediction_score_scaled(self, pred=None) -> float:
        y_test = self.y_test
        pred = pred if pred is not None else self.knn_classifierScaled.predict(
            self.x_test_scaled)
        return metrics.accuracy_score(y_test, pred)

    def prediction_score_grid(self, pred=None) -> float:
        y_test = self.y_test
        pred = pred if pred is not None else self.grid.predict(
            self.x_test)
        return metrics.accuracy_score(y_test, pred)

    def get_best_grid_params(self):
        return self.grid.best_params_

    # Private functions

    def __trainTestSplit(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = self.dataset.data
        y = self.dataset.target

        # spliteamos datos para entreno y testeo, 70% para entrenar y 30% para testear
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=42, stratify=y)

        return x_train, x_test, y_train, y_test
