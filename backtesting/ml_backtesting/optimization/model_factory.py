from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier


class ModelFactory:
    CLASSIFICATION_MODELS = {
        "LogisticRegression": LogisticRegression,
        "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis,
        "KNeighborsClassifier": KNeighborsClassifier,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "SVC": SVC,
        "AdaBoostClassifier": AdaBoostClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "MLPClassifier": MLPClassifier
    }

    REGRESSION_MODELS = {
        "LinearRegression": LinearRegression
    }

    @staticmethod
    def get_model(model_name: str, task_type: str):
        if task_type == "classification":
            return ModelFactory.CLASSIFICATION_MODELS.get(model_name)
        elif task_type == "regression":
            return ModelFactory.REGRESSION_MODELS.get(model_name)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
