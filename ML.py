from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, LearningCurveDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted


class ML:
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    svc = SVC(kernel='linear', probability=True, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    lgbm = LGBMClassifier(n_estimators=100, random_state=42)
    estimators = [
        ('rf', rf),
        ('svc', svc),
        ('knn', knn),
        ('lgbm', lgbm)]
    ens = StackingClassifier(estimators)

    def __init__(self):
        self.y_pred = None
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.dataset = None
        self.modelnames_to_models = {'Random Forest': ML.rf, 'SVM': ML.svc, 'KNN': ML.knn, 'LightGBM': ML.lgbm,
                                     'Ensemble': ML.ens}
        self.modelnames_to_plot = dict()

        # rf = RandomForestClassifier(n_estimators=100, random_state=42)
        # svm = SVC(kernel='linear', probability=True, random_state=42)
        # knn = KNeighborsClassifier(n_neighbors=5)
        # lgbm = LGBMClassifier(n_estimators=100, random_state=42)
        # models = [self.rf, self.svm, self.knn, self.lgbm]

    def set_dataset(self, dataset):
        self.dataset = dataset
        X = self.dataset.iloc[:, :-1]
        y = self.dataset.iloc[:, -1]
        print(y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, model):
        # if self.dataset is None:
        #     return
        try:
            check_is_fitted(model)
        except NotFittedError as e:
            model.fit(self.X_train, self.y_train)
            self.y_pred = model.predict(self.X_test)

    def get_info(self, model_name: str):
        model = self.modelnames_to_models[model_name]
        self.train_model(model)

        accuracy = accuracy_score(self.y_test, self.y_pred)
        # precision = precision_score(y_test, y_pred)
        # recall = recall_score(y_test, y_pred)
        # f1 = f1_score(y_test, y_pred)

        # fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        # auc = roc_auc_score(y_test, y_pred)

        # RocCurveDisplay.from_estimator(model, X_test, y_test)
        # plt.title(model_names[i])
        # plt.show()
        if model_name not in self.modelnames_to_plot:
            self.modelnames_to_plot[model_name] = LearningCurveDisplay.from_estimator(model, self.X_test, self.y_test)
        return accuracy, self.modelnames_to_plot[model_name].figure_

    # def train_all(self):
    #     # Plot training graphs for each model
    #     model_names = ['Random Forest', 'SVM', 'KNN', 'LightGBM']
    #     for i in range(len(models)):
    #         # model = models[i](self.X_train, y_train)
    #         model = models[i].fit(self.X_train, self.y_train)
    #         y_pred = model.predict(self.X_test)
    #
    #         acc = accuracy_score(self.y_test, y_pred)
    #         # precision = precision_score(y_test, y_pred)
    #         # recall = recall_score(y_test, y_pred)
    #         # f1 = f1_score(y_test, y_pred)
    #
    #         # fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(self.X_test)[:, 1])
    #         # auc = roc_auc_score(y_test, y_pred)
    #
    #         # RocCurveDisplay.from_estimator(model, X_test, y_test)
    #         # plt.title(model_names[i])
    #         # plt.show()
    #         LearningCurveDisplay.from_estimator(model, X, y)
    #         plt.show()
