from tkinter import W, BOTH

import numpy as np
import pandas as pd
import tkinter as tk
import tkinter.filedialog as filedialog
from tkinter import ttk

from lightgbm import LGBMClassifier
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import matplotlib.pyplot as plt
from ML import ML


def normalize_df(dataframe):
    for column in dataframe:
        if is_numeric_dtype(dataframe[column]):
            continue
        dataframe[column] = LabelEncoder().fit_transform(dataframe[column])
    dataframe = dataframe.dropna()
    return dataframe


class ClassificationGUI:
    def __init__(self, master):
        self.modelnames_to_tabs = None
        self.modelnames_to_models = None
        self.dataset = None
        self.master = master
        self.model_names = ['Random Forest', 'SVM', 'KNN', 'LightGBM', 'Ensemble']
        self.ML = ML()

        self.init_gui()

        self.master.mainloop()

    def init_gui(self):
        self.master.title("Classification GUI")
        self.master.geometry("500x500")

        self.file_label = tk.Label(self.master, text="Choose a dataset")
        self.file_label.pack()

        self.file_button = tk.Button(self.master, text="Browse...", command=self.choose_dataset)
        self.file_button.pack()

        self.label1 = tk.Label(self.master, text="Category col:")
        self.label1.pack()

        self.col = tk.Entry(self.master)
        self.col.pack()

        # создаем набор вкладок
        self.tabs = ttk.Notebook()

        self.tabs.pack(expand=True, fill=BOTH)
        self.modelnames_to_tabs = dict()
        for name in self.model_names:
            frame = tk.Frame(self.tabs)
            frame.pack(fill=BOTH, expand=True)

            self.modelnames_to_tabs[name] = frame
            # добавляем фреймы в качестве вкладок
            self.tabs.add(frame, text=name)

        self.tabs.bind("<<NotebookTabChanged>>", self.on_tab_selected)
        self.set_tabs_state("disabled")

    def set_tabs_state(self, state):
        for i in self.tabs.tabs():
            self.tabs.tab(i, state=state)

    def choose_dataset(self):
        file_path = filedialog.askopenfilename()
        self.dataset = pd.read_csv(file_path, on_bad_lines='skip')
        print(self.dataset)
        self.dataset = normalize_df(self.dataset)
        self.ML.set_dataset(self.dataset)
        self.modelnames_to_models = {'Random Forest': ML.rf, 'SVM': ML.svc, 'KNN': ML.knn, 'LightGBM': ML.lgbm,
                                     'Ensemble': ML.ens}
        self.set_tabs_state("normal")

    def on_tab_selected(self, event):
        selected_tab = event.widget.select()
        if len(selected_tab) == 0:
            return
        tab_text = event.widget.tab(selected_tab, "text")
        active_frame = self.tabs.nametowidget(selected_tab)
        # active_frame.children

        active_frame.children = dict()
        if len(active_frame.children) == 0:
            accuracy, figure = self.ML.get_info(tab_text)
            text = tk.Text(active_frame, width=25, height=5)
            text.pack()
            # text.delete('1.0', "end")
            text.insert('1.0', f"Accuracy: {accuracy}")
            chart_type = FigureCanvasTkAgg(figure, active_frame)
            chart_type.get_tk_widget().pack()

    # def train_model(self, model_name):
    #     if self.dataset is None:
    #         return
    #     X = self.dataset.iloc[:, :-1]
    #     y = self.dataset.iloc[:, -1]
    #
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #
    #     # if model_name == "rf":
    #     #     model = RandomForestClassifier(n_estimators=100, random_state=42)
    #     # elif model_name == "svm":
    #     #     model = SVC(kernel='linear', probability=True, random_state=42)
    #     # elif model_name == "knn":
    #     #     model = KNeighborsClassifier(n_neighbors=5)
    #     # elif model_name == "lgbm":
    #     #     model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
    #
    #     # model.fit(X_train, y_train)
    #     # y_pred = model.predict(X_test)
    #
    # def train_all(self):
    #     rf = RandomForestClassifier()
    #     svm = SVC()
    #     knn = KNeighborsClassifier()
    #     lgbm = LGBMClassifier()
    #     models = [rf, svm, knn, lgbm]
    #
    #     # Plot training graphs for each model
    #     model_names = ['Random Forest', 'SVM', 'KNN', 'LightGBM']
    #     for i in range(len(models)):
    #         # model = models[i](X_train, y_train)
    #         model = models[i].fit(X_train, y_train)
    #         y_pred = model.predict(X_test)
    #
    #         acc = accuracy_score(y_test, y_pred)
    #         # precision = precision_score(y_test, y_pred)
    #         # recall = recall_score(y_test, y_pred)
    #         # f1 = f1_score(y_test, y_pred)
    #
    #         # fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    #         # auc = roc_auc_score(y_test, y_pred)
    #
    #         # RocCurveDisplay.from_estimator(model, X_test, y_test)
    #         # plt.title(model_names[i])
    #         # plt.show()
    #         LearningCurveDisplay.from_estimator(model, X, y)
    #         plt.show()
    #
    # def ensemble_model(self):
    #     models = [self.rf_model, self.svm_model, self.knn_model, self.lgbm_model]
    #     y_preds = []
    #     for model in models:
    #         y_pred = model.predict(X_test)
    #         y_preds.append(y_pred)
    #
    #     y_pred_ensemble = []
    #     for i in range(len(y_preds[0])):
    #         votes = [y_preds[j][i] for j in range(len(models))]
    #         y_pred_ensemble.append(max(set(votes), key=votes.count))
    #
    #     acc = accuracy_score(y_test, y_pred_ensemble)
    #     precision = precision_score(y_test, y_pred_ensemble)
    #     recall = recall_score(y_test, y_pred_ensemble)
    #     f1 = f1_score(y_test, y_pred_ensemble)
    #
    #     fpr, tpr, thresholds = roc_curve(y_test, y_pred_ensemble)
    #     auc = roc_auc_score(y_test, y_pred_ensemble)
    #
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(fpr, tpr, label="AUC = %.3f" % auc)
    #     plt.xlabel("False Positive Rate")
    #     plt.ylabel("True Positive Rate")
    #     plt.title("ROC Curve")
    #     plt.legend()
    #     plt.show()
    #
    #     self.result_label.config(
    #         text="Ensemble Model\nAccuracy: {}\nPrecision: {}\nRecall: {}\nF1 Score: {}\nAUC: {}".format(
    #             round(acc, 3), round(precision, 3), round(recall, 3), round(f1, 3), round(auc, 3)
    #         ))


if __name__ == '__main__':
    root = tk.Tk()
    app = ClassificationGUI(root)
