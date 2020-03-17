#-*- coding : utf-8-*-
#
import sys
import os
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import QAbstractTableModel, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QTableView, QWidget
from PyQt5.QtWidgets import QPushButton, QFileDialog, QMessageBox
from PyQt5.QtWidgets import QLabel, QComboBox, QLineEdit, QListWidget, QCheckBox, QListWidgetItem
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn import svm
from sklearn.decomposition import SparsePCA
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
# from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


class WineClassify(QWidget):
    def __init__(self):
        super().__init__()
        self.button_widget = QWidget()
        self.main_widget = QWidget()
        self.message_box = QMessageBox()
        self.figure_able = 0

        self.init_ui()

    def init_ui(self):
        warnings.filterwarnings('ignore')
        self.input_data = pd.DataFrame()
        self.info_data = pd.DataFrame()
        self.label_data = pd.DataFrame()
        self.all_train_data = pd.DataFrame()
        self.train_data = pd.DataFrame()
        self.all_test_data = pd.DataFrame()
        self.test_data = pd.DataFrame()
        self.result_data = pd.DataFrame()
        self.test_has_label = False

        # self.theme = self.themes[0]
        # self.model = []
        # self.pca_top = 20
        # self.knn_top = 7
        # self.bp_lr = 0.2
        # self.bp_epoch = 10000
        # self.svm_type = self.svm_types[0]
        #
        self.messages = ['请导入Excel文件', '请导入包含标签的有效数据', '请点击链接', '请点击训练']
        self.themes = ['浅色', '深色', '白色']
        self.svm_types = ['ovo', 'ovr']

        self.import_methods = ['导入标签信息', '导入训练数据', '导入测试数据']
        self.import_method = self.import_methods[0]

        self.show_data_methods = ['显示标签信息', '显示全部训练数据', '显示全部测试数据', '显示训练数据', '显示测试数据']
        self.show_data_method = self.show_data_methods[0]

        self.data_linked = False
        self.data_classified = False

        self.test_figure_types = ['折线图', '条形图', '面积图']
        self.result_figure_types = ['折线图', '散点图']
        self.test_figure_type = self.test_figure_types[0]
        self.result_figure_type = self.result_figure_types[0]
        self.model_trained = False
        self.data_classified = False

        self.classify_methods = ['KNN', 'PCA-KNN', 'sPCA-KNN', 'SVM', 'PCA-SVM', 'sPCA-SVM',
                                 'BP', 'PCA-BP', 'sPCA-BP', 'Decision Tree', 'PCA-DT', 'SPCA-DT']
        # self.classify_methods = ['KNN', 'PCA-KNN', 'sPCA-KNN', 'SVM',
        #                          'PCA-SVM', 'sPCA-SVM', 'BP', 'PCA-BP', 'sPCA-BP']
        # self.classify_methods = ['KNN', 'PCA-KNN', 'SVM', 'PCA-SVM', 'BP', 'PCA-BP']
        self.classify_method = self.classify_methods[0]

        self.wine_grades = ['特级', '一级', '二级', '优级']
        self.wine_grade = self.wine_grades[0]
        self.wine_marks = [93, 88, 80, 70]

        self.import_btn = QPushButton('导入文件')
        self.import_btn.setToolTip('导入白酒数据文件')
        self.import_btn.clicked.connect(self.import_data)

        self.link_btn = QPushButton('链接文件')
        self.link_btn.setToolTip('链接导入的数据和信息')
        self.link_btn.clicked.connect(self.link_data)

        self.show_data_btn = QPushButton('显示数据')
        self.show_data_btn.setToolTip('显示表格数据')
        self.show_data_btn.clicked.connect(self.show_data_slot)

        self.figure_btn = QPushButton('绘制图像')
        self.figure_btn.setToolTip('绘制可视化图像')
        self.figure_btn.clicked.connect(self.figure_func)

        self.train_btn = QPushButton('开始训练')
        self.train_btn.setToolTip('开始训练分类模型')
        self.train_btn.clicked.connect(self.start_training)

        self.classify_btn = QPushButton('开始分类')
        self.classify_btn.setToolTip('对数据进行分类')
        self.classify_btn.clicked.connect(self.start_testing)

        self.save_btn = QPushButton('保存')
        self.save_btn.setToolTip('保存图像或数据')
        self.save_btn.clicked.connect(self.save_data)

        self.settint_btn = QPushButton('设置')
        self.settint_btn.setToolTip('设置显示的图像和数据格式')
        self.settint_btn.clicked.connect(self.setting_func)

        self.clear_btn = QPushButton('清空')
        self.clear_btn.setToolTip('清空图像和数据')
        self.clear_btn.clicked.connect(self.clear_func)

        self.exit_btn = QPushButton('退出')
        self.exit_btn.setToolTip('退出程序')
        self.exit_btn.clicked.connect(self.exit_func)

        self.import_method_cb = QComboBox()
        self.import_method_cb.setToolTip('选择要导入的数据')
        self.import_method_cb.addItems(self.import_methods)
        self.import_method_cb.activated[str].connect(self.change_type)

        self.show_data_method_cb = QComboBox()
        self.show_data_method_cb.setToolTip('选择要查看的数据')
        self.show_data_method_cb.addItems(self.show_data_methods)
        self.show_data_method_cb.activated[str].connect(self.change_type)

        self.test_figure_type_cb = QComboBox()
        self.test_figure_type_cb.setToolTip('选择可视化绘图方法')
        self.test_figure_type_cb.addItems(self.test_figure_types)
        self.test_figure_type_cb.activated[str].connect(self.change_type)

        self.result_figure_type_cb = QComboBox()
        self.result_figure_type_cb.setToolTip('选择结果可视化方法')
        self.result_figure_type_cb.addItems(self.result_figure_types)
        self.result_figure_type_cb.activated[str].connect(self.change_type)

        self.classify_type_cb = QComboBox()
        self.classify_type_cb.setToolTip('选择分类方法')
        self.classify_type_cb.addItems(self.classify_methods)
        self.classify_type_cb.activated[str].connect(self.change_type)

        self.button_layout = QHBoxLayout()
        self.button_layout.addWidget(self.import_btn)
        self.button_layout.addWidget(self.import_method_cb)
        self.button_layout.addWidget(self.link_btn)
        self.button_layout.addWidget(self.show_data_btn)
        self.button_layout.addWidget(self.show_data_method_cb)
        self.button_layout.addWidget(self.figure_btn)
        self.button_layout.addWidget(self.test_figure_type_cb)
        self.button_layout.addWidget(self.train_btn)
        self.button_layout.addWidget(self.classify_btn)
        self.button_layout.addWidget(self.classify_type_cb)
        self.button_layout.addWidget(self.result_figure_type_cb)
        self.button_layout.addWidget(self.settint_btn)
        self.button_layout.addWidget(self.save_btn)
        self.button_layout.addWidget(self.clear_btn)
        self.button_layout.addWidget(self.exit_btn)
        self.button_widget.setLayout(self.button_layout)


        self.setting_widget = Setting_widget(self)
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot()
        self.canvas = FigureCanvas(self.figure)
        self.table = QTableView()
        self.reset_setting()

        self.figure_layout = QHBoxLayout()
        self.figure_layout.addWidget(self.table)
        self.figure_layout.addWidget(self.canvas)
        self.figure_layout.addWidget(self.setting_widget)
        self.table.setVisible(False)
        self.canvas.setVisible(False)
        self.setting_widget.setVisible(False)
        # main_state 0:空, 1:表格 2:图 3: 结果 4：设置
        self.main_state = 0
        self.main_widget.setLayout(self.figure_layout)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.button_widget)
        self.layout.addWidget(self.main_widget)
        self.setLayout(self.layout)

        self.button_widget.setFixedHeight(60)
        self.setWindowTitle('基酒等级自动分类系统')
        self.setWindowIcon(QIcon('themes\\wine.ico'))
        desktop = QApplication.desktop()
        self.resize(desktop.width(), desktop.height()-80)
        self.move(0, 0)
        self.show()

    def change_type(self, text):
        print("changing type")
        if self.sender() == self.import_method_cb:
            self.import_method = text
        elif self.sender() == self.show_data_method_cb:
            self.show_data_method = text
        elif self.sender() == self.test_figure_type_cb:
            self.test_figure_type = text
        elif self.sender() == self.result_figure_type_cb:
            self.result_figure_type = text
        elif self.sender() == self.classify_type_cb:
            self.classify_method = text
            self.model_trained = False
        else:
            pass

    def import_data(self):
        print("importing data")
        file_name = QFileDialog.getOpenFileName(self, '打开文件')[0]
        if not os.path.splitext(file_name)[1] in ['.xls', '.xlsx']:
            self.show_message(self.messages[0])
        else:
            try:
                self.input_data = pd.read_excel(file_name, encoding="utf-8").fillna("未知")
            except BaseException:
                self.show_message("数据读取失败")
            else:
                if len(self.input_data.index) == 0:
                    self.show_message(self.messages[1])
                elif not ('编号' in self.input_data.columns):
                    self.show_message("请导入包含编号列的有效数据")
                else:
                    # self.import_methods = ['导入白酒信息', '导入训练数据', '导入测试数据']
                    if self.import_method == self.import_methods[0]:
                        if '标签' in self.input_data.columns:
                            self.show_message("已" + self.import_method)
                            self.data_linked = False
                            self.label_data = self.input_data
                            self.show_data(self.label_data)
                        else:
                            self.show_message("请导入包含标签列的白酒信息")
                    elif self.import_method == self.import_methods[1]:
                        self.show_message("已" + self.import_method)
                        self.data_linked = False
                        self.model_trained = False
                        self.all_train_data = self.input_data
                        self.show_data(self.all_train_data)
                    elif self.import_method == self.import_methods[2]:
                        self.show_message("已" + self.import_method)
                        self.data_linked = False
                        self.data_classified = False
                        self.all_test_data = self.input_data
                        self.show_data(self.all_test_data)
                        # self.test_has_label = False
                    else:
                        pass

    def link_data(self):
        print("linking data")
        # self.info_data = pd.read_excel("data\\wine_i.xlsx")
        # self.label_data = pd.read_excel("data\\wine_label.xlsx").fillna("未知")
        # self.all_train_data = pd.read_excel("data\\wine_train.xlsx").fillna("未知")
        # self.all_test_data = pd.read_excel("data\\wine_train.xlsx").fillna("未知")
        # self.all_test_data = pd.read_excel("data\\wine_test_single1.xlsx")
        # self.label_data = pd.read_excel("data\\wine_l.xlsx")
        # self.all_train_data = pd.read_excel("data\\wine_d.xlsx")
        # self.all_test_data = pd.read_excel("data\\wine_d.xlsx")
        # self.data_linked = False
        # self.check_data(self.all_data)
        # self.show_data(self.all_data)
        #
        if  len(self.label_data.index) == 0:
            self.show_message("请" + self.import_methods[0])
        elif len(self.all_train_data.index) == 0:
            self.show_message("请" + self.import_methods[1])
        elif len(self.all_test_data.index) == 0:
            self.show_message("请" + self.import_methods[2])
        elif self.data_linked:
            self.show_message("数据已链接")
            pass
        elif self.all_train_data.shape[1] != self.all_test_data.shape[1]:
            self.show_message("训练数据和测试数据列数不同！")
        else:
            cols = self.label_data.columns.tolist()
            for i in range(1, self.label_data.shape[1]):
                col = cols[i]
                # train_data
                if col in self.all_train_data.columns:
                    self.all_train_data.drop(col, axis=1, inplace=True)
                values = []
                for index, row in self.all_train_data.iterrows():
                    if row['编号'] in self.label_data.编号.values.tolist():
                        # print("in")
                        v = self.label_data.loc[self.label_data.编号 == row['编号'], col].tolist()[0]
                    else:
                        # print("not in ")
                        v = "未知"
                    values.append(v)
                self.all_train_data.insert(i, col, values)
                # print("train ok")

                # test_data
                if col in self.all_test_data.columns:
                    self.all_test_data.drop(col, axis=1, inplace=True)
                values = []
                for index, row in self.all_test_data.iterrows():
                    if row['编号'] in self.label_data.编号.values.tolist():
                        # print("in")
                        v = self.label_data.loc[self.label_data.编号 == row['编号'], col].tolist()[0]
                    else:
                        # print("not in ")
                        v = "未知"
                    values.append(v)
                self.all_test_data.insert(i, col, values)
                # print("test ok")

            # self.show_data(self.test_data)
            self.data_linked = True

            cols = self.all_train_data.columns.tolist()[self.label_data.shape[1]:]
            cols.insert(0, '编号')
            cols.insert(1, '标签')
            self.train_data = self.all_train_data.loc[self.all_train_data['标签'].isin(self.wine_grades), cols]

            cols = self.all_test_data.columns.tolist()[self.label_data.shape[1]:]
            cols.insert(0, '编号')
            cols.insert(1, '标签')
            self.test_data = self.all_test_data[cols]

            # if 'results' in self.test_data.columns:
            #     self.test_data.drop('results', axis=1, inplace=True)
            # if 'marks' in self.test_data.columns:
            #     self.test_data.drop('marks', axis=1, inplace=True)
            # print(self.all_train_data.shape)
            # print(self.all_test_data.shape)
            # print(self.train_data.shape)
            # print(self.test_data.shape)
            self.show_message("数据链接成功")

    def show_data_slot(self):
        # self.show_data_methods = ['显示白酒信息', '显示全部训练数据', '显示全部测试数据', '显示训练数据', '显示测试数据']
        if self.show_data_method == self.show_data_methods[0]:
            if len(self.label_data.index) == 0:
                self.show_message("请" + self.import_methods[0])
            else:
                self.show_data(self.label_data)
        elif self.show_data_method == self.show_data_methods[1]:
            if len(self.all_train_data.index) == 0:
                self.show_message("请" + self.import_methods[1])
            else:
                self.show_data(self.all_train_data)
        elif self.show_data_method == self.show_data_methods[2]:
            if len(self.all_test_data.index) == 0:
                self.show_message("请" + self.import_methods[2])
            else:
                self.show_data(self.all_test_data)

        # self.messages[0] = '请点击链接'
        elif self.show_data_method == self.show_data_methods[3]:
            if not self.data_linked:
                self.show_message(self.messages[2])
            else:
                self.show_data(self.train_data)
        elif self.show_data_method == self.show_data_methods[4]:
            if not self.data_linked:
                self.show_message(self.messages[2])
            else:
                self.show_data(self.test_data)
        else:
            self.show_message('显示数据错误')

    def show_data(self, data):
        print("showing data")
        if len(data.index) == 0:
            self.show_message("请先导入要显示的数据")
        else:
            model = QtTable(data)
            self.table.setModel(model)
            self.canvas.setVisible(False)
            self.setting_widget.setVisible(False)
            self.table.setVisible(True)
            self.main_state = 1
            # try:
            #     self.figure_layout.removeWidget(self.canvas)
            # except BaseException:
            #     pass
            # else:
            #     self.figure_layout.addWidget(self.table)
            #     self.ui_state = 1

    def figure_func(self):
        self.clear_func()
        print("figuring")
        if len(self.all_test_data.index) == 0:
            self.show_message("请" + self.import_methods[2])
        elif len(self.test_data.index) == 0:
            self.show_message(self.messages[2])
        elif not self.data_linked:
            self.show_message(self.messages[2])
        else:
            # 解决无法显示中文
            # plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
            plt.rcParams['font.sans-serif']=['SimHei'] #指定默认字体,SimHei为黑体
            # 解决无法显示负号
            plt.rcParams['axes.unicode_minus'] = False
            plt.title("测试数据")
            # plt.title(self.figure_type)
            label = self.test_data.编号.tolist()[:8]
            n=2
            if self.data_classified:
                n = 4
            if len(self.test_data.index > 8):
                data = self.test_data.iloc[:8, n:].T
            else:
                data = self.test_data.iloc[:, n:].T
            # print(data)
            if self.test_figure_type == self.test_figure_types[0]:
                data.plot(kind='line', ax=self.ax, rot=90, xticks=range(len(data.index)), fontsize=10)
                # pass
            elif self.test_figure_type == self.test_figure_types[1]:
                data.plot(kind='bar', ax=self.ax, fontsize=10)
            elif self.test_figure_type == self.test_figure_types[2]:
                data.plot(kind='area', ax=self.ax, rot=90, xticks=range(len(data.index)), fontsize=10)

            # plt.xlabel(data.index.tolist())
            plt.legend(label)

            self.table.setVisible(False)
            self.setting_widget.setVisible(False)
            self.canvas.setVisible(True)
            self.main_state = 2
            # self.figure_layout.removeWidget(self.table)
            # self.figure_layout.addWidget(self.canvas)
            self.canvas.draw()

    def start_training(self):
        if not self.data_linked:
            self.show_message("请链接数据")
            return
        # self.link_data()
        self.model = []
        print("training")
        # self.classify_methods =
        # ['KNN', 'PCA-KNN', 'sPCA-KNN', 'SVM', 'PCA-SVM', 'sPCA-SVM', 'BP', 'PCA-BP', 'sPCA-BP']
        train_data = self.train_data.iloc[:, 2:].values
        train_label = [self.wine_grades.index(grade) for grade in self.train_data['标签'].values]
        # start training
        if self.classify_method == self.classify_methods[0]:
            pass
        elif self.classify_method == self.classify_methods[1]:
            pass
        elif self.classify_method == self.classify_methods[2]:
            pass

        elif self.classify_method == self.classify_methods[3]:
            self.svm_train(train_data, train_label)
        elif self.classify_method == self.classify_methods[4]:
            lower_train_data = self.pca_train(train_data)
            self.svm_train(lower_train_data, train_label)
        elif self.classify_method == self.classify_methods[5]:
            lower_train_data = self.spca_train(train_data, train_label)
            self.svm_train(lower_train_data, train_label)

        elif self.classify_method == self.classify_methods[6]:
            self.bp_train(train_data, train_label)
        elif self.classify_method == self.classify_methods[7]:
            lower_train_data = self.pca_train(train_data)
            self.bp_train(lower_train_data, train_label)
            # print(type(train_data))
            # print(type(lower_train_data))
        elif self.classify_method == self.classify_methods[8]:
            lower_train_data = self.spca_train(train_data, train_label)
            self.bp_train(lower_train_data, train_label)
            # print(type(train_data))
            # print(type(lower_train_data))
        elif self.classify_method == self.classify_methods[9]:
            self.new_train(train_data, train_label)
        elif self.classify_method == self.classify_methods[10]:
            lower_train_data = self.pca_train(train_data)
            self.new_train(lower_train_data, train_label)
        elif self.classify_method == self.classify_methods[11]:
            lower_train_data = self.spca_train(train_data, train_label)
            self.new_train(lower_train_data, train_label)
        else:
            pass

        # self.show_data(pd.DataFrame(self.lower_train_data))
        self.show_message("训练完成")
        self.model_trained = True
        return

    def start_testing(self):
        if not self.model_trained:
            self.show_message("请训练模型")
            return
        print("testing")
        if '分类' in self.test_data.columns:
            self.test_data.drop('分类', axis=1, inplace=True)
        if '得分' in self.test_data.columns:
            self.test_data.drop('得分', axis=1, inplace=True)
        # self.classify_methods =
        # ['KNN', 'PCA-KNN', 'sPCA-KNN', 'SVM', 'PCA-SVM', 'sPCA-SVM', 'BP', 'PCA-BP', 'sPCA-BP']
        test_data = self.test_data.iloc[:, 2:].values
        train_data = self.train_data.iloc[:, 2:].values
        train_label = [self.wine_grades.index(grade) for grade in self.train_data['标签'].values]
        if self.classify_method == self.classify_methods[0]:
            results = self.knn_test(train_data, train_label, test_data)
        elif self.classify_method == self.classify_methods[1]:
            lower_train_data = self.pca_train(train_data)
            lower_test_data = self.pca_test(test_data)
            results = self.knn_test(lower_train_data, train_label, lower_test_data)
        elif self.classify_method == self.classify_methods[2]:
            lower_train_data = self.spca_train(train_data, train_label)
            lower_test_data = self.spca_test(test_data)
            results = self.knn_test(lower_train_data, train_label, lower_test_data)
        elif self.classify_method == self.classify_methods[3]:
            results = self.svm_test(test_data)
        elif self.classify_method == self.classify_methods[4]:
            lower_test_data = self.pca_test(test_data)
            results = self.svm_test(lower_test_data)
        elif self.classify_method == self.classify_methods[5]:
            lower_test_data = self.spca_test(test_data)
            results = self.svm_test(lower_test_data)
        elif self.classify_method == self.classify_methods[6]:
            results = self.bp_test(test_data)
        elif self.classify_method == self.classify_methods[7]:
            lower_test_data = self.pca_test(test_data)
            results = self.bp_test(lower_test_data)
        elif self.classify_method == self.classify_methods[8]:
            lower_test_data = self.spca_test(test_data)
            results = self.bp_test(lower_test_data)
        elif self.classify_method == self.classify_methods[9]:
            results = self.new_test(test_data)
        elif self.classify_method == self.classify_methods[10]:
            lower_test_data = self.pca_test(test_data)
            results = self.new_test(lower_test_data)
        elif self.classify_method == self.classify_methods[11]:
            lower_test_data = self.spca_test(test_data)
            results = self.new_test(lower_test_data)
        else:
            pass
        print("result figuring")

        self.data_classified = True
        result_visual = [self.wine_grades[result] for result in results]
        result_marks = [self.wine_marks[result] + random.randint(1, 8) for result in results]
        self.test_data = self.test_data.copy()
        self.test_data.insert(1, '分类', result_visual)
        self.test_data.insert(2, '得分', result_marks)

        all_test_label_visual = self.test_data['标签']
        test_label = [self.wine_grades.index(grade) for grade in all_test_label_visual if grade in self.wine_grades]
        test_result = [results[i] for i in range(len(all_test_label_visual)) if all_test_label_visual[i] in self.wine_grades]
        predict_result = [results[i] for i in range(len(all_test_label_visual)) if all_test_label_visual[i] not in self.wine_grades]
        self.clear_func()
        # 解决无法显示中文
        plt.rcParams['font.sans-serif']=['SimHei'] #指定默认字体,SimHei为黑体
        # 解决无法显示负号
        plt.rcParams['axes.unicode_minus'] = False
        if self.result_figure_type == self.result_figure_types[0]:
            legend = []
            if(len(test_result) > 0):
                result_mat, result_report = self.precision(test_label, test_result)
                result_mat.T.plot(kind='line', ax=self.ax, rot=90, xticks=range(len(result_mat.index)), fontsize=14)
                plt.text(1.5, max(result_mat.max()) / 2, result_report, fontsize=12)
                legend.extend(self.wine_grades)

            if(len(predict_result) > 0):
                dict = {}
                for key in predict_result:
                    dict[key] = dict.get(key, 0) + 1
                for i in range(4):
                    if i not in dict.keys():
                        dict[i] = 0
                # print(dict)
                predict = [dict[i] for i in range(4)]
                p_mat = pd.DataFrame(predict)
                # print(p_mat)
                p_mat.plot(kind='line', ax=self.ax, rot=90, xticks=range(4), fontsize=14)
                # plt.plot(range(4), predict)
                plt.xlim(0, 3)
                legend.append("未知")
            self.ax.set_xticklabels(self.wine_grades)
            plt.legend(legend)
        elif self.result_figure_type == self.result_figure_types[1]:
            y_labels = [' ']
            y_labels.extend(self.wine_grades)
            y_labels.append("未知")
            # print(y_labels)
            all_label = [y_labels.index(grade)-1 for grade in all_test_label_visual if grade in y_labels]
            # print(len(all_label))
            # print(len(results))
            plt.scatter(range(len(results)), all_label, marker='o', c='red')
            plt.scatter(range(len(results)), results, marker='x', c='blue')
            plt.legend(["分类标签", "分类结果"])
            plt.xlabel("测试数据")
            plt.ylabel("等级分类")
            plt.ylim(-0.5, 4.5)
            self.ax.set_yticklabels(y_labels)
            self.ax.xaxis.set_major_locator(plt.MultipleLocator(10))
            self.ax.yaxis.set_major_locator(plt.MultipleLocator(1))
            # self.ax.set_xticklabels(range(len(results)))

        self.table.setVisible(False)
        self.setting_widget.setVisible(False)
        self.canvas.setVisible(True)
        self.canvas.draw()
        self.main_state = 2
        self.show_message("测试完成")

    def gen_marks(self):
        marks = [self.wine_marks[result] + random.randint(1, 8) for result in results]
        return marks

    def save_data(self):
        if self.main_state == 0:
            self.show_message("没有数据可以保存")
        else:
            file_name = QFileDialog.getSaveFileName(self, '保存文件')[0]
            try:
                if self.main_state == 1:
                    self.train_data.to_excel(file_name)
                elif self.main_state == 2:
                    plt.savefig(file_name)
            except BaseException:
                self.show_message("保存失败")
            else:
                self.show_message("保存成功")

    def clear_func(self):
        sender = self.sender()
        if sender == self.clear_btn:
            self.show_message("清理成功")
        print("clearing data")
        self.main_state = 0
        self.table.setVisible(False)
        self.canvas.setVisible(False)
        self.setting_widget.setVisible(False)
        plt.cla()

    def exit_func(self):
        print("exited")
        self.close()
        # sys.exit(app.exec_())

    def show_message(self, text):
        self.message_box.setText(text)
        self.message_box.show()

    def check_data(self, data):
        print("checking data")
        # print(data)
        cols = data.columns.tolist()
        # print(cols)
        cols.remove('ID')
        # print(cols)
        for col in cols:
            # v = data[col].values.tolist()
            if not is_numeric_dtype(data[col]):
                # print("data cant")
                self.figure_able = 0
                return
        else:
            # print("data okk")
            self.figure_able = 1

    def pca_train(self, train_data):
        print("pca training")
        # normalize(train_data, axis=1)
        # train_data -= np.min(train_data)
        # train_data /= np.max(train_data)
        train_data -= np.mean(train_data, axis=0)
        # train_data /= np.std(train_data, axis=0)
        cov_mat = np.cov(train_data, rowvar=False)
        eig_vals, eig_vects = np.linalg.eig(np.mat(cov_mat))
        eig_val_indice = np.argsort(-eig_vals)
        #
        top = self.pca_top
        n_eig_val_indice = eig_val_indice[:top]
        # n_eig_val_indice = range(top)
        self.n_eig_vects = eig_vects[:, n_eig_val_indice]
        lower_pca_data = train_data.dot(self.n_eig_vects.real)
        # lower_pca_data = train_data * n_eig_vects.real
        lower_pca_data = np.array(lower_pca_data)
        lower_pca_data -= np.mean(lower_pca_data, axis=0)
        # lower_pca_data -= np.std(lower_pca_data, axis=0)
        #
        # print('eig_vals: ', eig_vals)
        # print('eig_vects: ', eig_vects.shape)
        # print("data", np.shape(train_data))
        # print("cov:", cov_mat.shape)
        # print("eig_val_indice", eig_val_indice)
        # print("n_eig_val_indice", n_eig_val_indice)
        # print("n_eig:",n_eig_vects.shape)
        # print("low:", lower_pca_data.shape)
        # recon_mat = (low_data_mat * eig_vects) + np.mean(train_data, axis=0)
        # print("rec:", recon_mat.shape)
        # return low_data_mat
        # transformer = PCA(n_components=self.pca_top)
        # lower_pca_data = transformer.fit_transform(train_data)
        return lower_pca_data

    def pca_test(self, test_data):
        print("pca testing")
        test_data -= np.mean(test_data, axis=0)
        lower_pca_data = test_data.dot(self.n_eig_vects.real)
        # lower_pca_data = train_data * n_eig_vects.real
        lower_pca_data = np.array(lower_pca_data)
        lower_pca_data -= np.mean(lower_pca_data, axis=0)
        return lower_pca_data

    def spca_train(self, train_data, train_label):
        print("spca training")
        # std = StandardScaler()
        # train_data = std.fit_transform(train_data)
        train_data -= np.mean(train_data, axis=0)
        # train_data /= np.std(train_data, axis=0)
        # rng = np.random.RandomState(0)
        self.spca_model = SparsePCA(n_components=self.pca_top, random_state=0, alpha=0, ridge_alpha=0)
        self.spca_model.fit(train_data)
        # self.spca_model.fit(train_data, train_label)
        lower_spca_data = self.spca_model.transform(train_data)
        # lower_spca_data = std.fit_transform(lower_spca_data)
        # print(type(lower_spca_data))
        # print(np.shape(lower_spca_data))
        # max = np.max(lower_spca_data)
        # print(max)
        # lower_spca_data /= max
        lower_spca_data /= max(np.abs(lower_spca_data[0,0]), 0.01)
        # lower_spca_data -= np.mean(lower_spca_data, axis=0)
        # normalize(lower_spca_data, axis=0)
        # print(np.shape(lower_spca_data))
        # print(lower_spca_data)
        return lower_spca_data

    def spca_test(self, test_data):
        print("spca testing")
        # std = StandardScaler()
        # test_data = std.fit_transform(test_data)
        test_data -= np.mean(test_data, axis=0)
        lower_spca_data = self.spca_model.transform(test_data)
        # lower_spca_data = std.fit_transform(lower_spca_data)
        # print(np.shape(lower_spca_data))
        # max = np.max(lower_spca_data)
        # lower_spca_data /= max
        lower_spca_data /= max(np.abs(lower_spca_data[0,0]), 0.01)
        # lower_spca_data -= np.mean(lower_spca_data, axis=0)
        # normalize(lower_spca_data, axis=0)
        # print(lower_spca_data)
        # lower_spca_data2 = self.spca_model.transform(test_data[0:5])
        # lower_spca_data2 /= lower_spca_data2[0,0]
        # print(lower_spca_data2)
        return lower_spca_data

    def knn_test(self, train_data, train_label, test_data):
        print("knn testing")
        train_data -= np.mean(train_data, axis=0)
        test_data -= np.mean(test_data, axis=0)
        # print(lower_train_data.shape)
        # print(lower_test_data.shape)
        result = []
        for i in range(test_data.shape[0]):
            nearest_index = -1 * np.ones(self.knn_top)
            nearest_class = np.ones(self.knn_top, dtype=np.int64)
            nearest_dists = np.ones(self.knn_top) * float('inf')
            for j in range(train_data.shape[0]):
                dist = np.sqrt(np.sum(np.square(test_data[i, :] - train_data[j, :])))
                if dist < max(nearest_dists):
                    nearest_index[np.argmax(nearest_dists)] = j
                    nearest_class[np.argmax(nearest_dists)] = train_label[j]
                    nearest_dists[np.argmax(nearest_dists)] = dist
            counts = np.bincount(nearest_class)
            result.append(np.argmax(counts))
            # print(nearest_index)
            # print(nearest_class)
            # print(nearest_dists)
            # print(len(result))
        # print(result)
        return result

    def svm_train(self, train_data, train_label):
        print("svm training")
        # train_data -= np.mean(train_data, axis=0)
        # train_data /= np.std(train_data, axis=0)
        self.svm_model = svm.SVC(decision_function_shape=self.svm_type, gamma='auto')
        # print(np.shape(train_data))
        # print(np.shape(train_label))
        # print("label: ", train_label)
        self.svm_model.fit(train_data, train_label)

    def svm_test(self, test_data):
        print("svm testing")
        # test_data -= np.mean(test_data, axis=0)
        # test_data /= np.std(test_data, axis=0)
        # print(np.shape(test_data))
        # print(np.shape(test_label))
        result = self.svm_model.predict(test_data)
        # print('result', result)
        # print(len(result))
        # print(len(self.test_data))
        return result


    # BP训练模块
    def bp_train(self, train_data, train_label):
        print("bp training")
        train_data -= np.mean(train_data, axis=0)
        # train_label_fit = LabelBinarizer().fit_transform(train_label)
        self.bp_model = MLPClassifier(random_state = 1, max_iter=self.bp_epoch ,learning_rate_init=self.bp_lr, hidden_layer_sizes=(self.bp_layers))
        self.bp_model.fit(train_data, train_label)
        # train_data -= np.min(train_data)
        # train_data /= np.max(train_data)
        # print(np.shape(train_label))
        # print(train_label)
        # layers = [np.shape(train_data)[1], self.bp_neuron_num, 4]
        # self.bp_model = NeuralNetwork(layers, 'logistic')
        # self.bp_model.fit(train_data, train_label_fit, learning_rate=self.bp_lr, epochs=self.bp_epoch)

    # BP测试模块
    def bp_test(self, test_data):
        print("bp testing")
        # print(type(test_data))
        test_data -= np.mean(test_data, axis=0)
        # test_data -= np.min(test_data)
        # test_data /= np.max(test_data)
        model = self.bp_model
        result = self.bp_model.predict(test_data)
        # print(model.trained)
        # result = []
        # for i in range(np.shape(test_data)[0]):
        #     print(type(test_data[i]))
            # o = model.predict(test_data[i])
            # np.argmax:第几个数对应最大概率值
            # result.append(np.argmax(o))
            # print(results)
        return result

    def new_train(self, train_data, train_label):
        # classifiers = [
        #     KNeighborsClassifier(self.knn_top),
        #     SVC(kernel="linear", C=0.025),
        #     SVC(gamma=2, C=1),
        #     GaussianProcessClassifier(1.0 * RBF(1.0)),
        #     DecisionTreeClassifier(max_depth=self.dt_depth),
        #     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        #     MLPClassifier(alpha=1, learning_rate_init=self.bp_lr, max_iter=self.bp_epoch),
        #     AdaBoostClassifier(),
        #     GaussianNB(),
        #     QuadraticDiscriminantAnalysis()]
        self.new_model = DecisionTreeClassifier(max_depth=self.dt_depth)
        self.new_model.fit(train_data, train_label)

    def new_test(self, test_data):
        result = self.new_model.predict(test_data)
        return result

    # 计算精度
    def precision(self, test_label, test_result):
        print("precision")
        if len(test_label) > 0:
            self.clear_func()
            # m = confusion_matrix(test_label, test_result)
            m = np.zeros((4, 4))
            for i in range(len(test_label)):
                m[test_label[i], test_result[i]] += 1
            result_mat = pd.DataFrame(m)
            result_report = classification_report(test_label, test_result)
            # print(m)
            # print(result_mat)
            print(result_report)
            return result_mat, result_report
        else:
            return None,None

    # 显示设置
    def setting_func(self):
        self.table.setVisible(False)
        self.canvas.setVisible(False)
        self.setting_widget.setVisible(True)
        # self.figure_layout.removeWidget(self.table)
        # self.figure_layout.removeWidget(self.canvas)
        # self.figure_layout.addWidget(self.setting_widget)
        self.main_state = 4
        # self.setting_widget.show()

    # 改变设置
    def change_setting(self):
        print("changing settings")
        try:
            self.theme = self.setting_widget.theme_cb.currentText()
            self.pca_top = int(self.setting_widget.pca_top_le.text())
            self.knn_top = int(self.setting_widget.knn_top_le.text())
            self.bp_lr = float(self.setting_widget.bp_lr_le.text())
            self.bp_epoch = int(self.setting_widget.bp_epoch_le.text())
            self.bp_layers = int(self.setting_widget.bp_layers_le.text())
            self.dt_depth = int(self.setting_widget.dt_depth_le.text())
            self.svm_type = self.setting_widget.svm_type_cb.currentText()
            self.model_trained = False
            self.data_classified = False
        except:
            self.reset_setting()
        # print(self.theme, self.pca_top, self.knn_top, self.svm_type, self.bp_lr, self.bp_epoch)
        if self.theme == self.themes[0]:
            # self.setStyle(QStyleFactory.create("Windows"))
            self.setStyleSheet("QWidget{color: black; background-color:#e0eaef;}QPushButton, QComboBox, QLineEdit{background-color: #d0dadf}")
            plt.cla()
            plt.style.use('themes\\qlight_color')
            # plt.gcf().set_textcolor('black')
            plt.gcf().set_facecolor('#e0eaef')
            plt.gca().set_facecolor('#e0eaef')
            # plt.gcf().set_edgecolor('#e0eaef')
        elif self.theme == self.themes[1]:
            # self.themes = ['浅色', '深色', '白色']
            # self.setStyleSheet(load_stylesheet_pyqt5())
            # QApplication.setStyle(QStyleFactory.create("Fusion"))
            self.setStyleSheet("QWidget{color: #b0b0b0; background-color:#203040;}QPushButton, QComboBox, QLineEdit{ background-color: #304050}")
            plt.cla()
            plt.style.use('themes\\qdark_color')
            # plt.gcf().set_textcolor('white')
            plt.gcf().set_facecolor('#203040')
            plt.gca().set_facecolor('#203040')
        elif self.theme == self.themes[2]:
            # self.themes = ['浅色', '深色', '白色']
            # self.setStyleSheet(load_stylesheet_pyqt5())
            # QApplication.setStyle(QStyleFactory.create("Fusion"))
            self.setStyleSheet("QWidget{color: black; background-color:white;}QPushButton, QComboBox, QLineEdit{ background-color: white}")
            plt.cla()
            plt.style.use('themes\\qwhite_color')
            # plt.gcf().set_textcolor('white')
            plt.gcf().set_facecolor('white')
            plt.gca().set_facecolor('white')
        # self.show()
        self.model_trained = False
        self.show_message("已改变设置")

    # 重置设置
    def reset_setting(self):
        print("reseting settings")
        self.theme = self.themes[0]
        self.pca_top = 12
        self.knn_top = 7
        self.bp_lr = 0.1
        self.bp_layers = 100
        self.bp_epoch = 3000
        self.dt_depth = 10
        self.svm_type = self.svm_types[0]
        self.setting_widget.pca_top_le.setText(str(self.pca_top))
        self.setting_widget.knn_top_le.setText(str(self.knn_top))
        self.setting_widget.bp_layers_le.setText(str(self.bp_layers))
        self.setting_widget.bp_lr_le.setText(str(self.bp_lr))
        self.setting_widget.svm_type_cb.setCurrentText(self.svm_type)
        self.setting_widget.theme_cb.setCurrentText(self.theme)
        # print(self.theme, self.pca_top, self.knn_top, self.svm_type, self.bp_lr, self.bp_epoch)
        self.setStyleSheet("QWidget{color: black; background-color:#e0eaef;}QPushButton, QComboBox, QLineEdit{background-color: #d0dadf}")
        plt.cla()
        plt.style.use('themes\\qlight_color')
        plt.gcf().set_facecolor('#e0eaef')
        plt.gca().set_facecolor('#e0eaef')
        self.model_trained = False
        self.main_state = 0
        sender = self.sender()
        if sender == self.setting_widget.reset_btn:
            self.show_message("已恢复默认设置")

# 表格模型
class QtTable(QAbstractTableModel):
    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None

# 下拉框
class ComboCheckBox(QComboBox):
    def __init__(self, items):  # items==[str,str...]
        super(ComboCheckBox, self).__init__()
        self.items = items
        self.items.insert(0, '全部')
        self.row_num = len(self.items)
        self.Selectedrow_num = 0
        self.qCheckBox = []
        self.qLineEdit = QLineEdit()
        self.qLineEdit.setReadOnly(True)
        self.qListWidget = QListWidget()
        self.addQCheckBox(0)
        self.qCheckBox[0].setChecked(True)
        self.qCheckBox[0].stateChanged.connect(self.All)
        for i in range(1, self.row_num):
            self.addQCheckBox(i)
            self.qCheckBox[i].setChecked(True)
            self.qCheckBox[i].stateChanged.connect(self.show)
        self.setModel(self.qListWidget.model())
        self.setView(self.qListWidget)
        self.setLineEdit(self.qLineEdit)

    def addQCheckBox(self, i):
        self.qCheckBox.append(QCheckBox())
        qItem = QListWidgetItem(self.qListWidget)
        self.qCheckBox[i].setText(self.items[i])
        self.qListWidget.setItemWidget(qItem, self.qCheckBox[i])

    def Selectlist(self):
        Outputlist = []
        for i in range(1, self.row_num):
            if self.qCheckBox[i].isChecked() == True:
                Outputlist.append(self.qCheckBox[i].text())
        self.Selectedrow_num = len(Outputlist)
        return Outputlist

    def show(self):
        show = ''
        Outputlist = self.Selectlist()
        self.qLineEdit.setReadOnly(True)
        # self.qLineEdit.clear()
        for i in Outputlist:
            show += i + ';'
        if self.Selectedrow_num == 0:
            self.qCheckBox[0].setCheckState(0)
        elif self.Selectedrow_num == self.row_num - 1:
            self.qCheckBox[0].setCheckState(2)
        else:
            self.qCheckBox[0].setCheckState(1)
        # self.qLineEdit.setText(show)
        # self.qLineEdit.setReadOnly(True)

    def All(self, zhuangtai):
        if zhuangtai == 2:
            for i in range(1, self.row_num):
                self.qCheckBox[i].setChecked(True)
        elif zhuangtai == 1:
            if self.Selectedrow_num == 0:
                self.qCheckBox[0].setCheckState(2)
        elif zhuangtai == 0:
            self.clear_check()

    def clear_check(self):
        for i in range(self.row_num):
            self.qCheckBox[i].setChecked(False)

    def setItems(self, items):
        self.clear()
        self.items = items
        self.items.insert(0, '全部')
        # print(self.items)
        self.row_num = len(self.items)
        self.Selectedrow_num = 0
        self.qCheckBox = []
        self.qLineEdit = QLineEdit()
        self.qLineEdit.setReadOnly(True)
        self.qListWidget = QListWidget()
        self.addQCheckBox(0)
        self.qCheckBox[0].setChecked(True)
        self.qCheckBox[0].stateChanged.connect(self.All)
        for i in range(1, self.row_num):
            self.addQCheckBox(i)
            self.qCheckBox[i].setChecked(True)
            self.qCheckBox[i].stateChanged.connect(self.show)
        self.setModel(self.qListWidget.model())
        self.setView(self.qListWidget)
        self.setLineEdit(self.qLineEdit)

# 设置窗口
class Setting_widget(QWidget):
    def __init__(self, parent):
        super().__init__()
        # settings self.themes
        self.parent_widget = parent
        # self.themes = ['浅色', '深色']
        self.themes = ['浅色', '深色', '白色']
        self.theme = self.themes[1]
        self.theme_cb = QComboBox()
        self.theme_cb.setStatusTip("选择主题")
        self.theme_cb.addItems(self.themes)
        self.pca_top_le = QLineEdit("10")
        self.pca_top_le.setToolTip("PCA参数，3-15, 默认为10")
        self.knn_top_le = QLineEdit("7")
        self.knn_top_le.setToolTip("KNN参数，1-19, 默认为7")
        self.svm_types = ['ovo', 'ovr']
        self.svm_type = self.svm_types[0]
        self.svm_type_cb = QComboBox()
        self.svm_type_cb.addItems(self.svm_types)
        self.svm_type_cb.setToolTip("SVM种类（默认为ovo)")
        self.bp_lr_le = QLineEdit("0.1")
        self.bp_lr_le.setToolTip("BP-learning rate参数（默认为0.1）")
        self.bp_epoch_le = QLineEdit("3000")
        self.bp_epoch_le.setToolTip("BP-最大迭代次数（默认为3000）")
        self.bp_layers_le = QLineEdit("100")
        self.bp_layers_le.setToolTip("BP-隐藏层个数参数（默认为100）")
        self.dt_depth_le = QLineEdit("10")
        self.dt_depth_le.setToolTip("Desition tree深度（默认为10）")
        self.confirm_btn = QPushButton("确认")
        self.confirm_btn.clicked.connect(self.parent_widget.change_setting)
        self.cancel_btn = QPushButton("取消")
        self.confirm_btn.clicked.connect(self.parent_widget.change_setting)
        self.reset_btn = QPushButton("恢复默认")
        self.reset_btn.clicked.connect(self.parent_widget.reset_setting)
        self.theme_label = QLabel("主题")
        self.pca_label = QLabel("PCA(SPCA) 主成分数量:")
        self.knn_label = QLabel("KNN 最近邻数量:")
        self.svm_label = QLabel("SVM 多分类模型类型:")
        self.bp_layers_label = QLabel("BP hidden layers:")
        self.bp_lr_label = QLabel("BP learning rate:")
        self.bp_epoch_label = QLabel("BP max epochs:")
        self.dt_depth_label = QLabel("Desition tree max depth:")

        self.setting_layout = QGridLayout()
        self.setting_layout.addWidget(self.theme_label, 1, 2)
        self.setting_layout.addWidget(self.pca_label, 3, 2)
        self.setting_layout.addWidget(self.knn_label, 4, 2)
        self.setting_layout.addWidget(self.svm_label, 5, 2)
        self.setting_layout.addWidget(self.bp_layers_label, 6, 2)
        self.setting_layout.addWidget(self.bp_lr_label, 7, 2)
        self.setting_layout.addWidget(self.bp_epoch_label, 8, 2)
        self.setting_layout.addWidget(self.dt_depth_label, 9, 2)
        self.setting_layout.addWidget(self.confirm_btn, 1, 8)
        self.setting_layout.addWidget(self.cancel_btn, 2, 8)
        self.setting_layout.addWidget(self.reset_btn, 3, 8)
        self.setting_layout.addWidget(self.theme_cb, 1, 3)
        self.setting_layout.addWidget(self.pca_top_le, 3, 3)
        self.setting_layout.addWidget(self.knn_top_le, 4, 3)
        self.setting_layout.addWidget(self.svm_type_cb, 5, 3)
        self.setting_layout.addWidget(self.bp_layers_le, 6, 3)
        self.setting_layout.addWidget(self.bp_lr_le, 7, 3)
        self.setting_layout.addWidget(self.bp_epoch_le, 8, 3)
        self.setting_layout.addWidget(self.dt_depth_le, 9, 3)
        self.setting_layout.setRowStretch(0, 1)
        self.setting_layout.setRowStretch(1, 1)
        self.setting_layout.setRowStretch(2, 1)
        self.setting_layout.setRowStretch(3, 1)
        self.setting_layout.setRowStretch(4, 1)
        self.setting_layout.setRowStretch(5, 1)
        self.setting_layout.setRowStretch(6, 1)
        self.setting_layout.setRowStretch(7, 1)
        self.setting_layout.setRowStretch(8, 1)
        self.setting_layout.setRowStretch(9, 1)
        self.setting_layout.setRowStretch(10, 1)
        self.setting_layout.setColumnStretch(0,3)
        self.setting_layout.setColumnStretch(1,3)
        self.setting_layout.setColumnStretch(2,3)
        self.setting_layout.setColumnStretch(3,1)
        self.setting_layout.setColumnStretch(4,2)
        self.setting_layout.setColumnStretch(5,2)
        self.setting_layout.setColumnStretch(6,2)
        self.setting_layout.setColumnStretch(7,2)
        self.setting_layout.setColumnStretch(8,1)
        self.setting_layout.setColumnStretch(9,3)
        self.setting_layout.setColumnStretch(10,3)
        self.setLayout(self.setting_layout)
        self.resize(800, 600)
        # self.show()


# BP 手动实现
# 定义tanh函数
def tanh(x):
    return np.tanh(x)


# tanh函数的导数
def tan_deriv(x):
    return 1.0 - np.tanh(x) * np.tan(x)


# sigmoid函数
def logistic(x):
    return 1 / (1 + np.exp(-x))


# sigmoid函数的导数
def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))

# 神经网络模型类
class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        """
        神经网络算法构造函数
        :param layers: 神经元层数
        :param activation: 使用的函数（默认tanh函数）
        :return:none
        """
        self.trained = False
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tan_deriv

        # 权重列表
        self.weights = []
        # 初始化权重（随机）
        for i in range(1, len(layers) - 1):
            self.weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)
            self.weights.append((2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1) * 0.25)

    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        """
        训练神经网络
        :param X: 数据集（通常是二维）
        :param y: 分类标记
        :param learning_rate: 学习率（默认0.2）
        :param epochs: 训练次数（最大循环次数，默认10000）
        :return: none
        """
        # 确保数据集是二维的
        X = np.atleast_2d(X)

        temp = np.ones([X.shape[0], X.shape[1] + 1])
        temp[:, 0: -1] = X
        X = temp
        y = np.array(y)

        for k in range(epochs):
            # 随机抽取X的一行
            i = np.random.randint(X.shape[0])
            # 用随机抽取的这一组数据对神经网络更新
            a = [X[i]]
            # 正向更新
            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l], self.weights[l])))
            error = y[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]

            # 反向更新
            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))
                deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

        self.trained = True

    def predict(self, x):
        x = np.array(x)
        # x = x.flatten()
        # print("x : ", x.shape)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

if  __name__ == '__main__':
    app = QApplication(sys.argv)
    WC = WineClassify()
    print("running")
    sys.exit(app.exec_())
