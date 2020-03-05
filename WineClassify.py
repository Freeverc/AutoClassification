import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn import svm
from PyQt5.QtWidgets import QApplication, QTableView, QWidget, QStyleFactory
from PyQt5.QtCore import QAbstractTableModel, Qt
from PyQt5.QtWidgets import QPushButton, QFileDialog, QMessageBox
from PyQt5.QtWidgets import QLabel, QComboBox, QLineEdit, QListWidget, QCheckBox, QListWidgetItem, QTableWidget
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout
from qdarkstyle import load_stylesheet_pyqt5
import pandas as pd
from pandas.api.types import is_numeric_dtype
import bp
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report

# 原酒鉴评规则
# TY级酒分段93分以上
# YY级酒分段88 - 92.9
# 分
# RY级酒分段80 - 87.9
# 分
# SY级酒分段70 - 79.9
# 分
#

class WineClassify(QWidget):
    def __init__(self, app):
        super().__init__()
        self.button_widget = QWidget()
        self.main_widget = QWidget()
        self.message_box = QMessageBox()
        self.figure_able = 0

        self.init_ui()

    def init_ui(self):
        self.input_data = pd.DataFrame()
        self.info_data = pd.DataFrame()
        self.label_data = pd.DataFrame()
        self.all_train_data = pd.DataFrame()
        self.train_data = pd.DataFrame()
        self.all_test_data = pd.DataFrame()
        self.test_data = pd.DataFrame()
        self.result_data = pd.DataFrame()

        self.themes = ['浅色', '深色']
        self.theme = self.themes[0]
        self.model = []
        self.pca_top = 20
        self.knn_top = 7
        self.bp_lr = 0.2
        self.bp_epoch = 10000
        self.svm_types = ['ovo', 'ova']
        self.svm_type = self.svm_types[0]

        self.messages = ['请导入Excel文件', '请导入包含感官鉴定的有效数据', '请点击链接', '请点击训练']

        self.import_methods = ['导入白酒信息', '导入训练数据', '导入测试数据']
        self.import_method = self.import_methods[0]

        self.show_data_methods = ['显示白酒信息', '显示全部训练数据', '显示全部测试数据', '显示训练数据', '显示测试数据']
        self.show_data_method = self.show_data_methods[0]

        self.data_linked = False
        self.data_classified = False

        self.figure_types = ['折线图', '条形图', '面积图']
        self.figure_type = self.figure_types[0]
        self.data_classified = False

        self.classify_methods = ['KNN', 'PCA-KNN', 'SVM', 'PCA-SVM', 'BP', 'PCA-BP', 'sPCA', 'sPCA-SVM']
        self.classify_method = self.classify_methods[0]

        self.wine_grades = ['特级', '一级', '二级', '优级']
        self.wine_grade = self.wine_grades[0]

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

        self.figure_type_cb = QComboBox()
        self.figure_type_cb.setToolTip('选择可视化绘图方法')
        self.figure_type_cb.addItems(self.figure_types)
        self.figure_type_cb.activated[str].connect(self.change_type)

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
        self.button_layout.addWidget(self.figure_type_cb)
        self.button_layout.addWidget(self.train_btn)
        self.button_layout.addWidget(self.classify_btn)
        self.button_layout.addWidget(self.classify_type_cb)
        self.button_layout.addWidget(self.settint_btn)
        self.button_layout.addWidget(self.save_btn)
        # self.button_layout.addWidget(self.clear_btn)
        self.button_layout.addWidget(self.exit_btn)
        self.button_widget.setLayout(self.button_layout)


        self.setting_widget = Setting_widget(self)
        if self.theme == self.themes[1]:
            self.setStyleSheet(load_stylesheet_pyqt5())
            plt.style.use('qdark_color')
        # plt.style.use('dark_background')
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot()
        self.canvas = FigureCanvas(self.figure)
        self.table = QTableView()

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
        self.setWindowTitle('智能白酒分析鉴定系统')
        desktop = QApplication.desktop()
        self.resize(desktop.width(), desktop.height())
        # self.move(200, 200)
        self.show()

    def change_type(self, text):
        print("changing type")
        if self.sender() == self.import_method_cb:
            self.import_method = text
        elif self.sender() == self.show_data_method_cb:
            self.show_data_method = text
        elif self.sender() == self.figure_type_cb:
            self.figure_type = text
        elif self.sender() == self.classify_type_cb:
            self.classify_method = text
        else:
            pass

    def import_data(self):
        print("importing data")
        file_name = QFileDialog.getOpenFileName(self, '打开文件')[0]
        if not os.path.splitext(file_name)[1] in ['.xls', '.xlsx']:
            self.show_message(self.messages[0])
        else:
            try:
                self.input_data = pd.read_excel(file_name)
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
                        if '感官鉴定' in self.input_data.columns:
                            self.show_message("已" + self.import_method)
                            self.data_linked = False
                            self.label_data = self.input_data
                            self.show_data(self.all_train_data)
                        else:
                            self.show_message("请导入包含感官鉴定的白酒信息")
                    elif self.import_method == self.import_methods[1]:
                        self.show_message("已" + self.import_method)
                        self.data_linked = False
                        self.all_train_data = self.input_data
                        self.show_data(self.label_data)
                    elif self.import_method == self.import_methods[2]:
                        self.show_message("已" + self.import_method)
                        self.data_linked = False
                        self.all_test_data = self.input_data
                        self.show_data(self.all_test_data)
                    else:
                        pass

    def link_data(self):
        print("linking data")
        self.info_data = pd.read_excel("wine_i.xlsx")
        self.label_data = pd.read_excel("wine_l.xlsx")
        self.all_train_data = pd.read_excel("wine_d.xlsx")
        self.all_test_data = pd.read_excel("wine_d.xlsx")
        self.data_linked = False
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
        else:
            for i in range(1, self.label_data.shape[1]):
                col = self.label_data.columns.tolist()[i]
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
            # self.show_data(self.test_data)
            self.data_linked = True

            cols = self.all_train_data.columns.tolist()[self.label_data.shape[1]:]
            cols.insert(0, '编号')
            cols.insert(1, '感官鉴定')
            self.train_data = self.all_train_data.loc[self.all_train_data['感官鉴定'].isin(self.wine_grades), cols]
            train_labels = [self.wine_grades.index(grade) for grade in self.train_data['感官鉴定'].values]
            self.train_data.insert(2, 'label', train_labels)

            cols = self.all_test_data.columns.tolist()[self.label_data.shape[1]:]
            cols.insert(0, '编号')
            cols.insert(1, '感官鉴定')
            self.test_data = self.all_test_data.loc[self.all_test_data['感官鉴定'].isin(self.wine_grades), cols]
            test_labels = [self.wine_grades.index(grade) for grade in self.test_data['感官鉴定'].values]
            self.test_data.insert(2, 'label', test_labels)
            print(self.all_train_data.shape)
            print(self.all_test_data.shape)
            print(self.train_data.shape)
            print(self.test_data.shape)
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
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
            # plt.rcParams['font.sans-serif']=['SimHei'] #指定默认字体,SimHei为黑体
            # 解决无法显示负号
            plt.rcParams['axes.unicode_minus'] = False
            plt.title(self.figure_type)
            label = self.test_data.编号.tolist()
            if len(self.test_data.index > 8):
                data = self.test_data.iloc[:8, 3:].T
            else:
                data = self.test_data.iloc[:, 3:].T
            # print(data)
            if self.figure_type == self.figure_types[0]:
                data.plot(kind='line', ax=self.ax, rot=90, xticks=range(len(data.index)), fontsize=10)
                # pass
            elif self.figure_type == self.figure_types[1]:
                data.plot(kind='bar', ax=self.ax, fontsize=10)
            elif self.figure_type == self.figure_types[2]:
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
        self.link_data()
        print("training")
        # self.classify_methods = ['KNN', 'PCA-KNN', 'SVM', 'PCA-SVM', 'BP', 'sPCA', 'sPCA-SVM']
        train_label = self.train_data.iloc[:, 2].values.tolist()
        train_data = self.train_data.iloc[:, 3:].values
        # start training
        if self.classify_method == self.classify_methods[0]:
            pass
        elif self.classify_method == self.classify_methods[1]:
            pass
        elif self.classify_method == self.classify_methods[2]:
            self.model = self.svm_train(train_data, train_label)
        elif self.classify_method == self.classify_methods[3]:
            lower_train_data = self.pca_train(train_data, train_label)
            self.model = self.svm_train(lower_train_data, train_label)
        elif self.classify_method == self.classify_methods[4]:
            self.model = self.bp_train(train_data, train_label)
        elif self.classify_method == self.classify_methods[5]:
            lower_train_data = self.pca_train(train_data, train_label)
            self.model = self.bp_train(lower_train_data, train_label)
        else:
            pass

        # self.show_data(pd.DataFrame(self.lower_train_data))
        self.show_message("训练完成")
        self.data_classified = False

    def start_testing(self):
        print("testing")
        if 'result' in self.test_data.columns:
            self.test_data.drop('result', axis=1, inplace=True)
        # self.classify_methods = ['KNN', 'PCA-KNN', 'SVM', 'PCA-SVM', 'BP', 'PCA-BP', 'sPCA', 'sPCA-SVM']
        test_label = self.test_data.iloc[:, 2].values.tolist()
        test_data = self.test_data.iloc[:, 3:].values
        if self.classify_method == self.classify_methods[0]:
            train_data = self.train_data.iloc[:, 3:].values
            train_label = self.train_data.iloc[:, 2].values.tolist()
            result = self.knn_test(train_data, train_label, test_data, test_label)
        elif self.classify_method == self.classify_methods[1]:
            train_data = self.train_data.iloc[:, 3:].values
            train_label = self.train_data.iloc[:, 2].values.tolist()
            lower_train_data = self.pca_train(train_data, test_label)
            lower_test_data = self.pca_train(test_data, test_label)
            result = self.knn_test(lower_train_data, train_label, lower_test_data, test_label)
        elif self.classify_method == self.classify_methods[2]:
            result = self.svm_test(test_data, test_label)
        elif self.classify_method == self.classify_methods[3]:
            lower_test_data = self.pca_train(test_data, test_label)
            result = self.svm_test(lower_test_data, test_label)
        elif self.classify_method == self.classify_methods[4]:
            result = self.bp_test(test_data, test_label)
        elif self.classify_method == self.classify_methods[5]:
            lower_test_data = self.pca_train(test_data, test_label)
            result = self.bp_test(lower_test_data, test_label)
        else:
            pass

        self.data_classified = True
        if 'result' in self.test_data.columns:
            self.test_data.drop('result', axis=1, inplace=True)
        self.test_data.insert(1, 'result', result)
        self.precision(test_data, test_label, result)
        # self.show_data(self.test_data)
        self.show_message("测试完成")

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

    def pca_train(self, train_data, train_label):
        print("pca training")
        train_data = train_data - np.mean(train_data, axis=0)
        cov_mat = np.cov(train_data, rowvar=0)
        eig_vals, eig_vects = np.linalg.eig(np.mat(cov_mat))
        # eig_val_indice = np.argsort(eig_vals)

        top = self.pca_top
        n_eig_val_indice = range(top)
        n_eig_vects = eig_vects[:, n_eig_val_indice]
        low_data_mat = train_data * n_eig_vects.real
        # print('eig_vals: ', eig_vals.shape)
        # print('eig_vects: ', eig_vects.shape)
        # print("data", np.shape(train_data))
        # print("data", np.shape(train_label))
        # print("cov:", cov_mat.shape)
        # print("eig_val_indice: ", eig_val_indice)
        # print("n_eig_val_indice", n_eig_val_indice)
        # print("n_eig:",n_eig_vects.shape)
        # print("low:", low_data_mat.shape)
        # recon_mat = (low_data_mat * eig_vects) + np.mean(train_data, axis=0)
        # print("rec:", recon_mat.shape)
        return low_data_mat

    def knn_test(self, lower_train_data, train_label, lower_test_data, test_label):
        print("knn testing")
        print(lower_train_data.shape)
        print(lower_test_data.shape)
        result = []
        for i in range(lower_test_data.shape[0]):
            nearest_index = -1 * np.ones(self.knn_top)
            nearest_class = np.ones(self.knn_top, dtype=np.int64)
            nearest_dists = np.ones(self.knn_top) * float('inf')
            for j in range(lower_train_data.shape[0]):
                dist = np.sqrt(np.sum(np.square(lower_test_data[i, :] - lower_train_data[j, :])))
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
        print("svm trainging")
        clf = svm.SVC(decision_function_shape=self.svm_type)
        print(np.shape(train_data))
        print(np.shape(train_label))
        # print("label: ", train_label)
        clf.fit(train_data, train_label)
        return clf

    def svm_test(self, test_data, test_label):
        print("svm testing")
        print(np.shape(test_data))
        print(np.shape(test_label))
        result = self.model.predict(test_data)
        # print('result', result)
        # print(len(result))
        # print(len(self.test_data))
        return result

    def bp_train(self, train_data, train_label):
        print("bp training")
        train_data = train_data - np.mean(train_data, axis=0)
        # print(np.shape(train_data))
        # print(np.shape(train_label))
        # print(train_label)
        train_label_fit = LabelBinarizer().fit_transform(train_label)
        layers = [np.shape(train_data)[1], 100, 4]
        nn_model = bp.NeuralNetwork(layers, 'logistic')
        nn_model.fit(train_data, train_label_fit, learning_rate=self.bp_lr, epochs=self.bp_epoch)
        return nn_model

    def bp_test(self, test_data, test_label):
        print("bp testing")
        test_data = test_data - np.mean(test_data, axis=0)
        model = self.model
        predictions = []
        for i in range(np.shape(test_data)[0]):
            o = model.predict(test_data[i])
            # np.argmax:第几个数对应最大概率值
            predictions.append(np.argmax(o))
        return predictions

    def precision(self, test_data, test_label, results):
        if len(test_label) > 0:
            print(confusion_matrix(test_label, results))
            # print(classification_report(test_label, results))
            report = classification_report(test_label, results)
            self.clear_func()
            # 解决无法显示中文
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
            # plt.rcParams['font.sans-serif']=['SimHei'] #指定默认字体,SimHei为黑体
            # 解决无法显示负号
            plt.rcParams['axes.unicode_minus'] = False
            plt.title(self.figure_type)
            # font = {'weight': 'normal',
            #         'size': 11,
            #         'color': 'r',
            #         }
            # f = self.ax
            # f.text(10, 10, "report", fontdict=font)
            plt.plot(range(10), range(20, 10, -1))
            plt.text(100, 100, "report ")
            self.table.setVisible(False)
            self.setting_widget.setVisible(False)
            self.canvas.setVisible(True)
            print(report)
            self.canvas.draw()
            self.main_state = 2
        else:
            pass

    def setting_func(self):
        self.table.setVisible(False)
        self.canvas.setVisible(False)
        self.setting_widget.setVisible(True)
        # self.figure_layout.removeWidget(self.table)
        # self.figure_layout.removeWidget(self.canvas)
        # self.figure_layout.addWidget(self.setting_widget)
        self.main_state = 4
        # self.setting_widget.show()

    def change_setting(self):
        print("changing settings")
        self.theme = self.setting_widget.theme_cb.currentText()
        self.pca_top = self.setting_widget.pca_top_le.text()
        self.knn_top = self.setting_widget.knn_top_le.text()
        self.bp_lr = self.setting_widget.bp_lr_le.text()
        self.bp_epoch = self.setting_widget.bp_epoch_le.text()
        self.svm_type = self.setting_widget.svm_type_cb.currentText()
        print(self.theme, self.pca_top, self.knn_top, self.svm_type, self.bp_lr, self.bp_epoch)
        if self.theme == self.themes[0]:
            self.setStyle(QStyleFactory.create("Macintosh"))
            plt.cla()
            plt.style.use('qwhite_color')
            plt.gcf().set_facecolor('white')
            plt.gca().set_facecolor('white')
        elif self.theme == self.themes[1]:
            self.setStyleSheet(load_stylesheet_pyqt5())
            # QApplication.setStyle(QStyleFactory.create("Fusion"))
            plt.cla()
            plt.style.use('qdark_color')
            plt.gcf().set_facecolor('#19232d')
            plt.gca().set_facecolor('#19232d')
        self.show()

    def reset_setting(self):
        print("reseting settings")
        self.theme = self.themes[0]
        self.model = []
        self.pca_top = 20
        self.knn_top = 7
        self.bp_lr = 0.2
        self.bp_epoch = 10000
        self.svm_type = self.svm_types[0]
        print(self.theme, self.pca_top, self.knn_top, self.svm_type, self.bp_lr, self.bp_epoch)
        if self.theme == self.themes[0]:
            self.setStyle(QStyleFactory.create("Macintosh"))
            plt.cla()
            plt.style.use('qwhite_color')
            plt.gcf().set_facecolor('white')
            plt.gca().set_facecolor('white')
        elif self.theme == self.themes[1]:
            self.setStylesheet(load_stylesheet_pyqt5())
            # QApplication.setStyle(QStyleFactory.create("Fusion"))
            plt.cla()
            plt.style.use('qdark_color')
            plt.gcf().set_facecolor('#19232d')
            plt.gca().set_facecolor('#19232d')
        self.show()

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

class Setting_widget(QWidget):
    def __init__(self, parent):
        super().__init__()
        # settings self.themes
        self.parent_widget = parent
        self.themes = ['浅色', '深色']
        self.theme = self.themes[1]
        self.theme_cb = QComboBox()
        self.theme_cb.setStatusTip("选择主题")
        self.theme_cb.addItems(self.themes)
        self.pca_top_le = QLineEdit()
        self.pca_top_le.setPlaceholderText("请输入PCA参数（默认为10）")
        self.knn_top_le = QLineEdit()
        self.knn_top_le.setPlaceholderText("请输入KNN参数（默认为7）")
        self.svm_types = ['ovo', 'ova']
        self.svm_type = self.svm_types[0]
        self.svm_type_cb = QComboBox()
        self.svm_type_cb.addItems(self.svm_types)
        self.svm_type_cb.setToolTip("请选择SVM种类（默认为ovo)")
        self.bp_lr_le = QLineEdit()
        self.bp_lr_le.setPlaceholderText("请输入BP-learning rate参数（默认为0.2）")
        self.bp_epoch_le = QLineEdit()
        self.bp_epoch_le.setPlaceholderText("请输入BP-epoch参数（默认为10000）")
        self.confirm_btn = QPushButton("确认")
        self.confirm_btn.clicked.connect(self.parent_widget.change_setting)
        self.cancel_btn = QPushButton("取消")
        self.confirm_btn.clicked.connect(self.parent_widget.change_setting)
        self.reset_btn = QPushButton("恢复默认")
        self.reset_btn.clicked.connect(self.parent_widget.reset_setting)
        self.theme_label = QLabel("主题")
        self.pca_label = QLabel("PCA 主成分数量")
        self.knn_label = QLabel("KNN 最近邻数量")
        self.svm_label = QLabel("SVM 多分类模型类型")
        self.bp_lr_label = QLabel("BP:learning rate")
        self.bp_epoch_label = QLabel("BP:epoch")

        self.setting_layout = QGridLayout()
        self.setting_layout.addWidget(self.theme_label, 1, 2)
        self.setting_layout.addWidget(self.pca_label, 3, 2)
        self.setting_layout.addWidget(self.knn_label, 4, 2)
        self.setting_layout.addWidget(self.svm_label, 5, 2)
        self.setting_layout.addWidget(self.bp_lr_label, 6, 2)
        self.setting_layout.addWidget(self.bp_epoch_label, 7, 2)
        self.setting_layout.addWidget(self.confirm_btn, 1, 8)
        self.setting_layout.addWidget(self.cancel_btn, 2, 8)
        self.setting_layout.addWidget(self.reset_btn, 3, 8)
        self.setting_layout.addWidget(self.theme_cb, 1, 3)
        self.setting_layout.addWidget(self.pca_top_le, 3, 3)
        self.setting_layout.addWidget(self.knn_top_le, 4, 3)
        self.setting_layout.addWidget(self.svm_type_cb, 5, 3)
        self.setting_layout.addWidget(self.bp_lr_le, 6, 3)
        self.setting_layout.addWidget(self.bp_epoch_le, 7, 3)
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
        self.setting_layout.setColumnStretch(0,2)
        self.setting_layout.setColumnStretch(1,2)
        self.setting_layout.setColumnStretch(2,6)
        self.setting_layout.setColumnStretch(3,3)
        self.setting_layout.setColumnStretch(4,3)
        self.setting_layout.setColumnStretch(5,2)
        self.setting_layout.setColumnStretch(6,2)
        self.setting_layout.setColumnStretch(7,2)
        self.setting_layout.setColumnStretch(8,2)
        self.setting_layout.setColumnStretch(9,2)
        self.setting_layout.setColumnStretch(10,2)
        self.setLayout(self.setting_layout)
        self.resize(800, 600)
        # self.show()


if  __name__ == '__main__':
    app = QApplication(sys.argv)
    WC = WineClassify(app)
    print("running")
    sys.exit(app.exec_())
