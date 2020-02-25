from qdarkstyle import load_stylesheet_pyqt5
import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QTableView, QWidget
from PyQt5.QtCore import QAbstractTableModel, Qt
from PyQt5.QtWidgets import QPushButton, QFileDialog, QMessageBox
from PyQt5.QtWidgets import QComboBox, QLineEdit, QListWidget, QCheckBox, QListWidgetItem
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout
import pandas as pd
from pandas.plotting import parallel_coordinates, radviz, andrews_curves, scatter_matrix
from pandas.api.types import is_numeric_dtype
import seaborn as sns
import rpy2.robjects as robjects

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
    def __init__(self):
        super().__init__()
        self.button_widget = QWidget()
        self.figure_widget = QWidget()
        self.message_box = QMessageBox()
        self.table = QTableView()
        # ui jiemian 0:空, 1:表格 2:图 3: 结果 4：设置
        self.ui_state = 0
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

        self.messages = ['请导入Excel文件', '请导入包含感官鉴定的有效数据', '请点击链接', '请点击训练']

        self.import_methods = ['导入白酒信息', '导入训练数据', '导入测试数据']
        self.import_method = self.import_methods[0]

        self.show_data_methods = ['显示白酒信息', '显示全部训练数据', '显示全部测试数据', '显示训练数据', '显示测试数据']
        self.show_data_method = self.show_data_methods[0]

        self.data_linked = False

        self.figure_types = ['折线图', '条形图', '面积图']
        self.figure_type = self.figure_types[0]
        self.data_classified = False

        self.classify_types = ['PCA', 'PCA-SVM', 'PCA-BP', 'sPCA-SVM', 'sPCA-BP', 'BP']
        self.classify_type = self.classify_types[0]

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
        self.classify_btn.clicked.connect(self.classify_func)

        self.save_btn = QPushButton('保存')
        self.save_btn.setToolTip('保存图像或数据')
        self.save_btn.clicked.connect(self.save_data)

        self.settint_btn = QPushButton('设置')
        self.settint_btn.setToolTip('设置显示的图像和数据格式')
        self.settint_btn.clicked.connect(self.clear_func)

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
        self.classify_type_cb.addItems(self.classify_types)
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
        self.button_layout.addWidget(self.clear_btn)
        self.button_layout.addWidget(self.exit_btn)
        self.button_widget.setLayout(self.button_layout)

        self.figure = plt.figure()
        self.ax = self.figure.add_subplot()
        self.canvas = FigureCanvas(self.figure)

        self.figure_layout = QHBoxLayout()
        self.figure_layout.addWidget(self.table)
        self.figure_widget.setLayout(self.figure_layout)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.button_widget)
        self.layout.addWidget(self.figure_widget)
        self.setLayout(self.layout)

        self.button_widget.setFixedHeight(60)
        self.setWindowTitle('智能白酒分析鉴定系统')
        self.resize(1300, 800)
        # self.move(200, 200)
        self.show()

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
                        self.show_message("已" + self.import_method)
                        self.data_linked = False
                        self.label_data = self.input_data
                        self.show_data(self.label_data)
                    elif self.import_method == self.import_methods[1]:
                        if '感官鉴定' in self.input_data.columns:
                            self.show_message("已" + self.import_method)
                            self.data_linked = False
                            self.all_train_data = self.input_data
                            self.show_data(self.all_train_data)
                        else:
                            self.show_message("请导入包含感官鉴定的白酒信息")
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
        self.all_test_data = pd.read_excel("wine_t.xlsx")
        self.data_linked = False
        # self.check_data(self.all_data)
        # self.show_data(self.all_data)
        #
        if len(self.all_train_data.index) == 0:
            self.show_message("请" + self.import_methods[0])
        elif len(self.label_data.index) == 0:
            self.show_message("请" + self.import_methods[1])
        elif len(self.all_test_data.index) == 0:
            self.show_message("请" + self.import_methods[2])
        elif self.data_linked:
            self.show_message("数据已链接")
            pass
        else:
            for i in range(1, self.label_data.shape[1]):
                col = self.label_data.columns.tolist()[i]
                # all_data
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
            if len(self.train_data.index) == 0:
                self.show_message(self.messages[2])
            elif not self.data_linked:
                self.show_message(self.messages[2])
            else:
                self.show_data(self.train_data)
        elif self.show_data_method == self.show_data_methods[4]:
            if len(self.test_data.index) == 0:
                self.show_message(self.messages[2])
            elif not self.data_linked:
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
            self.table.setVisible(True)
            try:
                self.figure_layout.removeWidget(self.canvas)
            except BaseException:
                pass
            else:
                self.figure_layout.addWidget(self.table)
                self.ui_state = 1

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
            print(data)
            if self.figure_type == self.figure_types[0]:
                data.plot(kind='line', ax=self.ax, rot=90, xticks=range(len(data.index)), fontsize=10)
            elif self.figure_type == self.figure_types[1]:
                data.plot(kind='bar', ax=self.ax, fontsize=10)
            elif self.figure_type == self.figure_types[2]:
                data.plot(kind='area', ax=self.ax, rot=90, xticks=range(len(data.index)), fontsize=10)

            # plt.xlabel(data.index.tolist())
            plt.legend(label)

            self.table.setVisible(False)
            self.canvas.setVisible(True)
            self.figure_layout.removeWidget(self.table)
            self.figure_layout.addWidget(self.canvas)
            self.canvas.draw()
            self.ui_state = 2

    def start_training(self):
        pass

    def classify_func(self):
        pass

    def change_type(self, text):
        print("changing type")
        if self.sender() == self.import_method_cb:
            self.import_method = text
        elif self.sender() == self.show_data_method_cb:
            self.show_data_method = text
        elif self.sender() == self.figure_type_cb:
            self.figure_type = text
        elif self.sender() == self.classify_type_cb:
            self.classify_type = text
        else:
            pass

    def save_data(self):
        if self.figure_state == 0:
            self.show_message("没有数据可以保存")
        else:
            file_name = QFileDialog.getSaveFileName(self, '保存文件')[0]
            try:
                if self.figure_state == 1:
                    self.all_train_data.to_excel(file_name)
                elif self.figure_state == 2:
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
        self.ui_state = 0
        self.table.setVisible(False)
        self.canvas.setVisible(False)
        plt.cla()
        # plt.clf()
        # plt.close(self.figure)
        # self.figure = plt.figure()
        # self.ax = self.figure.add_subplot()
        # self.canvas = FigureCanvas(self.figure)

    def exit_func(self):
        print("exited")
        sys.exit(app.exec_())

    def show_message(self, text):
        self.message_box.setText(text)
        self.message_box.show()



    def cur_slice(self):
        pass

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

    def figure_func_old(self):
        if len(self.all_train_data.index) == 0:
            self.show_message('请导入采样数据')
        elif len(self.info_data.index) == 0:
            self.show_message('请导入采样信息')
        elif not self.data_linked:
            self.show_message("请点击链接")
        elif not self.figure_able:
            self.show_message("数据包含非数值类型，不可画图！")
        else:
            self.clear_func()
            self.cur_slice()
            # 解决无法显示中文
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
            # plt.rcParams['font.sans-serif']=['SimHei'] #指定默认字体,SimHei为黑体
            # 解决无法显示负号
            plt.rcParams['axes.unicode_minus'] = False
            plt.title(self.figure_type)
            if self.figure_type == "主成分分析":
                region_data = self.all_test_data.iloc[:, 0].values.tolist()
                print(region_data)
                regions = list(set(region_data))
                print(regions)
                region_color = [(int(regions.index(i) * 255 / len(regions))) for i in region_data]
                # region_color = [regions.index[i] for i in region_data]
                print(region_color)
                data = self.all_test_data.iloc[:, 1:].values
                data = data - np.mean(data, axis=0)
                print("data",data.shape)
                cov_mat = np.cov(data, rowvar=0)
                print("cov:", cov_mat.shape)

                eig_vals, eig_vects = np.linalg.eig(np.mat(cov_mat))
                low_data_mat = data * eig_vects
                print("low:", low_data_mat.shape)
                eig_val_indice = np.argsort(eig_vals)

                top = 2
                n_eig_val_indice = range(top)
                print("n_eig_val_indice", n_eig_val_indice)
                n_eig_vects = eig_vects[:, n_eig_val_indice]
                print("n_eig:",n_eig_vects.shape)
                recon_mat = (low_data_mat * eig_vects) + np.mean(data, axis=0)
                print("rec:", recon_mat.shape)
                x = np.array(low_data_mat)[:, 0]
                y = np.array(low_data_mat)[:, 1]
                # z = np.array(low_data_mat)[:, 2]
                for region in regions:
                    index = [i for i, data in enumerate(region_data) if data == region]
                    plt.scatter(x[index], y[index])
                plt.legend(regions)
            elif self.figure_type == '平行坐标图':
                parallel_coordinates(self.all_test_data, self.region_method)
            elif self.figure_type == "Andrews图":
                colors = ['b', 'g', 'r', 'orange']
                andrews_curves(self.all_test_data, self.region_method, color=colors)
            elif self.figure_type == 'Radiv图':
                radviz(self.all_test_data, self.region_method)
            elif self.figure_type == '矩阵散点图':
                print("绘制矩阵散点图")
                sns.pairplot(data=self.all_test_data, hue=self.region_method)
                f = plt.gcf()
                self.ax = f
                self.canvas = FigureCanvas(f)
            elif self.figure_type == 'Chernoff脸谱图':
                self.all_test_data.to_excel('cur_data.xlsx')
                print("data out")
                # goto_r()
                os.system("python ./PyToR.py")
                face_info = pd.read_csv('face_info.csv')
                # f_str = face_info.to_string()

                font = {'weight': 'normal',
                         'size': 11,
                         }

                plt.text(500, 0 , "脸谱图条目                 数据列", fontdict=font)
                for index, row in face_info.iterrows():
                    f_str = row[0] + " : "
                    plt.text(500, 20 + 20 * index, f_str, fontdict=font)
                    f_str = row[1]
                    plt.text(650, 30 + 20 * index, f_str, fontdict=font)
                plt.imshow(Image.open('face.png'))
                plt.gca().add_patch(plt.Rectangle(xy=(500, 20), width=100, height=300,
                                                  edgecolor=[1, 1, 1],
                                                  fill=False,
                                                  linewidth=2))
                # print("文件命名为:face.jpg")
                # info=pd.read_csv('face_info.csv',encoding='GBK')
                # print("effect of variables:\n{}".format(info))

            self.table.setVisible(False)
            self.canvas.setVisible(True)
            self.figure_layout.removeWidget(self.table)
            self.figure_layout.addWidget(self.canvas)
            self.canvas.draw()
            self.figure_state = 2

    def pca(self):
        train_label = self.train_data.iloc[:, 0].values.tolist()
        print(train_label)
        # region_color = [(int(regions.index(i) * 255 / len(regions))) for i in region_data]
        # region_color = [regions.index[i] for i in region_data]
        # print(region_color)
        data = self.train_data.iloc[:, 1:]
        data = data - np.mean(data, axis=0)
        print("data",data.shape)
        cov_mat = np.cov(data, rowvar=0)
        print("cov:", cov_mat.shape)

        eig_vals, eig_vects = np.linalg.eig(np.mat(cov_mat))
        low_data_mat = data * eig_vects
        print("low:", low_data_mat.shape)
        eig_val_indice = np.argsort(eig_vals)

        top = 2
        n_eig_val_indice = range(top)
        print("n_eig_val_indice", n_eig_val_indice)
        n_eig_vects = eig_vects[:, n_eig_val_indice]
        print("n_eig:",n_eig_vects.shape)
        recon_mat = (low_data_mat * eig_vects) + np.mean(data, axis=0)
        print("rec:", recon_mat.shape)
        x = np.array(low_data_mat)[:, 0]
        y = np.array(low_data_mat)[:, 1]
        # z = np.array(low_data_mat)[:, 2]
        for region in regions:
            index = [i for i, data in enumerate(region_data) if data == region]
            plt.scatter(x[index], y[index])
        plt.legend(regions)


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

def goto_r():
    r_script = '''
                library(aplpack) 
                library(openxlsx)
                cur_data<-read.xlsx("cur_data.xlsx", sheet = 1)
                print("read sucess")
                # 保存图片
                png('face.png')
                col_num=ncol(cur_data)
                face_model=faces(cur_data[,3:col_num],labels=cur_data[[2]])
                dev.off()
                # 导出数据
                # cur_data=cur_data.frame(face_model$info)
                # write.csv(info_data,'face_info.csv',row.names=F,col.names=F,sep='')
               '''
    robjects.r(r_script)


if  __name__ == '__main__':
    app = QApplication(sys.argv)
    # app.setStyleSheet(load_stylesheet_pyqt5())
    DV = WineClassify()
    sys.exit(app.exec_())
