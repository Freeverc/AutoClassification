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


class DataVisual(QWidget):
    def __init__(self):
        super().__init__()
        self.button_widget = QWidget()
        self.figure_widget = QWidget()
        self.message_box = QMessageBox()
        # self.message_box.move(600, 360)
        self.table_view = QTableView()
        self.figure_state = 0
        self.figure_able = 0
        self.init_ui()

    def init_ui(self):

        self.info_data = pd.DataFrame()
        self.input_data = pd.DataFrame()
        self.all_data = pd.DataFrame()
        self.cur_data = pd.DataFrame()
        # self.file_name = ''
        self.show_data_method = '显示全部采样数据'
        self.show_data_methods = ['显示全部采样数据', '显示采样信息', '显示可视化数据']

        self.region_linked = False
        self.region_method = '采样省'
        self.region_methods = ['采样省', '采样市县', '采样区县']
        self.region_method_cb_list = ['按省显示', '按市显示', '按区县显示']

        self.figure_type = '平行坐标图'
        self.figure_types = ['平行坐标图', '矩阵散点图', '主成分分析', 'Chernoff脸谱图', 'Andrews图', 'Radiv图']

        self.row_name_list = ['第一行', '第二行', '第三列', '第四列']
        self.col_name_list = ['第一列', '第二列', '第三列', '第四列']
        self.cur_rows = []
        self.cur_cols = []

        self.import_btn = QPushButton('导入文件')
        self.import_btn.setToolTip('导入高纬度数据文件')
        self.import_btn.clicked.connect(self.import_data)

        self.link_btn = QPushButton('链接文件')
        self.link_btn.setToolTip('链接导入的文件和地区')
        self.link_btn.clicked.connect(self.link_data)

        self.run_btn = QPushButton('画图')
        self.run_btn.setToolTip('绘制可视化图像')
        self.run_btn.clicked.connect(self.draw_func)

        self.save_btn = QPushButton('保存')
        self.save_btn.setToolTip('保存图像或数据')
        self.save_btn.clicked.connect(self.save_data)

        self.clear_btn = QPushButton('清空')
        self.clear_btn.setToolTip('清空图像和数据')
        self.clear_btn.clicked.connect(self.clear_func)

        self.exit_btn = QPushButton('退出')
        self.exit_btn.setToolTip('退出程序')
        self.exit_btn.clicked.connect(self.exit_func)

        self.show_data_method_cb = QComboBox()
        self.show_data_method_cb.setToolTip('选择要查看的数据')
        self.show_data_method_cb.addItems(self.show_data_methods)
        self.show_data_method_cb.activated[str].connect(self.show_data_slot)

        self.region_method_cb = QComboBox()
        self.region_method_cb.setToolTip('选择按区域查看方式')
        self.region_method_cb.addItems(self.region_method_cb_list)
        self.region_method_cb.activated[str].connect(self.change_region)

        self.figure_type_cb = QComboBox()
        self.show_data_method_cb.setToolTip('选择可视化绘图方法')
        self.figure_type_cb.addItems(self.figure_types)
        self.figure_type_cb.activated[str].connect(self.change_type)

        self.row_box = ComboCheckBox(self.row_name_list)
        self.col_box = ComboCheckBox(self.col_name_list)
        self.row_box.setEditText("选择行")
        self.col_box.setEditText("选择列")

        self.button_layout = QHBoxLayout()
        self.button_layout.addWidget(self.import_btn)
        self.button_layout.addWidget(self.link_btn)
        self.button_layout.addWidget(self.show_data_method_cb)
        self.button_layout.addWidget(self.run_btn)
        self.button_layout.addWidget(self.region_method_cb)
        self.button_layout.addWidget(self.figure_type_cb)
        self.button_layout.addWidget(self.row_box)
        self.button_layout.addWidget(self.col_box)
        self.button_layout.addWidget(self.clear_btn)
        self.button_layout.addWidget(self.save_btn)
        self.button_layout.addWidget(self.exit_btn)
        self.button_widget.setLayout(self.button_layout)

        self.figure = plt.figure()
        self.ax = self.figure.add_subplot()
        self.canvas = FigureCanvas(self.figure)

        self.figure_layout = QHBoxLayout()
        self.figure_layout.addWidget(self.table_view)
        self.figure_widget.setLayout(self.figure_layout)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.button_widget)
        self.layout.addWidget(self.figure_widget)
        self.setLayout(self.layout)

        self.button_widget.setFixedHeight(60)
        self.setWindowTitle('高纬数据可视化系统')
        self.resize(1300, 800)
        # self.move(200, 200)
        self.show()

    def import_data(self):
        print("importing data")
        file_name = QFileDialog.getOpenFileName(self, '打开文件')[0]
        if not os.path.splitext(file_name)[1] in ['.xls', '.xlsx']:
            self.show_message("请导入Excel文件")
        else:
            try:
                self.input_data = pd.read_excel(file_name)
            except BaseException:
                self.show_message("数据读取失败")
            else:
                if not ('ID' in self.input_data.columns):
                    self.show_message("请导入包含ID列的有效数据")
                else:
                    self.region_linked = False
                    if ('采样省' in self.input_data.columns) \
                            or ('采样市县' in self.input_data.columns)\
                            or ('采样区县' in self.input_data.columns):
                        self.info_data = self.input_data
                        self.show_data(self.info_data)
                    else:
                        self.all_data = self.input_data
                        self.check_data(self.all_data)
                        self.show_data(self.all_data)

                        self.row_name_list = self.all_data['ID'].values.tolist()
                        self.col_name_list = self.all_data.columns.tolist()
                        self.col_name_list.remove('ID')
                        self.row_box.setItems(self.row_name_list)
                        self.col_box.setItems(self.col_name_list)
                        self.row_box.setEditText("选择行")
                        self.col_box.setEditText("选择列")


    def link_data(self):
        print("linking data")
        # self.info_data = pd.read_excel("infor.xlsx")
        # self.all_data = pd.read_excel("zb2.xlsx")
        # self.check_data(self.all_data)
        # self.show_data(self.all_data)
        #
        # self.row_name_list = self.all_data['ID'].values.tolist()
        # self.col_name_list = self.all_data.columns.tolist()
        # self.col_name_list.remove('ID')
        # self.row_box.setItems(self.row_name_list)
        # self.col_box.setItems(self.col_name_list)
        # self.row_box.setEditText("选择行")
        # self.col_box.setEditText("选择列")

        if len(self.info_data.index) == 0:
            self.show_message("请导入采样信息")
        elif len(self.all_data.index) == 0:
            self.show_message("请导入采样数据")
        elif self.region_linked:
            self.show_message("数据已链接")
            pass
        else:
            for i in range(1, 4):
                col = self.info_data.columns.tolist()[i]
                if col in self.all_data.columns:
                    self.all_data.drop(col, axis=1, inplace=True)
                values = []
                for index, row in self.all_data.iterrows():
                    # print(self.info_data.ID.tolist())
                    # print(self.info_data.ID.values.tolist())
                    # print(type(row['ID']))
                    # print(row['ID'])
                    if row['ID'] in self.info_data.ID.values.tolist():
                        # print("in")
                        v = self.info_data.loc[self.info_data.ID == row['ID'], col].tolist()[0]
                    else:
                        # print("not in ")
                        v = "unknown"
                    values.append(v)
                self.all_data.insert(i, self.info_data.columns[i], values)
                self.show_data(self.all_data)
                self.region_linked = True
                self.show_message("数据链接成功")


    def show_data_slot(self, text):
        if text == '显示全部采样数据':
            if len(self.all_data.index) == 0:
                self.show_message('请导入采样数据')
            else:
                self.show_data(self.all_data)
        elif text == '显示采样信息':
            if len(self.info_data.index) == 0:
                self.show_message('请导入采样信息')
            else:
                self.show_data(self.info_data)
        elif text == '显示可视化数据':
            if len(self.all_data.index) == 0:
                self.show_message('请导入采样数据')
            elif len(self.info_data.index) == 0:
                self.show_message('请导入采样信息')
            elif not self.region_linked:
                self.show_message("请点击链接")
            else:
                self.cur_slice()
                if len(self.cur_data.index) == 0:
                    self.show_message('选择的数据不能为空！')
                else:
                    self.show_data(self.cur_data)
        else:
            self.show_message('显示数据错误')

    def show_data(self, data):
        print("showing data")
        if len(data.index) == 0:
            self.show_message("请先导入要显示的数据")
        else:
            model = QtTable(data)
            self.table_view.setModel(model)
            self.canvas.setVisible(False)
            self.table_view.setVisible(True)
            try:
                self.figure_layout.removeWidget(self.canvas)
            except BaseException:
                pass
            else:
                self.figure_layout.addWidget(self.table_view)
                self.figure_state = 1

    def change_region(self, text):
        print("changing region")
        i = self.region_method_cb_list.index(text)
        self.region_method = self.region_methods[i]

    def change_type(self, text):
        print("changing figure type")
        self.figure_type = text

    def save_data(self):
        if self.figure_state == 0:
            self.show_message("没有数据可以保存")
        else:
            file_name = QFileDialog.getSaveFileName(self, '保存文件')[0]
            try:
                if self.figure_state == 1:
                    self.all_data.to_excel(file_name)
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
        self.figure_state = 0
        self.table_view.setVisible(False)
        self.canvas.setVisible(False)
        plt.cla()
        plt.clf()
        plt.close(self.figure)
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot()
        self.canvas = FigureCanvas(self.figure)

    def exit_func(self):
        print("exited")
        sys.exit(app.exec_())

    def show_message(self, text):
        self.message_box.setText(text)
        self.message_box.show()

    def cur_slice(self):
        print("slicing")
        if len(self.all_data.index) == 0:
            self.show_message('请导入采样数据')
        elif len(self.info_data.index) == 0:
            self.show_message('请导入采样信息')
        elif not self.region_linked:
            self.show_message("请点击链接")
        else:
            self.cur_rows = self.row_box.Selectlist()
            self.cur_cols = self.col_box.Selectlist()
            self.cur_cols.insert(0, self.region_method)
            # print(self.cur_rows)
            # print(self.cur_cols)
            self.cur_data = self.all_data.loc[self.all_data['ID'].isin(self.cur_rows), self.cur_cols]
            print(self.cur_data.shape)

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

    def draw_func(self):
        if len(self.all_data.index) == 0:
            self.show_message('请导入采样数据')
        elif len(self.info_data.index) == 0:
            self.show_message('请导入采样信息')
        elif not self.region_linked:
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
                region_data = self.cur_data.iloc[:, 0].values.tolist()
                print(region_data)
                regions = list(set(region_data))
                print(regions)
                region_color = [(int(regions.index(i) * 255 / len(regions))) for i in region_data]
                # region_color = [regions.index[i] for i in region_data]
                print(region_color)
                data = self.cur_data.iloc[:, 1:].values
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
                parallel_coordinates(self.cur_data, self.region_method)
            elif self.figure_type == "Andrews图":
                colors = ['b', 'g', 'r', 'orange']
                andrews_curves(self.cur_data, self.region_method, color=colors)
            elif self.figure_type == 'Radiv图':
                radviz(self.cur_data, self.region_method)
            elif self.figure_type == '矩阵散点图':
                print("绘制矩阵散点图")
                sns.pairplot(data=self.cur_data, hue=self.region_method)
                f = plt.gcf()
                self.ax = f
                self.canvas = FigureCanvas(f)
            elif self.figure_type == 'Chernoff脸谱图':
                self.cur_data.to_excel('cur_data.xlsx')
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

            self.table_view.setVisible(False)
            self.canvas.setVisible(True)
            self.figure_layout.removeWidget(self.table_view)
            self.figure_layout.addWidget(self.canvas)
            self.canvas.draw()
            self.figure_state = 2


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

#
# def PCA(dataMat, top):
#     # 数据中心化
#     meanVal = np.mean(dataMat, axis=0)
#     newData = dataMat - meanVal
#     print(np.shape(newData))
#     covMat = np.cov(newData, rowvar=0)
#     print(np.shape(covMat))
#     eigVals, eigVects = np.linalg.eig(np.mat(covMat))
#     eigValIndice = np.argsort(eigVals)
#     n_eigValIndice = eigValIndice[-1:-(top + 1):-1]
#     n_eigVects = eigVects[:, n_eigValIndice]
#     lowDataMata = newData * n_eigVects
#     reconMat = (lowDataMata * n_eigVects.T) + meanVal
#     return lowDataMata, reconMat
#

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
    DV = DataVisual()
    sys.exit(app.exec_())
