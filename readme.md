###
功能说明
______
本软件是一个简单的数据可视化与自动分类软件，源自一个基酒等级分类的项目。  
其中自行实现了PCA、BP等算法，也从sklearn引入了SparcePCA、SVM等一些算法作为比较，界面支持一些基本的参数设置。  
基于pyQt5开发界面，实现图像、表格数据的显示与简单的页面切换，支持白色、浅色、深色三种主题设置。  
结合Matplotlib做数据可视化，并保证主题色一致，对强迫症较友好。  
此软件适用于初学QT开发、数据可视化、机器学习的伙伴用于学习和玩耍。  
###
依赖
sklearn
pyQt5
numpy

###
使用说明
______
#####
**1. 导入数据**

手动选择导入的数据类型，['导入标签信息', '导入训练数据', '导入测试数据']

软件会自动判断数据类型是否合法：要求导入Excel类型数据，所有数据必须包含'编号'列，白酒信息类别的数据应当包含“感官鉴定”条目

数据格式模板参考文件夹中的文件格式
![](https://github.com/Freeverc/AutoClassification/blob/master/images/data_table.png)

**2. 链接文件**

将白酒信息数据加到白酒训练数据上生成全部训练数据和全部测试数据，并同时整理生成训练数据和测试数据

**3. 查看数据**

可以手动选择要查看的数据类型：['显示白酒信息', '显示全部训练数据', '显示全部测试数据', '显示训练数据', '显示测试数据']


**4. 绘制图像**

如果导入的数据为纯数字类型，则可以画可视化图表,画图方式有三种
1. 折线图
2. 条形图
3. 面积图
![](https://github.com/Freeverc/AutoClassification/blob/master/images/data_line.png)
![](https://github.com/Freeverc/AutoClassification/blob/master/images/data_area.png)

**5. 训练模型**

用训练数据集训练分类模型，支持以下机器学习分类方法
+ KNN
+ PCA-KNN
+ SPCA-KNN
+ SVM
+ PCA-SVM
+ SPCA-SVM
+ BP
+ PCA-BP
+ SPCA-BP
+ Decision Tree
+ PCA-Decision Tree
+ SPCA-Decision Tree


**6. 测试模型**

支持任意数量测试数据, 在测试数据中增加分类结果和打分结果。
![](https://github.com/Freeverc/AutoClassification/blob/master/images/white.png)

以散点图可视化显示分类结果，以折线图统计分类准确率，并呈现
![](https://github.com/Freeverc/AutoClassification/blob/master/images/pca_dt_zhe.png)
![](https://github.com/Freeverc/AutoClassification/blob/master/images/pca_dt_san.png)

**7. 设置**

|设置内容| 可选范围 |
|---------- |-------|
|设置主题 |白色、浅色和深色 |
|设置PCA算法主成分数量 | 1-15 |
|设置KNN算法最近邻居数量 | 1-10 |
|设置svm算法多分类模型类型 | 'ovo', 'ovo' |
|设置BP算法学习率 | 0.01-0.5 |
|设置BP算法迭代周期数 | 1000-100000 |
|设置BP算法隐藏层个数| 10-1000 |
|设置Decision Tree算法最大深度| 10-100 |

![](https://github.com/Freeverc/AutoClassification/blob/master/images/dark.png)
![](https://github.com/Freeverc/AutoClassification/blob/master/images/light.png)

**8. 保存**

可以对画的图和表格进行保存

**9. 清空**

清空显示内容，但是不会清空变量

**10. 退出**

退出程序


### 附：

基酒鉴评规则:

|基酒等级 |分段 |
|----|----|
|TY级 |93分以上|
|YY级|88 - 92.9分|
|RY级|80 - 87.9分|
|SY级|70 - 79.9分|
