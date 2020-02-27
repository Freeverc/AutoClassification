import numpy as np
import pandas as pd


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


class NeuralNetwork:
    def __init__(self, layers, activation='logistic'):
        """
        神经网络算法构造函数
        :param layers: 神经元层数
        :param activation: 使用的函数（默认tanh函数）
        :return:none
        """
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
        print(np.shape(self.weights[0]))
        print(np.shape(self.weights[1]))

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
        # print(X)
        # print(y)

        for k in range(epochs):
            # 随机抽取X的一行
            i = np.random.randint(X.shape[0])
            # 用随机抽取的这一组数据对神经网络更新
            a = [X[i]]
            # 正向更新
            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l], self.weights[l])))
            print(np.shape(a[0]), np.shape(a[1]), np.shape(a[2]))
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
            # print("training :", k)

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a


from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report


all_train_data = pd.read_excel("train_data.xlsx")

train_label = all_train_data.iloc[:, 3].values.tolist()
train_data = all_train_data.iloc[:, 4:].values
train_data = train_data - np.mean(train_data, axis=0)

print(np.shape(train_data))
print(np.shape(train_label))
print(train_label)
train_label_fit = LabelBinarizer().fit_transform(train_label)
layers = [30, 50, 4]
nn = NeuralNetwork(layers, 'logistic')
nn.fit(train_data, train_label_fit)


predictions = []
for i in range(np.shape(train_data)[0]):
    o = nn.predict(train_data[i])
    # np.argmax:第几个数对应最大概率值
    predictions.append(np.argmax(o))

print(np.shape(predictions))
print(np.shape(train_label))
# 打印预测相关信息
print(confusion_matrix(train_label, predictions))
print(classification_report(train_label, predictions))

# nn = NeuralNetwork([2, 2, 1], 'tanh')
# temp = [[0, 0], [0, 1], [1, 0], [1, 1]]
# X = np.array(temp)
# y = np.array([0, 1, 1, 0])
# nn.fit(X, y)
# for i in temp:
#     print(i, nn.predict(i))