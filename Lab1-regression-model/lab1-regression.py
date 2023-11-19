import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class LinearRegression(object):
    def __init__(self):
        self.w = 0  # 斜率
        self.b = 0  # 截距
        self.sqrLoss = 0  # 最小均方误差
        self.trainSet = 0  # 训练集特征
        self.label = 0  # 训练集标签
        self.iters = 0
        self.learning_rate = 0
        self.loss_list = []

    def train(
        self, x, y, learning_rate=0.1, iters=5000, min_loss=1e-5, method="matrix_sol"
    ):
        self.trainSet = x
        self.label = y
        if method == "formula_sol":
            self.formula_sol()
        elif method == "matrix_sol":
            self.matrix_sol()
        elif method == "gradient_descent":
            self.gradient_descent(learning_rate, iters, min_loss)
        else:
            print("method错误")

    def formula_sol(self):
        sample_num = self.trainSet.shape[0]  # 样本数量
        x = self.trainSet.flatten()  # 化为一维数组
        y = self.label  # 标签
        xmean = np.mean(x)  # x平均
        ymean = np.mean(y)  # y平均
        # 求w公式,x点乘y
        self.w = (np.dot(x, y) - sample_num * xmean * ymean) / (
            np.power(x, 2).sum() - sample_num * xmean**2
        )
        self.b = ymean - self.w * xmean
        # 求损失
        self.sqrLoss = np.power((y - np.dot(x, self.w) - self.b), 2).sum()
        return

    def matrix_sol(self):
        sample_num = self.trainSet.shape[0]  # 样本数量
        x = np.hstack(
            (self.trainSet, np.ones((sample_num, 1)))
        )  # 水平方向拼接数组，在x最后添加一列为1的特征
        y = self.label
        xTx = np.linalg.inv(np.dot(x.T, x))  # (xT*x)^-1
        what = np.dot(np.dot(xTx, x.T), y)  #  (xT*x)^-1*xT*y
        self.w = what[:-1]
        self.b = what[-1]
        self.sqrLoss = np.power((y - np.dot(x, what).flatten()), 2).sum()  # 损失值
        return

    def gradient_descent(self, learning_rate, iters, min_loss):
        sample_num, feature_num = self.trainSet.shape  # 样本数量
        x = self.trainSet
        y = self.label
        n = 0
        # 初始化w和b为1
        w = np.ones(feature_num)
        b = 1
        lst_loss = np.power((y - np.dot(x, w).flatten() - b), 2).sum()  # 上一次损失
        delta_loss = np.inf  # 损失变化
        while n < iters and lst_loss > min_loss and abs(delta_loss) > min_loss:
            y_pred = np.dot(x, w) + b  # 预测值
            # 求梯度
            w_gradient = np.dot((y_pred - y), x) / sample_num
            b_gradient = sum(y_pred - y) / sample_num
            # 更新
            w = w - learning_rate * w_gradient
            b = b - learning_rate * b_gradient
            cur_loss = np.power((y - np.dot(x, w).flatten() - b), 2).sum()  # 当前损失
            delta_loss = lst_loss - cur_loss
            lst_loss = cur_loss
            self.loss_list.append(cur_loss)
            n += 1
            print(f"第{n}次迭代，损失平方和为{cur_loss}，损失前后差为{delta_loss}")
        print("训练完成")
        self.w = w
        self.b = b
        self.sqrLoss = lst_loss
        self.learning_rate = learning_rate
        self.iters = n


# 单元测试
def test(w, b, size=100):
    # 随机数据
    x = np.expand_dims(np.linspace(-10, 10, size), axis=1)
    y = x.flatten() * w + b + (np.random.random(size) - 1) * 3
    # 求解
    lr1 = LinearRegression()
    lr1.train(x, y)
    print(f"矩阵方法：\nw:{lr1.w}, b:{lr1.b}, square loss:{lr1.sqrLoss}")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x, y)
    ax.plot(x, x * lr1.w + lr1.b, color="r", linewidth=3)
    plt.show()


# 多元线性回归测试
def multitest():
    from sklearn.datasets import load_digits, load_iris

    df = pd.read_csv("abalone_train.csv", sep=",")
    # x = load_iris().data
    # y = load_iris().target
    # 将特征x标准化，方便收敛
    # x = (x - x.mean(axis=0)) / x.std(axis=0)

    train_x = df.values[:, 1:]
    train_y = df.values[:, 0:1].flatten()
    df = pd.read_csv("abalone_test.csv", sep=",")
    test_x = df.values[:, 1:]
    test_y = df.values[:, 0:1].flatten()

    # 矩阵法求解
    # lr2 = LinearRegression()
    # lr2.train(
    #     train_x,
    #     train_y,
    #     method="matrix_sol",
    # )
    # print(f"矩阵法：\nw:{lr2.w}, b:{ lr2.b}, square loss:{lr2.sqrLoss}")

    # 梯度下降法求解
    lr2 = LinearRegression()
    lr2.train(
        train_x,
        train_y,
        learning_rate=0.1,
        iters=5000,
        min_loss=1e-7,
        method="gradient_descent",
    )
    print(f"梯度下降法：\nw:{lr2.w}, b:{ lr2.b}, square loss:{lr2.sqrLoss}")
    # 画梯度下降的误差下降图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(lr2.iters), lr2.loss_list, linewidth=3)
    ax.set_title("Square Loss")
    plt.show()

    test_num = test_x.shape[0]
    w = lr2.w
    b = lr2.b
    tol = 0
    for _ in range(0, test_num):
        pred = w.dot(test_x[_]) + b
        label = test_y[_]
        devi = abs(pred - test_y[_]) / label
        print(f"第{_+1}次测试：predict={pred}，label={label}，百分比相对误差为：{devi}")

        tol += devi

    print(f"相对误差为{tol/test_num}")
    return


if __name__ == "__main__":
    # test(3, -2)
    multitest()
