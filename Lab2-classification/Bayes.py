import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings

warnings.filterwarnings("ignore")  # 消除警告


# 高斯朴素贝叶斯，主要用于连续数据
class GaussianNaiveBayes:
    def fit(self, X, y):
        # 计算每个类别的先验概率、均值和方差
        self.classes = np.unique(y)
        self.parameters = {}
        for c in self.classes:
            X_c = X[y == c]
            self.parameters[c] = {
                "mean": X_c.mean(axis=0),
                "var": X_c.var(axis=0),
                "prior": X_c.shape[0] / X.shape[0],  # 先验概率
            }

    def calculate_likelihood(self, mean, var, x):
        # 计算高斯分布的似然概率
        exponent = np.exp(-((x - mean) ** 2 / (2 * var)))
        return (1 / np.sqrt(2 * np.pi * var)) * exponent

    def predict(self, X, y):
        # 预测给定数据的类别
        y_pred = [self.classify(sample) for sample in X]
        acc = np.sum(y == y_pred) / len(y)
        print(acc)
        return np.array(y_pred)

    def classify(self, sample):
        # 计算每个类别的后验概率并选择最高的
        posteriors = []
        for c in self.classes:
            # 由于概率值很小，直接相乘可能导致数值下溢。因此，在实际的算法实现中
            # 通常使用对数来转换乘法为加法，从而避免这个问题。
            prior = np.log(self.parameters[c]["prior"])
            class_likelihood = np.sum(
                np.log(
                    self.calculate_likelihood(
                        self.parameters[c]["mean"], self.parameters[c]["var"], sample
                    )
                )
            )
            posterior = prior + class_likelihood
            posteriors.append(posterior)
        # print(posteriors)
        return self.classes[np.argmax(posteriors)]


# 多项式朴素贝叶斯，主要用于离散数据
class MultinomialNaiveBayes:
    def fit(self, X, y):
        # 计算每个类别的先验概率和特征计数
        self.classes = np.unique(y)
        self.parameters = {}
        for c in self.classes:
            X_c = X[y == c]
            total_count = X_c.sum()
            self.parameters[c] = {
                "class_count": X_c.sum(axis=0),
                "total_count": total_count,
                "prior": X_c.shape[0] / X.shape[0],
            }

    def calculate_likelihood(self, class_count, total_count, x_i):
        # 计算多项分布的似然概率,使用拉普拉斯平滑
        return (class_count + 1) / (total_count + len(class_count))

    def predict(self, X, y):
        # 预测给定数据的类别
        y_pred = [self.classify(sample) for sample in X]
        acc = np.sum(y == y_pred) / len(y)
        # print(y_pred)
        print(acc)
        return np.array(y_pred)

    def classify(self, sample):
        # 计算每个类别的后验概率并选择最高的
        posteriors = []
        for c in self.classes:
            prior = np.log(self.parameters[c]["prior"])
            likelihood = np.sum(
                np.log(
                    self.calculate_likelihood(
                        self.parameters[c]["class_count"],
                        self.parameters[c]["total_count"],
                        sample,
                    )
                ).dot(
                    sample
                )  # 后验概率还要点乘sample
            )
            posterior = prior + likelihood
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]


def targetAndtargetNames(Dataframe, targetColumnIndex):
    target_dict = dict()
    target = list()
    count = -1
    for i in range(len(Dataframe.values)):
        if Dataframe.values[i][targetColumnIndex] not in target_dict:
            count += 1
            target_dict[Dataframe.values[i][targetColumnIndex]] = count
        target.append(target_dict[Dataframe.values[i][targetColumnIndex]])
    return np.asarray(target)


def show_plt(y_pred, y_true, model=None):
    # 创建混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

    # 打印分类报告
    print(classification_report(y_true, y_pred))

    if model != None:
        # 使用条形图来展示每个类别的准确率
        accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        plt.bar(range(len(model.classes)), accuracies)
        plt.xlabel("Class")
        plt.ylabel("Accuracy")
        plt.xticks(ticks=np.arange(len(model.classes)), labels=model.classes)
        plt.title("Per-Class Accuracy")
        plt.show()


if __name__ == "__main__":
    df = pd.read_csv("buddymove_holidayiq.csv", delimiter=",")
    y = pd.DataFrame(targetAndtargetNames(df, 1))

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(["User Id", "Sports"], axis=1),
        y[0],
        test_size=0.3,
        random_state=7,
    )
    # print(len(set(df["Sports"].values)))

    # print(y_train.values)
    print("---手写高斯朴素贝叶斯---")
    my_gnb = GaussianNaiveBayes()
    my_gnb.fit(X_train.values, y_train.values)
    predict = my_gnb.predict(X_test.values, y_test.values)
    show_plt(predict, y_test.values, my_gnb)

    print("---手写多项式朴素贝叶斯---")
    my_mnb = MultinomialNaiveBayes()
    my_mnb.fit(X_train.values, y_train.values)
    predict = my_mnb.predict(X_test.values, y_test.values)
    show_plt(predict, y_test.values, my_mnb)

    print("---sklearn库高斯朴素贝叶斯---")
    sk_gnb = GaussianNB()
    sk_gnb.fit(X_train.values, y_train.values)
    predict = sk_gnb.predict(X_test.values)
    acc = accuracy_score(y_test, predict)
    print(acc)
    show_plt(predict, y_test.values)

    print("---sklearn库多项式朴素贝叶斯---")
    sk_mnb = MultinomialNB()
    sk_mnb.fit(X_train.values, y_train.values)
    predict = sk_mnb.predict(X_test.values)
    # print(predict)
    acc = accuracy_score(y_test, predict)
    print(acc)
    show_plt(predict, y_test.values)
