import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import copy

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# torchvision的models中有很多与训练好的模型，如resnet、vgg、alexnet等
data_dir = "./Sort_1000pics-main/Sort_1000pics-main/image_data"
model_name = "resnet"
num_classes = 10
batch_size = 32
num_epochs = 10
input_size = 224
lr = 1e-3
momentum = 0.9
is_fixed = True
use_pretrained = True
is_train = True
is_test = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []


# 验证
def test(model, test_loader, loss_func):
    model.eval()
    loss_val = 0.0
    corrects = 0.0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            loss = loss_func(outputs, labels)

        _, predicts = torch.max(outputs, 1)

        loss_val += loss.item() * images.size(0)
        corrects += torch.sum(predicts.view(-1) == labels.view(-1)).item()

    test_loss = loss_val / len(test_loader.dataset)
    test_acc = corrects / len(test_loader.dataset)

    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    print("Test Loss: {}, Test Acc: {}".format(test_loss, test_acc))

    return test_acc


# 训练
def train(model, train_loader, test_loader, loss_func, optimizer, num_epochs):
    # 初始化最好的验证准确率
    best_val_acc = 0.0
    # 初始化最好的模型参数，采用deepcopy为防止优化过程中修改到best_model_params
    best_model_params = copy.deepcopy(model.state_dict())
    for epoch in range(num_epochs):
        model.train()
        loss_val = 0.0
        corrects = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_func(outputs, labels)

            # 找出输出的最大概率所在的位置
            _, predicts = torch.max(outputs, 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss.item()为一个batch的平均loss的值
            # images.size(0)为当前batch中有多少样本量
            # loss.item() * images.size(0)表示一个batch的总loss值
            loss_val += loss.item() * images.size(0)

            # view(-1)表示将tensor resize成一个维度为[batch_size]的tensor
            # 计算预测值与标签值相同的数量
            corrects += torch.sum(predicts.view(-1) == labels.view(-1)).item()

        # 计算每个epoch的平均loss
        train_loss = loss_val / len(train_loader.dataset)
        # 预测准确的数量除以总的样本量即为准确率
        train_acc = corrects / len(train_loader.dataset)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        print("Train Loss: {}, Train Acc: {}".format(train_loss, train_acc))

        # 调用测试
        test_acc = test(model, test_loader, loss_func)
        # 根据测试准确率跟新最佳模型的参数
        if best_val_acc < test_acc:
            best_val_acc = test_acc
            best_model_params = copy.deepcopy(model.state_dict())
    # 将模型的最优参数载入模型
    try:
        model.load_state_dict(best_model_params)
    except Exception as e:
        print("Error saving the model:", e)
    return model


def set_parameters_require_grad(model, is_fixed):
    # 默认parameter.requires_grad = True
    # 当采用固定预训练模型参数的方法进行训练时，将预训练模型的参数设置成不需要计算梯度
    if is_fixed:
        for parameter in model.parameters():
            parameter.requires_grad = False


def init_model(model_name, num_classes, is_fixed, use_pretrained):
    if model_name == "resnet":
        # 调用resnet模型，resnet18表示18层的resnet模型，
        # pretrained=True表示需要加载预训练好的模型参数，pretrained=False表示不加载预训练好的模型参数
        model = models.resnet18(pretrained=use_pretrained)  # 调用预训练的resnet18模型
        # 设置参数是否需要计算梯度
        # is_fixed=True表示模型参数不需要跟新（不需要计算梯度）
        # is_fixed=False表示模型参数需要fineturn（需要计算梯度）
        set_parameters_require_grad(model, is_fixed)

        in_features = model.fc.in_features  # 取出全连接层的输入特征维度

        # 重新定义resnet18模型的全连接层,使其满足新的分类任务
        # 此时模型的全连接层默认需要计算梯度
        model.fc = nn.Linear(in_features, num_classes)

    return model


# 获取数据，并对数据做预处理
# 该数据集已经被预处理成了可用ImageFolder处理的形式
def get_datasets(data_dir, input_size, is_train_data):
    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    # 读取数据
    dataset = datasets.ImageFolder(data_dir, transform)

    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size

    # 划分数据集
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    if is_train_data:
        return train_dataset
    else:
        return test_dataset


"""
#图片展示
unloader = transforms.ToPILImage()

def image_show(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.figure()
    plt.imshow(image)
"""


# 获取需要更新的模型参数
def get_require_updated_params(model, is_fixed):
    if is_fixed:
        require_update_params = []
        for param in model.parameters():
            if param.requires_grad:
                require_update_params.append(param)
        return require_update_params
    else:
        return model.parameters()


train_images = get_datasets(data_dir, input_size, is_train_data=True)
test_images = get_datasets(data_dir, input_size, is_train_data=False)

train_loader = torch.utils.data.DataLoader(
    train_images, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(test_images, batch_size=batch_size)

model = init_model(model_name, num_classes, is_fixed, use_pretrained)
model = model.to(device)

require_update_params = get_require_updated_params(model, is_fixed)

# 将需要跟新的参数放入优化器中进行优化
optimizer = torch.optim.SGD(require_update_params, lr=lr, momentum=momentum)
# 交叉熵损失函数
loss_func = nn.CrossEntropyLoss()

if is_train:
    model = train(model, train_loader, test_loader, loss_func, optimizer, num_epochs)
    torch.save(model.state_dict(), "resnet.pt")
    # print(train_losses, test_losses)
    # 绘制损失图
    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.plot(train_losses, label="train")
    plt.plot(test_losses, label="test")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # 绘制准确率图
    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Accuracy")
    plt.plot(train_accuracies, label="train")
    plt.plot(test_accuracies, label="test")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
if is_test:
    model.load_state_dict(torch.load("resnet.pt"))
    acc = test(model, test_loader, loss_func)
    print("Best Test Acc: {}".format(acc))
