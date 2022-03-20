import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
# from torchinfo import summary
from tqdm import tqdm

torch.manual_seed(0)

# device = torch.device("cuda" if torch.cuda.is_available() else "pcu")

# 使用cuDNN加速卷积运算
torch.backends.cudnn.benchmark = True

# 载入MNIST数据集
# 载入训练集
train_dataset = torchvision.datasets.MNIST(
    root="dataset/",
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
# 载入测试集
test_dataset = torchvision.datasets.MNIST(
    root="dataset/",
    train=False,
    transform=transforms.ToTensor(),
    download=True
)
train_loder = DataLoader(dataset=train_dataset,batch_size=32,shuffle=True)
test_loder  = DataLoader(dataset=test_dataset, batch_size=32,shuffle=False)


class TeacherModel(nn.Module):
    def __init__(self,in_channels=1,num_classes=10):
        super(TeacherModel, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(784,1200)
        self.fc2 = nn.Linear(1200,1200)
        self.fc3 = nn.Linear(1200,num_classes)
        self.dropout = nn.Dropout(p = 0.5)

    def forward(self,x):
        x = x.view(-1,784)
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.fc3(x)

        return x

model = TeacherModel() # 先训练一下学生模型

criterion = nn.CrossEntropyLoss() # 设置使用交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4) # 使用Adam优化器，学习率为lr=1e-4
#
epochs = 6 # 训练6轮
for epoch in range(epochs):
    model.train()

    for data,targets in tqdm(train_loder):
        # 前向预测
        preds = model(data)
        loss = criterion(preds,targets)

        # 反向传播，优化权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 测试集上评估性能
    model.eval()
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for x,y in test_loder:
            preds = model(x)
            predictions = preds.max(1).indices
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        acc = (num_correct / num_samples).item()

    model.train()
    print(("Epoch:{}\t Accuracy:{:4f}").format(epoch+1,acc))

teacher_model = model
# 学生模型
class StudentModel(nn.Module):
    def __init__( self,inchannels=1,num_class=10):
        super(StudentModel, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(784, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, num_class)

    def forward(self,x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        #x = self.dropout(x)
        x = self.relu(x)

        x = self.fc2(x)
        #x = self.dropout(x)
        x = self.relu(x)

        x = self.fc3(x)
        return x

model = StudentModel() # 从头先训练一下学生模型
# 设置交叉损失函数 和 激活函数
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
# 学生模型训练
# epochs = 3
# # 训练集上训练权重
# for epoch in range(epochs):
#     model.train()
#
#     for data,targets in tqdm(train_loder):
#         # 前向预测
#         preds = model(data)
#         loss = criterion(preds,targets)
#
#         # 反向传播，优化权重
#         optimizer.zero_grad() # 把梯度置为0
#         loss.backward()
#         optimizer.step()
#
#     with torch.no_grad():
#         for x,y in  test_loder:
#             preds = model(x)
#             predictions = preds.max(1).indices
#             num_correct += (predictions==y).sum()
#             num_samples += predictions.size(0)
#             acc = (num_correct / num_samples).item()
#
#     model.train()
#     print(("Epoch:{}\t Accuracy:{:4f}").format(epoch+1,acc))

student_model_scratch = model

# 知识蒸馏训练学生模型

# 准备好预训练好的教师模型
teacher_model.eval()

# 准备新的学生模型
model = StudentModel()
model.train()

# 蒸馏温度
temp = 7

# hard_loss
hard_loss = nn.CrossEntropyLoss()
# hard_loss权重
alpha = 0.3
# soft_loss
soft_loss = nn.KLDivLoss(reduction="batchmean")

optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

epochs = 3
for epoch in range(epochs):
    for data,targets in tqdm(train_loder):
        # 教师模型预测
        with torch.no_grad():
            teacher_preds = teacher_model(data)

        # 学生模型预测
        student_preds = student_model_scratch(data)

        student_loss = hard_loss(student_preds,targets)

        # 计算蒸馏后的预测结果及soft_loss
        distillation_loss = soft_loss(
            F.softmax(student_preds/temp,dim=1),
            F.softmax(teacher_preds/temp,dim=1)
        )

        # 将 hard_loss 和 soft_loss 加权求和
        loss = alpha * student_loss + (1-alpha) * distillation_loss

        # 反向传播,优化权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 测试集上评估性能
    model.eval()
    num_correct = 0

    with torch.no_grad():
        for x,y in test_loder:
            preds = model(x)
            predictions = preds.max(0).indices
            num_correct += (predictions==y).sum()
            num_samples += predictions,size(0)
        acc = (num_correct/num_samples).item()

    model.train()
    print(("Epoch:{}\t Accuracy:{:4f}").format(epoch+1,acc))



























































