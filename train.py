import torch
import torchvision.transforms as transforms
import torch.optim as optim#优化器
from torch.utils.data import DataLoader
from dataload import Mnist
from net import MNet
import matplotlib.pyplot as plt
lr=0.001
# 生成训练集
train_set = Mnist(
    root= "Mnist",
    train=True,#训练集
    transform=transforms.Compose([#Compose方法是将多种变换组合在一起。
        transforms.ToTensor(),#函数接受PIL Image或numpy.ndarray，将其先由HWC转置为CHW格式
        transforms.Normalize((0.1037,), (0.3081,))#灰度图像，一个通道，均值和方差，标准化
    ])
)
train_loader = DataLoader(#主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor
    dataset=train_set,#输出的数据
    batch_size=32,
    shuffle=True#将元素随机排序
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#判断能否调用GPU
print(device)
# 实例化一个网络
nets = MNet().to(device) #将网络放进GPU

# 定义损失函数和优化器
loss_function = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(nets.parameters(), lr=lr)
#optimizer = optim.SGD(
#    nets.parameters(),#网络参数
#    lr=0.001,#学习率
#    momentum=0.9#Momentum 用于加速 SGD（随机梯度下降）在某一方向上的搜索以及抑制震荡的发生。
#)

# 3 训练模型
loss_list = []#保存损失函数的值
for epoch in range(10):#训练10次
    running_loss = 0.0#误差清零？
    for batch_idx, data in enumerate(train_loader, start=0):#enumerate索引函数，start下标开始位置

        images, labels = data                       # 读取一个batch的数据
        images=images.to(device)  #将images放进GPU
        labels=labels.to(device)  #将labels放进GPU
        optimizer.zero_grad()                       # 梯度清零，初始化,如果不初始化，则梯度会叠加
        outputs = nets(images)                      # 前向传播
        loss = loss_function(outputs, labels)       # 计算误差，label标准？
        loss.backward()                             # 反向传播
        optimizer.step()                            # 权重更新
        running_loss += loss.item()                 # 误差累计

        # 每300个batch 打印一次损失值
        if batch_idx % 300 == 299:#（0-299）（300-599）
            print('epoch:{} batch_idx:{} loss:{}'
                  .format(epoch+1, batch_idx+1, running_loss/300))
            loss_list.append(running_loss/300)#将新的每个平均误差加到损失函数列表后面
            running_loss = 0.0                  #误差清零
print('Finished Training.')

torch.save(nets.state_dict(),"Linear.pth")#保存训练模型

# 打印损失值变化曲线

plt.plot(loss_list)
plt.title('traning loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

# 测试
test_set = Mnist(#生成测试集
    root="Mnist",
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1037,), (0.3081,))
    ])
)
test_loader = DataLoader(
    dataset=test_set,
    batch_size=32,
    shuffle=True
)
correct = 0  # 预测正确数
total = 0    # 总图片数
for data in test_loader:
    images, labels = data
    images=images.to(device)
    labels=labels.to(device)
    outputs = nets(images)
    _, predict = torch.max(outputs.data, 1)#1是指按行，0是按列，-1是指最后一个维度，一般也是按行
    total += labels.size(0)
    correct += (predict == labels).sum()

print('测试集准确率 {}%'.format(100*correct // total))

#检测网络
test_output=nets(images[:10])#在测试集中选择10张图片输入网络，得到结果
pred_y = torch.max(test_output, 1)[1].data#对得到的结果进行预测
print(pred_y, 'prediction numbe')#输出预测结果
print(labels[:10], 'real number')#输出真实结果
