import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataload import Mnist
from net import MNet
import matplotlib.pyplot as plt

nets = MNet()
nets.load_state_dict(torch.load("Linear.pth"))  # 调用训练好的网络

# 检测网络
test_set = Mnist(  # 生成测试集
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

for data in test_loader:
    images, labels = data

test_output = nets(images[:10])  # 在测试集中选择10张图片输入网络，得到结果
pred_y = torch.max(test_output, 1)[1].data  # 对得到的结果进行预测
print(pred_y, 'prediction number')  # 输出预测结果
print(labels[:10], 'real number')  # 输出真实结果


def plt_image(image):  # 定义一个函数，将需要预测的手写数字图画出来
    n = 10
    plt.figure(figsize=(10, 4))
    for i in range(n):
        ax = plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


plt_image(images)

# 测试自己手动设计的手写数字
from PIL import Image

I = Image.open('2.jpg')
L = I.convert('L')  # 转化为二值图像
plt.imshow(L, cmap='gray')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1037,), (0.3081,))
])
im = transform(L)  # [C, H, W]
im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]，扩展维度
with torch.no_grad():
    outputs = nets(im)
    _, predict = torch.max(outputs.data, 1)
    print(predict)
