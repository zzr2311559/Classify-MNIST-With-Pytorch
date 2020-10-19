#step 1 加载必要的库
import torch
import torch.nn as nn #用于构建网络层，包括了卷积，池化，RNN等计算,以及其他如Loss计算，
                      #可以把torch.nn包内的各个类想象成神经网络的一层
import torch.nn.functional as F
import torch.optim as optim #内含优化算法
from torchvision import datasets, transforms

#step 2 定义超参数
#参数：模型f（x，a）中的a称为模型的参数，可以通过优化算法进行学习，比如对 y = ax + b，中的x就是参数，参数是由模型自身
#确定的
#超参数：用来定义模型结构或优化策略，是外部（我）决定的
BATCH_SIZE = 16 #每批处理的数据，起到限流的作用，大小看电脑配置自行决定
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")#用GPU或者CPU训练
print(DEVICE)
EPOCHS = 10 #训练数据集的轮次

#step 3 构建pipeline（transform），对图像进行处理
pipeline = transforms.Compose([
    transforms.ToTensor(),#将图片转换成tensor
    transforms.Normalize((0.1307,),(0.3081,))#normalize正则化：当模型出现过拟合时，降低模型复杂度
])
#step 4 下载，加载数据
from torch.utils.data import DataLoader
#下载数据集
train_set = datasets.MNIST("data", train=True,download=True, transform=pipeline)# 训练集

test_set = datasets.MNIST("data", train=False, download=True, transform=pipeline)# 测试集
#加载数据
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) #训练加载，shuffle表示打乱数据集进入网络的顺序，有利于提高识别精度

test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True) #测试加载
#step 4.2 插入代码，显示MNIST中的图片
with open("./data/MNIST/raw/train-images-idx3-ubyte", "rb") as f:
    file = f.read()
image1 = [int(str(item).encode('ascii'),16) for item in file[16: 16+784]]
print(image1)
import cv2
import numpy as np
image1_np = np.array(image1, dtype=np.uint8).reshape(28,28,1)

print(image1_np.shape)
cv2.imwrite("digit.jpg",image1_np)

#step 5 构建网络模型
class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5) #1表示输入灰度图片的通道，10表示输出通道，5表示卷积核kernel
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3) # output feature map 仍是一个张量，具有高度和宽度，深度可以任意取值，因为输出深度是层的参数，深度轴的不同通道不再像RGB那样代表颜色，而是代表filter（过滤器），
        self.fc1 = nn.Linear(in_features=20*10*10, out_features=500)#连接层，20*10*10表示输入通道，500表示输出通道
        self.fc2 = nn.Linear(in_features=500, out_features=10)#500表示输入通道，10表示输出通道，由于最后是识别在0到9的某个数字，故是10
    def forward(self, x):
        input_size = x.size(0) # batch_size

        x = self.conv1(x) # 输入：batch_size * 1 * 28 * 28, 输出：batch*10*24*24（28-5+1）
        x = F.relu(x)#通过在所有隐藏层间添加激活函数，使表达能力更强，保持shape不变，输出：batch*10*24*24（28-5+1）,其他激活函数有torch。sigmoid()等
        x = F.max_pool2d(x, 2, 2)#池化层，此处为最大池化，使用下采样，用于：一、减少需要处理的特征图的元素个数，二、通过让连续卷积层的观察窗口越来越大（窗口覆盖原始输入的比例增大），从而引入空间过滤器的层级结构，即对图片进行压缩，输入：batch*10*24*24（28-5+1），输出：batch_size*10*12*12

        x = self.conv2(x)#输入：batch_size*10*12*12，输出：batch_size*20*10*10（12-3+1）
        x = F.relu(x)

        x = x.view(input_size, -1) # 拉伸拉平，-1表示自动计算维度，20*20*10=2000*1

        x = self.fc1(x) # 输入：batch_size*2000，输出：batch_size*500
        x = F.relu(x) # 保持shape不变

        x = self.fc2(x) # 输入：batch_size*500，输出：batch_size*10

        output = F.log_softmax(x, dim=1) # 计算分类后，每个数字的概率值，softmax本质上是activation function，用于实现矩阵值范围变换至0~1
        return output

#another way to build a net:
#import torch
#import torch.nn as nn
#net = nn.Sequential(
    #nn.Conv2d(1,20,5),
    #nn.ReLU(),
    #nn.Conv2d(20,64,5),
    #nn.ReLU()
        #)
#print(net)



#step 6定义优化器
net = NET().to(DEVICE)

# torch.save(net, "net.pkl")   # save entire net
# torch.save(net.state_dict(), "net_params.pkl")   # save parameters
# def restore_net():
    # net_2 = torch.load("net.pkl")             #load entire net
# def restore_params():
    # net_3 = #和被提取网络格式一样的形式
    # net_3.load_state_dict(torch.load("net_params.pkl"))             #load params to a new net without params beforesabe save sa#sdddsdawdsdawsd wasdwasdwa


optimizer = optim.Adam(net.parameters())
#step 7定义训练函数方法
def train_model(net, device, train_loader, optimizer, epoch):
    net.train()#如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()，在测试时添加model.eval()。其中model.train()是保证BN层用每一批数据的均值和方差，而model.eval()是保证BN用全部训练数据的均值和方差；而对于Dropout，model.train()是随机取一部分网络连接来训练更新参数，而model.eval()是利用到了所有网络连接。
    for batch_index, (inputs, labels) in enumerate(train_loader):

        inputs,labels = inputs.to(device), labels.to(device) # 部署至device
        optimizer.zero_grad() # 初始化梯度为0
        output = net(inputs) # 预测
        loss = F.cross_entropy(output, labels) # 以loss函数为反馈信号对权重值进行调节，该调节由optimizer完成，它实现了back propagation（反向传播）算法，
                                               # cross_entropy函数实现了每个prediction的概率值
                                               # output表示能表征预测值的矩阵， target表示0~10的标签
        loss.backward()#反向传播
        optimizer.step()#参数优化
        if batch_index % 3000 == 0: # MNIST有60000个测试集，60000/10=6000，6000/3000=2，每轮可以检测2个loss
            print("Train Epoch: {} \t Loss: {:.6f}".format(epoch,loss.item()))

#step 8 定义测试函数方法
def test_model(net, device, test_loader):
    net.eval()
    correct = 0.0
    test_loss = 0.0
    with torch.no_grad():#在使用pytorch时，并不是所有的操作都需要进行计算图的生成（计算过程的构建，以便梯度反向传播等操作）。而对于tensor的计算操作，默认是要进行计算图的构建的，在这种情况下，可以使用 with torch.no_grad():，强制之后的内容不进行计算图构建。
        for inputs, labels in test_loader:
            inputs,labels = inputs.to(device), labels.to(device)
            output = net(inputs)
            test_loss += F.cross_entropy(output, labels).item()
            pred = output.max(1,keepdim=True)[1]#找到概率值最大的下标（0是值，1是索引，故选用1）
            correct += pred.eq(labels.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print("Test-Average loss : {:.4f}, Accuracy : {:,.3f}%\n".format(test_loss, 100.0 * correct / len(test_loader.dataset)))

#step 9 调用方法
for epoch in range(1, EPOCHS + 1):
    train_model(net, DEVICE, train_loader, optimizer, epoch)
    test_model(net, DEVICE, test_loader)
