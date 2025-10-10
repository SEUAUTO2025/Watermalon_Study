#根据式(5.18)和(5.19),试构造一个能解决异或问题的单层RBF神经网络
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
import torch
import math
from matplotlib import pyplot as plt

#需要自己写一个RBF层，而不是用torch.nn.Linear
class RBFLayer(nn.Module):
    def __init__(self,in_features,out_features):
        super(RBFLayer, self).__init__()
        self.beta = nn.Parameter(torch.abs(torch.randn(out_features)))#beta不能是正数，指数之后会梯度爆炸
        self.center = nn.Parameter(torch.randn(out_features,in_features))
    def forward(self,x):
        x_expand = x.unsqueeze(1)#(batch_size,1,in_features)
        center_expand = self.center.unsqueeze(0) #(1,out_features,in_features),对齐，之后用广播机制相减
        #相减之后(batch_size,out_features,in_features)
        distance = (torch.sum((x_expand-center_expand)**2,dim = 2))#(batch_size,out_features)
        beta_expand = self.beta.unsqueeze(0)
        output = torch.exp(-beta_expand*distance)#(batch_size,out_features)
        return output

class RBFNet(nn.Module):
    def __init__(self,n_feature):
        super().__init__()
        self.RBFlayer = RBFLayer(n_feature,50)
        self.Output = nn.Linear(50,1)
        #self.softmax = nn.Softmax(dim=1)
        nn.init.xavier_normal_(self.Output.weight)# Xavier正态分布   
        nn.init.constant_(self.Output.bias,val=0)
    
    def forward(self,x):#前向传播的时候会自动调用forward方法
        y = self.RBFlayer(x)
        y = self.Output(y)
        return y
     
    #标准BP：每次针对一个样例更新
    def optimiser_sgd_lr(self,x,y,lr=0.01,num_epochs=150,batch_size=1):
        costs = []
        loss = nn.CrossEntropyLoss()
        dataset = Data.TensorDataset(x,y)
        iter = Data.DataLoader(dataset,batch_size,shuffle=True)
        for i in range(num_epochs):
            #动态调整学习率
            if(i%10==0):
                lr=lr*0.9
            for x1,y1 in iter:
                y_hat = self.forward(x1)
                l = loss(y_hat,y1)
                l.backward()
                #参数更新
                for param in self.parameters():
                    param.data -= lr * param.grad
                #梯度清零
                for param in self.parameters():
                    param.grad.zero_()
            y_hat = self.forward(x)
            l = loss(y_hat,y)
            costs.append(l.item())
        return costs
    
    def optimiser_sgd(self,x,y,lr=0.01,num_epochs=150,batch_size=1):
        costs = []
        loss = nn.CrossEntropyLoss()
        dataset = Data.TensorDataset(x,y)
        iter = Data.DataLoader(dataset,batch_size,shuffle=True)
        for i in range(num_epochs):
            for x1,y1 in iter:
                y_hat = self.forward(x1)
                l = loss(y_hat,y1)
                l.backward()
                #参数更新
                for param in self.parameters():
                    param.data -= lr * param.grad
                #梯度清零
                for param in self.parameters():
                    param.grad.zero_()
            y_hat = self.forward(x)
            l = loss(y_hat,y)
            costs.append(l.item())
        return costs
    
    def optimiser_gd(self,x,y,lr=0.3,num_epochs=300):
        costs = []
        loss = nn.BCEWithLogitsLoss()
        for i in range(num_epochs):
            y_hat = self.forward(x)
            l = loss(y_hat,y)
            l.backward()
            #参数更新
            for param in self.parameters():
                param.data -= lr * param.grad
            #梯度清零
            costs.append(l.item())
            for param in self.parameters():
                param.grad.zero_()
        return costs

    def optimiser_gd_lr(self,x,y,lr=0.3,num_epochs=300):
        costs = []
        loss = nn.BCEWithLogitsLoss()
        for i in range(num_epochs):
            #lr = lr_max*(1+math.cos(i*math.pi/num_epochs))/2 
            if(i % 10==0):
                lr = lr * 0.9
            y_hat = self.forward(x)
            l = loss(y_hat,y)
            l.backward()
            #参数更新
            for param in self.parameters():
                param.data -= lr * param.grad
            #梯度清零
            costs.append(l.item())
            for param in self.parameters():
                param.grad.zero_()
        return costs
    
    def predict(self,x):
        """
        对输入x进行预测
        """
        self.eval() # 将模型设置为评估模式
        with torch.no_grad(): # 在预测时不需要计算梯度
            logits = self.forward(x) # 从前向传播中获取原始输出
            
            # 使用sigmoid函数将logits转换为概率
            probabilities = torch.sigmoid(logits)
            
            # 将概率转换为二进制预测（0或1）
            predictions = (probabilities > 0.5).float()
            
            #print("输出概率:", probabilities.squeeze())
            print(predictions.squeeze())
            
            return predictions

# 创建异或问题的完整数据集
def create_xor_dataset():
    """
    创建异或问题的数据集
    异或真值表:
    输入1  输入2  输出
      0      0      0
      0      1      1
      1      0      1
      1      1      0
    """
    # 基础的4个样本点
    X_base = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=np.float32)

    y_base = np.array([0, 1, 1, 0], dtype=np.float32)

    # 为了有足够的训练、验证和测试数据，我们在基础样本周围添加一些带噪声的样本
    np.random.seed(42)
    X_augmented = []
    y_augmented = []

    # 每个基础样本生成多个带轻微噪声的样本
    n_samples_per_point = 50
    noise_level = 0.1

    for i in range(len(X_base)):
        for _ in range(n_samples_per_point):
            # 添加高斯噪声
            noise = np.random.normal(0, noise_level, size=2)
            X_augmented.append(X_base[i] + noise)
            y_augmented.append(y_base[i])

    X = np.array(X_augmented)
    y = np.array(y_augmented)

    return X, y

# 生成数据集
X, y = create_xor_dataset()
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 * 0.85 ≈ 0.15
)

# 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1,1)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(y_val)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

net = RBFNet(X_train_tensor.size()[1])
net2 = RBFNet(X_train_tensor.size()[1])
costs = net.optimiser_gd_lr(X_train_tensor,y_train_tensor)
costs2 = net2.optimiser_gd(X_train_tensor,y_train_tensor)

net.predict(X_test_tensor)
print(y_test_tensor)

#动态调整学习率可能再比较复杂的情况下比较好用，学习率调得很高都能收敛的情况下可能不用降低学习率
plt.plot(costs, label='GD_LR (lr=0.01)')
plt.plot(costs2, label='GD (lr=0.01)')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.show()

