# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 17:18:06 2021

@author: Yiyu Gong
"""

import torch
import numpy as np
#%%
# 创建一个 numpy ndarray
numpy_tensor = np.random.randn(10, 20)
#%%
#2种方法将numpy数组值传递给pytorch张量
pytorch_tensor = torch.tensor(numpy_tensor)
pytorch_tensor1 = torch.from_numpy(numpy_tensor)
#%%
#张量转换为numpy数组
numpy_array = pytorch_tensor.numpy()
numpy_array1 = pytorch_tensor1.numpy()
#%%
#将tensor放到GPU上加速
#第一种方法
dtype = torch.cuda.FloatTensor#定义数据类型
gpu_tensor = torch.tensor(numpy_tensor).type(dtype)
#第二种方法
gpu_tensor1 = torch.tensor(numpy_tensor).cuda(0)
#%%
#创建随机二维张量
tensor1 = torch.randn(10,20)#创建随机（10,20）二维张量
#%%
#将gputensor放给cpu
cpu_tensor = gpu_tensor.cpu()
#%%
# 可以通过下面两种方式得到 tensor 的大小
print(pytorch_tensor1.shape)
print(pytorch_tensor1.size())
#%%
# 得到 tensor 的数据类型
print(pytorch_tensor1.type())
#%%
# 得到 tensor 的维度
print(pytorch_tensor1.dim())
#%%
# 得到 tensor 的所有元素个数
print(pytorch_tensor1.numel())
#%%
#作业
homework = torch.randn(3,2, dtype = torch.double).numpy()
print(homework.dtype)
#%%
#生成全为一的张量
x = torch.ones(2,2)
print(x)
#%%
#转换torch数据类型
x = x.long()
print(x.dtype)
x = x.double()
print(x.dtype)
#%%
#取极值
# 沿着行取最大值
x = torch.randn(10,10)
#%%
print(x)
max_value, max_idx = torch.max(x, dim=1)#输出也为张量，先输出最大值张量，后输出最大值张量下标
print(max_value)
print(max_idx)
#%%
# 沿着行对 x 求和
sum_x = torch.sum(x, dim=1)
print(sum_x)
#%%
# 增加维度或者减少维度
print(x.shape)
x = x.unsqueeze(0) # 在第一维增加
print(x.shape)
x = x.unsqueeze(1) # 在第二维增加
print(x.shape)
#%%
x = x.squeeze(0) # 减少第一维
print(x.shape)
#%%
x = x.squeeze() # 将 tensor 中所有的维数为一的维度全部都去掉
print(x.shape)
#%%
x = torch.randn(3, 4, 5)
print(x.shape)

# 使用permute和transpose进行维度交换
x = x.permute(1, 0, 2) # permute 可以重新排列 tensor 的维度
print(x.shape)

x = x.transpose(0, 2)  # transpose 交换 tensor 中的两个维度
print(x.shape)
#%%
# 使用 view 对 tensor 进行 reshape
x = torch.randn(3, 4, 5)
print(x.shape)

x = x.view(-1, 12) # -1 表示任意的大小，5 表示第二维变成 5
print(x.shape)

x = x.view(3, 20) # 重新 reshape 成 (3, 20) 的大小
print(x.shape)
#%%
#操作符号后面加_可以直接对参数本体做修改不需要传递到新的内存地址
x = torch.ones(3, 3)
print(x.shape)

# unsqueeze 进行 inplace
x.unsqueeze_(0)
print(x.shape)

# transpose 进行 inplace
x.transpose_(1, 0)
print(x.shape)
#%%
from torch.autograd import Variable
#%%
x_tensor = torch.randn(10, 5)
y_tensor = torch.randn(10, 5)

# 将 tensor 变成 Variable
x = Variable(x_tensor, requires_grad=True) # 默认 Variable 是不需要求梯度的，所以我们用这个方式申明需要对其进行求梯度
y = Variable(y_tensor, requires_grad=True)
z = torch.sum(x + y)
print(z.data)
print(z.grad_fn)
#%%
z.backward()

print(x.grad)
print(y.grad)
#%%
