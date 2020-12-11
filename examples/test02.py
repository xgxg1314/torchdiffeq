#  # -*- coding:utf-8 -*-
# # import numpy as np
# # # 输出0-899的列表
# # print(np.arange(900))
# #
# # # np.random.choice
# # #在[0, 5)内输出3个数字并组成一维数组（ndarray
# # # 第一个参数必须是一维数组？
# # print(np.random.choice(5, 3))
# #
# # import torch
# # t = torch.linspace(0., 25., 1000)
# # print('t:',t)
# # batch_t = t[:10]
# # print(batch_t)
# #
# # true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]])
# # # 取第一行
# # print(true_A[1])
#
# import numpy as np
# import torch
#
# y, x = np.mgrid[-2:2:5j, -2:2:5j]
# print(x.shape)
# # print(y)
# #
# # z = torch.Tensor(np.stack([x, y], -1))
# # print(z)
#
#
# a = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
# b = np.array([[4, 5, 6], [4, 5, 6], [4, 5, 6]])
# print(a.shape)
# print(b.shape)
#
# # 这里的2就是指的a和b，而2放在什么位置是根据axis来确定的
# print(np.stack((a, b), axis=0).shape)
# # (2, 3, 3)
# print(np.stack((a, b), axis=1).shape)
# # (3, 2, 3)
# print(np.stack((a, b), axis=2).shape)
# # (3, 3, 2)


import numpy as np
import matplotlib.pyplot as plt

# plt.axis([0, 10, 0, 1])
# plt.ion()
#
# for i in range(10):
#     y = np.random.random()
#     plt.scatter(i, y)
#     plt.pause(0.05)
#
# # while True:
# #     plt.pause(0.05)
#
# a1 = np.eye(2,2)
# print(a1**3)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    # 定义一个新的类
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


for itr in range(1, 20):
    print(itr)


class Lambda(nn.Module):
    def forward(self, t, y):
        # torch.mm(a, b)是矩阵a和b矩阵相乘
        return torch.mm(y ** 3, true_A)

with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='dopri5')




# ode初值[2,0]
true_y0 = torch.tensor([[2., 0.]]).to(device)
# 时间范围0-25，data_size的default=1000，就是把时间分成1000份
t = torch.linspace(0., 25., args.data_size).to(device)
#
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)

class Lambda(nn.Module):
    def forward(self, t, y):
        # torch.mm(a, b)是矩阵a和b矩阵相乘
        return torch.mm(y**3, true_A)

# 计算y值
with torch.no_grad():
    # odeint(func, y0, t, rtol=1e-7, atol=1e-9, method=None, options=None):
    true_y = odeint(Lambda(), true_y0, t, method='dopri5')