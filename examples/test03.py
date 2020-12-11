# -*- coding:utf-8 -*-

import torch
import numpy as np
# 假设是时间步T1
T1 = torch.tensor([[1, 2, 3],
        		[4, 5, 6],
        		[7, 8, 9]])
# 假设是时间步T2
T2 = torch.tensor([[10, 20, 30],
        		[40, 50, 60],
        		[70, 80, 90]])

print(torch.stack((T1,T2),dim=0).shape)
# 拼接到行上
print(torch.stack((T1,T2),dim=0))

print(torch.stack((T1,T2),dim=1).shape)
print(torch.stack((T1,T2),dim=2).shape)
# print(torch.stack((T1,T2),dim=3).shape)

# # outputs:
# torch.Size([2, 3, 3])
# torch.Size([3, 2, 3])
# torch.Size([3, 3, 2])
# '选择的dim>len(outputs)，所以报错'
# IndexError: Dimension out of range (expected to be in range of [-3, 2], but got 3)


s = torch.from_numpy(np.random.choice(np.arange(1000 - 10, dtype=np.int64),
                                      20, replace=False))
print(s)
print(s+1)

true_y = torch.randn(1000,3)

s2 = [true_y[s + i] for i in range(10)]
print('s2',s2)


batch_y = torch.stack([true_y[s + i] for i in range(10)], dim=0)
print('batch_y',batch_y)


import os
print(os.path)

a = np.random.rand(4,3)
a = [a]
print(a[:,0,0])
print(a[:,0,1])