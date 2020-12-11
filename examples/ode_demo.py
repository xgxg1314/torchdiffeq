import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

# 设置系统参数
parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)

# 在终端运行时，不加--viz，就会认为--viz=false
# 是否使用可视化
parser.add_argument('--viz', action='store_true')

# 是否使用gpu
parser.add_argument('--gpu', type=int, default=0)

# 伴随方法
parser.add_argument('--adjoint', action='store_true')
# 解析所有参数
args = parser.parse_args()

# 是否使用伴随方法
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

# 是否使用gpu
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

# 初值[2,0]
true_y0 = torch.tensor([[2., 0.]]).to(device)
# 时间范围0-25，data_size的default=1000，就是把时间分成1000份
t = torch.linspace(0., 25., args.data_size).to(device)
#
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)

class Lambda(nn.Module):
    def forward(self, t, y):
        # torch.mm(a, b)是矩阵a和b矩阵相乘
        return torch.mm(y**3, true_A)

# 计算y真实值
with torch.no_grad():
    # 调用odeint
    # 返货真实的true_y
    # Lambda()——> torch.mm(y**3, true_A)
    # odeint(func, y0, t, rtol=1e-7, atol=1e-9, method=None, options=None):

    true_y = odeint(Lambda(), true_y0, t, method='dopri5')


def get_batch():
    # '--data_size', type=int, default=1000
    # '--batch_time', type=int, default=10
    # '--batch_size', type=int, default=20
    # 一共1000组数据，1000-10=990，随机从0-989中选择出20个数组成一个新的数组
    # False表示不可以取相同数字
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    # 从真实值中拿出20个作为一批初值
    batch_y0 = true_y[s]  # (M, D)

    # 取前10个做时间批次,t[:10]
    batch_t = t[:args.batch_time]  # (T)
    # 获取一批y的值，每一个小批次里面是20行数据
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


# 创建新的文件夹
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

# 如果调用可视化，则创建文件夹png
if args.viz:
    makedirs('png')
    import matplotlib.pyplot as plt
    # 定义画布大小，和背景颜色
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    #画出时间历程图，相图和向量场
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    # 画图的同时，执行后面的代码
    # 先创建空的图
    plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr):

    if args.viz:
       # cla()   即Clear axis即清除当前图形中的当前活动轴。其他轴不受影响。
        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        # 画图的时候，需要把数据转为numpy数组类型，并且放回到cpu上面
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.set_ylim(-2, 2)
        ax_traj.legend()
        # 相位图
        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)

        # 向量场
        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        # 将-2-2分成20等分
        y, x = np.mgrid[-2:2:21j, -2:2:21j]
        dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)

        # 画出向量场
        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        ax_vecfield.set_xlim(-2, 2)
        ax_vecfield.set_ylim(-2, 2)

        # tight_layout会自动调整子图参数，使之填充整个图像区域
        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr))
        # plt.draw是一种交互式的展示图方式
        plt.draw()
        # 画图停顿
        plt.pause(0.001)


# 定义前向传播
class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()
        # 设置
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2),
        )

        for m in self.net.modules():
            # 遍历子模块
            # 判断对象是否是特定类的示例，对线性层进行初始化设置
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    # 重写forward方法
    def forward(self, t, y):
        # 输出y^3
        return self.net(y**3)

        # 遍历子模块
        # import torch.nn as nn
        # class Model(nn.Module):
        #     def __init__(self):
        #         super(Model, self).__init__()
        #         self.add_module("conv", nn.Conv2d(10, 20, 4))
        #         self.add_module("conv1", nn.Conv2d(20, 10, 4))
        #
        # model = Model()
        #
        # for sub_module in model.children():
        #     print(sub_module)

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


if __name__ == '__main__':

    ii = 0

    # ode函数，前向传播过程，实际上就是我们构建好的神经网络的model，把模型放在设备上运行
    func = ODEFunc().to(device)
    # 为了构建一个Optimizer，你需要给它一个包含了需要优化的参数（必须都是Variable对象）
    # 的iterable。然后，你可以设置optimizer的参
    # 数选项，比如学习率，权重衰减，等等。
    # 例子：
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # optimizer = optim.Adam([var1, var2], lr=0.0001)

    # 设置优化器和学习率
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    # 计算时间
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    
    loss_meter = RunningAverageMeter(0.97)

    # niters = 2000
    # 训练，遍历2000次，1-2000
    for itr in range(1, args.niters + 1):
        # 梯度归零
        optimizer.zero_grad()
        # 批次获取数据，每批获取
        batch_y0, batch_t, batch_y = get_batch()
        pred_y = odeint(func, batch_y0, batch_t).to(device)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        # 更新时间
        time_meter.update(time.time() - end)
        # 更新误差
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, true_y0, t)
                loss = torch.mean(torch.abs(pred_y - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(true_y, pred_y, func, ii)
                ii += 1

        end = time.time()
