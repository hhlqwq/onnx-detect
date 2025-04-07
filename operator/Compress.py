import torch
import torch.nn as nn
import numpy as np


class LinearCompress(nn.Module):
    '''通过全连接层将高维特征映射到低维空间，适用于特征压缩或编码'''

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # 输入形状: [batch_size, input_dim]
        # 输出形状: [batch_size, output_dim]
        return self.linear(x)


class ConvCompress(nn.Module):
    '''使用卷积操作压缩特征图的空间尺寸，常用于图像处理'''

    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1
        )

    def forward(self, x):
        # 输入形状: [batch_size, in_channels, H, W]
        # 输出形状: [batch_size, out_channels, H//stride, W//stride]
        return self.conv(x)


class ThresholdCompress(nn.Module):
    '''保留绝对值大于阈值的元素，其余置零，适用于稀疏化处理'''

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        mask = (x.abs() > self.threshold).float()
        return x * mask


class TopKCompress(nn.Module):
    '''保留每个样本中最大的K个元素，其余置零，适用于特征选择'''

    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):
        batch_size = x.size(0)
        _, indices = torch.topk(x.view(batch_size, -1), self.k, dim=1)
        mask = torch.zeros_like(x.view(batch_size, -1))
        mask.scatter_(1, indices, 1)
        mask = mask.view_as(x)
        return x * mask


class GlobalPoolCompress(nn.Module):
    '''通过全局平均池化或最大池化压缩空间维度，常用于分类任务'''

    def __init__(self, mode='avg'):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        if self.mode == 'avg':
            return x.mean(dim=(2, 3))  # 全局平均池化
        elif self.mode == 'max':
            return x.amax(dim=(2, 3))  # 全局最大池化
        else:
            raise ValueError("Invalid mode. Use 'avg' or 'max'.")


class CompressFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, condition, dim=0):
        # 输入合法性检查
        if condition.dim() != 1:
            raise ValueError("Condition must be a 1D tensor")
        if len(condition) != input.shape[dim]:
            raise ValueError(
                f"Condition length {len(condition)} != input dim {dim} size {input.shape[dim]}")

        # 记录前向传播信息用于反向
        ctx.dim = dim
        ctx.input_shape = input.shape

        # 获取符合条件的索引
        indices = torch.where(condition)[0]
        ctx.save_for_backward(indices)

        # 执行压缩
        return torch.index_select(input, dim, indices)

    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors
        grad_input = torch.zeros(
            ctx.input_shape, dtype=grad_output.dtype, device=grad_output.device)
        grad_input.index_add_(ctx.dim, indices, grad_output)
        return grad_input, None, None


def compress(input, condition, dim=0):
    return CompressFunction.apply(input, condition, dim)

# -------------------------- 测试函数 --------------------------


def test_compress(shape=(5, 3), dim=0, threshold=1e-6):
    # 生成随机数据
    np_input = np.random.randn(*shape).astype(np.float32)
    np_condition = np.random.choice([True, False], size=shape[dim])

    # NumPy 官方结果
    np_result = np.compress(np_condition, np_input, axis=dim)

    # PyTorch 自定义算子结果
    torch_input = torch.tensor(np_input, requires_grad=True)
    torch_condition = torch.tensor(np_condition)
    torch_result = compress(torch_input, torch_condition, dim=dim)

    # -------------------------- 前向精度对比 --------------------------
    forward_diff = torch.abs(
        torch_result - torch.tensor(np_result)).max().item()
    assert forward_diff <= threshold, \
        f"前向传播误差过大: {forward_diff} > {threshold}"

    # -------------------------- 反向梯度检查 --------------------------
    # 计算梯度
    torch_result.sum().backward()
    grad_custom = torch_input.grad.numpy()

    # PyTorch 原生方式计算梯度 (对比基准)
    torch_input_grad = torch.zeros_like(torch_input)
    indices = torch.where(torch_condition)[0]
    selected = torch.index_select(torch_input, dim, indices)
    selected.sum().backward()
    grad_native = torch_input.grad.numpy()

    # 梯度对比
    grad_diff = np.abs(grad_custom - grad_native).max()
    assert grad_diff <= threshold, \
        f"反向传播梯度误差过大: {grad_diff} > {threshold}"

    print(f"测试通过！前向误差={forward_diff:.2e}, 梯度误差={grad_diff:.2e}")


# -------------------------- 多维度测试用例 --------------------------
if __name__ == "__main__":
    # 测试不同维度
    test_compress(shape=(5, 3), dim=0)      # 压缩第0维
    test_compress(shape=(4, 6), dim=1)      # 压缩第1维

    # 边界测试：全True条件
    np_input = np.random.randn(3, 2)
    np_condition = np.array([True, True, True])
    torch_input = torch.tensor(np_input, requires_grad=True)
    torch_result = compress(torch_input, torch.tensor(np_condition), dim=0)
    assert torch_result.shape == (3, 2), "全True条件测试失败"

    # 边界测试：全False条件
    np_condition = np.array([False, False])
    torch_result = compress(torch_input, torch.tensor(np_condition), dim=1)
    assert torch_result.shape == (3, 0), "全False条件测试失败"

    print("所有测试通过！")
