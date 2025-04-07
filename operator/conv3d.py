import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MyConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(MyConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # 处理kernel_size、stride、padding参数，确保为三元组
        self.kernel_size = self._to_3tuple(kernel_size)
        self.stride = self._to_3tuple(stride)
        self.padding = self._to_3tuple(padding)
        # 初始化权重和偏置
        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels, *self.kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    @staticmethod
    def _to_3tuple(value):
        if isinstance(value, int):
            return (value, value, value)
        elif isinstance(value, (list, tuple)) and len(value) == 3:
            return tuple(value)
        else:
            raise ValueError("Value must be an int or a 3-element tuple/list")

    def reset_parameters(self):
        # 使用Kaiming初始化权重
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * \
                self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        batch_size, in_channels, D_in, H_in, W_in = input.shape
        kd, kh, kw = self.kernel_size
        stride_d, stride_h, stride_w = self.stride
        pad_d, pad_h, pad_w = self.padding
        # 计算输出尺寸
        D_out = (D_in + 2 * pad_d - kd) // stride_d + 1
        H_out = (H_in + 2 * pad_h - kh) // stride_h + 1
        W_out = (W_in + 2 * pad_w - kw) // stride_w + 1

        # 输入填充
        input_padded = F.pad(input, (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d))

        # 三维展开
        input_unfold = input_padded.unfold(2, kd, stride_d)  # 沿D维度展开
        input_unfold = input_unfold.unfold(3, kh, stride_h)  # 沿H维度展开
        input_unfold = input_unfold.unfold(4, kw, stride_w)  # 沿W维度展开

        # 调整维度顺序并展平
        input_unfold = input_unfold.permute(
            0, 1, 5, 6, 7, 2, 3, 4).contiguous()
        input_unfold = input_unfold.view(
            batch_size, in_channels * kd * kh * kw, -1)

        # 展平权重
        weight_flat = self.weight.view(self.out_channels, -1)

        # 矩阵乘法计算输出
        output = torch.einsum('oi,bil->bol', weight_flat, input_unfold)
        output = output.view(
            batch_size, self.out_channels, D_out, H_out, W_out)

        # 添加偏置
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)

        return output


# 测试代码
if __name__ == "__main__":
    # 参数设置
    in_channels = 3
    out_channels = 2
    kernel_size = (3, 1, 3)
    stride = (2, 1, 2)
    padding = (1, 2, 1)
    bias = True
    input_tensor = torch.randn(1, in_channels, 3, 5, 5)  # 输入尺寸5x5x5

    # 官方实现
    conv_official = nn.Conv3d(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

    # 自定义实现
    conv_custom = MyConv3d(in_channels, out_channels,
                           kernel_size, stride=stride, padding=padding, bias=bias)

    # 复制权重和偏置
    with torch.no_grad():
        conv_custom.weight.copy_(conv_official.weight)
        conv_custom.bias.copy_(conv_official.bias)

    # 前向计算
    output_official = conv_official(input_tensor)
    output_custom = conv_custom(input_tensor)

    # 比较结果
    print(output_official.shape, output_custom.shape)
    print("输出形状是否一致:", output_official.shape == output_custom.shape)
    print("最大误差:", torch.max(torch.abs(output_official - output_custom)))
    print("平均误差:", torch.abs(output_official-output_custom).mean())
    print("输出值是否接近:", torch.allclose(output_official, output_custom, atol=1e-4))
