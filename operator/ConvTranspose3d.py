import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math

class MyConvTranspose3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        output_padding: Union[int, Tuple[int, int, int]] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        padding_mode: str = 'zeros'
    ):
        super().__init__()
        
        # 参数检查
        if padding_mode != 'zeros':
            raise NotImplementedError("Only 'zeros' padding mode is supported")
            
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(output_padding, int):
            output_padding = (output_padding, output_padding, output_padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)
            
        # 保存参数
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = dilation
        self.padding_mode = padding_mode
        
        # 检查参数有效性
        assert in_channels % groups == 0, 'in_channels must be divisible by groups'
        assert out_channels % groups == 0, 'out_channels must be divisible by groups'
        assert all(op < max(s, d) for op, s, d in 
                  zip(output_padding, stride, dilation)), \
            'output_padding must be smaller than max(stride, dilation)'
        
        # 初始化权重和偏置
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels // groups, *kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        # 使用kaiming初始化权重
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入形状检查
        assert x.dim() == 5, "Input must be 5D (batch, channels, depth, height, width)"
        
        # 计算输出形状
        input_shape = x.shape
        output_shape = self._compute_output_shape(input_shape)
        
        # 使用unfold和matmul实现转置卷积
        output = self._conv_transpose3d(x, self.weight, output_shape)
        
        # 添加偏置
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)
            
        return output
    
    def _compute_output_shape(self, input_shape):
        # 计算输出形状 (N, C_out, D_out, H_out, W_out)
        N, C_in, D_in, H_in, W_in = input_shape
        C_out = self.out_channels
        
        D_out = (D_in - 1) * self.stride[0] - 2 * self.padding[0] + \
                self.dilation[0] * (self.kernel_size[0] - 1) + self.output_padding[0] + 1
        H_out = (H_in - 1) * self.stride[1] - 2 * self.padding[1] + \
                self.dilation[1] * (self.kernel_size[1] - 1) + self.output_padding[1] + 1
        W_out = (W_in - 1) * self.stride[2] - 2 * self.padding[2] + \
                self.dilation[2] * (self.kernel_size[2] - 1) + self.output_padding[2] + 1
        
        return (N, C_out, D_out, H_out, W_out)
    
    def _conv_transpose3d(
        self, 
        x: torch.Tensor, 
        weight: torch.Tensor, 
        output_shape: Tuple[int, int, int, int, int]
    ) -> torch.Tensor:
        # 实现转置卷积的核心计算
        
        # 1. 对输入进行im2col操作
        # 这里我们使用PyTorch的unfold函数来实现
        # 注意：转置卷积实际上是对输入进行补零后执行常规卷积
        
        # 计算需要的padding
        # 转置卷积相当于在输入之间插入零，然后执行常规卷积
        # 我们需要计算等效的padding
        
        # 等效的输入padding
        pad_input = []
        for i in range(3):
            pad_input.extend([
                self.dilation[i] * (self.kernel_size[i] - 1) - self.padding[i],
                self.dilation[i] * (self.kernel_size[i] - 1) - self.padding[i]
            ])
        
        # 对输入进行补零
        x_padded = F.pad(x, pad_input, mode='constant', value=0)
        
        # 计算等效的dilation和stride
        # 转置卷积相当于将stride和dilation互换
        # 这里我们使用分组卷积来实现转置卷积
        
        # 将输入和权重重新排列以适应分组卷积
        # 转置卷积的权重形状是 (in_channels, out_channels // groups, *kernel_size)
        # 我们需要将其转换为 (out_channels, in_channels // groups, *kernel_size)
        
        # 交换输入和输出通道维度
        weight_t = weight.transpose(0, 1).contiguous()
        
        # 执行分组卷积
        output = F.conv3d(
            x_padded, 
            weight_t, 
            bias=None,
            stride=self.dilation,  # 注意：这里交换了stride和dilation
            padding=0,
            dilation=self.stride,  # 注意：这里交换了stride和dilation
            groups=self.groups
        )
        
        # 由于我们使用了等效的padding计算，输出形状应该已经匹配
        assert output.shape == output_shape, \
            f"Output shape mismatch: expected {output_shape}, got {output.shape}"
            
        return output
    
    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)
    
# 测试代码
if __name__ == "__main__":
    # 设置随机种子以便复现结果
    torch.manual_seed(42)
    
    # 测试参数
    batch_size = 2
    in_channels = 3
    out_channels = 4
    kernel_size = (3, 3, 3)
    stride = (2, 2, 2)
    padding = (1, 1, 1)
    output_padding = (1, 1, 1)
    dilation = (1, 1, 1)
    input_size = (5, 6, 7)  # D, H, W
    
    # 创建输入
    x = torch.randn(batch_size, in_channels, *input_size)
    
    # 官方实现
    conv_official = nn.ConvTranspose3d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding, 
        output_padding=output_padding, dilation=dilation
    )
    
    # 自定义实现
    conv_custom = MyConvTranspose3d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding, 
        output_padding=output_padding, dilation=dilation
    )
    
    # 复制权重和偏置
    with torch.no_grad():
        conv_custom.weight.copy_(conv_official.weight)
        if conv_official.bias is not None:
            conv_custom.bias.copy_(conv_official.bias)
    
    # 前向传播
    out_official = conv_official(x)
    out_custom = conv_custom(x)
    
    # 比较输出
    print("Output shape - Official:", out_official.shape)
    print("Output shape - Custom:", out_custom.shape)
    
    # 计算差异
    diff = torch.abs(out_official - out_custom).max().item()
    print(f"Max difference between outputs: {diff:.6f}")
    
    # 反向传播测试
    out_official.sum().backward()
    out_custom.sum().backward()
    
    # 比较梯度
    weight_grad_diff = torch.abs(conv_official.weight.grad - conv_custom.weight.grad).max().item()
    print(f"Max difference in weight gradients: {weight_grad_diff:.6f}")
    
    if conv_official.bias is not None:
        bias_grad_diff = torch.abs(conv_official.bias.grad - conv_custom.bias.grad).max().item()
        print(f"Max difference in bias gradients: {bias_grad_diff:.6f}")