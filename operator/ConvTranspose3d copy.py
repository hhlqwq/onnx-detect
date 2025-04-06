import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, dilation=1,
                 bias=True, padding_mode='zeros'):
        super().__init__()
        if padding_mode != 'zeros':
            raise NotImplementedError("Only padding_mode='zeros' is supported.")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        self.stride = stride if isinstance(stride, tuple) else (stride,) * 3
        self.padding = padding if isinstance(padding, tuple) else (padding,) * 3
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding,) * 3
        self.groups = groups
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation,) * 3
        self.padding_mode = padding_mode

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        # 初始化权重参数
        self.weight = nn.Parameter(torch.Tensor(
            in_channels,
            out_channels // groups,
            self.kernel_size[0],
            self.kernel_size[1],
            self.kernel_size[2]
        ))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        N, C_in, D_in, H_in, W_in = x.shape

        # Step 1: 插入零进行上采样
        D_up = (D_in - 1) * self.stride[0] + 1
        H_up = (H_in - 1) * self.stride[1] + 1
        W_up = (W_in - 1) * self.stride[2] + 1

        x_up = torch.zeros(N, C_in, D_up, H_up, W_up, dtype=x.dtype, device=x.device)
        x_up[:, :, ::self.stride[0], ::self.stride[1], ::self.stride[2]] = x

        # Step 2: 计算填充并填充
        pad_d = self.dilation[0] * (self.kernel_size[0] - 1) - self.padding[0]
        pad_h = self.dilation[1] * (self.kernel_size[1] - 1) - self.padding[1]
        pad_w = self.dilation[2] * (self.kernel_size[2] - 1) - self.padding[2]

        pad_d, pad_h, pad_w = max(0, pad_d), max(0, pad_h), max(0, pad_w)
        x_pad = F.pad(x_up, (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d))

        # Step 3: 调整权重形状并翻转卷积核
        G = self.groups
        in_g = self.in_channels // G
        out_g = self.out_channels // G

        conv_weight = self.weight.view(G, in_g, out_g, *self.kernel_size)
        conv_weight = conv_weight.permute(0, 2, 1, 3, 4, 5).contiguous()
        conv_weight = conv_weight.view(G * out_g, in_g, *self.kernel_size)
        conv_weight = conv_weight.flip(dims=[2, 3, 4])

        # Step 4: 应用普通卷积
        output = F.conv3d(
            x_pad,
            conv_weight,
            bias=None,
            stride=1,
            padding=0,
            dilation=self.dilation,
            groups=self.groups
        )

        # Step 5: 添加输出填充
        op_d, op_h, op_w = self.output_padding
        output = F.pad(output, (0, op_w, 0, op_h, 0, op_d))

        # Step 6: 添加偏置
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)

        return output
    

# 参数设置
in_channels = 3
out_channels = 6
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
groups = 1
dilation = 2

# 官方实现
conv_official = nn.ConvTranspose3d(
    in_channels, out_channels, kernel_size,
    stride, padding, output_padding, groups, bias=False, dilation=dilation
)

# 自定义实现
conv_custom = ConvTranspose3d(
    in_channels, out_channels, kernel_size,
    stride, padding, output_padding, groups, dilation, bias=False
)

# 同步参数
conv_custom.load_state_dict(conv_official.state_dict())

# 测试输入
x = torch.randn(2, in_channels, 5, 5, 5)

# 前向传播
output_official = conv_official(x)
output_custom = conv_custom(x)

# 验证输出是否一致
print(torch.allclose(output_official, output_custom, atol=1e-4))  # 应输出True
print("输出形状是否一致:", output_official.shape == output_custom.shape)
print("最大误差:", torch.max(torch.abs(output_official - output_custom)))
print("输出值是否接近:", torch.allclose(output_official, output_custom, atol=1e-4))