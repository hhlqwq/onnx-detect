import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union, Tuple

class ConvTranspose3d(nn.Module):
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
        if padding_mode != 'zeros':
            raise NotImplementedError("仅支持 padding_mode='zeros'")
        
        # 参数初始化
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = self._triple(kernel_size)
        self.stride = self._triple(stride)
        self.padding = self._triple(padding)
        self.output_padding = self._triple(output_padding)
        self.groups = groups
        self.dilation = self._triple(dilation)

        # 通道数校验
        if in_channels % groups != 0:
            raise ValueError(f"in_channels ({in_channels}) 必须能被 groups ({groups}) 整除")
        if out_channels % groups != 0:
            raise ValueError(f"out_channels ({out_channels}) 必须能被 groups ({groups}) 整除")

        # 初始化权重
        self.weight = nn.Parameter(torch.empty(
            in_channels,
            out_channels // groups,
            *self.kernel_size
        ))

        # 初始化偏置
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def _triple(self, x: Union[int, Tuple[int, int, int]]) -> Tuple[int, int, int]:
        if isinstance(x, int):
            return (x, x, x)
        elif isinstance(x, tuple) and len(x) == 3:
            return x
        else:
            raise ValueError(f"参数应为 int 或三元组，实际类型为 {type(x)}")

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C_in, D_in, H_in, W_in = x.shape

        # Step 1: 上采样（插入零）
        D_up = (D_in - 1) * self.stride[0] + 1
        H_up = (H_in - 1) * self.stride[1] + 1
        W_up = (W_in - 1) * self.stride[2] + 1

        x_up = torch.zeros(N, C_in, D_up, H_up, W_up, dtype=x.dtype, device=x.device)
        x_up[:, :, ::self.stride[0], ::self.stride[1], ::self.stride[2]] = x

        # Step 2: 重新计算填充量（关键修复：填充量乘2）
        def get_padding(total_pad: int) -> Tuple[int, int]:
            return (total_pad // 2, total_pad - total_pad // 2)

        pad_d = 2 * ((self.kernel_size[0] - 1) * self.dilation[0] - self.padding[0])
        pad_h = 2 * ((self.kernel_size[1] - 1) * self.dilation[1] - self.padding[1])
        pad_w = 2 * ((self.kernel_size[2] - 1) * self.dilation[2] - self.padding[2])

        pad_d = max(0, pad_d)
        pad_h = max(0, pad_h)
        pad_w = max(0, pad_w)

        # 应用非对称填充
        pad_front, pad_back = get_padding(pad_d)
        pad_top, pad_bottom = get_padding(pad_h)
        pad_left, pad_right = get_padding(pad_w)

        x_pad = F.pad(x_up, (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back))

        # Step 3: 权重变换（转置通道 + 翻转卷积核）
        G = self.groups
        conv_weight = self.weight.view(
            G,
            self.in_channels // G,
            self.out_channels // G,
            *self.kernel_size
        ).permute(0, 2, 1, 3, 4, 5).contiguous().flatten(0, 1)
        conv_weight = conv_weight.flip(dims=[2, 3, 4])

        # Step 4: 执行普通卷积
        output = F.conv3d(
            x_pad,
            conv_weight,
            bias=None,
            stride=1,
            padding=0,
            dilation=self.dilation,
            groups=self.groups
        )

        # Step 5: 添加输出填充（仅右侧）
        op_d = (0, self.output_padding[0])
        op_h = (0, self.output_padding[1])
        op_w = (0, self.output_padding[2])
        output = F.pad(output, (op_w[0], op_w[1], op_h[0], op_h[1], op_d[0], op_d[1]))

        # Step 6: 添加偏置
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)

        return output

# ------------------------ 测试代码 ------------------------
def test_case(case_name: str, **kwargs):
    print(f"\n===== 测试案例: {case_name} =====")
    
    # 官方实现
    conv_official = nn.ConvTranspose3d(
        in_channels=3,
        out_channels=6,
        **kwargs,
        bias=True
    )
    
    # 自定义实现
    conv_custom = ConvTranspose3d(
        in_channels=3,
        out_channels=6,
        **kwargs,
        bias=True
    )
    
    # 参数同步
    conv_custom.load_state_dict(conv_official.state_dict())
    
    # 测试输入
    x = torch.randn(2, 3, 5, 7, 11)
    
    # 前向传播
    out_official = conv_official(x)
    out_custom = conv_custom(x)
    
    # 验证形状
    shape_match = out_official.shape == out_custom.shape
    print(f"形状一致性: {shape_match}")
    
    # 验证数值
    numerical_match = torch.allclose(out_official, out_custom, atol=1e-5)
    print(f"数值一致性: {numerical_match}")

if __name__ == "__main__":
    test_cases = {
        "基础测试": {
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "output_padding": 1,
            "dilation": 1
        },
        "非对称参数": {
            "kernel_size": (3, 5, 2),
            "stride": (2, 1, 3),
            "padding": (0, 2, 1),
            "output_padding": (1, 0, 1),
            "dilation": (2, 1, 3)
        },
        "无输出填充": {
            "kernel_size": 4,
            "stride": 3,
            "padding": 2,
            "output_padding": 0,
            "dilation": 3
        }
    }

    for name, params in test_cases.items():
        test_case(name, **params)