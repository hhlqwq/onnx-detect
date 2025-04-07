import torch
import torch.nn as nn
import torch.nn.functional as F


def manual_conv_transpose3d(x, weight, bias, stride, padding, output_padding, groups, dilation, kernel_size):
    # 上采样输入（插入零）
    batch, in_c, D_in, H_in, W_in = x.shape
    s_d, s_h, s_w = stride
    D_up = (D_in - 1) * s_d + 1
    H_up = (H_in - 1) * s_h + 1
    W_up = (W_in - 1) * s_w + 1

    up_x = torch.zeros(batch, in_c, D_up, H_up, W_up,
                       dtype=x.dtype, device=x.device)
    up_x[:, :, ::s_d, ::s_h, ::s_w] = x

    # 调整权重维度（关键修正点）
    # [out,in/g,k] -> [in/g,out,k]
    weight_permuted = weight.permute(1, 0, 2, 3, 4)

    # 计算填充量（考虑dilation）
    kernel_d = (kernel_size[0] - 1) * dilation[0] + 1
    kernel_h = (kernel_size[1] - 1) * dilation[1] + 1
    kernel_w = (kernel_size[2] - 1) * dilation[2] + 1

    pad_d = kernel_d - padding[0] - 1
    pad_h = kernel_h - padding[1] - 1
    pad_w = kernel_w - padding[2] - 1

    # 对称填充
    pad_d_left, pad_d_right = pad_d//2, pad_d - pad_d//2
    pad_h_left, pad_h_right = pad_h//2, pad_h - pad_h//2
    pad_w_left, pad_w_right = pad_w//2, pad_w - pad_w//2

    padded_x = F.pad(up_x, (pad_w_left, pad_w_right,
                            pad_h_left, pad_h_right,
                            pad_d_left, pad_d_right))

    # 分组卷积处理
    in_c_per_group = in_c // groups
    out_c_per_group = weight.shape[0] // groups

    # 执行常规卷积
    conv_out = F.conv3d(
        padded_x, weight_permuted, bias=None,
        stride=1, padding=0, dilation=dilation,
        groups=groups
    )

    # 添加output_padding
    op_d, op_h, op_w = output_padding
    conv_out = F.pad(conv_out, (0, op_w, 0, op_h, 0, op_d))

    # 添加偏置
    if bias is not None:
        conv_out += bias.view(1, -1, 1, 1, 1)

    return conv_out


class TestCase:
    def __init__(self, in_c, out_c, kernel_size, stride, padding, output_padding=(0, 0, 0), dilation=1, groups=1, input_size=(2, 3, 4)):
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.input_size = input_size


test_cases = [
    TestCase(
        in_c=3, out_c=6, kernel_size=3,
        stride=2, padding=1
    ),
    TestCase(
        in_c=4, out_c=8, kernel_size=4,
        stride=2, padding=1, output_padding=1
    ),
    TestCase(
        in_c=4, out_c=8, kernel_size=3,
        stride=2, padding=2, dilation=2
    ),
    TestCase(
        in_c=6, out_c=12, kernel_size=3,
        stride=2, padding=1, groups=3
    ),
    TestCase(
        in_c=4, out_c=8, kernel_size=(3, 5, 5),
        stride=(2, 3, 3), padding=(1, 2, 2),
        output_padding=(1, 1, 1)
    )
]


def test_case(tc):
    # 创建输入和官方层
    x = torch.randn(2, tc.in_c, *tc.input_size, requires_grad=True)

    # 官方实现
    conv = nn.ConvTranspose3d(
        tc.in_c, tc.out_c, tc.kernel_size,
        stride=tc.stride, padding=tc.padding,
        output_padding=tc.output_padding,
        dilation=tc.dilation, groups=tc.groups,
        bias=True
    )

    # 手动实现
    manual_out = manual_conv_transpose3d(
        x, conv.weight, conv.bias,
        stride=conv.stride,
        padding=conv.padding,
        output_padding=conv.output_padding,
        groups=conv.groups,
        dilation=conv.dilation,
        kernel_size=conv.kernel_size
    )

    # 官方前向计算
    official_out = conv(x)

    # 计算最大误差
    diff = (manual_out - official_out).abs().max().item()
    print(f"Max error: {diff:.5f}")
    assert diff < 1e-5, f"Test failed with max error {diff}"

    # 反向传播测试
    official_out.sum().backward()
    manual_out.sum().backward()

    # 检查权重梯度是否存在
    assert conv.weight.grad is not None
    print("Test passed!")


if __name__ == "__main__":
    for i, tc in enumerate(test_cases):
        print(f"Testing case {i+1}")
        test_case(tc)
