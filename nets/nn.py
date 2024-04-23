import torch


def fuse_conv(conv, norm):
    """
    [https://nenadmarkus.com/p/fusing-batchnorm-and-conv/]
    """
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 conv.kernel_size,
                                 conv.stride,
                                 conv.padding,
                                 conv.dilation,
                                 conv.groups, True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, activation, k=1, s=1, p=0):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch)
        self.relu = activation

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class Residual(torch.nn.Module):
    def __init__(self, in_ch, out_ch, s=1):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.add_m = s != 1 or in_ch != out_ch

        self.conv1 = Conv(in_ch, out_ch, torch.nn.ReLU(), k=3, s=s, p=1)
        self.conv2 = Conv(out_ch, out_ch, torch.nn.Identity(), k=3, s=1, p=1)

        if self.add_m:
            self.conv3 = Conv(in_ch, out_ch, torch.nn.Identity(), s=s)

    def zero_init(self):
        torch.nn.init.zeros_(self.conv2.norm.weight)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)

        if self.add_m:
            x = self.conv3(x)

        return self.relu(x + y)


class ResNet(torch.nn.Module):
    def __init__(self, filters):
        super().__init__()

        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []

        # p1/2
        self.p1.append(Conv(filters[0], filters[1], torch.nn.ReLU(), k=7, s=2, p=3))
        # p2/4
        self.p2.append(torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.p2.append(Residual(filters[1], filters[1], s=1))
        self.p2.append(Residual(filters[1], filters[1], s=1))
        # p3/8
        self.p3.append(Residual(filters[1], filters[2], s=2))
        self.p3.append(Residual(filters[2], filters[2], s=1))
        # p4/16
        self.p4.append(Residual(filters[2], filters[3], s=2))
        self.p4.append(Residual(filters[3], filters[3], s=1))
        # p5/32
        self.p5.append(Residual(filters[3], filters[4], s=2))
        self.p5.append(Residual(filters[4], filters[4], s=1))

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)
        self.p5 = torch.nn.Sequential(*self.p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)

        return p2, p3, p4, p5


class Neck(torch.nn.Module):
    def __init__(self, filters):
        super().__init__()

        self.up = torch.nn.Upsample(scale_factor=2)
        self.p2 = torch.nn.Conv2d(filters[1], filters[3], kernel_size=1, bias=False)
        self.p3 = torch.nn.Conv2d(filters[2], filters[3], kernel_size=1, bias=False)
        self.p4 = torch.nn.Conv2d(filters[3], filters[3], kernel_size=1, bias=False)
        self.p5 = torch.nn.Conv2d(filters[4], filters[3], kernel_size=1, bias=False)

        self.p2_out = torch.nn.Sequential(torch.nn.Conv2d(filters[3], filters[3] // 4,
                                                          kernel_size=3, padding=1, bias=False))
        self.p3_out = torch.nn.Sequential(torch.nn.Conv2d(filters[3], filters[3] // 4,
                                                          kernel_size=3, padding=1, bias=False),
                                          torch.nn.Upsample(scale_factor=2))
        self.p4_out = torch.nn.Sequential(torch.nn.Conv2d(filters[3], filters[3] // 4,
                                                          kernel_size=3, padding=1, bias=False),
                                          torch.nn.Upsample(scale_factor=4))
        self.p5_out = torch.nn.Sequential(torch.nn.Conv2d(filters[3], filters[3] // 4,
                                                          kernel_size=3, padding=1, bias=False),
                                          torch.nn.Upsample(scale_factor=8))

    def forward(self, x):
        p2, p3, p4, p5 = x

        p2 = self.p2(p2)
        p3 = self.p3(p3)
        p4 = self.p4(p4)
        p5 = self.p5(p5)

        p4 = self.up(p5) + p4
        p3 = self.up(p4) + p3
        p2 = self.up(p3) + p2

        p2 = self.p2_out(p2)
        p3 = self.p3_out(p3)
        p4 = self.p4_out(p4)
        p5 = self.p5_out(p5)

        return torch.cat(tensors=(p5, p4, p3, p2), dim=1)


class Head(torch.nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.b = torch.nn.Sequential(Conv(filters[3], filters[1], torch.nn.ReLU(True), k=3, p=1),
                                     torch.nn.ConvTranspose2d(filters[1], filters[1], kernel_size=2, stride=2),
                                     torch.nn.BatchNorm2d(filters[1]),
                                     torch.nn.ReLU(True),
                                     torch.nn.ConvTranspose2d(filters[1], out_channels=1, kernel_size=2, stride=2))

        self.t = torch.nn.Sequential(Conv(filters[3], filters[1], torch.nn.ReLU(True), k=3, p=1),
                                     torch.nn.ConvTranspose2d(filters[1], filters[1], kernel_size=2, stride=2),
                                     torch.nn.BatchNorm2d(filters[1]),
                                     torch.nn.ReLU(True),
                                     torch.nn.ConvTranspose2d(filters[1], out_channels=1, kernel_size=2, stride=2))

    def forward(self, x):
        if self.training:
            return self.b(x), self.t(x)
        else:
            return torch.sigmoid(self.b(x))


class DBNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet([3, 64, 128, 256, 512])
        self.neck = Neck([3, 64, 128, 256, 512])
        self.head = Head([3, 64, 128, 256, 512])

        for m in self.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                torch.nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.ones_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self
