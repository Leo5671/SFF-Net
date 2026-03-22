import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
import numbers


class AttentionBase(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
    ):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )
        out = self.proj(out)
        return out


class Mlp(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, ffn_expansion_factor=2, bias=False
    ):
        super().__init__()
        hidden_features = int(in_features * ffn_expansion_factor)
        self.project_in = nn.Conv2d(
            in_features, hidden_features * 2, kernel_size=1, bias=bias
        )
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features,
            bias=bias,
        )
        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias
        )

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class BaseFeatureExtractor(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        ffn_expansion_factor=1.0,
        qkv_bias=False,
    ):
        super(BaseFeatureExtractor, self).__init__()
        self.norm1 = LayerNorm(dim, "WithBias")
        self.attn = AttentionBase(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self.norm2 = LayerNorm(dim, "WithBias")
        self.mlp = Mlp(
            in_features=dim,
            ffn_expansion_factor=ffn_expansion_factor,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class AttFuseLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        ffn_expansion_factor=1.0,
        qkv_bias=False,
    ):
        super(AttFuseLayer, self).__init__()
        concat_dim = dim * 2
        self.norm1 = LayerNorm(concat_dim, "WithBias")
        self.attn = AttentionBase(
            concat_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self.norm2 = LayerNorm(concat_dim, "WithBias")
        self.mlp = Mlp(
            in_features=concat_dim,
            ffn_expansion_factor=ffn_expansion_factor,
        )
        self.reduce_dim = nn.Conv2d(concat_dim, dim, kernel_size=1)

    def forward(self, feature_I_B, feature_V_B):
        x = torch.cat((feature_I_B, feature_V_B), dim=1)  # [B, 2*DIM, H, W]
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        x = self.reduce_dim(x)

        return x


class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return self.bottleneckBlock(x)


class DetailNode(nn.Module):
    def __init__(self):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.shffleconv = nn.Conv2d(
            64, 64, kernel_size=1, stride=1, padding=0, bias=True
        )

    def separateFeature(self, x):
        z1, z2 = x[:, : x.shape[1] // 2], x[:, x.shape[1] // 2 : x.shape[1]]
        return z1, z2

    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2


class DetailFeatureExtractor(nn.Module):
    def __init__(self, num_layers=3):
        super(DetailFeatureExtractor, self).__init__()
        INNmodules = [DetailNode() for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)

    def forward(self, x):
        z1, z2 = x[:, : x.shape[1] // 2], x[:, x.shape[1] // 2 : x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class IFFT_Block(nn.Module):
    def __init__(self, out_channels=8):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, out_channels // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, amp, pha, H, W):
        real = amp * torch.cos(pha) + 1e-8
        imag = amp * torch.sin(pha) + 1e-8
        x = torch.complex(real, imag)
        # 指定逆变换的输出尺寸为原图的H和W
        x = torch.abs(torch.fft.irfftn(x, s=(H, W), dim=(-2, -1)))
        x = torch.cat((torch.max(x, 1)[0].unsqueeze(1),
                      torch.mean(x, 1).unsqueeze(1)), dim=1)
        return self.conv1(x)
def fft(input):
    '''
    input: tensor of shape (batch_size, 1, height, width)
    mask: tensor of shape (height, width)
    '''
    # 执行2D FFT
    img_fft = torch.fft.rfftn(input, dim=(-2, -1))
    amp = torch.abs(img_fft)
    pha = torch.angle(img_fft)
    return amp, pha


class PhaseCompensation(nn.Module):#PAB
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.LeakyReLU(0.1),  # 改为与原始模块一致
            nn.Conv2d(8, 1, 3, padding=1),
            nn.Identity()  # 返回输入本身
        )

    def forward(self, x):
        return x + self.conv(x)  # 残差式补偿



class ChannelAttention(nn.Module):#SeNet
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualConvBlock(nn.Module):#RCB
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(4, channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        return x + self.conv(x)




class DepthwiseSeparableConv(nn.Module):# 深度可分离卷积DCconve3*3
    """减少参数同时保持细节"""

    def __init__(self, in_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3,
                                   padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class DynamicDetailEnhance(nn.Module):
    """自适应高频细节强化"""

    def __init__(self, channel):
        super().__init__()
        self.high_pass = nn.Conv2d(channel, channel, 3, padding=1)  # 高通滤波器
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // 8, 1),
            nn.LeakyReLU(0.1),  # 与原始模块一致
            nn.Conv2d(channel // 8, channel, 1),
            nn.Softplus()  # 替换Sigmoid，允许大于1的增强
        )

    def forward(self, x):
        # 高通特征提取
        high = self.high_pass(x)
        # 自适应门控
        gate = self.gate(x)
        return x + high * gate  # 自适应增强高频

class ChannelWiseAttention(nn.Module):#输入的是64*3维度，池化，mlp及卷积处理
    def __init__(self, channel, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel // ratio),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_out = self.mlp(self.max_pool(x).squeeze(-1).squeeze(-1))
        channel_weights = self.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)
        return x * channel_weights

class MultiScaleFusion(nn.Module):#最后一步mlp后的处理
    def __init__(self, dim):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.conv = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, x):
        original_size = x.shape[2:]
        x1 = self.down1(x)#先把图像缩小一半（Stride=2），这样卷积核看到的范围就大了一倍，能捕捉到更大的结构信息
        # 动态上采样到原始尺寸
        x1 = F.interpolate(x1, size=original_size, mode='bilinear', align_corners=True)# 再把缩小的图像拉回到原始尺寸。
        return self.conv(torch.cat([x, x1], dim=1))#把原始细节和刚才那个“大范围视野”的特征拼在一起。在1*1卷积，最后频域输出特征




class EnhancedFuse(nn.Module): #总流程
    def __init__(self, embed_dim=64):
        super().__init__()
        # 增强型幅度谱融合（宽通道+残差）
        self.amp_fuse = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.LeakyReLU(0.1),
            ResidualConvBlock(16),
            ChannelAttention(16),
            nn.Conv2d(16, 1, 1)  # 最终通道压缩到1
        )

        # 相位谱融合（带可学习补偿）
        self.pha_fuse = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.ReLU(),
            ResidualConvBlock(32),
            nn.Conv2d(32, 1, 1),
            PhaseCompensation()  # 新增相位补偿
        )

        # 改进的逆变换处理
        self.ifft_block = IFFT_Block(out_channels=8)
        self.freq_channel_adjust = nn.Conv2d(8, 64, 1)  # 新增与原始模块一致的线性映射
        self.ifft_conv = nn.Sequential(
            nn.Conv2d(64, embed_dim, 3, padding=1),
            DepthwiseSeparableConv(embed_dim),  # 深度可分离卷积
            LayerNorm(embed_dim, "WithBias")
        )

        self.detail_enhance = DynamicDetailEnhance(embed_dim)

        # 图像特征升维模块（关键修正点）
        self.img_project = nn.Conv2d(1, embed_dim, 3, padding=1)  # 单通道->embed_dim

        # 多模态融合层（MSAF）
        self.fuse_block = nn.Sequential(
            ChannelWiseAttention(embed_dim * 3),  # # 第一步：给特征打分并加权
            nn.Conv2d(embed_dim * 3, embed_dim, 1), #sigmod后的3X3conve
            MultiScaleFusion(embed_dim)
        )

    def forward(self, ir, vi):
        # 傅里叶分解（保持单通道输出）
        B, C, H, W = ir.shape  # 获取输入图像的原尺寸

        # 确保H和W是偶数，避免FFT问题
        H_even = H if H % 2 == 0 else H - 1
        W_even = W if W % 2 == 0 else W - 1
        ir = F.interpolate(ir, size=(H_even, W_even), mode='bilinear', align_corners=True)
        vi = F.interpolate(vi, size=(H_even, W_even), mode='bilinear', align_corners=True)

        ir_amp, ir_pha = fft(ir)
        vi_amp, vi_pha = fft(vi)

        # 幅度谱融合
        amp_fused = self.amp_fuse(torch.cat([ir_amp, vi_amp], dim=1))  # [B,1,H,W]

        # 相位谱融合（残差连接）
        pha_fused = self.pha_fuse(torch.cat([ir_pha, vi_pha], dim=1)) + ir_pha  # [B,1,H,W]

        # 逆傅里叶变换流程
        freq_feat = self.ifft_block(amp_fused, pha_fused, H_even, W_even)
        freq_feat = self.freq_channel_adjust(freq_feat)  # 先线性映射
        freq_feat = self.ifft_conv(freq_feat)  # 再进行卷积处理

        # 图像特征升维（关键步骤）
        ir_feat = self.img_project(ir)  # [B,1,H,W] -> [B,64,H,W]
        vi_feat = self.img_project(vi)  # [B,1,H,W] -> [B,64,H,W]

        if freq_feat.shape[-2:] != (H_even, W_even):
            freq_feat = F.interpolate(freq_feat, size=(H_even, W_even), mode='bilinear', align_corners=True)

        # 多模态融合（通道数64+64+64=192）
        fused = torch.cat([ir_feat, vi_feat, freq_feat], dim=1)
        fused = self.fuse_block(fused)  # [B,64,H,W]
        return fused  # [B,64,H,W]






class Restormer_Encoder(nn.Module):
    def __init__(
        self,
        inp_channels=1,
        out_channels=1,
        dim=64,
        num_blocks=[4, 4],
        heads=[8, 8, 8],
        ffn_expansion_factor=2,
        bias=False,
        LayerNorm_type="WithBias",
    ):

        super(Restormer_Encoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )
        self.baseFeature = BaseFeatureExtractor(dim=dim, num_heads=heads[2])
        self.detailFeature = DetailFeatureExtractor()

    def forward(self, inp_img):

        # 将输入图像进行补丁嵌入
        inp_enc_level1 = self.patch_embed(inp_img)#torch.Size([1,1, 128, 128])

        # 使用第一级编码器处理嵌入后的输入
        out_enc_level1 = self.encoder_level1(inp_enc_level1) #torch.Size([1, 64, 128, 128])

        # 提取基础特征
        base_feature = self.baseFeature(out_enc_level1)#torch.Size([1, 64, 128, 128])

        # 提取细节特征
        detail_feature = self.detailFeature(out_enc_level1)#torch.Size([1, 64, 128, 128])

        # 返回基础特征、细节特征和第一级编码器的输出
        return base_feature, detail_feature, out_enc_level1


class Restormer_Decoder(nn.Module):
    def __init__(
        self,
        inp_channels=1,
        out_channels=1,
        dim=64,
        num_blocks=[4, 4],
        heads=[8, 8, 8],
        ffn_expansion_factor=2,
        bias=False,
        LayerNorm_type="WithBias",
    ):

        super(Restormer_Decoder, self).__init__()
        self.reduce_channel = nn.Conv2d(
            int(dim * 2), int(dim), kernel_size=1, bias=bias
        )
        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )
        self.output = nn.Sequential(
            nn.Conv2d(
                int(dim), int(dim) // 2, kernel_size=3, stride=1, padding=1, bias=bias
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                int(dim) // 2,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            ),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp_img, base_feature, detail_feature):
        #base_feature: torch.Size([1, 64, 128, 128])

        # 合并基础特征和细节特征，以进行特征维度的初步处理
        out_enc_level0 = torch.cat((base_feature, detail_feature), dim=1)#([1, 128, 128, 128])

        # 1*1卷积输入[B, 2*dim, H, W]，输出[B, dim, H, W]
        out_enc_level0 = self.reduce_channel(out_enc_level0)#([1, 64, 128, 128])

        # 使用多个 Transformer 块，维度保持为 [B, dim, H, W]
        out_enc_level1 = self.encoder_level2(out_enc_level0)#([1, 64, 128, 128])


        # 根据inp_img是否为空，决定是否将output模块的输出与inp_img相加
        if inp_img is not None:
            out_enc_level1 = self.output(out_enc_level1) + inp_img
        else:
            out_enc_level1 = self.output(out_enc_level1)
        # 使用sigmoid激活函数处理输出，并返回与out_enc_level0的特征图
        return self.sigmoid(out_enc_level1), out_enc_level0

class Restormer_Decoder1(nn.Module):
    def __init__(
        self,
        inp_channels=1,
        out_channels=1,
        dim=64,
        num_blocks=[4, 4],
        heads=[8, 8, 8],
        ffn_expansion_factor=2,
        bias=False,
        LayerNorm_type="WithBias",
    ):

        super(Restormer_Decoder1, self).__init__()
        self.reduce_channel = nn.Conv2d(
            int(dim * 3), int(dim), kernel_size=1, bias=bias
        )
        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )
        self.output = nn.Sequential(
            nn.Conv2d(
                int(dim), int(dim) // 2, kernel_size=3, stride=1, padding=1, bias=bias
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                int(dim) // 2,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            ),
        )
        self.sigmoid = nn.Sigmoid()
        self.freq_fuse = EnhancedFuse(embed_dim=64)

    def forward(self, inp_img, inp_img1, base_feature, detail_feature):
        # 确保 base_feature 和 detail_feature 的形状一致
        assert base_feature.shape == detail_feature.shape, "base_feature and detail_feature must have the same shape"

        # 合并基础特征和细节特征，以进行特征维度的初步处理
        fre= self.freq_fuse(inp_img, inp_img1)

        # 确保 fre 的形状与 base_feature 和 detail_feature 一致
        if fre.shape[2:] != base_feature.shape[2:]:
            fre = F.interpolate(fre, size=base_feature.shape[2:], mode='bilinear', align_corners=True)

        out_enc_level0 = torch.cat((base_feature, detail_feature, fre), dim=1)

        # 1*1卷积输入[B, 3*dim, H, W]，输出[B, dim, H, W]
        out_enc_level0 = self.reduce_channel(out_enc_level0)

        # 使用多个 Transformer 块，维度保持为 [B, dim, H, W]
        out_enc_level1 = self.encoder_level2(out_enc_level0)

        # 根据inp_img是否为空，决定是否将output模块的输出与inp_img相加
        if inp_img is not None:
            out_enc_level1 = self.output(out_enc_level1) + inp_img
        else:
            out_enc_level1 = self.output(out_enc_level1)
        # 使用sigmoid激活函数处理输出，并返回与out_enc_level0的特征图
        return self.sigmoid(out_enc_level1), out_enc_level0

if __name__ == "__main__":
    height = 128
    width = 128
    window_size = 8
    modelE = Restormer_Encoder().cuda()
    modelD = Restormer_Decoder().cuda()
