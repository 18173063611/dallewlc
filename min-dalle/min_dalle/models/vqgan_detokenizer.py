import torch
from torch import nn
from torch import FloatTensor, LongTensor
from math import sqrt


class ResnetBlock(nn.Module):
    def __init__(self, log2_count_in: int, log2_count_out: int):
        super().__init__()
        m, n = 2 ** log2_count_in, 2 ** log2_count_out  # 讨论情况中 log2_count_in = 9  log2_count_out = 9 
        # 因此m = 512, n = 512
        self.is_middle = m == n
        self.norm1 = nn.GroupNorm(2 ** 5, m)
        self.conv1 = nn.Conv2d(m, n, 3, padding=1)
        self.norm2 = nn.GroupNorm(2 ** 5, n)
        self.conv2 = nn.Conv2d(n, n, 3, padding=1)
        if not self.is_middle:
            self.nin_shortcut = nn.Conv2d(m, n, 1)

    def forward(self, x: FloatTensor) -> FloatTensor:
        '''
        x 形如 (16, 512, 16, 16)
        '''
        h = x
        h = self.norm1.forward(h)  # shape不变
        h *= torch.sigmoid(h)
        h = self.conv1.forward(h)  # 讨论情况中shape不变
        h = self.norm2.forward(h)  # 讨论情况中shape不变
        h *= torch.sigmoid(h)
        h = self.conv2(h)  # 讨论情况中shape不变
        if not self.is_middle:
            x = self.nin_shortcut.forward(x)  # 如果 m != n 则进行处理
        return x + h


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        n = 2 ** 9
        self.norm = nn.GroupNorm(2 ** 5, n)
        self.q = nn.Conv2d(n, n, 1)
        self.k = nn.Conv2d(n, n, 1)
        self.v = nn.Conv2d(n, n, 1)
        self.proj_out = nn.Conv2d(n, n, 1)

    def forward(self, x: FloatTensor) -> FloatTensor:
        '''
        x 形如 (16, 512, 16, 16)
        '''
        n, m = 2 ** 9, x.shape[0]
        h = x
        h = self.norm(h)
        k = self.k.forward(h)
        v = self.v.forward(h)
        q = self.q.forward(h)
        k = k.reshape(m, n, -1)
        v = v.reshape(m, n, -1)
        q = q.reshape(m, n, -1)
        q = q.permute(0, 2, 1)
        w = torch.bmm(q, k)
        w /= n ** 0.5
        w = torch.softmax(w, dim=2)
        w = w.permute(0, 2, 1)
        h = torch.bmm(v, w)
        token_count = int(sqrt(h.shape[-1]))
        h = h.reshape(m, n, token_count, token_count)
        h = self.proj_out.forward(h)
        return x + h


class MiddleLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_1 = ResnetBlock(9, 9)
        self.attn_1 = AttentionBlock()
        self.block_2 = ResnetBlock(9, 9)
    
    def forward(self, h: FloatTensor) -> FloatTensor:
        '''
        h形如 (16, 512, 16, 16)
        '''
        h = self.block_1.forward(h)  # 讨论情况中shape不变
        h = self.attn_1.forward(h)
        h = self.block_2.forward(h)
        return h


class Upsample(nn.Module):
    def __init__(self, log2_count):
        super().__init__()
        n = 2 ** log2_count
        self.upsample = torch.nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = nn.Conv2d(n, n, 3, padding=1)

    def forward(self, x: FloatTensor) -> FloatTensor:
        x = self.upsample.forward(x.to(torch.float32))
        x = self.conv.forward(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(
        self, 
        log2_count_in: int, 
        log2_count_out: int, 
        has_attention: bool, 
        has_upsample: bool
    ):
        super().__init__()
        self.has_attention = has_attention
        self.has_upsample = has_upsample
        
        self.block = nn.ModuleList([
            ResnetBlock(log2_count_in, log2_count_out),
            ResnetBlock(log2_count_out, log2_count_out),
            ResnetBlock(log2_count_out, log2_count_out)
        ])

        if has_attention:
            self.attn = nn.ModuleList([
                AttentionBlock(),
                AttentionBlock(),
                AttentionBlock()
            ])

        if has_upsample:
            self.upsample = Upsample(log2_count_out)


    def forward(self, h: FloatTensor) -> FloatTensor:
        for j in range(3):
            h = self.block[j].forward(h)
            if self.has_attention:
                h = self.attn[j].forward(h)
        if self.has_upsample:
            h = self.upsample.forward(h)
        return h


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_in = nn.Conv2d(2 ** 8, 2 ** 9, 3, padding=1)  # channel从256到512 
        self.mid = MiddleLayer()

        self.up = nn.ModuleList([
            UpsampleBlock(7, 7, False, False),
            UpsampleBlock(8, 7, False, True),
            UpsampleBlock(8, 8, False, True),
            UpsampleBlock(9, 8, False, True),
            UpsampleBlock(9, 9, True, True)
        ])

        self.norm_out = nn.GroupNorm(2 ** 5, 2 ** 7)
        self.conv_out = nn.Conv2d(2 ** 7, 3, 3, padding=1)

    def forward(self, z: FloatTensor) -> FloatTensor:
        ''' 
        z 形如(16, 256, 16, 16)  后面两个16是图片TOKEN256的分解
        '''
        z = self.conv_in.forward(z)  # 处理后z形如(16, 512, 16, 16)
        z = self.mid.forward(z)  # 讨论情况中shape不变

        for i in reversed(range(5)):
            z = self.up[i].forward(z)

        z = self.norm_out.forward(z)  # 处理后z形如(16, 128, 256, 256)
        z *= torch.sigmoid(z)
        z = self.conv_out.forward(z) # 处理后z形如(16, 3, 256, 256) 3是RGB通道
        return z


class VQGanDetokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        vocab_count, embed_count = 2 ** 14, 2 ** 8
        self.vocab_count = vocab_count
        self.embedding = nn.Embedding(vocab_count, embed_count)  # 16384,  256
        self.post_quant_conv = nn.Conv2d(embed_count, embed_count, 1)
        self.decoder = Decoder()

    def forward(self, is_seamless: bool, z: LongTensor) -> FloatTensor:
        ''' 
        z: LongTensor, shape = (batch_size, IMAGE_TOKEN_COUNT)  形如(16, 256)
        '''
        grid_size = int(sqrt(z.shape[0]))
        token_count = grid_size * 2 ** 4    # 例如4 * 2 ** 4 = 64
        
        if is_seamless:
            z = z.view([grid_size, grid_size, 2 ** 4, 2 ** 4])  # 形如(4, 4, 16, 16)
            z = z.flatten(1, 2).transpose(1, 0).flatten(1, 2)  # (4, 64, 16) -> (64, 4, 16) -> (64, 64)
            z = z.flatten().unsqueeze(1) # (64, 64) -> (4096, 1)
            z = self.embedding.forward(z)
            z = z.view((1, token_count, token_count, 2 ** 8))  # 形如(1, 64, 64, 256)
        else:
            z = self.embedding.forward(z)   # 经过Embedding后shape形如(16, 256, 256)
            z = z.view((z.shape[0], 2 ** 4, 2 ** 4, 2 ** 8))  # 形如(16, 16, 16, 256)

        z = z.permute(0, 3, 1, 2).contiguous()  # not is_seamless情况下形如(16, 256, 16, 16)
        z = self.post_quant_conv.forward(z) # shape不变形如(16, 256, 16, 16)
        z = self.decoder.forward(z)    # 经过decoder后shape形如(16, 3, 256, 256)
        z = z.permute(0, 2, 3, 1)    # permute后形如(16, 256, 256, 3)
        z = z.clip(0.0, 1.0) * 255   # 数值剪裁到0-255之间

        if is_seamless:
            z = z[0]
        else:
            z = z.view([grid_size, grid_size, 2 ** 8, 2 ** 8, 3])  # 形如(4, 4, 256, 256, 3)
            z = z.flatten(1, 2).transpose(1, 0).flatten(1, 2)  # 处理后形如(1024, 1024, 3)
            # 先(4, 4, 256, 256, 3) -> (4, 1024, 256, 3) -> (1024, 4, 256, 3) -> (1024, 1024, 3)
        return z
