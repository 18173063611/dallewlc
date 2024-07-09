from typing import Tuple, List
import torch
from torch import nn, LongTensor, FloatTensor, BoolTensor
from .dalle_bart_encoder import GLU, AttentionBase

IMAGE_TOKEN_COUNT = 256


class DecoderCrossAttention(AttentionBase):
    def forward(
        self,
        decoder_state: FloatTensor,
        encoder_state: FloatTensor,
        attention_mask: BoolTensor
    ) -> FloatTensor:
        keys = self.k_proj.forward(encoder_state)
        values = self.v_proj.forward(encoder_state)
        queries = self.q_proj.forward(decoder_state)    # 这里keys values queries的布置和原版本Transformer一致
        return super().forward(keys, values, queries, attention_mask)


class DecoderSelfAttention(AttentionBase):
    def __init__(self, head_count: int, embed_count: int):
        super().__init__(head_count, embed_count)

    def forward(
        self, 
        decoder_state: FloatTensor,
        attention_state: FloatTensor,   # 形如(64, 256, 2048)  (24, 64, 256, 2048)其中的一个
        attention_mask: BoolTensor,     # 形如(32, 1, 1, 64)
        token_index: LongTensor         # 形如(32, 1)  在目前讨论情况下应该是一样的数
    ) -> Tuple[FloatTensor, FloatTensor]:
        keys = self.k_proj.forward(decoder_state)    # 形如(32, 1, 2048)
        values = self.v_proj.forward(decoder_state)  # 形如(32, 1, 2048)
        queries = self.q_proj.forward(decoder_state) # 形如(32, 1, 2048)
        
        token_count = token_index.shape[1]
        if token_count == 1:
            batch_count = decoder_state.shape[0]
            attn_state_new = torch.cat([keys, values]).to(attention_state.dtype)   # 形如(64, 1, 2048)  
                                                                                # 跟之前定义attention_state的时候对上了,keys和values的第一个维度都是两个images数，加起来一共4个images数
            attention_state[:, token_index[0]] = attn_state_new  # 将(64, 256, 2048)中的第二个维度的第token_index个位置替换为attn_state_new
            keys = attention_state[:batch_count]  # 取出前batch_count个，讨论情况中是32
            values = attention_state[batch_count:]  # 取出后batch_count个，讨论情况中是32

        decoder_state = super().forward(keys, values, queries, attention_mask)
        return decoder_state, attention_state


class DecoderLayer(nn.Module):
    def __init__(
        self, 
        head_count: int, 
        embed_count: int,
        glu_embed_count: int,
        device: str
    ):
        super().__init__()
        self.pre_self_attn_layer_norm = nn.LayerNorm(embed_count)
        self.self_attn = DecoderSelfAttention(head_count, embed_count)
        self.self_attn_layer_norm = nn.LayerNorm(embed_count)
        self.pre_encoder_attn_layer_norm = nn.LayerNorm(embed_count)
        self.encoder_attn = DecoderCrossAttention(head_count, embed_count)
        self.encoder_attn_layer_norm = nn.LayerNorm(embed_count)
        self.glu = GLU(embed_count, glu_embed_count)
        self.token_indices = torch.arange(IMAGE_TOKEN_COUNT, device=device)


    def forward(
        self,
        decoder_state: FloatTensor,   # 形如（32，1，2048）image token的embeddings
        encoder_state: FloatTensor,   # 形如(32, 64, 2048)
        attention_state: FloatTensor,   # 形如(64, 256, 2048) 讨论情况中是24个中的一个
        attention_mask: BoolTensor,   # 形如(32, 1, 1, 64)
        token_index: LongTensor
    ) -> Tuple[FloatTensor, FloatTensor]:
        # Self Attention
        token_count = token_index.shape[1]
        if token_count == 1:
            self_attn_mask = self.token_indices <= token_index   # 形如(32, 256)  左边是一个(256,) 右边是一个(32, 1)
            self_attn_mask = self_attn_mask[:, None, None, :]    # 变形后形如(32, 1, 1, 256)
        else:
            self_attn_mask = (
                self.token_indices[None, None, :token_count] <= 
                token_index[:, :, None]
            )
            self_attn_mask = self_attn_mask[:, None, :, :]
        
        residual = decoder_state
        decoder_state = self.pre_self_attn_layer_norm.forward(decoder_state)
        decoder_state, attention_state = self.self_attn.forward(
            decoder_state=decoder_state,
            attention_state=attention_state,
            attention_mask=self_attn_mask,
            token_index=token_index
        )
        decoder_state = self.self_attn_layer_norm.forward(decoder_state)
        decoder_state = residual + decoder_state   # 残差连接

        # Cross Attention
        residual = decoder_state
        decoder_state = self.pre_encoder_attn_layer_norm.forward(decoder_state)
        decoder_state = self.encoder_attn.forward(
            decoder_state=decoder_state,
            encoder_state=encoder_state,
            attention_mask=attention_mask
        )   # 这里涉及到与encoder的交互，使用了encoder_state来应用encoder的产出的信息
        decoder_state = self.encoder_attn_layer_norm.forward(decoder_state)
        decoder_state = residual + decoder_state

        # Feed forward
        residual = decoder_state
        decoder_state = self.glu.forward(decoder_state)
        decoder_state = residual + decoder_state

        return decoder_state, attention_state


class DalleBartDecoder(nn.Module):
    def __init__(
        self,
        image_vocab_count: int,
        embed_count: int,
        attention_head_count: int,
        glu_embed_count: int,
        layer_count: int,
        device: str
    ):
        super().__init__()
        self.layer_count = layer_count
        self.embed_count = embed_count
        self.image_vocab_count = image_vocab_count
        self.embed_tokens = nn.Embedding(image_vocab_count + 1, embed_count)
        self.embed_positions = nn.Embedding(IMAGE_TOKEN_COUNT, embed_count)
        self.layers: List[DecoderLayer] = nn.ModuleList([
            DecoderLayer(
                head_count=attention_head_count,
                embed_count=embed_count,
                glu_embed_count=glu_embed_count,
                device=device
            ) 
            for _ in range(layer_count)
        ])
        self.layernorm_embedding = nn.LayerNorm(embed_count)
        self.final_ln = nn.LayerNorm(embed_count)
        self.lm_head = nn.Linear(embed_count, image_vocab_count + 1, bias=False)  # 为何+1 ？？？
        self.token_indices = torch.arange(IMAGE_TOKEN_COUNT, device=device)


    def forward(
        self,
        attention_mask: BoolTensor,  # 形如(32, 1, 1, 64)
        encoder_state: FloatTensor,
        attention_state: FloatTensor,
        prev_tokens: LongTensor,
        token_index: LongTensor
    ) -> Tuple[FloatTensor, FloatTensor]:
        '''
        attention_mask: shape 形如(32, 1, 1, 64)
        encoder_state: shape 形如(32, 64, 2048)
        attention_state: shape 形如(24, 64, 256, 2048)  应该是做attention计算时每一层的K,V,Q的记录
        prev_tokens: shape 形如(16, 1)  所有生成的图片中的同一位置的token
        token_index: shape 形如(1, )  tokend的index   一个单独的数
        '''
        image_count = encoder_state.shape[0] // 2
        token_index = token_index.unsqueeze(0).repeat(image_count * 2, 1)  # token_index 形如(32, 1)  why？？？repeat
        prev_tokens = prev_tokens.repeat(2, 1) # prev_tokens 形如(32, 1)   ？？？why repeat
        decoder_state = self.embed_tokens.forward(prev_tokens)  # 创建时形如（32，1，2048） 从image token的index映射到embeddings
        decoder_state += self.embed_positions.forward(token_index)  # 添加位置信息
        decoder_state = self.layernorm_embedding.forward(decoder_state)  # 经过一系列处理后依然形如（32，1，2048）
        for i in range(self.layer_count):
            decoder_state, attention_state[i] = self.layers[i].forward(
                decoder_state,
                encoder_state,
                attention_state[i],
                attention_mask,
                token_index
            )
        decoder_state = self.final_ln(decoder_state)   # 经过final_ln前后shape不变，仍形如(32, 1, 2048)
        logits = self.lm_head(decoder_state) # logits 形如（32，1，16416）对于16415个image vocab的概率
        return logits, attention_state
    

    def sample_tokens(self, settings, **kwargs) -> Tuple[LongTensor, FloatTensor]:
        logits, attention_state = self.forward(**kwargs)   # logits 形如（32，1，16416)   attention_state 形如（24，64，256，2048）应该是做attention计算时每一层的K,V,Q的记录
        image_count = logits.shape[0] // 2
        temperature = settings[[0]]
        top_k = settings[[1]].to(torch.long)
        supercondition_factor = settings[[2]]
        logits = logits[:, -1, : 2 ** 14]   # 经过索引后形如（32，16384）  缩减了维度，第二个维度没有了
        logits: FloatTensor = (
            logits[:image_count] * (1 - supercondition_factor) + 
            logits[image_count:] * supercondition_factor
        )  # 分开前16和后16的含义？？？ supercondition_factor是什么？？？
        # 处理后的logits形如（16，16384）
        logits_sorted, _ = logits.sort(descending=True)  # 形如（16，16384）
        is_kept = logits >= logits_sorted[:, top_k - 1]   # 形如（16，16384）
        logits -= logits_sorted[:, [0]]  # 为什么要统一减去最大值？？？
        logits /= temperature    # 为什么要除以temperature？？？
        logits.exp_()   # 变成0-1之间的值
        logits *= is_kept.to(torch.float32)   # 使用is_kept过滤掉不需要的token
        image_tokens = torch.multinomial(logits, 1)[:, 0]  # image_tokens 形如（16，1） 从概率分布中采样得到 image tokens。
        return image_tokens, attention_state