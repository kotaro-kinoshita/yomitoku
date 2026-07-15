# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Optional

import torch
from timm.models.vision_transformer import PatchEmbed, VisionTransformer
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.modules import transformer


class DecoderLayer(nn.Module):
    """A Transformer decoder layer supporting two-stream attention (XLNet)
    This implements a pre-LN decoder, as opposed to the post-LN default in PyTorch.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_ratio=2048,
        dropout=0.1,
        activation="gelu",
        layer_norm_eps=1e-5,
    ):
        super().__init__()
        dim_feedforward = embed_dim * mlp_ratio
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm_q = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm_c = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = transformer._get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.gelu
        super().__setstate__(state)

    def forward_stream(
        self,
        tgt: Tensor,
        tgt_norm: Tensor,
        tgt_kv: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor],
        tgt_key_padding_mask: Optional[Tensor],
    ):
        """Forward pass for a single stream (i.e. content or query)
        tgt_norm is just a LayerNorm'd tgt. Added as a separate parameter for efficiency.
        Both tgt_kv and memory are expected to be LayerNorm'd too.
        memory is LayerNorm'd by ViT.
        """
        tgt2, sa_weights = self.self_attn(
            tgt_norm,
            tgt_kv,
            tgt_kv,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        tgt = tgt + self.dropout1(tgt2)

        tgt2, ca_weights = self.cross_attn(self.norm1(tgt), memory, memory)
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.linear2(
            self.dropout(self.activation(self.linear1(self.norm2(tgt))))
        )
        tgt = tgt + self.dropout3(tgt2)
        return tgt, sa_weights, ca_weights

    def forward(
        self,
        query,
        content,
        memory,
        query_mask: Optional[Tensor] = None,
        content_mask: Optional[Tensor] = None,
        content_key_padding_mask: Optional[Tensor] = None,
        update_content: bool = True,
    ):
        query_norm = self.norm_q(query)
        content_norm = self.norm_c(content)
        query = self.forward_stream(
            query,
            query_norm,
            content_norm,
            memory,
            query_mask,
            content_key_padding_mask,
        )[0]
        if update_content:
            content = self.forward_stream(
                content,
                content_norm,
                content_norm,
                memory,
                content_mask,
                content_key_padding_mask,
            )[0]
        return query, content


class Decoder(nn.Module):
    __constants__ = ["norm"]

    def __init__(self, norm, cfg):
        super().__init__()
        decoder_layer = DecoderLayer(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
        )

        self.layers = transformer._get_clones(decoder_layer, cfg.depth)
        self.num_layers = cfg.depth
        self.norm = norm

    def forward(
        self,
        query,
        content,
        memory,
        query_mask: Optional[Tensor] = None,
        content_mask: Optional[Tensor] = None,
        content_key_padding_mask: Optional[Tensor] = None,
    ):
        for i, mod in enumerate(self.layers):
            last = i == len(self.layers) - 1
            query, content = mod(
                query,
                content,
                memory,
                query_mask,
                content_mask,
                content_key_padding_mask,
                update_content=not last,
            )
        query = self.norm(query)
        return query


class Encoder(VisionTransformer):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        embed_layer=PatchEmbed,
    ):
        super().__init__(
            img_size,
            patch_size,
            in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            embed_layer=embed_layer,
            num_classes=0,  # These
            global_pool="",  # disable the
            class_token=False,  # classifier head.
        )

    def forward(self, x):
        if x.shape[-2:] == tuple(self.patch_embed.img_size):
            # Return all tokens
            return self.forward_features(x)
        return self.forward_features_dynamic(x)

    def forward_features_dynamic(self, x):
        """Forward pass for inputs narrower than the training canvas.

        Patchifies directly (bypassing PatchEmbed's strict size check) and
        adds the learned absolute position embedding cropped to the actual
        patch grid, so each patch keeps the same position embedding it had
        at training time.
        """
        x = self.patch_embed.proj(x)  # B, D, gh, gw
        gh, gw = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)  # B, N, D
        x = self.patch_embed.norm(x)

        full_gh, full_gw = self.patch_embed.grid_size
        pos_embed = self.pos_embed.reshape(1, full_gh, full_gw, -1)[:, :gh, :gw]
        x = x + pos_embed.reshape(1, gh * gw, -1)

        x = self.pos_drop(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x


class TokenEmbedding(nn.Module):
    def __init__(self, charset_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(charset_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, tokens: torch.Tensor):
        return math.sqrt(self.embed_dim) * self.embedding(tokens)
