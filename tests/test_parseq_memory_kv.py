from types import SimpleNamespace

import torch
from torch import nn

from yomitoku.models.layers.parseq_transformer import Decoder


def test_projected_memory_matches_multihead_attention():
    torch.manual_seed(0)
    cfg = SimpleNamespace(embed_dim=32, num_heads=4, mlp_ratio=2, depth=2)
    decoder = Decoder(norm=nn.LayerNorm(cfg.embed_dim), cfg=cfg).eval()

    query = torch.randn(2, 1, cfg.embed_dim)
    content = torch.randn(2, 3, cfg.embed_dim)
    memory = torch.randn(2, 10, cfg.embed_dim)
    content_mask = torch.triu(torch.ones(3, 3, dtype=torch.bool), 1)
    query_mask = content_mask[2:]

    with torch.inference_mode():
        expected = decoder(query, content, memory, query_mask, content_mask)
        actual = decoder(
            query,
            content,
            memory,
            query_mask,
            content_mask,
            memory_kv=decoder.project_memory(memory),
        )

    torch.testing.assert_close(actual, expected)
