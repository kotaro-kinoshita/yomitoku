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

from functools import partial
from typing import Optional, Sequence

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from timm.models.helpers import named_apply
from torch import Tensor

from .layers.parseq_transformer import Decoder, Encoder, TokenEmbedding


def init_weights(module: nn.Module, name: str = "", exclude: Sequence[str] = ()):
    """Initialize the weights using the typical initialization schemes used in SOTA models."""
    if any(map(name.startswith, exclude)):
        return
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class PARSeq(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        cfg,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.max_label_length = self.cfg.max_label_length
        self.decode_ar = self.cfg.decode_ar
        self.refine_iters = self.cfg.refine_iters
        embed_dim = self.cfg.decoder.embed_dim

        self.encoder = Encoder(
            self.cfg.data.img_size,
            **self.cfg.encoder,
        )

        self.decoder = Decoder(
            norm=nn.LayerNorm(self.cfg.decoder.embed_dim),
            cfg=self.cfg.decoder,
        )

        # We don't predict <bos> nor <pad>
        self.head = nn.Linear(embed_dim, self.cfg.num_tokens - 2)
        self.text_embed = TokenEmbedding(self.cfg.num_tokens, embed_dim)

        # +1 for <eos>
        self.pos_queries = nn.Parameter(
            torch.Tensor(1, self.max_label_length + 1, embed_dim)
        )
        self.dropout = nn.Dropout()
        # Encoder has its own init.
        named_apply(partial(init_weights, exclude=["encoder"]), self)
        nn.init.trunc_normal_(self.pos_queries, std=0.02)

        self.export_onnx = False

        # Decode-time repetition early-stop. When the AR decoder enters a
        # degenerate loop (a period-p token unit repeated many times, the
        # "EOS-hallucination" failure mode), detect it online, collapse the
        # repeated run to a single unit and force-stop that sequence. This
        # removes the runaway tail at the source and skips the wasted AR
        # steps. Thresholds are deliberately conservative (period-1 needs a
        # long run) so genuine text is never truncated. Tunable via config.
        self.repetition_stop = bool(getattr(self.cfg, "repetition_stop", True))
        self.rep_period_max = int(getattr(self.cfg, "rep_period_max", 8))
        self.rep_min_run_p1 = int(getattr(self.cfg, "rep_min_run_p1", 8))
        self.rep_min_repeats = int(getattr(self.cfg, "rep_min_repeats", 3))

    @property
    def _device(self) -> torch.device:
        return next(self.head.parameters(recurse=False)).device

    @torch.jit.ignore
    def no_weight_decay(self):
        param_names = {"text_embed.embedding.weight", "pos_queries"}
        enc_param_names = {"encoder." + n for n in self.encoder.no_weight_decay()}
        return param_names.union(enc_param_names)

    def _detect_repeat_onset(self, seq):
        """Detect a trailing periodic loop in a token-id list.

        Returns (onset, period) where ``onset`` is the index of the first
        token of the repeated run, or None if no qualifying loop is found.
        A period-1 unit must repeat at least ``rep_min_run_p1`` times; a
        multi-token unit at least ``rep_min_repeats`` times. The smallest
        qualifying period wins (the fundamental unit).
        """
        n = len(seq)
        for p in range(1, self.rep_period_max + 1):
            if n < 2 * p:
                continue
            unit = seq[n - p : n]
            k, t = 1, n - p
            while t - p >= 0 and seq[t - p : t] == unit:
                k += 1
                t -= p
            if k >= (self.rep_min_run_p1 if p == 1 else self.rep_min_repeats):
                return t, p  # run starts at index t
        return None

    def encode(self, img: torch.Tensor):
        return self.encoder(img)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
        tgt_query: Optional[Tensor] = None,
        tgt_query_mask: Optional[Tensor] = None,
    ):
        N, L = tgt.shape
        # <bos> stands for the null context. We only supply position information for characters after <bos>.
        null_ctx = self.text_embed(tgt[:, :1])
        tgt_emb = self.pos_queries[:, : L - 1] + self.text_embed(tgt[:, 1:])
        tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1))
        if tgt_query is None:
            tgt_query = self.pos_queries[:, :L].expand(N, -1, -1)
        tgt_query = self.dropout(tgt_query)
        return self.decoder(
            tgt_query,
            tgt_emb,
            memory,
            tgt_query_mask,
            tgt_mask,
            tgt_padding_mask,
        )

    def forward(
        self,
        images: Tensor,
        max_length: Optional[int] = None,
    ) -> Tensor:
        testing = max_length is None
        max_length = (
            self.max_label_length
            if max_length is None
            else min(max_length, self.max_label_length)
        )
        bs = images.shape[0]
        # +1 for <eos> at end of sequence.
        num_steps = max_length + 1
        memory = self.encode(images)

        # Query positions up to `num_steps`
        pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)

        # Special case for the forward permutation. Faster than using `generate_attn_masks()`
        tgt_mask = query_mask = torch.triu(
            torch.ones((num_steps, num_steps), dtype=torch.bool, device=self._device),
            1,
        )

        # Per-sequence repetition early-stop bookkeeping. ``rep_cut`` holds the
        # truncation position (keep one copy of the repeated unit) for any
        # sequence detected to be looping; applied to the logits after decoding
        # (and after refinement) so the runaway tail is removed from the output.
        rep_on = self.repetition_stop and testing and not self.export_onnx
        rep_cut = [None] * bs

        if self.decode_ar:
            tgt_in = torch.full(
                (bs, num_steps),
                self.tokenizer.pad_id,
                dtype=torch.long,
                device=self._device,
            )
            tgt_in[:, 0] = self.tokenizer.bos_id

            rep_done = [False] * bs

            logits = []
            for i in range(num_steps):
                j = i + 1  # next token index
                # Efficient decoding:
                # Input the context up to the ith token. We use only one query
                # (at position = i) at a time. This works because of the
                # lookahead masking effect of the canonical (forward) AR
                # context. Past tokens have no access to future tokens, hence
                # are fixed once computed.
                tgt_out = self.decode(
                    tgt_in[:, :j],
                    memory,
                    tgt_mask[:j, :j],
                    tgt_query=pos_queries[:, i:j],
                    tgt_query_mask=query_mask[i:j, :j],
                )
                # the next token probability is in the output's ith token position
                p_i = self.head(tgt_out)
                logits.append(p_i)
                if j < num_steps:
                    # greedy decode. add the next token index to the target input
                    tgt_in[:, j] = p_i.squeeze().argmax(-1)

                    if rep_on:
                        # ``i`` is the position of the token just emitted.
                        for b in range(bs):
                            if rep_done[b] or int(tgt_in[b, j]) == self.tokenizer.eos_id:
                                continue
                            seq = tgt_in[b, 1 : j + 1].tolist()
                            hit = self._detect_repeat_onset(seq)
                            if hit is not None:
                                onset, period = hit
                                # keep the valid prefix + a single unit
                                rep_cut[b] = onset + period
                                rep_done[b] = True
                                # force EOS context so the batch can finish early
                                tgt_in[b, j] = self.tokenizer.eos_id

                    # Efficient batch decoding: If all output words have at least one EOS token, end decoding.
                    if (
                        not self.export_onnx
                        and testing
                        and (tgt_in == self.tokenizer.eos_id).any(dim=-1).all()
                    ):
                        break

            logits = torch.cat(logits, dim=1)
        else:
            # No prior context, so input is just <bos>. We query all positions.
            tgt_in = torch.full(
                (bs, 1),
                self.tokenizer.bos_id,
                dtype=torch.long,
                device=self._device,
            )
            tgt_out = self.decode(tgt_in, memory, tgt_query=pos_queries)
            logits = self.head(tgt_out)

        if self.refine_iters:
            # For iterative refinement, we always use a 'cloze' mask.
            # We can derive it from the AR forward mask by unmasking the token context to the right.
            query_mask[
                torch.triu(
                    torch.ones(
                        num_steps,
                        num_steps,
                        dtype=torch.int64,
                        device=self._device,
                    ),
                    2,
                )
            ] = 0
            bos = torch.full(
                (bs, 1),
                self.tokenizer.bos_id,
                dtype=torch.long,
                device=self._device,
            )
            for i in range(self.refine_iters):
                # Prior context is the previous output.
                tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                # Mask tokens beyond the first EOS token.
                tgt_padding_mask = (tgt_in == self.tokenizer.eos_id).int().cumsum(
                    -1
                ) > 0
                tgt_out = self.decode(
                    tgt_in,
                    memory,
                    tgt_mask,
                    tgt_padding_mask,
                    pos_queries,
                    query_mask[:, : tgt_in.shape[1]],
                )
                logits = self.head(tgt_out)

        if rep_on and any(c is not None for c in rep_cut):
            # Force EOS at the loop-onset position so the tokenizer truncates
            # the repeated tail. Done after refinement, which would otherwise
            # regenerate the run. EOS is class index ``eos_id`` of the head.
            eos_id = self.tokenizer.eos_id
            for b, cut in enumerate(rep_cut):
                if cut is not None and cut < logits.shape[1]:
                    logits[b, cut, :] = -30.0
                    logits[b, cut, eos_id] = 30.0

        return logits
