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

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


class BaseTokenizer(ABC):
    def __init__(
        self,
        charset: str,
        specials_first: tuple = (),
        specials_last: tuple = (),
    ) -> None:
        self._itos = specials_first + tuple(charset) + specials_last
        self._stoi = {s: i for i, s in enumerate(self._itos)}

    def __len__(self):
        return len(self._itos)

    def _tok2ids(self, tokens: str) -> list[int]:
        return [self._stoi[s] for s in tokens]

    def _ids2tok(self, token_ids: list[int], join: bool = True) -> str:
        tokens = [self._itos[i] for i in token_ids]
        return "".join(tokens) if join else tokens

    @abstractmethod
    def encode(
        self, labels: list[str], device: Optional[torch.device] = None
    ) -> Tensor:
        """Encode a batch of labels to a representation suitable for the model.

        Args:
            labels: List of labels. Each can be of arbitrary length.
            device: Create tensor on this device.

        Returns:
            Batched tensor representation padded to the max label length. Shape: N, L
        """
        raise NotImplementedError

    @abstractmethod
    def _filter(self, probs: Tensor, ids: Tensor) -> tuple[Tensor, list[int]]:
        """Internal method which performs the necessary filtering prior to decoding."""
        raise NotImplementedError

    def decode(
        self, token_dists: Tensor, raw: bool = False
    ) -> tuple[list[str], list[Tensor]]:
        """Decode a batch of token distributions.

        Args:
            token_dists: softmax probabilities over the token distribution. Shape: N, L, C
            raw: return unprocessed labels (will return list of list of strings)

        Returns:
            list of string labels (arbitrary length) and
            their corresponding sequence probabilities as a list of Tensors
        """
        batch_tokens = []
        batch_probs = []
        for dist in token_dists:
            probs, ids = dist.max(-1)  # greedy selection
            if not raw:
                probs, ids = self._filter(probs, ids)
            tokens = self._ids2tok(ids, not raw)
            probs = probs.cpu().numpy()
            probs = float(probs.prod())
            batch_tokens.append(tokens)
            batch_probs.append(probs)
        return batch_tokens, batch_probs


class ParseqTokenizer(BaseTokenizer):
    BOS = "[B]"
    EOS = "[E]"
    PAD = "[P]"

    def __init__(self, charset: str) -> None:
        specials_first = (self.EOS,)
        specials_last = (self.BOS, self.PAD)
        super().__init__(charset, specials_first, specials_last)
        self.eos_id, self.bos_id, self.pad_id = [
            self._stoi[s] for s in specials_first + specials_last
        ]

    def encode(
        self, labels: list[str], device: Optional[torch.device] = None
    ) -> Tensor:
        batch = [
            torch.as_tensor(
                [self.bos_id] + self._tok2ids(y) + [self.eos_id],
                dtype=torch.long,
                device=device,
            )
            for y in labels
        ]
        return pad_sequence(batch, batch_first=True, padding_value=self.pad_id)

    def _filter(self, probs: Tensor, ids: Tensor) -> tuple[Tensor, list[int]]:
        ids = ids.tolist()
        try:
            eos_idx = ids.index(self.eos_id)
        except ValueError:
            eos_idx = len(ids)  # Nothing to truncate.
        # Truncate after EOS
        ids = ids[:eos_idx]
        probs = probs[
            : eos_idx + 1
        ]  # but include prob. for EOS (if it exists)
        return probs, ids
