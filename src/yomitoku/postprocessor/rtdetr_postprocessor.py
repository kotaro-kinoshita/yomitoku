# Copyright 2023 lyuwenyu
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


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def mod(a, b):
    out = a - a // b * b
    return out


class RTDETRPostProcessor(nn.Module):
    __share__ = [
        "num_classes",
        "use_focal_loss",
        "num_top_queries",
        "remap_mscoco_category",
    ]

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        remap_mscoco_category=False,
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False

    def extra_repr(self) -> str:
        return f"use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}"

    def clamp(self, boxes, h, w):
        boxes[:, 0] = torch.clamp(boxes[:, 0], min=torch.Tensor([0]), max=None)
        boxes[:, 1] = torch.clamp(boxes[:, 1], min=torch.Tensor([0]), max=None)
        boxes[:, 2] = torch.clamp(boxes[:, 2], min=torch.Tensor([0]), max=w)
        boxes[:, 3] = torch.clamp(boxes[:, 3], min=torch.Tensor([0]), max=h)
        return boxes

    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes: torch.Tensor, threshold):
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        w, h = orig_target_sizes.unbind(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            # TODO for older tensorrt
            # labels = index % self.num_classes
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(
                dim=1,
                index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]),
            )

        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(
                    boxes,
                    dim=1,
                    index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]),
                )

        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, scores

        # TODO
        if self.remap_mscoco_category:
            from ...data.dataset import mscoco_label2category

            labels = (
                torch.tensor(
                    [mscoco_label2category[int(x.item())] for x in labels.flatten()]
                )
                .to(boxes.device)
                .reshape(labels.shape)
            )

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            lab = lab[sco > threshold]
            box = box[sco > threshold]
            sco = sco[sco > threshold]

            lab = lab.cpu().numpy()
            sco = sco.cpu().numpy()

            box = self.clamp(box.cpu(), h.cpu(), w.cpu()).numpy()

            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)

        return results

    def deploy(
        self,
    ):
        self.eval()
        self.deploy_mode = True
        return self
