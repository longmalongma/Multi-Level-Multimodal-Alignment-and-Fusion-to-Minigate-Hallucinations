
import numpy as np
import torch
from torch import distributed as dist, nn as nn
import torch.nn.functional as F
from lavis.models.blip2_models.Qformer import BertEmbeddings, BertLMHeadModel
from transformers import BertTokenizer, BertConfig
import torch.distributed as dist

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False,
):
    if use_horovod:
        assert hvd is not None, "Please install horovod"
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(
                    all_image_features.chunk(world_size, dim=0)
                )
                gathered_text_features = list(
                    all_text_features.chunk(world_size, dim=0)
                )
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(
                torch.distributed.nn.all_gather(image_features), dim=0
            )
            all_text_features = torch.cat(
                torch.distributed.nn.all_gather(text_features), dim=0
            )
        else:
            gathered_image_features = [
                torch.zeros_like(image_features) for _ in range(world_size)
            ]
            gathered_text_features = [
                torch.zeros_like(text_features) for _ in range(world_size)
            ]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                self.local_loss,
                self.gather_with_grad,
                self.rank,
                self.world_size,
                self.use_horovod,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = (
                        logit_scale * all_image_features @ all_text_features.T
                )
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (
                             F.cross_entropy(logits_per_image, labels,label_smoothing=0.1)
                             + F.cross_entropy(logits_per_text, labels,label_smoothing=0.1)
                     ) / 2

        if total_loss == .0:
            print(f"logits_per_text:{logits_per_text}")
            print(f"logits_per_image:{logits_per_image}")
            # time.sleep(20)

        return total_loss


class Config:
    def __init__(self):
        self.vocab_size = 30522  # 例如，BERT 的词汇表大小
        self.hidden_size = 768  # 隐藏层的维度
        self.pad_token_id = 0  # 填充 token 的 ID
        self.max_position_embeddings = 512  # 最大位置嵌入的数量
        self.layer_norm_eps = 1e-12  # 层归一化的 epsilon 值
        self.hidden_dropout_prob = 0.1  # Dropout 概率


class T1(nn.Module):
    def __init__(self, config, way="avgpool", max_length=10,aim_hidden_size = 768, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if way not in ["linear", "avgpool"]:
            raise ValueError(f"way={way} is invalid input")
        self.aim_hidden_size = aim_hidden_size
        self.way = way
        self.max_length = max_length

        if way == "linear":
            # 线性投影
            self.text_projection = nn.Parameter(torch.empty(max_length * config.hidden_size, aim_hidden_size))
            self.qformer_projection = nn.Parameter(torch.empty(32 * 768, aim_hidden_size))
        elif way == "avgpool":
            self.text_projection = nn.Parameter(torch.empty(config.hidden_size, aim_hidden_size))
            self.qformer_projection = nn.Parameter(torch.empty(aim_hidden_size, aim_hidden_size))

        self.loss = ClipLoss(
            # world_size=dist.get_world_size(),
            # rank=dist.get_rank(),
            local_loss=False,
            gather_with_grad=False,
            use_horovod=False,
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # 获取设备类型
        self.device =torch.device(f"cuda:{dist.get_rank()}")  # 根据 local_rank 获取设备
        self.init_parameters()

    def init_parameters(self):
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))
        # 线性投影
        if self.way == "linear":
            nn.init.normal_(self.text_projection, std=(self.max_length * self.aim_hidden_size) ** -0.5)
            nn.init.normal_(self.qformer_projection, std=(32 * self.aim_hidden_size) ** -0.5)
        elif self.way == "avgpool":
            nn.init.normal_(self.text_projection, std=self.aim_hidden_size ** -0.5)
            nn.init.normal_(self.qformer_projection, std=self.aim_hidden_size ** -0.5)

    def align_qformer_features(self, qformer_features: torch.Tensor) -> torch.Tensor:
        """
        对齐后的 qformer 特征，保持与 T1 forward 中的处理方式一致。
        返回形状：[B, aim_hidden_size]，后续可按需扩展。
        """
        if self.way == "linear":
            aligned = qformer_features.reshape(qformer_features.shape[0], -1) @ self.qformer_projection
        else:
            aligned = qformer_features.mean(dim=1) @ self.qformer_projection

        return F.normalize(aligned, dim=-1)

    def forward(self, att_mask,text_features, qformer_features):
        if self.way == "linear":
            # print("linear")
            # text线性投影
            text_features = text_features.reshape(text_features.shape[0], -1)
            text_features = text_features @ self.text_projection
            # qformer线性投影
            qformer_features = qformer_features.reshape(qformer_features.shape[0], -1).to(self.device)
            qformer_features = qformer_features @ self.qformer_projection

        elif self.way == "avgpool":
            # print("avgpool")
            # 平均池化
            hidden_states = text_features
            mask_expanded = att_mask.unsqueeze(-1).expand_as(hidden_states)
            sum_hidden_states = (hidden_states * mask_expanded).sum(dim=1)
            text_features = sum_hidden_states / mask_expanded.sum(dim=1)
            qformer_features = qformer_features.mean(dim=1)
            # proj
            text_features = text_features @ self.text_projection
            qformer_features = qformer_features @ self.qformer_projection

        text_features = F.normalize(text_features, dim=-1)
        qformer_features = F.normalize(qformer_features, dim=-1)

        # print(f"text_features2:{text_features}")
        # print(f"qformer_features2:{qformer_features}")

        loss = self.loss(qformer_features, text_features, self.logit_scale.exp())

        # print(f"clip_one_loss:{loss}")
        # time.sleep(5)

        return loss


