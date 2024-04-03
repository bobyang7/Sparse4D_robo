# Copyright (c) Horizon Robotics. All rights reserved.
from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn

from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)
from mmcv.runner import BaseModule, force_fp32
from mmcv.utils import build_from_cfg
from mmdet.core.bbox.builder import BBOX_SAMPLERS
from mmdet.core.bbox.builder import BBOX_CODERS
from mmdet.models import HEADS, LOSSES
from mmdet.core import reduce_mean

from .blocks import DeformableFeatureAggregation as DFG

__all__ = ["Sparse4DHead"]


@HEADS.register_module()
class Sparse4DHead(BaseModule):
    def __init__(
        self,
        instance_bank: dict,
        anchor_encoder: dict,
        k_encoder: dict,
        graph_model: dict,
        norm_layer: dict,
        ffn: dict,
        deformable_model: dict,
        refine_layer: dict,
        num_decoder: int = 6,
        num_single_frame_decoder: int = -1,
        temp_graph_model: dict = None,
        loss_cls: dict = None,
        loss_reg: dict = None,
        loss_kt: dict = None,
        decoder: dict = None,
        sampler: dict = None,
        gt_cls_key: str = "gt_labels_3d",
        gt_reg_key: str = "gt_bboxes_3d",
        reg_weights: List = None,
        operation_order: Optional[List[str]] = None,
        cls_threshold_to_reg: float = -1,
        dn_loss_weight: float = 5.0,
        decouple_attn: bool = True,
        kal: bool = False,
        init_cfg: dict = None,
        **kwargs,
    ):
        super(Sparse4DHead, self).__init__(init_cfg)
        self.num_decoder = num_decoder
        self.num_single_frame_decoder = num_single_frame_decoder
        self.gt_cls_key = gt_cls_key
        self.gt_reg_key = gt_reg_key
        self.cls_threshold_to_reg = cls_threshold_to_reg
        self.dn_loss_weight = dn_loss_weight
        self.decouple_attn = decouple_attn
        self.kal = kal

        if reg_weights is None:
            self.reg_weights = [1.0] * 10
        else:
            self.reg_weights = reg_weights

        if operation_order is None:
            operation_order = [
                "temp_gnn",
                "gnn",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_decoder
            # delete the 'gnn' and 'norm' layers in the first transformer blocks
            operation_order = operation_order[3:]
        self.operation_order = operation_order

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.instance_bank = build(instance_bank, PLUGIN_LAYERS)
        self.anchor_encoder = build(anchor_encoder, POSITIONAL_ENCODING)
        self.sampler = build(sampler, BBOX_SAMPLERS)
        self.decoder = build(decoder, BBOX_CODERS)
        self.loss_cls = build(loss_cls, LOSSES)
        self.loss_reg = build(loss_reg, LOSSES)
        self.loss_kt = build(loss_kt, LOSSES)
        self.op_config_map = {
            "temp_gnn": [temp_graph_model, ATTENTION],
            "gnn": [graph_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "deformable": [deformable_model, ATTENTION],
            "refine": [refine_layer, PLUGIN_LAYERS],
            "kalman": [k_encoder, PLUGIN_LAYERS],
        }
        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )
        self.embed_dims = self.instance_bank.embed_dims
        if self.decouple_attn:
            self.fc_before = nn.Linear(self.embed_dims, self.embed_dims * 2, bias=False)
            self.fc_after = nn.Linear(self.embed_dims * 2, self.embed_dims, bias=False)
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(
            self.layers[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        )

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
    ):
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        batch_size = feature_maps[0].shape[0]

        # ========= get instance info ============
        if (
            self.sampler.dn_metas is not None
            and self.sampler.dn_metas["dn_anchor"].shape[0] != batch_size
        ):
            self.sampler.dn_metas = None
        (
            instance_feature,
            anchor,
            temp_instance_feature,
            temp_anchor,
            time_interval,
        ) = self.instance_bank.get(
            batch_size, metas, dn_metas=self.sampler.dn_metas
        )  # 每一帧初始化的900个query和上一帧传递过来的600个query

        cached_anchor_embedding = self.instance_bank.cached_anchor_embedding

        # ========= prepare for denosing training ============
        # 1. get dn metas: noisy-anchors and corresponding GT
        # 2. concat learnable instances and noisy instances
        # 3. get attention mask
        attn_mask = None
        dn_metas = None
        temp_dn_reg_target = None
        if self.training and hasattr(self.sampler, "get_dn_anchors"):
            if "instance_id" in metas["img_metas"][0]:
                gt_instance_id = [
                    torch.from_numpy(x["instance_id"]).cuda()
                    for x in metas["img_metas"]
                ]  # 获取一个batch的每个sample的所有实例id，每个sample实例id数量不等
            else:
                gt_instance_id = None
            dn_metas = self.sampler.get_dn_anchors(
                metas[self.gt_cls_key],
                metas[self.gt_reg_key],
                gt_instance_id,
            )  # 根据每个sample的实例id和对应的类别gt、位置gt，生成对应的噪声
        if dn_metas is not None:
            (
                dn_anchor,  # (正例噪声的位置，填补到32的正例噪声的位置[非零]，负例噪声的位置，填补到32的负例噪声的位置[非零]) * 5
                dn_reg_target,  # （正例噪声的要回归的gt位置，填补到32的正例噪声要回归的位置[零]，负例噪声要回归的位置[零]，填补到32的负例噪声要回归的位置[零]
                dn_cls_target,  # (正例噪声的类别，填补到32的正例噪声的类别[-1]，负例噪声的类别[-3]，填补到32的负例噪声的类别[-3]) * 5
                dn_attn_mask,
                valid_mask,  # (正例噪声的掩码[True]，填补到32的正例噪声的掩码[False]，负例噪声的掩码[True]，填补到32的负例噪声的掩码[False]) * 5
                dn_id_target,  # (正例噪声的id，填补到32的正例噪声的id[-1]，负例噪声的id[-1]，填补到32的负例噪声的id[-1]) * 5
            ) = dn_metas
            num_dn_anchor = dn_anchor.shape[1]
            if dn_anchor.shape[-1] != anchor.shape[-1]:  # dn_anchor是10维，anchor是11维
                remain_state_dims = anchor.shape[-1] - dn_anchor.shape[-1]
                dn_anchor = torch.cat(
                    [
                        dn_anchor,
                        dn_anchor.new_zeros(
                            batch_size, num_dn_anchor, remain_state_dims
                        ),
                    ],
                    dim=-1,
                )
            anchor = torch.cat([anchor, dn_anchor], dim=1)
            instance_feature = (
                torch.cat(  # anchor和dn_anchor的instance feature初始化都是0
                    [
                        instance_feature,
                        instance_feature.new_zeros(
                            batch_size, num_dn_anchor, instance_feature.shape[-1]
                        ),
                    ],
                    dim=1,
                )
            )
            num_instance = instance_feature.shape[1]
            num_free_instance = num_instance - num_dn_anchor
            attn_mask = anchor.new_ones((num_instance, num_instance), dtype=torch.bool)
            attn_mask[:num_free_instance, :num_free_instance] = False
            attn_mask[num_free_instance:, num_free_instance:] = (
                dn_attn_mask  # 这里最终构造的mask是对角线对称mask，和DN-DETR不同
            )

        anchor_embed = self.anchor_encoder(anchor)
        if temp_anchor is not None:  # 第一帧的时候没有temp_anchor
            temp_anchor_embed = self.anchor_encoder(temp_anchor)
        else:
            temp_anchor_embed = None

        # =================== forward the layers ====================
        prediction = []
        classification = []
        quality = []
        k_preds = []
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif (
                op == "temp_gnn"
            ):  # 当前的（900个query+噪声实例）和时序的600个query做交叉注意力
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    temp_instance_feature,  # key，第一帧为空
                    temp_instance_feature,  # value，第一帧为空
                    query_pos=anchor_embed,
                    key_pos=temp_anchor_embed,
                    attn_mask=attn_mask if temp_instance_feature is None else None,
                )
            elif op == "gnn":  # 当前的（900个query+噪声实例）做自注意力
                instance_feature = self.graph_model(
                    i,
                    instance_feature,  # 自注意力key和query相同
                    value=instance_feature,
                    query_pos=anchor_embed,
                    attn_mask=attn_mask,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "deformable":
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    metas,
                )
            elif op == "refine":
                anchor, cls, qt = self.layers[i](  # 都是对1220个query进行更新
                    instance_feature,
                    anchor,
                    anchor_embed,
                    time_interval=time_interval,
                    return_cls=(
                        self.training
                        or len(prediction) == self.num_single_frame_decoder - 1
                        or i == len(self.operation_order) - (1 + self.kal)
                    ),
                )
                prediction.append(anchor)  # 观测位置，用来计算kt_gt
                classification.append(cls)
                quality.append(qt)
                if len(prediction) == self.num_single_frame_decoder:  # 单帧层的操作
                    instance_feature, anchor = self.instance_bank.update(
                        instance_feature, anchor, cls
                    )  # 如果是单帧层，且不是第一帧，则将缓存的600个query和单帧层的top300个query拼接
                    if (
                        dn_metas is not None
                        and self.sampler.num_temp_dn_groups > 0
                        and dn_id_target is not None
                    ):  # 切记是只有单帧层才有的操作
                        (
                            instance_feature,
                            anchor,
                            temp_dn_reg_target,
                            temp_dn_cls_target,
                            temp_valid_mask,
                            dn_id_target,
                        ) = self.sampler.update_dn(
                            instance_feature,
                            anchor,
                            dn_reg_target,
                            dn_cls_target,
                            valid_mask,
                            dn_id_target,
                            self.instance_bank.num_anchor,
                            self.instance_bank.mask,  # 在第一帧的时候为None，第二帧的时候在运动补偿的部分设值
                        )  # 上一步是对600/900的query加入时序，这一步是对3组/5组的dn_query加入时序，
                        # 上一帧的batch中8个样本的id和这一帧的batch的8个样本的id可能不匹配
                anchor_embed = self.anchor_encoder(anchor)

                if (
                    len(prediction) > self.num_single_frame_decoder
                    and temp_anchor_embed is not None
                ):  # 传入下一个时序层的600个递归query的anchor_embed是会更新的，但是600个query的instance_feature不更新
                    temp_anchor_embed = anchor_embed[
                        :, : self.instance_bank.num_temp_instances
                    ]
            elif op == "kalman":
                if cached_anchor_embedding is not None:
                    k_pred = self.layers[i](cached_anchor_embedding, temp_anchor_embed)
                    k_preds.append(k_pred)
                else:
                    k_pred = self.layers[i](
                        anchor_embed[:, :600], anchor_embed[:, :600]
                    )
                    k_preds.append(k_pred)
            else:
                raise NotImplementedError(f"{op} is not supported.")

        output = {}

        # split predictions of learnable instances and noisy instances
        if dn_metas is not None:
            dn_classification = [x[:, num_free_instance:] for x in classification]
            classification = [x[:, :num_free_instance] for x in classification]
            dn_prediction = [x[:, num_free_instance:] for x in prediction]
            prediction = [x[:, :num_free_instance] for x in prediction]
            quality = [
                x[:, :num_free_instance] if x is not None else None for x in quality
            ]
            output.update(  # 这是320个噪声query的预测和gt
                {
                    "dn_prediction": dn_prediction,
                    "dn_classification": dn_classification,
                    "dn_reg_target": dn_reg_target,  # 这是取这一帧的gt时生成的5组噪声，没有考虑时序拼接过来的噪声
                    "dn_cls_target": dn_cls_target,
                    "dn_valid_mask": valid_mask,
                }
            )
            if temp_dn_reg_target is not None:
                output.update(
                    {
                        "temp_dn_reg_target": temp_dn_reg_target,
                        "temp_dn_cls_target": temp_dn_cls_target,
                        "temp_dn_valid_mask": temp_valid_mask,
                        "dn_id_target": dn_id_target,
                    }
                )
                # 为什么要单独保存时序dn query的gt，因为在计算损失时需要判断用哪一个
                dn_cls_target = (
                    temp_dn_cls_target  # 后面4行都是为5组噪声选3组的步骤做准备
                )
                valid_mask = temp_valid_mask
            dn_instance_feature = instance_feature[:, num_free_instance:]
            dn_anchor = anchor[:, num_free_instance:]
            instance_feature = instance_feature[
                :, :num_free_instance
            ]  # 最后一层时序层的实例特征
            anchor = anchor[:, :num_free_instance]  # 最后一层时序层回归的anchor
            cls = cls[:, :num_free_instance]  # 最后一层时序层的输出分类置信度

            # cache dn_metas for temporal denoising 5组噪声中选择3组作为时序噪声实例
            self.sampler.cache_dn(
                dn_instance_feature,
                dn_anchor,
                dn_cls_target,
                valid_mask,
                dn_id_target,
            )
        output.update(  # 这是900个query的预测
            {
                "classification": classification,
                "prediction": prediction,
                "quality": quality,
            }
        )

        # kal
        output.update(
            {
                "xt": temp_anchor,  # [8,600,11]
                "kt_preds": k_preds,  # [8,600,6]*时序层数
            }
        )

        # cache current instances for temporal modeling 当前帧时序层最终输出的top600个query作为缓存的query
        anchor_embed = anchor_embed[:, :900]
        self.instance_bank.cache(
            instance_feature, anchor, anchor_embed, cls, metas, feature_maps
        )
        if not self.training:
            instance_id = self.instance_bank.get_instance_id(
                cls, anchor, self.decoder.score_threshold
            )
            output["instance_id"] = instance_id
        return output

    @force_fp32(apply_to=("model_outs"))
    def loss(self, model_outs, data, feature_maps=None):
        # ===================== prediction losses ======================
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        quality = model_outs["quality"]
        xt = model_outs["xt"]
        kt_preds = model_outs["kt_preds"]
        output = {}
        for decoder_idx, (cls, reg, qt) in enumerate(
            zip(cls_scores, reg_preds, quality)
        ):  # 先算每一层的900个query的预测的损失
            reg = reg[..., : len(self.reg_weights)]
            cls_target, reg_target, reg_weights = self.sampler.sample(
                cls,
                reg,
                data[self.gt_cls_key],
                data[self.gt_reg_key],
            )
            reg_target = reg_target[..., : len(self.reg_weights)]  # [8,900,10]
            mask = torch.logical_not(
                torch.all(reg_target == 0, dim=-1)
            )  # 900个query中当某个10维向量全为0则该query没有匹配到gt，为负样本
            mask_valid = mask.clone()

            num_pos = max(reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0)
            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(
                    mask,
                    cls.max(dim=-1).values.sigmoid()
                    > threshold,  # 一个query类别置信度最大值sigmoid后小于阈值的也加为负样本
                )

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_loss = self.loss_cls(cls, cls_target, avg_factor=num_pos)

            mask = mask.reshape(-1)

            if self.kal and decoder_idx == len(reg_preds) - 1:
                if xt is not None:
                    zt = reg[:, :600]  # [8,600,10]
                    xt_target = reg_target[:, :600]  # [8,600,10]
                    xt_ = xt[..., :10]  # [8,600,10]
                    mask_temp = mask.reshape(8, 900)[:, :600]
                    mask_temp = mask_temp.reshape(-1)

                    zt = zt.flatten(end_dim=1)[mask_temp]
                    xt_target = xt_target.flatten(end_dim=1)[mask_temp]
                    xt_ = xt_.flatten(end_dim=1)[mask_temp]

                    # (xt_target - xt)*(zt - xt)^(-1) = k_gt，当H矩阵为单位阵时，K为对角阵
                    k_gt = (xt_target - xt_) / (zt - xt_)
                    k_gt = k_gt[:, :6].detach()
                    k_pred = kt_preds[-1].flatten(end_dim=1)[mask_temp]
                    kt_loss = self.loss_kt(k_pred, k_gt, avg_factor=num_pos)
                    output["loss_kt"] = kt_loss
                    # print(f"loss_kt_{decoder_idx} = {kt_loss}")
                else:
                    kt_loss = kt_preds[-1].sum() * 0
                    output["loss_kt"] = kt_loss
                print(kt_loss)

            reg_weights = reg_weights * reg.new_tensor(self.reg_weights)
            reg_target = reg_target.flatten(end_dim=1)[mask]
            reg = reg.flatten(end_dim=1)[mask]
            reg_weights = reg_weights.flatten(end_dim=1)[mask]
            reg_target = torch.where(
                reg_target.isnan(), reg.new_tensor(0.0), reg_target
            )
            cls_target = cls_target[mask]
            if qt is not None:
                qt = qt.flatten(end_dim=1)[mask]

            reg_loss = self.loss_reg(
                reg,
                reg_target,
                weight=reg_weights,
                avg_factor=num_pos,
                suffix=f"_{decoder_idx}",
                quality=qt,
                cls_target=cls_target,
            )

            output[f"loss_cls_{decoder_idx}"] = cls_loss
            output.update(reg_loss)

        if "dn_prediction" not in model_outs:
            return output

        # ===================== denoising losses ======================
        dn_cls_scores = model_outs["dn_classification"]
        dn_reg_preds = model_outs["dn_prediction"]

        (
            dn_valid_mask,  # 8*320
            dn_cls_target,  # 非填充噪声的分类目标
            dn_reg_target,  # 只有正例非填充噪声的回归目标
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        ) = self.prepare_for_dn_loss(model_outs)
        for decoder_idx, (cls, reg) in enumerate(zip(dn_cls_scores, dn_reg_preds)):
            if (
                "temp_dn_valid_mask" in model_outs
                and decoder_idx == self.num_single_frame_decoder
            ):
                (
                    dn_valid_mask,
                    dn_cls_target,
                    dn_reg_target,
                    dn_pos_mask,
                    reg_weights,
                    num_dn_pos,
                ) = self.prepare_for_dn_loss(model_outs, prefix="temp_")

            cls_loss = self.loss_cls(
                cls.flatten(end_dim=1)[dn_valid_mask],
                dn_cls_target,
                avg_factor=num_dn_pos,
            )  # 分类损失是在没有填补的正例噪声和负例噪声上计算
            reg_loss = self.loss_reg(
                reg.flatten(end_dim=1)[dn_valid_mask][dn_pos_mask][
                    ..., : len(self.reg_weights)
                ],
                dn_reg_target,
                avg_factor=num_dn_pos,  # num_dn_pos的数量实际包含了正例噪声和负例噪声的总和，各自占一半
                weight=reg_weights,
                suffix=f"_dn_{decoder_idx}",
            )  # 回归损失是在没有填补的正例噪声上计算
            output[f"loss_cls_dn_{decoder_idx}"] = cls_loss
            output.update(reg_loss)
        return output

    def prepare_for_dn_loss(self, model_outs, prefix=""):
        dn_valid_mask = model_outs[f"{prefix}dn_valid_mask"].flatten(end_dim=1)
        dn_cls_target = model_outs[f"{prefix}dn_cls_target"].flatten(end_dim=1)[
            dn_valid_mask
        ]  # [8,32]展平并只取非填充的
        dn_reg_target = model_outs[f"{prefix}dn_reg_target"].flatten(end_dim=1)[
            dn_valid_mask
        ][..., : len(self.reg_weights)]
        dn_pos_mask = dn_cls_target >= 0  # 正例噪声
        dn_reg_target = dn_reg_target[dn_pos_mask]  # 回归任务只考虑正例噪声
        reg_weights = dn_reg_target.new_tensor(self.reg_weights)[None].tile(
            dn_reg_target.shape[0], 1
        )
        num_dn_pos = max(
            reduce_mean(torch.sum(dn_valid_mask).to(dtype=reg_weights.dtype)),
            1.0,
        )
        return (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        )

    @force_fp32(apply_to=("model_outs"))
    def post_process(self, model_outs, output_idx=-1):
        return self.decoder.decode(
            model_outs["classification"],
            model_outs["prediction"],
            model_outs["xt"],
            model_outs["kt_preds"],
            model_outs.get("instance_id"),
            model_outs.get("quality"),
            output_idx=output_idx,
        )
