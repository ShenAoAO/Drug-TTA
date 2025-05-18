import math
import torch
import torch.nn.functional as F
import pandas as pd
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss
from unicore.losses.cross_entropy import CrossEntropyLoss
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import numpy as np
import warnings
from sklearn.metrics import top_k_accuracy_score
from rdkit.ML.Scoring.Scoring import CalcBEDROC
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcAUC, CalcEnrichment
from sklearn.metrics import roc_curve
from torch.nn.functional import log_softmax, softmax, kl_div
from unicore.modules.layer_norm import LayerNorm
import torch.nn as nn


def calculate_bedroc(y_true, y_score, alpha):
    """
    Calculate BEDROC score.

    Parameters:
    - y_true: true binary labels (0 or 1)
    - y_score: predicted scores or probabilities
    - alpha: parameter controlling the degree of early retrieval emphasis

    Returns:
    - BEDROC score
    """

    # concate res_single and labels
    scores = np.expand_dims(y_score, axis=1)
    y_true = np.expand_dims(y_true, axis=1)
    # print(scores.shape, y_true.shape)
    scores = np.concatenate((scores, y_true), axis=1)
    # inverse sort scores based on first column
    scores = scores[scores[:, 0].argsort()[::-1]]
    bedroc = CalcBEDROC(scores, 1, 80.5)
    return bedroc


def cal_dist_loss(sample, dist, masked_tokens, target_key, weights, normalize=False):
    dist_masked_tokens = masked_tokens
    masked_distance = dist[dist_masked_tokens, :]
    masked_distance_target = sample[target_key]["distance_target"][dist_masked_tokens]
    non_pad_pos = masked_distance_target > 0
    if normalize:
        masked_distance_target = (
                                         masked_distance_target.float() - 6.312581655060595
                                 ) / 3.3899264663911888
    masked_dist_loss = F.smooth_l1_loss(
        masked_distance[non_pad_pos].view(-1).float(),
        masked_distance_target[non_pad_pos].view(-1),
        reduction="none",
        beta=1.0,
    )
    masked_dist_loss = (masked_dist_loss * weights[torch.where(dist_masked_tokens != 0)[0]][
        torch.where(non_pad_pos != 0)[0]]).mean()
    return masked_dist_loss


def cal_dist_loss_ori(sample, dist, masked_tokens, target_key, normalize=False):
    dist_masked_tokens = masked_tokens
    masked_distance = dist[dist_masked_tokens, :]
    masked_distance_target = sample[target_key]["distance_target"][
        dist_masked_tokens
    ]
    non_pad_pos = masked_distance_target > 0
    if normalize:
        masked_distance_target = (
                                         masked_distance_target.float() - 6.312581655060595
                                 ) / 3.3899264663911888
    masked_dist_loss = F.smooth_l1_loss(
        masked_distance[non_pad_pos].view(-1).float(),
        masked_distance_target[non_pad_pos].view(-1),
        reduction="mean",
        beta=1.0,
    )
    # masked_dist_loss = (masked_dist_loss * weights[torch.where(dist_masked_tokens!=0)[0]][torch.where(non_pad_pos!=0)[0]]).mean()
    return masked_dist_loss


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    model.requires_grad_(False)
    for nm, m in model.named_modules():
        if isinstance(m, LayerNorm):
            # if isinstance(m, (nn.BatchNorm2d, InPlaceABN)):
            # print(nm)
            m.requires_grad_(True)
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def re_new(y_true, y_score, ratio):
    fp = 0
    tp = 0
    p = sum(y_true)
    n = len(y_true) - p
    num = ratio * n
    sort_index = np.argsort(y_score)[::-1]
    for i in range(len(sort_index)):
        index = sort_index[i]
        if y_true[index] == 1:
            tp += 1
        else:
            fp += 1
            if fp >= num:
                break
    return (tp * n) / (p * fp)


def calc_re(y_true, y_score, ratio_list):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    # print(fpr, tpr)
    res = {}
    res2 = {}
    total_active_compounds = sum(y_true)
    total_compounds = len(y_true)

    # for ratio in ratio_list:
    #     for i, t in enumerate(fpr):
    #         if t > ratio:
    #             #print(fpr[i], tpr[i])
    #             if fpr[i-1]==0:
    #                 res[str(ratio)]=tpr[i]/fpr[i]
    #             else:
    #                 res[str(ratio)]=tpr[i-1]/fpr[i-1]
    #             break

    for ratio in ratio_list:
        res2[str(ratio)] = re_new(y_true, y_score, ratio)

    # print(res)
    # print(res2)
    return res2


def cal_metrics(y_true, y_score, alpha):
    """
    Calculate BEDROC score.

    Parameters:
    - y_true: true binary labels (0 or 1)
    - y_score: predicted scores or probabilities
    - alpha: parameter controlling the degree of early retrieval emphasis

    Returns:
    - BEDROC score
    """

    # concate res_single and labels
    scores = np.expand_dims(y_score, axis=1)
    y_true = np.expand_dims(y_true, axis=1)
    scores = np.concatenate((scores, y_true), axis=1)
    # inverse sort scores based on first column
    scores = scores[scores[:, 0].argsort()[::-1]]
    bedroc = CalcBEDROC(scores, 1, 80.5)
    count = 0
    # sort y_score, return index
    index = np.argsort(y_score)[::-1]
    for i in range(int(len(index) * 0.005)):
        if y_true[index[i]] == 1:
            count += 1
    auc = CalcAUC(scores, 1)
    ef_list = CalcEnrichment(scores, 1, [0.005, 0.01, 0.02, 0.05])
    ef = {
        "0.005": ef_list[0],
        "0.01": ef_list[1],
        "0.02": ef_list[2],
        "0.05": ef_list[3]
    }
    re_list = calc_re(y_true, y_score, [0.005, 0.01, 0.02, 0.05])
    return auc, bedroc, ef, re_list


@register_loss("in_batch_softmax")
class IBSLoss(CrossEntropyLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, flag='main_train', reduce=True, fix_encoder=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        if flag == "ssl_train_mol":
            mol_dist = sample["net_input"]["mol_src_distance"]
            mol_et = sample["net_input"]["mol_src_edge_type"]
            mol_st = sample["net_input"]["mol_src_tokens"]

            mol_mask_coord = sample['mol_mask_input']['src_coord']
            mol_mask_et = sample['mol_mask_input']['src_edge_type']
            mol_mask_dist = sample['mol_mask_input']['src_distance']
            mol_mask_st = sample['mol_mask_input']['src_tokens']

            mol_masked_tokens = sample['mol_mask_input']["tokens_target"].ne(model.mol_model.padding_idx)
            [(mol_features_kl,
              mol_features_simclr),
             (mol_logits_encoder,
              mol_encoder_distance,
              mol_encoder_coord,
              mol_x_norm,
              mol_delta_encoder_pair_rep_norm,
              mask_mol_features_simclr,
              mol_features_mlp
              )] = model(mol_src_tokens=mol_st,
                         mol_src_distance=mol_dist,
                         mol_src_edge_type=mol_et,
                         pocket_src_tokens=None,
                         pocket_src_distance=None,
                         pocket_src_edge_type=None,

                         mask_mol_src_tokens=mol_mask_st,
                         mask_mol_src_distance=mol_mask_dist,
                         mask_mol_src_coord=mol_mask_coord,
                         mask_mol_src_edge_type=mol_mask_et,

                         mask_pocket_src_tokens=None,
                         mask_pocket_src_distance=None,
                         mask_pocket_src_coord=None,
                         mask_pocket_src_edge_type=None,

                         mol_tta=True,
                         mol=True,
                         mol_features_only=True,
                         encoder_masked_tokens=mol_masked_tokens)

            weights_mol = [torch.sigmoid(net(mol_features_mlp)) for net in model.weight_nets1]
            weights_tensor = torch.stack(weights_mol, dim=1)  # 将每个网络的输出按列堆叠
            weights = F.softmax(1 / (2 * weights_tensor.float() ** 2), dim=1) * 10
            loss = 0.0
            logging_output = {}

            targets = sample['mol_mask_input']["tokens_target"]
            targets = targets[mol_masked_tokens]
            if self.args.mol.masked_token_loss > 0 and mol_masked_tokens is not None:
                masked_token_loss = F.nll_loss(
                    F.log_softmax(mol_logits_encoder, dim=-1, dtype=torch.float32),
                    targets,
                    ignore_index=model.mol_model.padding_idx,
                    reduction="none",
                )
                masked_token_loss = (
                        masked_token_loss * weights[:, 0, :].squeeze(-1)[torch.where(mol_masked_tokens != 0)[0]]).mean()
                masked_pred = mol_logits_encoder.argmax(dim=-1)
                masked_hit = (masked_pred == targets).long().sum()
                sample_size = mol_masked_tokens.long().sum()
                masked_cnt = sample_size
                loss = loss + masked_token_loss * self.args.mol.masked_token_loss
                logging_output = {
                    "sample_size": 1,
                    "bsz": sample['mol_mask_input']["tokens_target"].size(0),
                    "seq_len": sample['mol_mask_input']["tokens_target"].size(1)
                               * sample['mol_mask_input']["tokens_target"].size(0),
                    "masked_token_loss": masked_token_loss.data,
                    "masked_token_hit": masked_hit.data,
                    "masked_token_cnt": masked_cnt,
                }

            if self.args.mol.masked_coord_loss > 0 and mol_encoder_coord is not None:
                # real = mask + delta
                coord_target = sample['mol_mask_input']["coord_target"]
                masked_coord_loss = F.smooth_l1_loss(
                    mol_encoder_coord[mol_masked_tokens].view(-1, 3).float(),
                    coord_target[mol_masked_tokens].view(-1, 3),
                    reduction="none",
                    beta=1.0,
                )
                masked_coord_loss = (masked_coord_loss.mean(-1) * weights[:, 1, :].squeeze(-1)[
                    torch.where(mol_masked_tokens != 0)[0]]).mean()

                loss = loss + masked_coord_loss * self.args.mol.masked_coord_loss
                # restore the scale of loss for displaying
                logging_output["masked_coord_loss"] = masked_coord_loss.data

            if self.args.mol.masked_dist_loss > 0 and mol_encoder_distance is not None:
                dist_masked_tokens = mol_masked_tokens
                masked_dist_loss = cal_dist_loss(
                    sample, mol_encoder_distance, dist_masked_tokens, 'mol_mask_input', weights[:, 2, :].squeeze(-1),
                    normalize=True
                )
                loss = loss + masked_dist_loss * self.args.mol.masked_dist_loss
                logging_output["masked_dist_loss"] = masked_dist_loss.data

            if self.args.mol.x_norm_loss > 0 and mol_x_norm is not None:
                loss = loss + self.args.mol.x_norm_loss * mol_x_norm * weights[:, 3, :].squeeze(-1).mean()
                logging_output["x_norm_loss"] = mol_x_norm.data

            if (
                    self.args.mol.delta_pair_repr_norm_loss > 0
                    and mol_delta_encoder_pair_rep_norm is not None
            ):
                loss = (
                        loss + self.args.mol.delta_pair_repr_norm_loss * mol_delta_encoder_pair_rep_norm * weights[:, 4,
                                                                                                           :].squeeze(
                    -1).mean()
                )
                logging_output[
                    "delta_pair_repr_norm_loss"
                ] = mol_delta_encoder_pair_rep_norm.data


            if self.args.mol.kl_loss > 0:
                latent_dim = mol_features_kl.shape[-1]

                uniform = (torch.ones(latent_dim) / latent_dim).to(mol_features_kl.device)

                softmax_uniform = softmax(uniform / self.args.mol.kl_temperature, dim=0)
                softmax_uniform = softmax_uniform.unsqueeze(dim=0).repeat(mol_features_kl.shape[0], 1)

                softmax_latents = log_softmax(mol_features_kl / self.args.mol.kl_temperature, dim=1)
                kl_loss = kl_div(softmax_latents.float(), softmax_uniform, reduction="none")

                loss = loss + self.args.mol.kl_loss * (kl_loss.mean(-1) * weights[:, 5, :].squeeze(-1)).mean()
            if self.args.mol.simclr_loss > 0:
                batch_size = mol_features_simclr.shape[0]

                combined_features = torch.cat([mol_features_simclr, mask_mol_features_simclr],
                                              dim=0)

                similarity_matrix = torch.mm(combined_features,
                                             combined_features.T)

                similarity_matrix = similarity_matrix / self.args.mol.simclr_temperature

                positive_pairs = torch.diag(similarity_matrix, batch_size)
                positive_pairs = torch.cat([positive_pairs, torch.diag(similarity_matrix, -batch_size)],
                                           dim=0)
                mask = torch.eye(2 * batch_size, dtype=torch.bool, device=mol_features_simclr.device)
                similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

                log_sum_exp_negatives = torch.logsumexp(similarity_matrix, dim=1)  # [2 * batch_size]

                simclr_loss = -positive_pairs + log_sum_exp_negatives
                loss = loss + self.args.mol.simclr_loss * (simclr_loss * torch.cat(
                    [weights[:, 6, :].squeeze(-1), weights[:, 6, :].squeeze(-1)])).mean()

            logging_output[
                "mol_total_loss"
            ] = loss.data
            return loss, logging_output

        if flag == "ssl_train_pocket":
            pocket_dist = sample["net_input"]["pocket_src_distance"]
            pocket_et = sample["net_input"]["pocket_src_edge_type"]
            pocket_st = sample["net_input"]["pocket_src_tokens"]

            pocket_mask_coord = sample['pocket_mask_input']['src_coord']
            pocket_mask_et = sample['pocket_mask_input']['src_edge_type']
            pocket_mask_dist = sample['pocket_mask_input']['src_distance']
            pocket_mask_st = sample['pocket_mask_input']['src_tokens']

            pocket_masked_tokens = sample['pocket_mask_input']["tokens_target"].ne(model.pocket_model.padding_idx)
            [(pocket_features_kl,
              pocket_features_simclr),
             (
                 pocket_logits_encoder,
                 pocket_encoder_distance,
                 pocket_encoder_coord,
                 pocket_x_norm,
                 pocket_delta_encoder_pair_rep_norm,
                 pocket_mask_pocket_features_simclr,
                 pocket_features_mlp
             )] = model(mol_src_tokens=None,
                        mol_src_distance=None,
                        mol_src_edge_type=None,
                        pocket_src_tokens=pocket_st,
                        pocket_src_distance=pocket_dist,
                        pocket_src_edge_type=pocket_et,

                        mask_mol_src_tokens=None,
                        mask_mol_src_distance=None,
                        mask_mol_src_coord=None,
                        mask_mol_src_edge_type=None,

                        mask_pocket_src_tokens=pocket_mask_st,
                        mask_pocket_src_distance=pocket_mask_dist,
                        mask_pocket_src_coord=pocket_mask_coord,
                        mask_pocket_src_edge_type=pocket_mask_et,

                        pocket_tta=True,
                        pocket=True,
                        pocket_features_only=True,
                        encoder_masked_tokens=pocket_masked_tokens)



            weights_mlp_p = [torch.sigmoid(net(pocket_features_mlp)) for net in model.weight_nets2]
            weights_tensor = torch.stack(weights_mlp_p, dim=1)
            weights_p = F.softmax(1 / (2 * weights_tensor.float() ** 2), dim=1) * 10
            pocket_loss = 0.0
            logging_output = {}
            pocket_targets = sample['pocket_mask_input']["tokens_target"]
            pocket_targets = pocket_targets[pocket_masked_tokens]
            if self.args.pocket.masked_token_loss > 0 and pocket_masked_tokens is not None:
                pocket_masked_token_loss = F.nll_loss(
                    F.log_softmax(pocket_logits_encoder, dim=-1, dtype=torch.float32),
                    pocket_targets,
                    ignore_index=model.pocket_model.padding_idx,
                    reduction="none",
                )
                pocket_masked_token_loss = (pocket_masked_token_loss * weights_p[:, 0, :].squeeze(-1)[
                    torch.where(pocket_masked_tokens != 0)[0]]).mean()
                pocket_masked_pred = pocket_logits_encoder.argmax(dim=-1)
                pocket_masked_hit = (pocket_masked_pred == pocket_targets).long().sum()
                pocket_sample_size = pocket_masked_tokens.long().sum()
                pocket_masked_cnt = pocket_sample_size

                pocket_loss = pocket_loss + pocket_masked_token_loss * self.args.pocket.masked_token_loss
                logging_output = {
                    "pocket_sample_size": 1,
                    "pocket_bsz": sample['pocket_mask_input']["tokens_target"].size(0),
                    "pocket_seq_len": sample['pocket_mask_input']["tokens_target"].size(1)
                                      * sample['pocket_mask_input']["tokens_target"].size(0),
                    "pocket_masked_token_loss": pocket_masked_token_loss.data,
                    "pocket_masked_token_hit": pocket_masked_hit.data,
                    "pocket_masked_token_cnt": pocket_masked_cnt,
                }

            if self.args.pocket.masked_coord_loss > 0 and pocket_encoder_coord is not None:
                # real = mask + delta
                pocket_coord_target = sample['pocket_mask_input']["coord_target"]
                pocket_masked_coord_loss = F.smooth_l1_loss(
                    pocket_encoder_coord[pocket_masked_tokens].view(-1, 3).float(),
                    pocket_coord_target[pocket_masked_tokens].view(-1, 3),
                    reduction="none",
                    beta=1.0,
                )
                pocket_masked_coord_loss = (pocket_masked_coord_loss.mean(-1) * weights_p[:, 1, :].squeeze(-1)[
                    torch.where(pocket_masked_tokens != 0)[0]]).mean()
                pocket_loss = pocket_loss + pocket_masked_coord_loss * self.args.pocket.masked_coord_loss
                # restore the scale of loss for displaying
                logging_output["pocket_masked_coord_loss"] = pocket_masked_coord_loss.data

            if self.args.pocket.masked_dist_loss > 0 and pocket_encoder_distance is not None:
                pocket_dist_masked_tokens = pocket_masked_tokens
                pocket_masked_dist_loss = cal_dist_loss(
                    sample, pocket_encoder_distance, pocket_dist_masked_tokens, 'pocket_mask_input',
                    weights_p[:, 2, :].squeeze(-1), normalize=True
                )
                pocket_loss = pocket_loss + pocket_masked_dist_loss * self.args.pocket.masked_dist_loss
                logging_output["pocket_masked_dist_loss"] = pocket_masked_dist_loss.data

            if self.args.pocket.x_norm_loss > 0 and pocket_x_norm is not None:
                pocket_loss = pocket_loss + self.args.pocket.x_norm_loss * pocket_x_norm * weights_p[:, 3, :].squeeze(
                    -1).mean()
                logging_output["pocket_x_norm_loss"] = pocket_x_norm.data

            if (
                    self.args.pocket.delta_pair_repr_norm_loss > 0
                    and pocket_delta_encoder_pair_rep_norm is not None
            ):
                pocket_loss = (
                        pocket_loss + self.args.pocket.delta_pair_repr_norm_loss * pocket_delta_encoder_pair_rep_norm * weights_p[
                                                                                                                        :,
                                                                                                                        4,
                                                                                                                        :].squeeze(
                    -1).mean()
                )
                logging_output[
                    "pocket_delta_pair_repr_norm_loss"
                ] = pocket_delta_encoder_pair_rep_norm.data
            if self.args.pocket.kl_loss > 0:
                pocket_latent_dim = pocket_features_kl.shape[-1]

                pocket_uniform = (torch.ones(pocket_latent_dim) / pocket_latent_dim).to(pocket_features_kl.device)

                pocket_softmax_uniform = softmax(pocket_uniform / self.args.pocket.kl_temperature, dim=0)
                pocket_softmax_uniform = pocket_softmax_uniform.unsqueeze(dim=0).repeat(pocket_features_kl.shape[0], 1)

                pocket_softmax_latents = log_softmax(pocket_features_kl / self.args.pocket.kl_temperature, dim=1)
                pocket_kl_loss = kl_div(pocket_softmax_latents.float(), pocket_softmax_uniform, reduction="none")
                pocket_loss = pocket_loss + self.args.pocket.kl_loss * (
                            pocket_kl_loss.mean(-1) * weights_p[:, 5, :].squeeze(-1)).mean()
            if self.args.pocket.simclr_loss > 0:
                batch_size = pocket_features_simclr.shape[0]

                combined_features = torch.cat([pocket_features_simclr, pocket_mask_pocket_features_simclr],
                                              dim=0)

                similarity_matrix = torch.mm(combined_features, combined_features.T)

                similarity_matrix = similarity_matrix / self.args.mol.simclr_temperature

                positive_pairs = torch.diag(similarity_matrix, batch_size)
                positive_pairs = torch.cat([positive_pairs, torch.diag(similarity_matrix, -batch_size)],
                                           dim=0)

                mask = torch.eye(2 * batch_size, dtype=torch.bool, device=pocket_features_simclr.device)
                similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

                log_sum_exp_negatives = torch.logsumexp(similarity_matrix, dim=1)  # [2 * batch_size]

                simclr_loss = -positive_pairs + log_sum_exp_negatives
                pocket_loss = pocket_loss + self.args.pocket.simclr_loss * (simclr_loss * torch.cat(
                    [weights_p[:, 6, :].squeeze(-1), weights_p[:, 6, :].squeeze(-1)])).mean()
            logging_output[
                "pocket_total_loss"
            ] = pocket_loss.data
            return pocket_loss, logging_output

        if flag == 'main_train':
            mol_dist = sample["net_input"]["mol_src_distance"]
            mol_et = sample["net_input"]["mol_src_edge_type"]
            mol_st = sample["net_input"]["mol_src_tokens"]

            mol_masked_tokens = sample['mol_mask_input']["tokens_target"].ne(model.mol_model.padding_idx)

            pocket_dist = sample["net_input"]["pocket_src_distance"]
            pocket_et = sample["net_input"]["pocket_src_edge_type"]
            pocket_st = sample["net_input"]["pocket_src_tokens"]

            net_output = model(
                mol_src_tokens=mol_st,
                mol_src_distance=mol_dist,
                mol_src_edge_type=mol_et,
                pocket_src_tokens=pocket_st,
                pocket_src_distance=pocket_dist,
                pocket_src_edge_type=pocket_et,

                mask_mol_src_tokens=None,
                mask_mol_src_distance=None,
                mask_mol_src_coord=None,
                mask_mol_src_edge_type=None,

                mask_pocket_src_tokens=None,
                mask_pocket_src_distance=None,
                mask_pocket_src_coord=None,
                mask_pocket_src_edge_type=None,
                smi_list=sample["smi_name"],
                pocket_list=sample["pocket_name"],

                encoder_masked_tokens=mol_masked_tokens,
                Drugclip_train=True,
                mol=True,
                pocket=True,
            )

            logit_output = net_output[0][0]
            loss = self.compute_loss(model, logit_output, sample, reduce=reduce)
            sample_size = logit_output.size(0)
            targets = torch.arange(sample_size, dtype=torch.long).cuda()
            affinities = sample["target"]["finetune_target"].view(-1)
            if not self.training:
                logit_output = logit_output[:, :sample_size]
                probs = F.softmax(logit_output.float(), dim=-1).view(
                    -1, logit_output.size(-1)
                )
                logging_output = {
                    "loss": loss.data,
                    "prob": probs.data,
                    "target": targets,
                    "smi_name": sample["smi_name"],
                    "sample_size": sample_size,
                    "bsz": targets.size(0),
                    "scale": net_output[0][1].data,
                    "affinity": affinities,
                }
            else:
                logging_output = {
                    "loss": loss.data,
                    "sample_size": sample_size,
                    "bsz": targets.size(0),
                    "scale": net_output[0][1].data
                }
            return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs_pocket = F.log_softmax(net_output.float(), dim=-1)
        lprobs_pocket = lprobs_pocket.view(-1, lprobs_pocket.size(-1))
        sample_size = lprobs_pocket.size(0)
        targets = torch.arange(sample_size, dtype=torch.long).view(-1).cuda()

        # pocket retrieve mol
        loss_pocket = F.nll_loss(
            lprobs_pocket,
            targets,
            reduction="sum" if reduce else "none",
        )

        lprobs_mol = F.log_softmax(torch.transpose(net_output.float(), 0, 1), dim=-1)
        lprobs_mol = lprobs_mol.view(-1, lprobs_mol.size(-1))
        lprobs_mol = lprobs_mol[:sample_size]

        # mol retrieve pocket
        loss_mol = F.nll_loss(
            lprobs_mol,
            targets,
            reduction="sum" if reduce else "none",
        )

        loss = 0.5 * loss_pocket + 0.5 * loss_mol
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        metrics.log_scalar("scale", logging_outputs[0].get("scale"), round=3)
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            acc_sum = sum(
                sum(log.get("prob").argmax(dim=-1) == log.get("target"))
                for log in logging_outputs
            )

            prob_list = []
            if len(logging_outputs) == 1:
                prob_list.append(logging_outputs[0].get("prob"))
            else:
                for i in range(len(logging_outputs) - 1):
                    prob_list.append(logging_outputs[i].get("prob"))
            probs = torch.cat(prob_list, dim=0)

            metrics.log_scalar(
                f"{split}_acc", acc_sum / sample_size, sample_size, round=3
            )

            metrics.log_scalar(
                "valid_neg_loss", -loss_sum / sample_size / math.log(2), sample_size, round=3
            )

            targets = torch.cat(
                [log.get("target", 0) for log in logging_outputs], dim=0
            )
            print(targets.shape, probs.shape)

            targets = targets[:len(probs)]
            bedroc_list = []
            auc_list = []
            for i in range(len(probs)):
                prob = probs[i]
                target = targets[i]
                label = torch.zeros_like(prob)
                label[target] = 1.0
                cur_auc = roc_auc_score(label.cpu(), prob.cpu())
                auc_list.append(cur_auc)
                bedroc = calculate_bedroc(label.cpu(), prob.cpu(), 80.5)
                bedroc_list.append(bedroc)
            bedroc = np.mean(bedroc_list)
            auc = np.mean(auc_list)

            top_k_acc = top_k_accuracy_score(targets.cpu(), probs.cpu(), k=3, normalize=True)
            metrics.log_scalar(f"{split}_auc", auc, sample_size, round=3)
            metrics.log_scalar(f"{split}_bedroc", bedroc, sample_size, round=3)
            metrics.log_scalar(f"{split}_top3_acc", top_k_acc, sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train