# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.data import Dictionary
from unicore.models import (BaseUnicoreModel, register_model,
                            register_model_architecture)
from unicore.modules import LayerNorm
import unicore

from .transformer_encoder_with_pair import TransformerEncoderWithPair
from .unimol import NonLinearHead, UniMolModel, base_architecture

logger = logging.getLogger(__name__)


@register_model("tta")
class BindingAffinityModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--mol-pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--pocket-pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--pocket-encoder-layers",
            type=int,
            help="pocket encoder layers",
        )
        parser.add_argument(
            "--recycling",
            type=int,
            default=1,
            help="recycling nums of decoder",
        )

        parser.add_argument(
            "--masked-token-loss",
            type=float,
            metavar="D",
            help="mask loss ratio",
        )
        parser.add_argument(
            "--masked-dist-loss",
            type=float,
            metavar="D",
            help="masked distance loss ratio",
        )
        parser.add_argument(
            "--masked-coord-loss",
            type=float,
            metavar="D",
            help="masked coord loss ratio",
        )
        parser.add_argument(
            "--x-norm-loss",
            type=float,
            metavar="D",
            help="x norm loss ratio",
        )
        parser.add_argument(
            "--delta-pair-repr-norm-loss",
            type=float,
            metavar="D",
            help="delta encoder pair repr norm loss ratio",
        )
        parser.add_argument(
            "--masked-coord-dist-loss",
            type=float,
            metavar="D",
            help="masked coord dist loss ratio",
        )

    def __init__(self, args, mol_dictionary, pocket_dictionary):
        super().__init__()
        drugclip_architecture(args)
        self.args = args
        self.mol_model = UniMolModel(args.mol, mol_dictionary)
        self.pocket_model = UniMolModel(args.pocket, pocket_dictionary)

        self.cross_distance_project = NonLinearHead(
            args.mol.encoder_embed_dim * 2 + args.mol.encoder_attention_heads, 1, "relu"
        )
        self.holo_distance_project = DistanceHead(
            args.mol.encoder_embed_dim + args.mol.encoder_attention_heads, "relu"
        )

        self.mol_project = NonLinearHead(
            args.mol.encoder_embed_dim, 128, "relu"
        )
        self.mol_project_new = NonLinearHead(
            args.mol.encoder_embed_dim, 128, "relu"
        )

        self.logit_scale = nn.Parameter(torch.ones([1], device="cuda") * np.log(14))

        self.pocket_project = NonLinearHead(
            args.pocket.encoder_embed_dim, 128, "relu"
        )
        self.pocket_project_new = NonLinearHead(
            args.pocket.encoder_embed_dim, 128, "relu"
        )

        self.fuse_project = NonLinearHead(
            256, 1, "relu"
        )
        self.classification_head = nn.Sequential(
            nn.Linear(args.pocket.encoder_embed_dim + args.pocket.encoder_embed_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.weight = nn.Parameter(torch.ones(7,),requires_grad = True)
        self.weight_p = nn.Parameter(torch.ones(7,),requires_grad = True)
        self.kl_head_mol = NonLinearHead(
            args.mol.encoder_embed_dim, 128, "relu"
        )
        self.simclr_head_mol = NonLinearHead(
            args.mol.encoder_embed_dim, 128, "relu"
        )
        self.kl_head_pocket = NonLinearHead(
            args.pocket.encoder_embed_dim, 128, "relu"
        )
        self.simclr_head_pocket = NonLinearHead(
            args.pocket.encoder_embed_dim, 128, "relu"
        )
        self.weight_nets1 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            for _ in range(7)
        ])
        self.weight_nets2 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            for _ in range(7)
        ])
        self.mlp_cat = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.mlp_cat_p = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128))
        # self.mlp_cat = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128)
        # )
        # self.mlp_cat_p = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128))
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary, task.pocket_dictionary)

    def forward(
            self,
            mol_src_tokens,
            mol_src_distance,
            mol_src_edge_type,
            pocket_src_tokens,
            pocket_src_distance,
            pocket_src_edge_type,

            mask_mol_src_tokens,
            mask_mol_src_distance,
            mask_mol_src_coord,
            mask_mol_src_edge_type,

            mask_pocket_src_tokens,
            mask_pocket_src_distance,
            mask_pocket_src_coord,
            mask_pocket_src_edge_type,


            mol_features_only = False,
            pocket_features_only = False,
            Drugclip_train = False,
            mol_tta = False,
            pocket_tta = False,


            smi_list=None,
            pocket_list=None,
            encoder_masked_tokens=None,

            features_only=True,
            is_train=True,
            mol = False,
            pocket = False,
            **kwargs
    ):

        def get_dist_features(dist, et, flag):
            if flag == "mol":
                n_node = dist.size(-1)
                gbf_feature = self.mol_model.gbf(dist, et)
                gbf_result = self.mol_model.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                return graph_attn_bias
            else:
                n_node = dist.size(-1)
                gbf_feature = self.pocket_model.gbf(dist, et)
                gbf_result = self.pocket_model.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                return graph_attn_bias

        if mol:

            mol_padding_mask = mol_src_tokens.eq(self.mol_model.padding_idx)
            mol_x = self.mol_model.embed_tokens(mol_src_tokens)
            mol_graph_attn_bias = get_dist_features(
                mol_src_distance, mol_src_edge_type, "mol"
            )
            mol_outputs = self.mol_model.encoder(
                mol_x, padding_mask=mol_padding_mask, attn_mask=mol_graph_attn_bias
            )
            mol_encoder_rep = mol_outputs[0]
            mol_rep = mol_encoder_rep[:, 0, :]

        if pocket:
            pocket_padding_mask = pocket_src_tokens.eq(self.pocket_model.padding_idx)
            pocket_x = self.pocket_model.embed_tokens(pocket_src_tokens)
            pocket_graph_attn_bias = get_dist_features(
                pocket_src_distance, pocket_src_edge_type, "pocket"
            )
            pocket_outputs = self.pocket_model.encoder(
                pocket_x, padding_mask=pocket_padding_mask, attn_mask=pocket_graph_attn_bias
            )
            pocket_encoder_rep = pocket_outputs[0]
            pocket_rep = pocket_encoder_rep[:, 0, :]

        results = []
        if mol_features_only:
            mol_emb_kl = self.kl_head_mol(mol_rep)
            mol_emb_kl = mol_emb_kl / mol_emb_kl.norm(dim=1, keepdim=True)
            mol_emb_simclr = self.simclr_head_mol(mol_rep)
            mol_emb_simclr = mol_emb_simclr / mol_emb_simclr.norm(dim=1, keepdim=True)
            results.append((mol_emb_kl,mol_emb_simclr))
        if pocket_features_only:
            pocket_emb_kl = self.kl_head_pocket(pocket_rep)
            pocket_emb_kl = pocket_emb_kl / pocket_emb_kl.norm(dim=1, keepdim=True)
            pocket_emb_simclr = self.simclr_head_pocket(pocket_rep)
            pocket_emb_simclr = pocket_emb_simclr / pocket_emb_simclr.norm(dim=1, keepdim=True)
            results.append((pocket_emb_kl, pocket_emb_simclr))
        if Drugclip_train:
            mol_emb = self.mol_project(mol_rep)
            mol_emb = mol_emb / mol_emb.norm(dim=1, keepdim=True)
            pocket_emb = self.pocket_project(pocket_rep)
            pocket_emb = pocket_emb / pocket_emb.norm(dim=1, keepdim=True)
            ba_predict = torch.matmul(pocket_emb, torch.transpose(mol_emb, 0, 1))

            # mask duplicate mols and pockets in same batch

            bsz = ba_predict.shape[0]

            pockets = np.array(pocket_list, dtype=str)
            pockets = np.expand_dims(pockets, 1)
            matrix1 = np.repeat(pockets, len(pockets), 1)
            matrix2 = np.repeat(np.transpose(pockets), len(pockets), 0)
            pocket_duplicate_matrix = matrix1 == matrix2
            pocket_duplicate_matrix = 1 * pocket_duplicate_matrix
            pocket_duplicate_matrix = torch.tensor(pocket_duplicate_matrix, dtype=ba_predict.dtype).cuda()

            mols = np.array(smi_list, dtype=str)
            mols = np.expand_dims(mols, 1)
            matrix1 = np.repeat(mols, len(mols), 1)
            matrix2 = np.repeat(np.transpose(mols), len(mols), 0)
            mol_duplicate_matrix = matrix1 == matrix2
            mol_duplicate_matrix = 1 * mol_duplicate_matrix
            mol_duplicate_matrix = torch.tensor(mol_duplicate_matrix, dtype=ba_predict.dtype).cuda()

            onehot_labels = torch.eye(bsz).cuda()
            indicater_matrix = pocket_duplicate_matrix + mol_duplicate_matrix - 2 * onehot_labels

            # print(ba_predict.shape)
            ba_predict = ba_predict * self.logit_scale.exp().detach()
            ba_predict = indicater_matrix * -1e6 + ba_predict

            results.append((ba_predict, self.logit_scale.exp())) # _pocket, ba_predict_mol
        if mol_tta:
            mask_mol_padding_mask = mask_mol_src_tokens.eq(self.mol_model.padding_idx)
            mask_mol_x = self.mol_model.embed_tokens(mask_mol_src_tokens)
            mask_mol_graph_attn_bias = get_dist_features(
                mask_mol_src_distance, mask_mol_src_edge_type, "mol"
            )
            (
                mask_mol_encoder_rep,
                mask_mol_encoder_pair_rep,
                mask_mol_delta_encoder_pair_rep,
                mask_mol_x_norm,
                mask_mol_delta_encoder_pair_rep_norm,_
            ) = self.mol_model.encoder(
                mask_mol_x, padding_mask=mask_mol_padding_mask, attn_mask=mask_mol_graph_attn_bias
            )

            mask_mol_rep = mask_mol_encoder_rep[:, 0, :]

            # mol_emb = self.mol_project_new(mol_outputs[-1][16][:,0,:])

            mol_list = mol_outputs[-1]
            # mol_cat = torch.cat([mol_list[0][:, 0, :], mol_list[8][:, 0, :]], dim=-1)
            mol_cat = torch.cat([mol_list[0][:, 0, :],mol_list[8][:, 0, :],mol_list[16][:, 0, :]],dim=-1)
            mol_emb = self.mlp_cat(mol_cat)
            mol_emb = mol_emb / mol_emb.norm(dim=1, keepdim=True)
            mask_mol_emb_simclr = self.simclr_head_mol(mask_mol_rep)
            mask_mol_emb_simclr = mask_mol_emb_simclr / mask_mol_emb_simclr.norm(dim=1, keepdim=True)
            mask_mol_encoder_pair_rep[mask_mol_encoder_pair_rep == float("-inf")] = 0
            mask_mol_encoder_distance = None
            mask_mol_encoder_coord = None
            mask_mol_logits = None
            if self.args.mol.masked_token_loss > 0:
                mask_mol_logits = self.mol_model.lm_head(mask_mol_encoder_rep, encoder_masked_tokens)
            if self.args.mol.masked_coord_loss > 0:
                coords_emb = mask_mol_src_coord
                if mask_mol_padding_mask is not None:
                    atom_num = (torch.sum(1 - mask_mol_padding_mask.type_as(mask_mol_x), dim=1) - 1).view(
                        -1, 1, 1, 1
                    )
                else:
                    atom_num = mask_mol_src_coord.shape[1] - 1
                delta_pos = coords_emb.unsqueeze(1) - coords_emb.unsqueeze(2)
                attn_probs = self.mol_model.pair2coord_proj(mask_mol_delta_encoder_pair_rep)
                coord_update = delta_pos / atom_num * attn_probs
                coord_update = torch.sum(coord_update, dim=2)
                mask_mol_encoder_coord = coords_emb + coord_update
            if self.args.mol.masked_dist_loss > 0:
                mask_mol_encoder_distance = self.mol_model.dist_head(mask_mol_encoder_pair_rep)
            results.append((
                mask_mol_logits,
                mask_mol_encoder_distance,
                mask_mol_encoder_coord,
                mask_mol_x_norm,
                mask_mol_delta_encoder_pair_rep_norm,
                mask_mol_emb_simclr,
                mol_emb,
                mol_list,
            ))
        if pocket_tta:
            mask_pocket_padding_mask = mask_pocket_src_tokens.eq(self.pocket_model.padding_idx)
            mask_pocket_x = self.pocket_model.embed_tokens(mask_pocket_src_tokens)
            mask_pocket_graph_attn_bias = get_dist_features(
                mask_pocket_src_distance, mask_pocket_src_edge_type, "pocket"
            )
            (
                mask_pocket_encoder_rep,
                mask_pocket_encoder_pair_rep,
                mask_pocket_delta_encoder_pair_rep,
                mask_pocket_x_norm,
                mask_pocket_delta_encoder_pair_rep_norm,_
            ) = self.pocket_model.encoder(
                mask_pocket_x, padding_mask=mask_pocket_padding_mask, attn_mask=mask_pocket_graph_attn_bias
            )

            mask_pocket_rep = mask_pocket_encoder_rep[:, 0, :]
            pocket_list = pocket_outputs[-1]
            # pocket_cat = torch.cat([pocket_list[0][:, 0, :], pocket_list[8][:, 0, :]], dim=-1)
            pocket_cat = torch.cat([pocket_list[0][:, 0, :], pocket_list[8][:, 0, :], pocket_list[16][:, 0, :]], dim=-1)
            pocket_emb = self.mlp_cat_p(pocket_cat)
            # pocket_emb = self.pocket_project_new(pocket_outputs[-1][16][:, 0, :])
            pocket_emb = pocket_emb / pocket_emb.norm(dim=1, keepdim=True)
            mask_pocket_emb_simclr = self.simclr_head_pocket(mask_pocket_rep)
            mask_pocket_emb_simclr = mask_pocket_emb_simclr / mask_pocket_emb_simclr.norm(dim=1, keepdim=True)

            mask_pocket_encoder_pair_rep[mask_pocket_encoder_pair_rep == float("-inf")] = 0
            mask_pocket_encoder_distance = None
            mask_pocket_encoder_coord = None
            mask_pocket_logits = None
            if self.args.pocket.masked_token_loss > 0:
                mask_pocket_logits = self.pocket_model.lm_head(mask_pocket_encoder_rep, encoder_masked_tokens)
            if self.args.pocket.masked_coord_loss > 0:
                coords_emb = mask_pocket_src_coord
                if mask_pocket_padding_mask is not None:
                    atom_num = (torch.sum(1 - mask_pocket_padding_mask.type_as(mask_pocket_x), dim=1) - 1).view(
                        -1, 1, 1, 1
                    )
                else:
                    atom_num = mask_pocket_src_coord.shape[1] - 1
                delta_pos = coords_emb.unsqueeze(1) - coords_emb.unsqueeze(2)
                attn_probs = self.pocket_model.pair2coord_proj(mask_pocket_delta_encoder_pair_rep)
                coord_update = delta_pos / atom_num * attn_probs
                coord_update = torch.sum(coord_update, dim=2)
                mask_pocket_encoder_coord = coords_emb + coord_update
            if self.args.pocket.masked_dist_loss > 0:
                mask_pocket_encoder_distance = self.pocket_model.dist_head(mask_pocket_encoder_pair_rep)
            results.append((
                mask_pocket_logits,
                mask_pocket_encoder_distance,
                mask_pocket_encoder_coord,
                mask_pocket_x_norm,
                mask_pocket_delta_encoder_pair_rep_norm,
                mask_pocket_emb_simclr,
                pocket_emb,
                pocket_list
            ))
        return results

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""

        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates

class DistanceHead(nn.Module):
    def __init__(
            self,
            heads,
            activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        x[x == float("-inf")] = 0
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len)
        x = (x + x.transpose(-1, -2)) * 0.5
        return x


@register_model_architecture("tta", "tta")
def drugclip_architecture(args):
    parser = argparse.ArgumentParser()
    args.mol = parser.parse_args([])
    args.pocket = parser.parse_args([])

    args.mol.encoder_layers = getattr(args, "mol_encoder_layers", 15)
    args.mol.encoder_embed_dim = getattr(args, "mol_encoder_embed_dim", 512)
    args.mol.encoder_ffn_embed_dim = getattr(args, "mol_encoder_ffn_embed_dim", 2048)
    args.mol.encoder_attention_heads = getattr(args, "mol_encoder_attention_heads", 64)
    args.mol.dropout = getattr(args, "mol_dropout", 0.1)
    args.mol.emb_dropout = getattr(args, "mol_emb_dropout", 0.1)
    args.mol.attention_dropout = getattr(args, "mol_attention_dropout", 0.1)
    args.mol.activation_dropout = getattr(args, "mol_activation_dropout", 0.0)
    args.mol.pooler_dropout = getattr(args, "mol_pooler_dropout", 0.0)
    args.mol.max_seq_len = getattr(args, "mol_max_seq_len", 512)
    args.mol.activation_fn = getattr(args, "mol_activation_fn", "gelu")
    args.mol.pooler_activation_fn = getattr(args, "mol_pooler_activation_fn", "tanh")
    args.mol.post_ln = getattr(args, "mol_post_ln", False)
    args.mol.masked_token_loss = 1.0
    args.mol.masked_coord_loss = -5
    args.mol.masked_dist_loss = -10
    args.mol.x_norm_loss = -0.01
    args.mol.delta_pair_repr_norm_loss = -0.01
    args.mol.kl_loss = 1000.0
    args.mol.simclr_loss = 0.01
    args.mol.kl_temperature = 1.0
    args.mol.simclr_temperature = 0.007



    args.pocket.encoder_layers = getattr(args, "pocket_encoder_layers", 15)
    args.pocket.encoder_embed_dim = getattr(args, "pocket_encoder_embed_dim", 512)
    args.pocket.encoder_ffn_embed_dim = getattr(
        args, "pocket_encoder_ffn_embed_dim", 2048
    )
    args.pocket.encoder_attention_heads = getattr(
        args, "pocket_encoder_attention_heads", 64
    )
    args.pocket.dropout = getattr(args, "pocket_dropout", 0.1)
    args.pocket.emb_dropout = getattr(args, "pocket_emb_dropout", 0.1)
    args.pocket.attention_dropout = getattr(args, "pocket_attention_dropout", 0.1)
    args.pocket.activation_dropout = getattr(args, "pocket_activation_dropout", 0.0)
    args.pocket.pooler_dropout = getattr(args, "pocket_pooler_dropout", 0.0)
    args.pocket.max_seq_len = getattr(args, "pocket_max_seq_len", 512)
    args.pocket.activation_fn = getattr(args, "pocket_activation_fn", "gelu")
    args.pocket.pooler_activation_fn = getattr(
        args, "pocket_pooler_activation_fn", "tanh"
    )
    args.pocket.post_ln = getattr(args, "pocket_post_ln", False)
    args.pocket.masked_token_loss = 1.0
    args.pocket.masked_coord_loss = -1.0
    args.pocket.masked_dist_loss = -1.0
    args.pocket.x_norm_loss = -0.01
    args.pocket.delta_pair_repr_norm_loss = -0.01
    args.pocket.kl_loss = 1000.0
    args.pocket.simclr_loss = 0.01
    args.pocket.kl_temperature = 1.0
    args.pocket.simclr_temperature = 0.007


    base_architecture(args)



