# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import unicore.optim.adam
from IPython import embed as debug_embedded
import logging
import os
from collections.abc import Iterable
from sklearn.metrics import roc_auc_score
from xmlrpc.client import Boolean
import numpy as np
import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
from unicore import checkpoint_utils
import unicore
from unicore.data import (AppendTokenDataset, Dictionary, EpochShuffleDataset,
                          FromNumpyDataset, NestedDictionaryDataset,
                          PrependTokenDataset, RawArrayDataset, LMDBDataset, RawLabelDataset,
                          RightPadDataset, RightPadDataset2D, TokenizeDataset, SortDataset, data_utils)
from unicore.tasks import UnicoreTask, register_task
from unimol.data import (AffinityDataset, CroppingPocketDataset,
                         CrossDistanceDataset, DistanceDataset,
                         EdgeTypeDataset, KeyDataset, LengthDataset,
                         NormalizeDataset, NormalizeDockingPoseDataset,
                         PrependAndAppend2DDataset, RemoveHydrogenDataset,
                         RemoveHydrogenPocketDataset, RightPadDatasetCoord,
                         RightPadDatasetCross2D, TTADockingPoseDataset, AffinityTestDataset, AffinityValidDataset,
                         AffinityMolDataset, AffinityPocketDataset, ResamplingDataset)
# from skchem.metrics import bedroc_score
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcAUC, CalcEnrichment
from sklearn.metrics import roc_curve
from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    EpochShuffleDataset,
    TokenizeDataset,
    RightPadDataset2D,
    FromNumpyDataset,
    RawArrayDataset,
)
from unimol.data import (
    KeyDataset,
    ConformerSampleDataset,
    DistanceDataset,
    EdgeTypeDataset,
    MaskPointsDataset,
    RemoveHydrogenDataset,
    AtomTypeDataset,
    NormalizeDataset,
    CroppingDataset,
    RightPadDatasetCoord,
    Add2DConformerDataset,
    LMDBDataset,
)
import torch.nn.functional as F
from torch.nn.functional import log_softmax, softmax, kl_div
logger = logging.getLogger(__name__)

from unicore.modules.layer_norm import LayerNorm
from unimol.losses.tta_loss import IBSLoss

def cal_dist_loss(sample, dist, masked_tokens, target_key, weights,normalize=False):
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
        reduction="none",
        beta=1.0,
    )
    masked_dist_loss = (masked_dist_loss * weights[torch.where(dist_masked_tokens!=0)[0]][torch.where(non_pad_pos!=0)[0]]).mean()
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
def collect_params( model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    # model.requires_grad_(False)
    for nm, m in model.named_modules():
        if isinstance(m, LayerNorm):
            # if isinstance(m, (nn.BatchNorm2d, InPlaceABN)):
            # print(nm)
            m.requires_grad_(True)
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    p.requires_grad_(True)# weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    for name,par in model.named_parameters():
        if name.startswith("lm_head") or name.startswith("pair2coord_proj") or name.startswith("dist_head") :
            par.requires_grad_(True)
            params.append(par)
            names.append(name)

    return set(params), set(names)

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


@register_task("tta")
class DrugCLIP(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="downstream data path",
        )
        parser.add_argument(
            "--finetune-mol-model",
            default=None,
            type=str,
            help="pretrained molecular model path",
        )
        parser.add_argument(
            "--finetune-pocket-model",
            default=None,
            type=str,
            help="pretrained pocket model path",
        )
        parser.add_argument(
            "--dist-threshold",
            type=float,
            default=6.0,
            help="threshold for the distance between the molecule and the pocket",
        )
        parser.add_argument(
            "--max-pocket-atoms",
            type=int,
            default=256,
            help="selected maximum number of atoms in a pocket",
        )
        parser.add_argument(
            "--test-model",
            default=False,
            type=Boolean,
            help="whether test model",
        )
        parser.add_argument("--reg", action="store_true", help="regression task")

        parser.add_argument(
            "--mask-prob",
            default=0.15,
            type=float,
            help="probability of replacing a token with mask",
        )
        parser.add_argument(
            "--leave-unmasked-prob",
            default=0.05,
            type=float,
            help="probability that a masked token is unmasked",
        )
        parser.add_argument(
            "--random-token-prob",
            default=0.05,
            type=float,
            help="probability of replacing a token with a random token",
        )
        parser.add_argument(
            "--noise-type",
            default="uniform",
            choices=["trunc_normal", "uniform", "normal", "none"],
            help="noise type in coordinate noise",
        )
        parser.add_argument(
            "--noise",
            default=1.0,
            type=float,
            help="coordinate noise for masked atoms",
        )
        parser.add_argument(
            "--remove-hydrogen",
            action="store_true",
            help="remove hydrogen atoms",
        )
        parser.add_argument(
            "--remove-polar-hydrogen",
            action="store_true",
            help="remove polar hydrogen atoms",
        )
        parser.add_argument(
            "--max-atoms",
            type=int,
            default=256,
            help="selected maximum number of atoms in a molecule",
        )
        parser.add_argument(
            "--dict-name",
            default="dict.txt",
            help="dictionary file",
        )
        parser.add_argument(
            "--only-polar",
            default=1,
            type=int,
            help="1: only polar hydrogen ; -1: all hydrogen ; 0: remove all hydrogen ",
        )

        parser.add_argument(
            "--mask-seed",
            default=42,
            type=int,
            help="mask make seed",
        )
        parser.add_argument(
            "--tta-time",
            default=1,
            type=int,
            help="tta time",
        )
        parser.add_argument(
            "--tta-time-p",
            default=1,
            type=int,
            help="tta time",
        )
        parser.add_argument(
            "--tta-lr",
            default=0.005,
            type=float,
            help="lr mol",
        )
        parser.add_argument(
            "--tta-lr-p",
            default=0.0001,
            type=float,
            help="lr pocket",
        )
        parser.add_argument(
            "--target-path",
            type=str,
            help="target path",
        )
        parser.add_argument(
            "--mol-path",
            type=str,
            help="mol path",
        )
        parser.add_argument(
            "--checkpoint-path",
            type=str,
            help="checkpoint path",
        )


    def __init__(self, args, dictionary, pocket_dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.pocket_dictionary = pocket_dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        self.pocket_mask_idx = pocket_dictionary.add_symbol("[MASK]", is_special=True)
        self.mol_reps = None
        self.keys = None

    @classmethod
    def setup_task(cls, args, **kwargs):
        mol_dictionary = Dictionary.load(os.path.join(args.data, "dict_mol.txt"))
        pocket_dictionary = Dictionary.load(os.path.join(args.data, "dict_pkt.txt"))
        logger.info("ligand dictionary: {} types".format(len(mol_dictionary)))
        logger.info("pocket dictionary: {} types".format(len(pocket_dictionary)))
        return cls(args, mol_dictionary, pocket_dictionary)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        'smi','pocket','atoms','coordinates','pocket_atoms','pocket_coordinates'
        Args:
            split (str): name of the data scoure (e.g., bppp)
        """
        data_path = os.path.join(self.args.data, split + ".lmdb")
        dataset = LMDBDataset(data_path)
        if split.startswith("train"):
            smi_dataset = KeyDataset(dataset, "smi")
            poc_dataset = KeyDataset(dataset, "pocket")

            dataset = AffinityDataset(
                dataset,
                self.args.seed,
                "atoms",
                "coordinates",
                "pocket_atoms",
                "pocket_coordinates",
                "label",
                True,
            )
            tgt_dataset = KeyDataset(dataset, "affinity")

        else:

            dataset = AffinityDataset(
                dataset,
                self.args.seed,
                "atoms",
                "coordinates",
                "pocket_atoms",
                "pocket_coordinates",
                "label",
            )
            tgt_dataset = KeyDataset(dataset, "affinity")
            smi_dataset = KeyDataset(dataset, "smi")
            poc_dataset = KeyDataset(dataset, "pocket")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        dataset = RemoveHydrogenPocketDataset(
            dataset,
            "pocket_atoms",
            "pocket_coordinates",
            True,
            True,
        )
        dataset = CroppingPocketDataset(
            dataset,
            self.seed,
            "pocket_atoms",
            "pocket_coordinates",
            self.args.max_pocket_atoms,
        )

        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, True)

        apo_dataset = NormalizeDataset(dataset, "coordinates")
        apo_dataset = NormalizeDataset(apo_dataset, "pocket_coordinates")

        src_dataset = KeyDataset(apo_dataset, "atoms")
        mol_len_dataset = LengthDataset(src_dataset)
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(apo_dataset, "coordinates")

        mol_expand_dataset = MaskPointsDataset(
            src_dataset,
            coord_dataset,
            self.dictionary,
            pad_idx=self.dictionary.pad(),
            mask_idx=self.mask_idx,
            noise_type=self.args.noise_type,
            noise=self.args.noise,
            seed=self.args.mask_seed,
            mask_prob=self.args.mask_prob,
            leave_unmasked_prob=self.args.leave_unmasked_prob,
            random_token_prob=self.args.random_token_prob,
        )
        mol_encoder_token_dataset = KeyDataset(mol_expand_dataset, "atoms")
        mol_encoder_target_dataset = KeyDataset(mol_expand_dataset, "targets")
        mol_encoder_coord_dataset = KeyDataset(mol_expand_dataset, "coordinates")

        mol_mask_src_dataset = PrependAndAppend(
            mol_encoder_token_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        mol_mask_tgt_dataset = PrependAndAppend(
            mol_encoder_target_dataset, self.dictionary.pad(), self.dictionary.pad()
        )
        mol_mask_encoder_coord_dataset = PrependAndAppend(mol_encoder_coord_dataset, 0.0, 0.0)
        mol_mask_encoder_distance_dataset = DistanceDataset(mol_mask_encoder_coord_dataset)
        mol_mask_edge_type = EdgeTypeDataset(mol_mask_src_dataset, len(self.dictionary))

        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)

        src_pocket_dataset = KeyDataset(apo_dataset, "pocket_atoms")
        pocket_len_dataset = LengthDataset(src_pocket_dataset)
        src_pocket_dataset = TokenizeDataset(
            src_pocket_dataset,
            self.pocket_dictionary,
            max_seq_len=self.args.max_seq_len,
        )
        coord_pocket_dataset = KeyDataset(apo_dataset, "pocket_coordinates")

        pocket_expand_dataset = MaskPointsDataset(
            src_pocket_dataset,
            coord_pocket_dataset,
            self.pocket_dictionary,
            pad_idx=self.pocket_dictionary.pad(),
            mask_idx=self.pocket_mask_idx,
            noise_type=self.args.noise_type,
            noise=self.args.noise,
            seed=self.args.mask_seed,
            mask_prob=self.args.mask_prob,
            leave_unmasked_prob=self.args.leave_unmasked_prob,
            random_token_prob=self.args.random_token_prob,
        )
        pocket_encoder_token_dataset = KeyDataset(pocket_expand_dataset, "atoms")
        pocket_encoder_target_dataset = KeyDataset(pocket_expand_dataset, "targets")
        pocket_encoder_coord_dataset = KeyDataset(pocket_expand_dataset, "coordinates")

        pocket_mask_src_dataset = PrependAndAppend(
            pocket_encoder_token_dataset, self.pocket_dictionary.bos(), self.pocket_dictionary.eos()
        )
        pocket_mask_tgt_dataset = PrependAndAppend(
            pocket_encoder_target_dataset, self.pocket_dictionary.pad(), self.pocket_dictionary.pad()
        )
        pocket_mask_encoder_coord_dataset = PrependAndAppend(pocket_encoder_coord_dataset, 0.0, 0.0)
        pocket_mask_encoder_distance_dataset = DistanceDataset(pocket_mask_encoder_coord_dataset)
        pocket_mask_edge_type = EdgeTypeDataset(pocket_mask_src_dataset, len(self.pocket_dictionary))

        src_pocket_dataset = PrependAndAppend(
            src_pocket_dataset,
            self.pocket_dictionary.bos(),
            self.pocket_dictionary.eos(),
        )
        pocket_edge_type = EdgeTypeDataset(
            src_pocket_dataset, len(self.pocket_dictionary)
        )
        coord_pocket_dataset = FromNumpyDataset(coord_pocket_dataset)
        distance_pocket_dataset = DistanceDataset(coord_pocket_dataset)
        coord_pocket_dataset = PrependAndAppend(coord_pocket_dataset, 0.0, 0.0)
        distance_pocket_dataset = PrependAndAppend2DDataset(
            distance_pocket_dataset, 0.0
        )






        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "mol_src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "mol_src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "mol_src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                    "pocket_src_tokens": RightPadDataset(
                        src_pocket_dataset,
                        pad_idx=self.pocket_dictionary.pad(),
                    ),
                    "pocket_src_distance": RightPadDataset2D(
                        distance_pocket_dataset,
                        pad_idx=0,
                    ),
                    "pocket_src_edge_type": RightPadDataset2D(
                        pocket_edge_type,
                        pad_idx=0,
                    ),
                    "pocket_src_coord": RightPadDatasetCoord(
                        coord_pocket_dataset,
                        pad_idx=0,
                    ),
                    "mol_len": RawArrayDataset(mol_len_dataset),
                    "pocket_len": RawArrayDataset(pocket_len_dataset)
                },
                "target": {
                    "finetune_target": RawLabelDataset(tgt_dataset),
                },
                "smi_name": RawArrayDataset(smi_dataset),
                "pocket_name": RawArrayDataset(poc_dataset),
                "mol_mask_input": {
                    "src_tokens": RightPadDataset(
                        mol_mask_src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "src_coord": RightPadDatasetCoord(
                        mol_mask_encoder_coord_dataset,
                        pad_idx=0),
                    "src_distance": RightPadDataset2D(
                        mol_mask_encoder_distance_dataset,
                        pad_idx=0,
                    ),
                    "src_edge_type": RightPadDataset2D(
                        mol_mask_edge_type,
                        pad_idx=0,
                    ),
                    "tokens_target": RightPadDataset(
                        mol_mask_tgt_dataset, pad_idx=self.dictionary.pad()),
                    "distance_target": RightPadDataset2D(distance_dataset, pad_idx=0),
                    "coord_target": RightPadDatasetCoord(coord_dataset, pad_idx=0),
                },
                "pocket_mask_input": {
                    "src_tokens": RightPadDataset(
                        pocket_mask_src_dataset,
                        pad_idx=self.pocket_dictionary.pad(),
                    ),
                    "src_coord": RightPadDatasetCoord(
                        pocket_mask_encoder_coord_dataset,
                        pad_idx=0),
                    "src_distance": RightPadDataset2D(
                        pocket_mask_encoder_distance_dataset,
                        pad_idx=0,
                    ),
                    "src_edge_type": RightPadDataset2D(
                        pocket_mask_edge_type,
                        pad_idx=0,
                    ),
                    "tokens_target": RightPadDataset(
                        pocket_mask_tgt_dataset, pad_idx=self.pocket_dictionary.pad()),
                    "distance_target": RightPadDataset2D(distance_pocket_dataset, pad_idx=0),
                    "coord_target": RightPadDatasetCoord(coord_pocket_dataset, pad_idx=0),
                }
            },
        )
        if split == "train":
            with data_utils.numpy_seed(self.args.seed):
                shuffle = np.random.permutation(len(src_dataset))

            self.datasets[split] = SortDataset(
                nest_dataset,
                sort_order=[shuffle],
            )
            self.datasets[split] = ResamplingDataset(
                self.datasets[split]
            )
        else:
            self.datasets[split] = nest_dataset

    def load_mols_dataset(self, data_path, atoms, coords, **kwargs):

        dataset = LMDBDataset(data_path)
        label_dataset = KeyDataset(dataset, "label")
        dataset = AffinityMolDataset(
            dataset,
            self.args.seed,
            atoms,
            coords,
            False,
        )

        smi_dataset = KeyDataset(dataset, "smi")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, True)

        apo_dataset = NormalizeDataset(dataset, "coordinates")

        src_dataset = KeyDataset(apo_dataset, "atoms")
        len_dataset = LengthDataset(src_dataset)
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(apo_dataset, "coordinates")
        expand_dataset = MaskPointsDataset(
            src_dataset,
            coord_dataset,
            self.dictionary,
            pad_idx=self.dictionary.pad(),
            mask_idx=self.mask_idx,
            noise_type=self.args.noise_type,
            noise=self.args.noise,
            seed=self.args.mask_seed,
            mask_prob=self.args.mask_prob,
            leave_unmasked_prob=self.args.leave_unmasked_prob,
            random_token_prob=self.args.random_token_prob,
        )
        encoder_token_dataset = KeyDataset(expand_dataset, "atoms")
        encoder_target_dataset = KeyDataset(expand_dataset, "targets")
        encoder_coord_dataset = KeyDataset(expand_dataset, "coordinates")



        mask_src_dataset = PrependAndAppend(
            encoder_token_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        mask_tgt_dataset = PrependAndAppend(
            encoder_target_dataset, self.dictionary.pad(), self.dictionary.pad()
        )
        mask_encoder_coord_dataset = PrependAndAppend(encoder_coord_dataset, 0.0, 0.0)
        mask_encoder_distance_dataset = DistanceDataset(mask_encoder_coord_dataset)
        mask_edge_type = EdgeTypeDataset(mask_src_dataset, len(self.dictionary))


        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)

        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)


        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "mol_src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "mol_src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "mol_src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                },
                "smi_name": RawArrayDataset(smi_dataset),
                "target": RawArrayDataset(label_dataset),
                "mol_len": RawArrayDataset(len_dataset),
                "mask_input": {
                    "src_tokens": RightPadDataset(
                        mask_src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "src_coord": RightPadDatasetCoord(
                        mask_encoder_coord_dataset,
                        pad_idx=0),
                    "src_distance": RightPadDataset2D(
                        mask_encoder_distance_dataset,
                        pad_idx=0,
                    ),
                    "src_edge_type": RightPadDataset2D(
                        mask_edge_type,
                        pad_idx=0,
                    ),
                    "tokens_target": RightPadDataset(
                        mask_tgt_dataset, pad_idx=self.dictionary.pad()),
                    "distance_target": RightPadDataset2D(distance_dataset, pad_idx=0),
                    "coord_target": RightPadDatasetCoord(coord_dataset, pad_idx=0),
                }
            },

        )
        return nest_dataset

    def load_retrieval_mols_dataset(self, data_path, atoms, coords, **kwargs):

        dataset = LMDBDataset(data_path)
        dataset = AffinityMolDataset(
            dataset,
            self.args.seed,
            atoms,
            coords,
            False,
        )

        smi_dataset = KeyDataset(dataset, "smi")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, True)

        apo_dataset = NormalizeDataset(dataset, "coordinates")

        src_dataset = KeyDataset(apo_dataset, "atoms")
        len_dataset = LengthDataset(src_dataset)
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(apo_dataset, "coordinates")
        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "mol_src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "mol_src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "mol_src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                },
                "smi_name": RawArrayDataset(smi_dataset),
                "mol_len": RawArrayDataset(len_dataset),
            },
        )
        return nest_dataset

    def load_pockets_dataset(self, data_path, **kwargs):

        dataset = LMDBDataset(data_path)

        dataset = AffinityPocketDataset(
            dataset,
            self.args.seed,
            "pocket_atoms",
            "pocket_coordinates",
            False,
            "pocket"
        )
        poc_dataset = KeyDataset(dataset, "pocket")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        dataset = RemoveHydrogenPocketDataset(
            dataset,
            "pocket_atoms",
            "pocket_coordinates",
            True,
            True,
        )
        dataset = CroppingPocketDataset(
            dataset,
            self.seed,
            "pocket_atoms",
            "pocket_coordinates",
            self.args.max_pocket_atoms,
        )

        apo_dataset = NormalizeDataset(dataset, "pocket_coordinates")

        src_pocket_dataset = KeyDataset(apo_dataset, "pocket_atoms")
        len_dataset = LengthDataset(src_pocket_dataset)
        src_pocket_dataset = TokenizeDataset(
            src_pocket_dataset,
            self.pocket_dictionary,
            max_seq_len=self.args.max_seq_len,
        )
        coord_pocket_dataset = KeyDataset(apo_dataset, "pocket_coordinates")

        expand_dataset = MaskPointsDataset(
            src_pocket_dataset,
            coord_pocket_dataset,
            self.pocket_dictionary,
            pad_idx=self.pocket_dictionary.pad(),
            mask_idx=self.pocket_mask_idx,
            noise_type=self.args.noise_type,
            noise=self.args.noise,
            seed=self.args.mask_seed,
            mask_prob=self.args.mask_prob,
            leave_unmasked_prob=self.args.leave_unmasked_prob,
            random_token_prob=self.args.random_token_prob,
        )
        encoder_token_dataset = KeyDataset(expand_dataset, "atoms")
        encoder_target_dataset = KeyDataset(expand_dataset, "targets")
        encoder_coord_dataset = KeyDataset(expand_dataset, "coordinates")

        mask_src_dataset = PrependAndAppend(
            encoder_token_dataset, self.pocket_dictionary.bos(), self.pocket_dictionary.eos()
        )
        mask_tgt_dataset = PrependAndAppend(
            encoder_target_dataset, self.pocket_dictionary.pad(), self.pocket_dictionary.pad()
        )
        mask_encoder_coord_dataset = PrependAndAppend(encoder_coord_dataset, 0.0, 0.0)
        mask_encoder_distance_dataset = DistanceDataset(mask_encoder_coord_dataset)
        mask_edge_type = EdgeTypeDataset(mask_src_dataset, len(self.pocket_dictionary))

        src_pocket_dataset = PrependAndAppend(
            src_pocket_dataset,
            self.pocket_dictionary.bos(),
            self.pocket_dictionary.eos(),
        )
        pocket_edge_type = EdgeTypeDataset(
            src_pocket_dataset, len(self.pocket_dictionary)
        )
        coord_pocket_dataset = FromNumpyDataset(coord_pocket_dataset)
        distance_pocket_dataset = DistanceDataset(coord_pocket_dataset)
        coord_pocket_dataset = PrependAndAppend(coord_pocket_dataset, 0.0, 0.0)
        distance_pocket_dataset = PrependAndAppend2DDataset(
            distance_pocket_dataset, 0.0
        )

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "pocket_src_tokens": RightPadDataset(
                        src_pocket_dataset,
                        pad_idx=self.pocket_dictionary.pad(),
                    ),
                    "pocket_src_distance": RightPadDataset2D(
                        distance_pocket_dataset,
                        pad_idx=0,
                    ),
                    "pocket_src_edge_type": RightPadDataset2D(
                        pocket_edge_type,
                        pad_idx=0,
                    ),
                    "pocket_src_coord": RightPadDatasetCoord(
                        coord_pocket_dataset,
                        pad_idx=0,
                    ),
                },
                "pocket_name": RawArrayDataset(poc_dataset),
                "pocket_len": RawArrayDataset(len_dataset),
                "mask_input": {
                    "src_tokens": RightPadDataset(
                        mask_src_dataset,
                        pad_idx=self.pocket_dictionary.pad(),
                    ),
                    "src_coord": RightPadDatasetCoord(
                        mask_encoder_coord_dataset,
                        pad_idx=0),
                    "src_distance": RightPadDataset2D(
                        mask_encoder_distance_dataset,
                        pad_idx=0,
                    ),
                    "src_edge_type": RightPadDataset2D(
                        mask_edge_type,
                        pad_idx=0,
                    ),
                    "tokens_target": RightPadDataset(
                        mask_tgt_dataset, pad_idx=self.pocket_dictionary.pad()),
                    "distance_target": RightPadDataset2D(distance_pocket_dataset, pad_idx=0),
                    "coord_target": RightPadDatasetCoord(coord_pocket_dataset, pad_idx=0),
                }
            },
        )
        return nest_dataset

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)


        if args.finetune_mol_model is not None:
            print("load pretrain model weight from...", args.finetune_mol_model)
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.finetune_mol_model,
            )
            model.mol_model.load_state_dict(state["model"], strict=False)

        if args.finetune_pocket_model is not None:
            print("load pretrain model weight from...", args.finetune_pocket_model)
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.finetune_pocket_model,
            )
            model.pocket_model.load_state_dict(state["model"], strict=False)
        # model.load_state_dict(torch.load('checkpoint_best.pt')["model"], strict=False)
        #
        return model

    def train_step(
            self, sample, model, loss, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *loss*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~unicore.data.UnicoreDataset`.
            model (~unicore.models.BaseUnicoreModel): the model
            loss (~unicore.losses.UnicoreLoss): the loss
            optimizer (~unicore.optim.UnicoreOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        # """

        # optimizer_mol = torch.optim.SGD(model.parameters(), lr=self.args.tta_lr, momentum=0.9)
        # optimizer_pocket = torch.optim.SGD(model.parameters(), lr=self.args.tta_lr_p, momentum=0.9)

        model.train()
        model.set_num_updates(update_num)
        model.requires_grad_(False)
        params_mol, param_names_mol = collect_params(model.mol_model)
        params_pocket, param_names_pocket = collect_params(model.pocket_model)
        params_mol = list(params_mol) + list(model.weight) + list(model.kl_head_mol.parameters()) + list(model.simclr_head_mol.parameters())+list(model.weight_nets1.parameters())
        params_pocket = list(params_pocket) + list(model.weight_p) + list(model.kl_head_pocket.parameters()) + list(model.simclr_head_pocket.parameters())+list(model.weight_nets2.parameters())
        model.weight.requires_grad , model.weight_p.requires_grad  = True,True
        model.kl_head_mol.requires_grad_(True)
        model.simclr_head_mol.requires_grad_(True)
        model.kl_head_pocket.requires_grad_(True)
        model.simclr_head_pocket.requires_grad_(True)
        model.weight_nets1.requires_grad_(True)
        model.weight_nets2.requires_grad_(True)
        optimizer_mol = torch.optim.SGD(params_mol, lr=self.args.tta_lr, momentum=0.9)
        optimizer_pocket = torch.optim.SGD(params_pocket, lr=self.args.tta_lr_p, momentum=0.9)
        # optimizer_mol = unicore.optim.adam.Adam(params_mol, lr=self.args.tta_lr)
        # optimizer_pocket = unicore.optim.adam.Adam(params_pocket, lr=self.args.tta_lr_p)
        with torch.autograd.profiler.record_function("forward"):
            loss_mol, logging_output = loss(model, sample,flag ='ssl_train_mol')
        if ignore_grad:
            loss_mol *= 0
        with torch.autograd.profiler.record_function("backward"):
            if not torch.isnan(loss_mol):
                loss_mol.backward()
                optimizer_mol.step()
                optimizer_mol.zero_grad()
            if torch.isnan(loss_mol):
                print('nan')
        with torch.autograd.profiler.record_function("forward"):
            loss_pocket, logging_output = loss(model, sample, flag='ssl_train_pocket')
        if ignore_grad:
            loss_pocket *= 0
        with torch.autograd.profiler.record_function("backward"):
            if not torch.isnan(loss_pocket):
                loss_pocket.backward()
                optimizer_pocket.step()
                optimizer_pocket.zero_grad()
            if torch.isnan(loss_pocket):
                print('nan')
        model.requires_grad_(True)
        with torch.autograd.profiler.record_function("forward"):
            loss_total, sample_size, logging_output = loss(model, sample,flag ='main_train')
        if ignore_grad:
            loss_total *= 0
        with torch.autograd.profiler.record_function("backward"):
            if not torch.isnan(loss_total):
                optimizer.backward(loss_total)
            if torch.isnan(loss_total):
                print('nan')

        return loss_total, sample_size, logging_output

    def valid_step(self, sample, model, loss, test=False):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = loss(model, sample)
        return loss, sample_size, logging_output

    def test_dude_target(self, target, model, **kwargs):

        data_path = self.args.mol_path + target + "/mols.lmdb"
        # data_path = "./data/DUD-E/raw/all/" + target + "/mols.lmdb"
        # data_path = "./data/casf2016/" + target + "/mols.lmdb"
        # data_path = "./data/DEKOIS2.0/" + target + "/mol.lmdb"
        mol_dataset = self.load_mols_dataset(data_path, "atoms", "coordinates")
        num_data = len(mol_dataset)
        bsz =64
        print(num_data // bsz)
        mol_reps = []
        mol_names = []
        labels = []

        # generate mol data
        mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater)
        model.requires_grad_(False)
        params_mol, param_names_mol = collect_params(model.mol_model)
        params_pocket, param_names_pocket = collect_params(model.pocket_model)
        optimizer_mol = torch.optim.SGD(params_mol, lr=self.args.tta_lr, momentum=0.9)
        optimizer_pocket = torch.optim.SGD(params_pocket, lr=self.args.tta_lr_p, momentum=0.9)
        model_original_weight = unicore.utils.move_to_cuda(torch.load(self.args.checkpoint_path)["model"])
        for _, sample in enumerate(tqdm(mol_data)):

            model.train()
            model.load_state_dict(model_original_weight, strict=False)
            sample = unicore.utils.move_to_cuda(sample)
            for _ in range(self.args.tta_time):

                # loss, logging_output = loss_func(model, sample,flag ='ssl_train_mol')
                dist = sample["net_input"]["mol_src_distance"]
                et = sample["net_input"]["mol_src_edge_type"]
                st = sample["net_input"]["mol_src_tokens"]


                mask_coord = sample['mask_input']['src_coord']
                mask_et = sample['mask_input']['src_edge_type']
                mask_dist = sample['mask_input']['src_distance']
                mask_st = sample['mask_input']['src_tokens']

                masked_tokens = sample['mask_input']["tokens_target"].ne(model.mol_model.padding_idx)

                [(mol_features_kl,
                  mol_features_simclr),
                 (logits_encoder,
                  encoder_distance,
                  encoder_coord,
                  x_norm,
                  delta_encoder_pair_rep_norm,
                  mask_mol_features_simclr,
                  mol_features_mlp,
                  mol_list
                  )] = model(mol_src_tokens=st,
                    mol_src_distance=dist,
                    mol_src_edge_type=et,
                    pocket_src_tokens=None,
                    pocket_src_distance=None,
                    pocket_src_edge_type=None,

                    mask_mol_src_tokens=mask_st,
                    mask_mol_src_distance=mask_dist,
                    mask_mol_src_coord=mask_coord,
                    mask_mol_src_edge_type=mask_et,

                    mask_pocket_src_tokens=None,
                    mask_pocket_src_distance=None,
                    mask_pocket_src_coord=None,
                    mask_pocket_src_edge_type=None,

                    mol_tta=True,
                    mol=True,
                    mol_features_only = True,
                    encoder_masked_tokens=masked_tokens)


                weights_mol = [torch.sigmoid(net(mol_features_mlp)) for net in model.weight_nets1]
                weights_tensor = torch.stack(weights_mol, dim=1)  # 将每个网络的输出按列堆叠
                weights = F.softmax(1 / (2 * weights_tensor.float() ** 2), dim=1) * 10
                # weights = torch.ones(7,).to('cuda')
                # weights = [torch.sigmoid(net(mol_features_mlp)) for net in model.weight_nets1]
                # weights = torch.stack(weights, dim=1)

                targets = sample['mask_input']["tokens_target"]
                loss = 0.0
                logging_output = {}

                if self.args.mol.masked_token_loss > 0 and masked_tokens is not None:
                    targets = targets[masked_tokens]

                    masked_token_loss = F.nll_loss(
                        F.log_softmax(logits_encoder, dim=-1, dtype=torch.float32),
                        targets,
                        ignore_index=model.mol_model.padding_idx,
                        reduction="none",
                    )
                    masked_token_loss = (masked_token_loss * weights[:, 0, :].squeeze(-1)[
                        torch.where(masked_tokens != 0)[0]]).mean()
                    masked_pred = logits_encoder.argmax(dim=-1)
                    masked_hit = (masked_pred == targets).long().sum()
                    sample_size = masked_tokens.long().sum()
                    masked_cnt = sample_size
                    loss = loss + masked_token_loss * self.args.mol.masked_token_loss
                    logging_output = {
                        "sample_size": 1,
                        "bsz": sample['mask_input']["tokens_target"].size(0),
                        "seq_len": sample['mask_input']["tokens_target"].size(1)
                                   * sample['mask_input']["tokens_target"].size(0),
                        "masked_token_loss": masked_token_loss.data,
                        "masked_token_hit": masked_hit.data,
                        "masked_token_cnt": masked_cnt,
                    }

                if self.args.mol.masked_coord_loss > 0 and encoder_coord is not None:
                    # real = mask + delta
                    coord_target = sample['mask_input']["coord_target"]
                    masked_coord_loss = F.smooth_l1_loss(
                        encoder_coord[masked_tokens].view(-1, 3).float(),
                        coord_target[masked_tokens].view(-1, 3),
                        reduction="none",
                        beta=1.0,
                    )
                    masked_coord_loss = (masked_coord_loss.mean(-1) * weights[:, 1, :].squeeze(-1)[
                        torch.where(masked_tokens != 0)[0]]).mean()

                    loss = loss + masked_coord_loss * self.args.mol.masked_coord_loss
                    # restore the scale of loss for displaying
                    logging_output["masked_coord_loss"] = masked_coord_loss.data

                if self.args.mol.masked_dist_loss > 0 and encoder_distance is not None:
                    dist_masked_tokens = masked_tokens
                    masked_dist_loss = cal_dist_loss(
                        sample, encoder_distance, dist_masked_tokens, 'mask_input', weights[:, 2, :].squeeze(-1),
                        normalize=True
                    )
                    loss = loss + masked_dist_loss * self.args.mol.masked_dist_loss
                    logging_output["masked_dist_loss"] = masked_dist_loss.data

                if self.args.mol.x_norm_loss > 0 and x_norm is not None:
                    loss = loss + self.args.mol.x_norm_loss * x_norm * weights[:, 3, :].squeeze(-1).mean()
                    logging_output["x_norm_loss"] = x_norm.data

                if (
                        self.args.mol.delta_pair_repr_norm_loss > 0
                        and delta_encoder_pair_rep_norm is not None
                ):
                    loss = (
                            loss + self.args.mol.delta_pair_repr_norm_loss * delta_encoder_pair_rep_norm * weights[:, 4,
                                                                                                           :].squeeze(
                        -1).mean()
                    )
                    logging_output[
                        "delta_pair_repr_norm_loss"
                    ] = delta_encoder_pair_rep_norm.data

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

                ####fixed weights ablation
                # weights = model.weight
                # targets = sample['mask_input']["tokens_target"]
                # if masked_tokens is not None:
                #     targets = targets[masked_tokens]
                # masked_token_loss = F.nll_loss(
                #     F.log_softmax(logits_encoder, dim=-1, dtype=torch.float32),
                #     targets,
                #     ignore_index=model.mol_model.padding_idx,
                #     reduction="mean",
                # )
                # masked_pred = logits_encoder.argmax(dim=-1)
                # masked_hit = (masked_pred == targets).long().sum()
                # sample_size = masked_tokens.long().sum()
                # masked_cnt = sample_size
                # loss = masked_token_loss * self.args.mol.masked_token_loss * weights[0]
                # logging_output = {
                #     "sample_size": 1,
                #     "bsz": sample['mask_input']["tokens_target"].size(0),
                #     "seq_len": sample['mask_input']["tokens_target"].size(1)
                #                * sample['mask_input']["tokens_target"].size(0),
                #     "masked_token_loss": masked_token_loss.data,
                #     "masked_token_hit": masked_hit.data,
                #     "masked_token_cnt": masked_cnt,
                # }
                # if encoder_coord is not None:
                #     # real = mask + delta
                #     coord_target = sample['mask_input']["coord_target"]
                #     masked_coord_loss = F.smooth_l1_loss(
                #         encoder_coord[masked_tokens].view(-1, 3).float(),
                #         coord_target[masked_tokens].view(-1, 3),
                #         reduction="mean",
                #         beta=1.0,
                #     )
                #     masked_coord_loss = F.smooth_l1_loss(
                #         encoder_coord[masked_tokens].view(-1, 3).float(),
                #         coord_target[masked_tokens].view(-1, 3),
                #         reduction="mean",
                #         beta=1.0,
                #     )
                #
                #     loss = loss + masked_coord_loss * self.args.mol.masked_coord_loss * weights[1]
                #     # restore the scale of loss for displaying
                #     logging_output["masked_coord_loss"] = masked_coord_loss.data
                #
                # if encoder_distance is not None:
                #     dist_masked_tokens = masked_tokens
                #     masked_dist_loss = cal_dist_loss_ori(
                #         sample, encoder_distance, dist_masked_tokens, 'mask_input', normalize=True
                #     )
                #     loss = loss + masked_dist_loss * self.args.mol.masked_dist_loss * weights[2]
                #     logging_output["masked_dist_loss"] = masked_dist_loss.data
                #
                # if self.args.mol.x_norm_loss > 0 and x_norm is not None:
                #     loss = loss + self.args.mol.x_norm_loss * x_norm * weights[3]
                #     logging_output["x_norm_loss"] = x_norm.data
                #
                # if (
                #         self.args.mol.delta_pair_repr_norm_loss > 0
                #         and delta_encoder_pair_rep_norm is not None
                # ):
                #     loss = (
                #             loss + self.args.mol.delta_pair_repr_norm_loss * delta_encoder_pair_rep_norm * weights[4]
                #     )
                #     logging_output[
                #         "delta_pair_repr_norm_loss"
                #     ] = delta_encoder_pair_rep_norm.data
                #
                #
                # if self.args.mol.kl_loss > 0:
                #     latent_dim = mol_features_kl.shape[-1]
                #
                #     uniform = (torch.ones(latent_dim) / latent_dim).to(mol_features_kl.device)
                #
                #     softmax_uniform = softmax(uniform / self.args.mol.kl_temperature, dim=0)
                #     softmax_uniform = softmax_uniform.unsqueeze(dim=0).repeat(mol_features_kl.shape[0], 1)
                #
                #     softmax_latents = log_softmax(mol_features_kl / self.args.mol.kl_temperature, dim=1)
                #     kl_loss = kl_div(softmax_latents.float(), softmax_uniform, reduction="batchmean")
                #
                #     loss = loss + self.args.mol.kl_loss * kl_loss * weights[5]
                # if self.args.mol.simclr_loss > 0:
                #     batch_size = mol_features_simclr.shape[0]
                #     combined_features = torch.cat([mol_features_simclr, mask_mol_features_simclr],
                #                                   dim=0)
                #     similarity_matrix = torch.mm(combined_features,
                #                                  combined_features.T)
                #     similarity_matrix = similarity_matrix / self.args.mol.simclr_temperature
                #     positive_pairs = torch.diag(similarity_matrix, batch_size)
                #     positive_pairs = torch.cat([positive_pairs, torch.diag(similarity_matrix, -batch_size)],
                #                                dim=0)
                #     mask = torch.eye(2 * batch_size, dtype=torch.bool, device=mol_features_simclr.device)
                #     similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))
                #     log_sum_exp_negatives = torch.logsumexp(similarity_matrix, dim=1)
                #     simclr_loss = -positive_pairs + log_sum_exp_negatives
                #     loss = loss + self.args.mol.simclr_loss * simclr_loss.mean() * weights[6]


                logging_output["loss"] = loss.data
                loss.backward()
                optimizer_mol.step()
                optimizer_mol.zero_grad()
            model.eval()
            with torch.no_grad():
                mol_padding_mask = st.eq(model.mol_model.padding_idx)
                mol_x = model.mol_model.embed_tokens(st)
                n_node = dist.size(-1)
                gbf_feature = model.mol_model.gbf(dist, et)
                gbf_result = model.mol_model.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                mol_outputs = model.mol_model.encoder(
                    mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
                )
                mol_encoder_rep = mol_outputs[0][:, 0, :]
                mol_emb = mol_encoder_rep
                mol_emb = model.mol_project(mol_encoder_rep)
                mol_emb = mol_emb / mol_emb.norm(dim=-1, keepdim=True)
                mol_emb = mol_emb.detach().cpu().numpy()
                mol_reps.append(mol_emb)
                mol_names.extend(sample["smi_name"])
                labels.extend(sample["target"].detach().cpu().numpy())
        mol_reps = np.concatenate(mol_reps, axis=0)
        labels = np.array(labels, dtype=np.int32)

        # generate pocket data
        data_path = self.args.mol_path + target + "/pocket.lmdb"
        # data_path = "./data/casf2016/" + target + "/pocket.lmdb"
        # data_path = "./data/DEKOIS2.0/" + target + "/pocket.lmdb"
        pocket_dataset = self.load_pockets_dataset(data_path)
        pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=bsz, collate_fn=pocket_dataset.collater)
        pocket_reps = []

        for _, sample in enumerate(tqdm(pocket_data)):
            model.train()
            model.load_state_dict(model_original_weight, strict=False)
            sample = unicore.utils.move_to_cuda(sample)
            for _ in range(self.args.tta_time_p):
                dist = sample["net_input"]["pocket_src_distance"]
                et = sample["net_input"]["pocket_src_edge_type"]
                st = sample["net_input"]["pocket_src_tokens"]

                mask_coord = sample['mask_input']['src_coord']
                mask_et = sample['mask_input']['src_edge_type']
                mask_dist = sample['mask_input']['src_distance']
                mask_st = sample['mask_input']['src_tokens']

                masked_tokens = sample['mask_input']["tokens_target"].ne(model.pocket_model.padding_idx)

                [(pocket_features_kl,
                    pocket_features_simclr),(
                    logits_encoder,
                    encoder_distance,
                    encoder_coord,
                    x_norm,
                    delta_encoder_pair_rep_norm,
                    mask_pocket_features_simclr,
                    pocket_features_mlp,
                    pocket_list
                )] = model(mol_src_tokens=None,
                           mol_src_distance=None,
                           mol_src_edge_type=None,
                           pocket_src_tokens=st,
                           pocket_src_distance=dist,
                           pocket_src_edge_type=et,

                           mask_mol_src_tokens=None,
                           mask_mol_src_distance=None,
                           mask_mol_src_coord=None,
                           mask_mol_src_edge_type=None,

                           mask_pocket_src_tokens=mask_st,
                           mask_pocket_src_distance=mask_dist,
                           mask_pocket_src_coord=mask_coord,
                           mask_pocket_src_edge_type=mask_et,

                           pocket_tta=True,
                           pocket=True,
                           pocket_features_only = True,
                           encoder_masked_tokens=masked_tokens
                           )

                targets = sample['mask_input']["tokens_target"]
                weights_pocket = [torch.sigmoid(net(pocket_features_mlp)).mean() for net in model.weight_nets2]
                weights_p = F.softmax(1 / (2 * torch.tensor(weights_pocket).float() ** 2), dim=0) * 10

                # weights_p = [torch.sigmoid(net(pocket_features_mlp)) for net in model.weight_nets2]
                # weights_p = torch.ones(7, ).to('cuda')
                # weights_p = model.weight_p
                loss=0.0
                logging_output = {}

                if self.args.pocket.masked_token_loss>0 and masked_tokens is not None:
                    targets = targets[masked_tokens]
                    masked_token_loss = F.nll_loss(
                        F.log_softmax(logits_encoder, dim=-1, dtype=torch.float32),
                        targets,
                        ignore_index=model.pocket_model.padding_idx,
                        reduction="mean",
                    )
                    masked_pred = logits_encoder.argmax(dim=-1)
                    masked_hit = (masked_pred == targets).long().sum()
                    sample_size = masked_tokens.long().sum()
                    masked_cnt = sample_size
                    loss = loss+ masked_token_loss * self.args.pocket.masked_token_loss * weights_p[0]
                    logging_output = {
                        "sample_size": 1,
                        "bsz": sample['mask_input']["tokens_target"].size(0),
                        "seq_len": sample['mask_input']["tokens_target"].size(1)
                                   * sample['mask_input']["tokens_target"].size(0),
                        "masked_token_loss": masked_token_loss.data,
                        "masked_token_hit": masked_hit.data,
                        "masked_token_cnt": masked_cnt,
                    }

                if self.args.pocket.masked_coord_loss>0 and encoder_coord is not None:
                    # real = mask + delta
                    coord_target = sample['mask_input']["coord_target"]
                    masked_coord_loss = F.smooth_l1_loss(
                        encoder_coord[masked_tokens].view(-1, 3).float(),
                        coord_target[masked_tokens].view(-1, 3),
                        reduction="mean",
                        beta=1.0,
                    )
                    loss = loss + masked_coord_loss * self.args.pocket.masked_coord_loss * weights_p[1]
                    # restore the scale of loss for displaying
                    logging_output["masked_coord_loss"] = masked_coord_loss.data

                if  self.args.pocket.masked_dist_loss >0 and encoder_distance is not None:
                    dist_masked_tokens = masked_tokens
                    masked_dist_loss = cal_dist_loss_ori(
                        sample, encoder_distance, dist_masked_tokens, 'mask_input', normalize=True
                    )
                    loss = loss + masked_dist_loss * self.args.pocket.masked_dist_loss * weights_p[2]
                    logging_output["masked_dist_loss"] = masked_dist_loss.data

                if self.args.pocket.x_norm_loss > 0 and x_norm is not None:
                    loss = loss + self.args.pocket.x_norm_loss * x_norm * weights_p[3]
                    logging_output["x_norm_loss"] = x_norm.data

                if (
                        self.args.pocket.delta_pair_repr_norm_loss > 0
                        and delta_encoder_pair_rep_norm is not None
                ):
                    loss = (
                            loss + self.args.pocket.delta_pair_repr_norm_loss * delta_encoder_pair_rep_norm * weights_p[4]
                    )
                    logging_output[
                        "delta_pair_repr_norm_loss"
                    ] = delta_encoder_pair_rep_norm.data

                if self.args.pocket.kl_loss>0:
                    latent_dim = pocket_features_kl.shape[-1]

                    uniform = (torch.ones(latent_dim) / latent_dim).to(pocket_features_kl.device)

                    softmax_uniform = softmax(uniform / self.args.pocket.kl_temperature, dim=0)
                    softmax_uniform = softmax_uniform.unsqueeze(dim=0).repeat(pocket_features_kl.shape[0], 1)

                    softmax_latents = log_softmax(pocket_features_kl / self.args.pocket.kl_temperature, dim=1)
                    kl_loss = kl_div(softmax_latents.float(), softmax_uniform, reduction="batchmean")
                    loss = loss + self.args.pocket.kl_loss * kl_loss * weights_p[5]
                if self.args.pocket.simclr_loss > 0:
                    batch_size = pocket_features_simclr.shape[0]
                    combined_features = torch.cat([pocket_features_simclr, mask_pocket_features_simclr],
                                                  dim=0)
                    similarity_matrix = torch.mm(combined_features,
                                                 combined_features.T)
                    similarity_matrix = similarity_matrix / self.args.mol.simclr_temperature
                    positive_pairs = torch.diag(similarity_matrix, batch_size)
                    positive_pairs = torch.cat([positive_pairs, torch.diag(similarity_matrix, -batch_size)],
                                               dim=0)
                    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=pocket_features_simclr.device)
                    similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))
                    log_sum_exp_negatives = torch.logsumexp(similarity_matrix, dim=1)
                    simclr_loss = -positive_pairs + log_sum_exp_negatives
                    loss = loss + self.args.pocket.simclr_loss * simclr_loss.mean() * weights_p[6]

                logging_output["loss"] = loss.data
                loss.backward()
                optimizer_pocket.step()
                optimizer_pocket.zero_grad()
            model.eval()

            with torch.no_grad():
                pocket_padding_mask = st.eq(model.pocket_model.padding_idx)
                pocket_x = model.pocket_model.embed_tokens(st)
                n_node = dist.size(-1)
                gbf_feature = model.pocket_model.gbf(dist, et)
                gbf_result = model.pocket_model.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                pocket_outputs = model.pocket_model.encoder(
                    pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
                )
                pocket_encoder_rep = pocket_outputs[0][:, 0, :]
                # pocket_emb = pocket_encoder_rep
                pocket_emb = model.pocket_project(pocket_encoder_rep)
                pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
                pocket_emb = pocket_emb.detach().cpu().numpy()
                pocket_reps.append(pocket_emb)
        pocket_reps = np.concatenate(pocket_reps, axis=0)
        print(pocket_reps.shape)
        res = pocket_reps @ mol_reps.T

        res_single = res.max(axis=0)

        auc, bedroc, ef_list, re_list = cal_metrics(labels, res_single, 80.5)

        print(target)

        print(np.sum(labels), len(labels) - np.sum(labels))

        return auc, bedroc, ef_list, re_list, res_single, labels,mol_reps,pocket_reps

    def test_dude(self, model, **kwargs):

        targets = os.listdir(self.args.target_path)
        # targets = os.listdir("./data/DUD-E/raw/all/")
        # targets = os.listdir("./data/casf2016/")
        # targets = os.listdir("./data/DEKOIS2.0/")
        auc_list = []
        bedroc_list = []
        ef_list = []
        res_list = []
        labels_list = []
        re_list = {
            "0.005": [],
            "0.01": [],
            "0.02": [],
            "0.05": [],
        }
        ef_list = {
            "0.005": [],
            "0.01": [],
            "0.02": [],
            "0.05": [],
        }
        for i, target in enumerate(targets):
            auc, bedroc, ef, re, res_single, labels,mol_reps,pocket_reps = self.test_dude_target(target, model)
            auc_list.append(auc)
            bedroc_list.append(bedroc)
            for key in ef:
                ef_list[key].append(ef[key])
            for key in re_list:
                re_list[key].append(re[key])
            res_list.append(res_single)
            labels_list.append(labels)
        res = np.concatenate(res_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        print("auc mean", np.mean(auc_list))
        print("bedroc mean", np.mean(bedroc_list))

        for key in ef_list:
            print("ef", key, "mean", np.mean(ef_list[key]))

        for key in re_list:
            print("re", key, "mean", np.mean(re_list[key]))

        # save printed results

        return



















