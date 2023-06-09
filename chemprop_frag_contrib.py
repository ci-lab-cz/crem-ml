import chemprop
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from rdkit import Chem
from torch import Tensor
import torch
import numpy as np
from  sirms.files import LoadFragments
from collections import OrderedDict
import chemprop_predict
import typing


mol_frag_sep = "###"


def main_params(x_fname:str,
                out_fname:str,
                model_dir:str,
                model_type:str,
                model_names=["MPNN"],  # not used
                prop_names=None,
                activity_file=None,
                verbose=False,
                save_pred=False,
                save_frag_ids=True
                ):
    """

    :param x_fname: input csv file with descriptors of fragments
    :param out_fname: output file with contributions in same format as SPCI

    :param model_names: list with model names, i.e.  ['MPNN']; no effect; used only for consistency wuith SPCI
    :param model_dir: path to models directories
    :param prop_names: no effect; used only for consistency with SPCI
    :param model_type: 'reg' or 'class'
    :param activity_file:no effect
    :param verbose::no effect
    :param save_pred: save file with predictions for molecules and counterfragments (complement to target fragments)
    :param save_frag_ids: save frag id in each molecule. it will return additional column named Frag_ID
    """
    frag_preds = chemprop_predict.main_params(x_fname=x_fname,
                                  out_fname=out_fname,
                                  model_dir=model_dir,
                                  model_type=model_type,
                                  save_pred=save_pred,
                                  fragments_mode=True,
                                  num_frag_id=True
                                  )


    # rename copounds->compound (spci consuistent)
    frag_preds.columns = ['Compound']+ frag_preds.columns[1:].tolist()
    # add mol_pred column (pandas)
    compound_preds = frag_preds.loc[pd.isnull(frag_preds.Fragment), ["Compound", "consensus"]]

    frag_preds = pd.merge(frag_preds, compound_preds, on="Compound", suffixes=["_f", "_c"])

    # diff
    frag_preds["Contribution_value"] = frag_preds['consensus_c'] - frag_preds["consensus_f"]
    frag_preds['Contribution_type'] = "overall"  # for format compatibility
    frag_preds['Model'] = "MPNN"  # for format compatibility
    # remove molecules , leave only frags
    frag_preds = frag_preds.loc[~pd.isnull(frag_preds.Fragment),:]

    #  write (pandas)
    frag_preds.to_csv(out_fname, sep="\t", index=False)

    return frag_preds

