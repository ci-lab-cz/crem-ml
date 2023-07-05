import chemprop
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from rdkit import Chem
from torch import Tensor
import torch
from torch.nn.functional import sigmoid
import numpy as np
from  sirms.files import LoadFragments
from collections import OrderedDict
import re
import os

import optimizer_utils

mol_frag_sep = "###"


def main_params(x_fname,
                     out_fname,
                     model_dir,
                     model_type,
                     # ad# uncertainty? bb?,
                     multitask,
                     input_format="csv",  # for crem-ml compatability, not used
                     model_names="MPNN",  # for crem-ml compatability, not used
                     verbose=None,  # for crem-ml compatability, not used
                     title=None,  # for crem-ml compatability, not used
                     save_pred=True,
                     fragments_mode=False,
                     num_frag_id=False):

    # load model
    arguments = [
        '--test_path', '/dev/null',
        '--preds_path', '/dev/null',
        '--checkpoint_dir', model_dir
    ]

    args = chemprop.args.PredictArgs().parse_args(arguments)

    _, __, models, scalers, ___, prop_names = chemprop.train.load_model(args=args)
    if len(prop_names)>1 and not multitask\
            or  len(prop_names)==1 and  multitask: # todo move this sanity check to process config
        print("Stopping! Model is multitask, but in config multitask is Fasle - or vice versa - model is single task but multitask=True"
              " this will lead to incorrect results!"
              " Please, change the bool  value or use appropriate  models")
        return None
    sclrs = [i[0] for i in scalers]
    ffns = [i.ffn for i in models]

    outs = []
    for i,(scl, ffn) in enumerate(zip(sclrs, ffns)):
        # read fingerprint and coerce to tensor of 2 dims
        tmp = x_fname.split(".")
        x_fname_i = tmp[0] + "_"+str(i)+"."+tmp[1]
        fp_names = pd.read_csv(x_fname_i, header=None)
        fp_names.columns = ['Compounds'] + fp_names.columns[1:].tolist()
        # take only fp, strip names
        fp = Tensor(fp_names.values[:, 1:].astype(float))

        # separate mol and frag name use only for fragments file
        if fragments_mode:
            fp_names[['Compounds', 'Fragment']] = fp_names.Compounds.str.split(mol_frag_sep, expand=True)
            if num_frag_id: # separate  piece after last # - numerical fragment id
                fp_names[['Fragment', 'Frag_id']] = fp_names.Fragment.str.rsplit("#", n=1, expand=True)

        # predict FP
        ffn.eval()
        with torch.no_grad():
            out = ffn.forward(fp)
        if model_type == "reg":
            out = scl.inverse_transform(out)

        elif model_type == "class":
            out = np.asarray(sigmoid(out))

        outs.append(out)
    print(np.array(outs).shape)
    outs = np.mean(np.array(outs), axis=0)
    print(np.array(outs).shape)
    # construct df and  write to file
    if fragments_mode:
        if num_frag_id:
            outs = pd.DataFrame(pd.concat(
                (fp_names.Compounds, fp_names.Fragment, fp_names.Frag_id, pd.DataFrame(outs, columns=prop_names)),
                axis=1))
        else:
            outs = pd.DataFrame(
                pd.concat((fp_names.Compounds, fp_names.Fragment, pd.DataFrame(outs, columns=prop_names)), axis=1))

    else:
        outs = pd.DataFrame(pd.concat((fp_names.Compounds, pd.DataFrame(outs, columns=prop_names)), axis=1))

    outs["bound_box"] = 1
    # write down file for each property - if multitask
    if not multitask:
        outs["consensus"] = outs[prop_names[0]]# there should be only 1 property
        if save_pred:
            outs.to_csv(out_fname, sep="\t", index=False)
        return outs
    else:
        out_fname_param = optimizer_utils.retrieve_out_fname_param(out_fname)
        print(out_fname_param)
        outs_list = []
        for  prop_name in prop_names:
            print(prop_name)
            none_names =  set(prop_names) - {prop_name}#to exclude these later
            print(none_names)
            # todo fix _activity - itss hardcode
            out_fname_tmp = os.path.join(os.path.dirname(out_fname),re.sub(out_fname_param , re.sub("activity_","",prop_name), os.path.basename(out_fname))) # replace old param name with actually to be written
            outs["consensus"] = outs[prop_name]
            print(outs)
            outs_tmp = outs.loc[:, [i for i in outs.columns if i  not in none_names]] # leave only cur prop and consensus, rest is unneded and may be miused by proba consensus
            print(outs_tmp, "removed_prop_cols")
            outs_list.append(outs_tmp)
            if save_pred:
                outs_tmp.to_csv(out_fname_tmp, sep="\t", index=False)

        return outs_list



def entry_point():
    parser = argparse.ArgumentParser(description='Predict parameters (properties) using chemprop fingerprint and '
                                                 'chemprop model (passing FP through lastFFN of the model)')
    parser.add_argument('-i', '--x_fname', metavar='param_x.txt', required=True,
                        help='input file with fingerprints for compounds for  a given parameter in pkl format')
    parser.add_argument('-o', '--out', metavar='param_pred.txt', required=True,
                        help='output file with predictions, tsv.')
    parser.add_argument('-d', '--model_dir',
                        help='Path to model '
                             )
    parser.add_argument('-t', '--type', metavar='regression/classification', required=True,
                        help='')
    parser.add_argument('-m', '--multitask', metavar='True/False', required=True,
                        help='')

    # parser.add_argument('-a', '--applicability_domain', metavar='none|bound_box', required=False, nargs='*', default=None,
    #                     help='name(s) of applicability domain(s) to apply. If several - provide a space separated '
    #                          'list. Possible values: none - do not compute; bound_box - compounds with descriptor '
    #                          'values out of those for training set compounds are outside AD. Default: none')
    args = vars(parser.parse_args())

    for o, v in args.items():
        if o == "x_fname": x_fname = v
        if o == "out": out_fname = v
        if o == "model_dir": model_dir = v
        if o == "model_type": model_type= v
        if o == "multitask": multitask = bool(v)

    # if ad is not None and 'none' in ad:
    #         ad.remove('none')
    #         if not ad:
    #             ad = None

    main_params(x_fname=x_fname, out_fname=out_fname, model_dir=model_dir, model_type=model_type, multitask=multitask)


if __name__ == '__main__':
    entry_point()

