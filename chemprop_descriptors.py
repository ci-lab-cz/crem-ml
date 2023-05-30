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


mol_frag_sep = "###"


def chemprop_CalcMolFP(m, model, i, opt_noH, frags=None, per_atom_fragments=None, id_field_name=None):
    def get_fp_as_dict(mol, model, opt_noH):
        """calc specified fingerprint for input molecule and return nonzero elements of it as dict
        """

        if opt_noH:
            # Chem.RemoveHs(mol)  # !it doesnt help, anyway next line gets them Hs back
            mol = Chem.RWMol(mol)
            for idx in reversed(range(mol.GetNumAtoms())):  # reverse because ids of atoms change while iter
                if mol.GetAtomWithIdx(idx).GetAtomicNum() == 1:
                    mol.RemoveAtom(idx)
            Chem.FastFindRings(mol)  # needs to calc morganfp, otherwise err "no ringinfo"
        with torch.no_grad():
            fp = model.fingerprint([[mol]], fingerprint_type="MPN")  # list of 1 rdkit.mol -> torch.tesor
        return np.array(fp).squeeze(0)

    mol_dict = OrderedDict()
    if id_field_name is not None:
        nm = m.GetProp(id_field_name)
    elif m.GetProp("_Name") == "":
        nm = 'auto_generated_id_' + str(i + 1)  # 1-based as in sirms.py
    else:
        nm = m.GetProp("_Name")
    res = get_fp_as_dict(m, model, opt_noH)
    mol_dict[nm] = res
    if per_atom_fragments:
        counter = 0
        for idx in range(m.GetNumAtoms()):
            if m.GetAtomWithIdx(idx).GetAtomicNum() > 1:
                rw_m = Chem.RWMol(m)
                rw_m.GetAtoms()[idx].SetAtomicNum(0)
                mol_dict[nm + mol_frag_sep + str(idx + 1) + "#" + str(counter)] = get_fp_as_dict(rw_m, model, opt_noH)
                counter += 1
    elif frags and nm in frags:
        for k, v in frags[nm].items():
            rw_m = Chem.RWMol(m)
            for idx in sorted(v, reverse=True):  # note we don't check if atom== H (is it ok?)
                rw_m.GetAtoms()[idx - 1].SetAtomicNum(0)
            mol_dict[nm + mol_frag_sep + k] = get_fp_as_dict(rw_m, model, opt_noH)
    return mol_dict

def main_params(in_fname, out_fname, model_path, opt_noH, frag_fname,
                per_atom_fragments, id_field_name):
    # load model
    arguments = [
        '--test_path', '/dev/null',
        '--preds_path', '/dev/null',
        '--checkpoint_dir', model_path
    ]
    args = chemprop.args.PredictArgs().parse_args(arguments)
    model_objects = chemprop.train.load_model(args=args)
    model = model_objects[2][0]  # mpnn  part of the model:"encoder"

    # load sdf and get dict of fp (like sirms dict)
    input_file_extension = in_fname.strip().split(".")[-1].lower()
    if input_file_extension == 'sdf':
        mols = None
        mols = OrderedDict()  # key - molname, val- mol; if frags: key - molname or mol+fragname, val-mol for mol or part b
        frags = LoadFragments(frag_fname)
        for i, m in enumerate(Chem.SDMolSupplier(in_fname, removeHs=False)):
            if m is not None:
                res = chemprop_CalcMolFP(m, model, i, opt_noH=opt_noH, frags=frags,
                                         per_atom_fragments=per_atom_fragments,
                                         id_field_name=id_field_name)

                mols.update(res)
        # save to file
        pd.DataFrame.from_dict(mols, orient="index").to_csv(out_fname, header=False)
    else:
        print("Input file extension should be SDF Current file has %s. Please check it." %
              input_file_extension.upper())
        return None


def entry_point():
    parser = argparse.ArgumentParser(description='Calculate chemprop fingerprint descriptors (MPNN part,aka encoder)')
    parser.add_argument('-i', '--in', metavar='input.sdf', required=True,
                        help='input file ( sdf with standardized structures')
    parser.add_argument('-o', '--out', metavar='output.txt', required=True,
                        help='output file with calculated descriptors (csv format).')
    parser.add_argument('-x', '--noH', action='store_true', default=False,
                        help='if set this flag hydrogen atoms will be excluded from the descriptors calculation.')
    parser.add_argument('-f', '--fragments', metavar='fragments.txt', default=None,
                        help='text file containing list of names of single compounds, fragment names and atom '
                             'indexes of fragment to remove (all values are tab-separated).')
    parser.add_argument('--per_atom_fragments', action='store_true', default=False,
                        help='if set this flag input fragments will be omitted and single atoms will be considered '
                             'as fragments.')
    parser.add_argument('-w', '--id_field_name', metavar='field_name', default=None,
                        help='field name of unique ID for compounds. '
                             'If omitted for sdf molecule titles will be used or auto-generated names')

    args = vars(parser.parse_args())

    for o, v in args.items():
        if o == "in": in_fname = v
        if o == "out": out_fname = v
        if o == "noH": opt_noH = v
        if o == "fragments": frag_fname = v
        if o == "per_atom_fragments": per_atom_fragments = v
        if o == "id_field_name": id_field_name = v

    main_params(in_fname=in_fname, out_fname=out_fname, opt_noH=opt_noH, frag_fname=frag_fname,
                per_atom_fragments=per_atom_fragments, id_field_name=id_field_name)


if __name__ == '__main__':
    entry_point()
