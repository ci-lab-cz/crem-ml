import sys
import argparse
from collections import OrderedDict
from crem.crem import mutate_mol
from optimizer_utils import  get_child_protected_atom_ids  # todo import from parent module
import numpy as np
from rdkit import Chem
from  sirms.files import LoadFragments
import pandas  as pd
from multiprocessing import Pool
from functools import  partial
np.random.seed(1)

# def read_worst_and_ids(input_worst, input_ids):
#     """
#     Prepare list of fragments with their ids
#     :param input_worst: file name of selected fragments
#     :param input_ids: file name with fragment ids
#     :return: list of fragments with ids, e.g. [[compound_id, fragment_id, core, env, (fragment_ids)], [...], ...]
#     """
#
#     list_of_fragments = []
#     d = OrderedDict()
#
#     # prepare list of the worst fragments
#     with open(input_worst, 'r') as f_worst:
#         f_worst.readline()
#         for line in f_worst:
#             line = line.split('\t')
#             frag_core = line[2].split('|')[0]
#             d[int(line[1])] = line[:2] + [frag_core]
#
#     # prepare list of ids and connect them with fragments
#     with open(input_ids, 'r') as f_ids:
#         for i, line in enumerate(f_ids):
#             line = line.strip().split('\t')
#             if i in d:
#                 d[i].append(tuple(j-1 for j in map(int, line[2:])))
#     return list(d.values())

def read_worst_and_ids(input_worst, input_ids):
    """
    Prepare dataframe of fragments with their ids
    :param input_worst: file name of selected fragments
    :param input_ids: file name with fragment ids
    :return: pd.df of fragments with ids, columns: Compound,Frag_id,Fragment,Average,ids (atom ids)
    """


    d = pd.read_csv(input_worst, sep="\t")
    d["Fragment"] = d["Fragment"].apply(lambda x: x.split('|')[0]) # take only core (in case there is some context after '|'; no need in env)

    f_ids = LoadFragments(input_ids)
    f_ids = pd.json_normalize(f_ids, sep='#').transpose() # squash keys of nested dicts
    # print(f_ids.tail())
    f_ids[0]  = f_ids[0].apply(lambda i: [int(x)-1 for x in i]) # turn 1based (spci style) to 0based (rdkit style)
    # print(f_ids.tail())

    f_ids["Compound"]  = [x.split("#")[0] for x in f_ids.index]
    f_ids["Frag_id"]  = [int(x.split("#")[-1]) for x in f_ids.index] # frag id of given fragment in given molecule
    f_ids.columns = ['ids','Compound','Frag_id']

    return pd.merge(d, f_ids, how='inner')

def make_replacements(mol, prot_ids, df_of_fragments, path_to_db,
                        radius,

                        max_size):
        products = []
        try:
            mol_id = str(mol.GetProp('ID'))
            if prot_ids is not None:
                protected_ids = list(map(int, mol.GetProp(prot_ids).split(',')))
            else:
                protected_ids = prot_ids

            for row in df_of_fragments.loc[df_of_fragments.Compound ==mol_id ,:].iterrows():
                bad_mol_name, bad_frag_id = row[1].Compound, row[1].ids  # molid and atom ids

                out = mutate_mol(
                        mol=mol,
                        db_name=path_to_db,
                        radius=radius,
                        min_size=0,
                        max_size=max_size,
                        min_rel_size=0,
                        max_rel_size=1,
                        max_replacements=None,
                        replace_cycles=False,
                        min_inc=-2,
                        max_inc=2,
                        min_freq=0,
                        protected_ids=protected_ids,
                        symmetry_fixes=False,
                        return_rxn=True,
                        ncores=1,
                        return_rxn_freq=False,
                        return_mol=True,
                        replace_ids=bad_frag_id

                    )
                out = list(out)

                for new_smile, transformation, molobj in out:
                        # new_mol = Chem.MolFromSmiles(new_smile)
                        new_mol = molobj
                        new_mol.SetProp('parent_name', bad_mol_name)
                        new_mol.SetProp('transformation', transformation)
                        if prot_ids is not None:
                            new_mol.SetProp('protected_ids',','.join(map(str,get_child_protected_atom_ids(new_mol, protected_ids))))
                        products.append( new_mol)


        except:
            pass
        return products

def make_replacements_mp(input_sdf, input_worst, input_ids, path_to_db, radius, max_size, ncores, prot_ids=None):
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps) # mp for molobjects with props
    p = Pool(ncores)

    new_products = []
    compounds = Chem.SDMolSupplier(input_sdf, removeHs=False, sanitize=True)
    df_of_fragments = read_worst_and_ids(input_worst, input_ids)
    print(df_of_fragments)
    for res in p.imap_unordered(partial(make_replacements, prot_ids=prot_ids, df_of_fragments=df_of_fragments, path_to_db=path_to_db,
                        radius=radius, max_size=max_size), compounds):
        new_products.extend(res)
    return new_products


def main(input_sdf, input_worst, input_ids, path_to_db, radius,max_size,  output_product_file, ncores,prot_ids):

    print('Replacing fragments ...')
    products = make_replacements_mp(input_sdf, input_worst, input_ids, path_to_db, radius,max_size, ncores,prot_ids)
    w = Chem.SDWriter(output_product_file)
    for m in products:
        w.write(m)
    w.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
                                      'Create new file with replaced fragments.')
    parser.add_argument('-is', '--in_sdf', metavar='output_process_predictions.sdf', required=True,
                         help='path to file which contains selected compounds from pareto/desirability/...')
    parser.add_argument('-iw', '--in_worst', metavar='worst_fragments.txt', required=True,
                         help='path to the file where you store the worst fragments')
    parser.add_argument('-id', '--in_ids', metavar='fragments_ids.txt', required=True,
                        help='path to the file where you store ids of fragments')
    parser.add_argument('-ic', '--in_con', metavar='database.db', required=True,
                        help='path to the database with fragment replacements')
    parser.add_argument('-oc', '--out_compounds', metavar='new_compounds.sdf', required=True,
                        help='file name where you want to store new compounds')
    parser.add_argument('-c', '--ncores', metavar='number_of_cores', default=1,
                        help='number of cpu cores used.')

    args = vars(parser.parse_args())

    main(args['in_sdf'], args['in_worst'], args['in_ids'],
         args['in_con'], args['radius'], args['max_frag_size'],args['out_compounds'],args['ncores'], args['prot_ids'])
