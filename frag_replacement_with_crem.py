import sys
import argparse
from collections import OrderedDict
from crem.crem import mutate_mol

import numpy as np
from rdkit import Chem


def read_worst_and_ids(input_worst, input_ids):
    """
    Prepare list of fragments with their ids
    :param input_worst: file name of selected fragments
    :param input_ids: file name with fragment ids
    :return: list of fragments with ids, e.g. [[compound_id, fragment_id, core, env, (fragment_ids)], [...], ...]
    """

    list_of_fragments = []
    d = OrderedDict()

    # prepare list of the worst fragments
    with open(input_worst, 'r') as f_worst:
        f_worst.readline()
        for line in f_worst:
            line = line.split('\t')
            frag_core = line[2].split('|')[0] # take only core (in case there is some context after '|'; no need in env)
            d[int(line[1])] = line[:2] + [frag_core]

    # prepare list of ids and connect them with fragments
    with open(input_ids, 'r') as f_ids:
        for i, line in enumerate(f_ids):
            line = line.strip().split('\t')
            if i in d:
                d[i].append(tuple(j-1 for j in map(int, line[2:])))
    return list(d.values())


def make_replacements(input_sdf, input_worst, input_ids, path_to_db,radius):
    new_products = []
    id_mol = 0
    compounds = Chem.SDMolSupplier(input_sdf, removeHs=False, sanitize=True)
    list_of_fragments = read_worst_and_ids(input_worst, input_ids)

    for mol in compounds:
        try:
            mol_id = str(mol.GetProp('ID'))
            for frag in list_of_fragments:
                bad_mol_name, bad_frag_id = frag[0], list(frag[-1])
                if mol_id == bad_mol_name:
                    out = mutate_mol(
                        mol,
                        path_to_db,
                        radius,
                        min_size=0,
                        max_size=10,
                        min_rel_size=0,
                        max_rel_size=1,
                        max_replacements=None,
                        replace_cycles=False,
                        min_inc=-2,
                        max_inc=2,
                        min_freq=0,
                        protected_ids=None,
                        symmetry_fixes=False,
                        return_rxn=True,
                        ncores=1,
                        return_rxn_freq=False,
                        return_mol=False,
                        replace_ids=bad_frag_id
                    )

                    for new_smile, transformation in out:
                        new_mol = Chem.MolFromSmiles(new_smile)
                        new_mol.SetProp('parent_name', bad_mol_name)
                        new_mol.SetProp('transformation', transformation)
                        new_products.append(new_mol)
        except:
            pass

    return new_products


def main(input_sdf, input_worst, input_ids, path_to_db, radius, output_product_file):

    print('Replacing fragments ...')

    products = make_replacements(input_sdf, input_worst, input_ids, path_to_db,radius)
    w = Chem.SDWriter(output_product_file)
    for m in products: w.write(m)
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

    args = vars(parser.parse_args())

    main(args['in_sdf'], args['in_worst'], args['in_ids'],
         args['in_con'], args['radius'], args['out_compounds'])
