import os
import sys
import shutil

from subprocess import call
import sqlite3 as lite
from rdkit import Chem
import pandas as pd

from typing import List
from typing import NewType
pandas_table = NewType('Processed pandas table with id of compound and predicted properties',
                      pd.DataFrame
                      )

# sys.path.insert(1, os.path.join(sys.path[0], 'spci'))
from spci import calc_atomic_properties_chemaxon
from spci import descriptors
from spci import predict
from spci import find_frags_auto_rdkit as find_frags
from spci import filter_descriptors
from spci import calc_frag_contrib as frag_contrib

import chemprop_descriptors
import chemprop_predict
import chemprop_frag_contrib

# sys.path.insert(1, os.path.join(sys.path[0], 'spci/sirms'))
from sirms import sirms


def save_output_poll(in_sdf: str, out_fname: str, output_poll: pandas_table) -> None:
    """
    Save list of selected compounds into file.

    :param in_sdf: path to input sdf file
    :param out_fname: path to output sdf file
    :output_poll: pandas table with selected compounds
    """

    output = Chem.SDWriter(out_fname)

    # get generator of mols in sdf file
    supplier = Chem.SDMolSupplier(in_sdf)

    for mol in supplier:
        if mol.GetProp('ID') in output_poll.index:
            for col in output_poll.columns:
                mol.SetProp('predicted_{}'.format(col),
                            str(output_poll.loc[mol.GetProp('ID'), col]))
            output.write(mol)

def create_database(working_dir: str, parameter_to_optimize: List) -> str:
    """
    Define new database and save it to working_dir.

    :param working_dir: path to working directory, new db will be stored there
    :param parameter_to_optimize: list of all parameters
    :return: path_to_database
    """
    parameter_to_optimize = ["predicted_{}".format(parameter) for parameter in parameter_to_optimize]
    path_to_database = os.path.join(working_dir, 'output.db')

    table_str = "CREATE TABLE optimizer_table"\
                    "(id TEXT NOT NULL,"\
                    "smi TEXT NOT NULL UNIQUE,"\
                    "generation INTEGER NOT NULL,"\
                    "parent TEXT,"\
                    "transformation TEXT,"\
                    "fit INTEGER,"
    table_str += " REAL,".join(parameter_to_optimize) + " REAL)"

    if os.path.isfile(path_to_database):
        os.remove(path_to_database)

    con = lite.connect(path_to_database)
    with con:
        cursor = con.cursor()
        cursor.execute(table_str)
        cursor.execute("CREATE INDEX idx ON optimizer_table (id)")
        cursor.execute("CREATE INDEX smi_idx ON optimizer_table (smi)")
        cursor.execute("DELETE FROM optimizer_table")

    return path_to_database

def add_mols_into_db(input_sdf: str, database: str, gen: int) -> int:
    """
    Read input sdf file, convert all mols into smiles, check if they are in DB,
    and if not add them with all possible options, such as transformation rules,
    parents, number of generation and so on.

    :param input_sdf: path to input sdf file
    :param database: path to output database
    :param gen: actual generation of optimization
    :return: number of compounds in database
    """

    num_of_compounds = 0

    # get generator of mols in sdf file
    supplier = Chem.SDMolSupplier(input_sdf,removeHs=False)

    new_sdf_path = os.path.join(os.path.dirname(input_sdf), 'tmp.sdf')
    new_sdf = Chem.SDWriter(new_sdf_path)

    con = lite.connect(database)
    with con:
        cursor = con.cursor()

        cursor.execute("SELECT smi FROM optimizer_table")
        mols_in_db = [mol[0] for mol in cursor.fetchall()]

        insert_query = []

        for mol in supplier:
            if not mol:
                return 0
            smile = Chem.MolToSmiles(mol)

            # mol doesn't have parent and transformation prop if it is in zero gen
            if gen == 0:
                mol.SetProp("_Name", "ID_{}_{}".format(gen, num_of_compounds))
                mol.SetProp("ID", "ID_{}_{}".format(gen, num_of_compounds))
                mol.SetProp("parent_name", "None")
                mol.SetProp("transformation", "None")
                new_sdf.write(mol)

            if smile not in mols_in_db:

                # change indexes
                if gen != 0:
                    mol.SetProp("_Name", "ID_{}_{}".format(gen, num_of_compounds))
                    mol.SetProp("ID", "ID_{}_{}".format(gen, num_of_compounds))
                    new_sdf.write(mol)

                insert_query.append((mol.GetProp("ID"),
                                     smile,
                                     gen,
                                     mol.GetProp('parent_name'),
                                     mol.GetProp('transformation')))
                mols_in_db.append(smile)
                num_of_compounds += 1

        cursor.executemany("INSERT INTO optimizer_table (id, smi, generation, parent, transformation) VALUES (?, ?, ?, ?, ?)", insert_query)
        con.commit()

        new_sdf.close()
        os.remove(input_sdf)
        os.rename(new_sdf_path, input_sdf)

    return num_of_compounds

def count_fitted_compounds(database: str) -> int:
    """
    Count fitted compounds from database

    :param database: path to output database
    :return: number of fitted compounds in database
    """

    con = lite.connect(database)
    with con:
        cursor = con.cursor()
        cursor.execute("SELECT count(*) FROM optimizer_table where fit=1")
        number_of_fitted_compounds = cursor.fetchone()

    return number_of_fitted_compounds[0]


def quote_str(s: str) -> str:
    """
    Quote string

    :param s: input string
    :return: quoted string
    """

    return "'%s'" % s

def standardize_sdf(input_sdf_file: str, std_rules_path: str,
                    chemaxon_path: str, copy_rules: bool=False) -> str:
    """
    Create file with standardized compounds

    :param input_sdf_file: path to sdf file with compounds
    :param std_rules_path: path to file with rules for standardization
    :param chemaxon_path: path to chemaxon bin
    :param copy_rules: if specified, copy rules to output directory
    :return: path to new sdf file
    """

    print('Standardization is in progress...')

    # copy xml-rules if specified
    if copy_rules:
        shutil.copyfile(
            std_rules_path, os.path.join(os.path.dirname(input_sdf_file), std_rules_path.split("/")[-1]))

    # run standardize
    std_sdf = os.path.join(os.path.dirname(input_sdf_file), 'input_dataset_std.sdf')
    run_params = [os.path.join(chemaxon_path, 'standardize'),
                  '-c',
                  quote_str(std_rules_path),  # path to rules
                  '--ignore-error',
                  quote_str(input_sdf_file),  # path to input file
                  '-f',
                  'sdf',  # type of output file
                  '-o',
                  quote_str(std_sdf)]  # name of output file
    call(' '.join(run_params), shell=True)

    return std_sdf

def calculate_atomic_prop(input_sdf_file: str, chemaxon_path: str, properties: List) -> str:
    """
    Calculate atomic properties with Chemaxon, it creates file with labeled compounds

    :param input_sdf_file: path to sdf file with compounds
    :param chemaxon_path: path to chemaxon bin
    :param properties: list of properties, e.g. ['charge', 'refractivity', 'logp', ...]
    :return: path to new sdf file
    """
    print('Atomic properties calculation is in progress...')
    lbl_sdf = os.path.join(os.path.dirname(input_sdf_file), 'input_dataset_std_lbl.sdf')
    calc_atomic_properties_chemaxon.main_params(input_sdf_file,
                                                lbl_sdf,
                                                properties,
                                                None,
                                                os.path.join(chemaxon_path, 'cxcalc'))
    return lbl_sdf


# noinspection PyStatementEffect

def calculate_fingerprints(input_sdf_file: str,
                            fingerprint_type: str, output_format: str, model_path:str=None,parameter_name:str=None,
                            fragments_ids=None, id_field_name: str = 'ID') -> None:
    """
    Create files with RDKIT fingerprints. Encoded as: ECFP4='MG2', atom pair fingerprint='AP', rdkit fingerprint: 'RDK',
    topological torsions: TT; binary (hashed) versions  are specified with 'b' prefix, e.g. 'bAP'.

    :param input_sdf_file: path to [optionally standardized] and labeled sdf file
    :param fingerprint_type: str  fingerprints to calculate e.g. 'bAP','MG2'
    :param output_format: svm
    :param model_path: provide this path iff calculating MPNN fingerprint
    :param parameter_name: provide this name (of target property corresponding to model) iff  calculating MPNN fingerprint
    :param fragments_ids: path to file with frag_ids; if specified, use fragments ids
    :param id_field_name: specifies name of parameter in which is id of mol saved
    """

    print("Descriptors calculation started. Please wait it can take some time")

    # define output files
    if fragments_ids is not None:
        if fingerprint_type != "MPNN_fingerprint":
            x_fname = os.path.join(os.path.dirname(input_sdf_file), 'new_x.txt')
        else: # indicate parameter for which fp is created in output file name
            if parameter_name is None: print("for MPNN fingerprint parameter_name must be specified"); return None
            x_fname = os.path.join(os.path.dirname(input_sdf_file), parameter_name+'_MPNN_fingerprint_new_x.txt')

    else:
        if fingerprint_type != "MPNN_fingerprint":
            x_fname = os.path.join(os.path.dirname(input_sdf_file), 'x.txt')
        else:  # indicate parameter for which fp is created in output file name
            if parameter_name is None: print("for MPNN fingerprint parameter_name must be specified"); return None
            x_fname = os.path.join(os.path.dirname(input_sdf_file), parameter_name+'_MPNN_fingerprint_x.txt')

    if fingerprint_type == "MPNN_fingerprint": # mpnn fingerprint
        chemprop_descriptors.main_params( in_fname=input_sdf_file,    # input
                          out_fname=x_fname,        # output
                          opt_noH=True, #  MPNN fingerprint with hs may  lead  (?) to wrong predictions
                          frag_fname=fragments_ids,
                          per_atom_fragments=False,
                          id_field_name=id_field_name,
                          model_path=model_path
                          )

    else: # rdkit fingerprint
        descriptors.main_params(  in_fname=input_sdf_file,    # input
                          out_fname=x_fname,        # output

                          opt_verbose=False,
                          opt_noH=False,
                          frag_fname=fragments_ids,
                          per_atom_fragments=False,
                          id_field_name=id_field_name,
                          output_format=output_format,
                          get_fp=fingerprint_type)


def get_child_protected_atom_ids(mol, protected_parent_ids):
    '''
    :param mol:
    :param protected_parent_ids: list[int]
    :type  protected_parent_ids: list[int]
    :return: sorted list of integers
    '''
    # After RDKit reaction procedure there is a field <react_atom_idx> with initial parent atom idx in product mol
    protected_product_ids = []
    for a in mol.GetAtoms():
        if a.HasProp('react_atom_idx') and int(a.GetProp('react_atom_idx')) in protected_parent_ids:
            protected_product_ids.append(a.GetIdx())
    return sorted(protected_product_ids)


def calculate_sirms_descriptors(input_sdf_file: str, setup_file: str,
                                properties: List, output_format: str,
                                n_cores: int, copy_setup: bool = True,
                                fragments_ids=None, id_field_name: str = 'ID') -> None:
    """
    Create files with descriptors

    :param input_sdf_file: path to standardized and labeled sdf file
    :param setup_file: path to file with setup for calculation of sirms descriptors
    :param properties: list of properties, e.g. ['CHARGE', 'REFRACTIVITY', 'LOGP', ...]
    :param output_format: svm
    :param n_cores: number of cores for computing
    :param copy_setup: if specified, copy setup file to output directory
    :param fragments_ids: path to frag_ids file; if specified, use fragments ids
    :param id_field_name: specifies name of parameter in which is id of mol saved
    """

    print("Descriptors calculation started. Please wait it can take some time")

    # copy setup file for sirms into generation dir
    if copy_setup:
        shutil.copyfile(setup_file,
                        os.path.join(os.path.dirname(input_sdf_file), os.path.basename(setup_file))
                        )

    # define output files
    if fragments_ids is not None:
        x_fname = os.path.join(os.path.dirname(input_sdf_file), 'new_x.txt')
    else:
        x_fname = os.path.join(os.path.dirname(input_sdf_file), 'x.txt')

    sirms.main_params(in_fname=input_sdf_file,    # input
                      out_fname=x_fname,        # output
                      opt_diff=properties,
                      min_num_atoms=2,
                      max_num_atoms=4,
                      min_num_components=1,
                      max_num_components=2,
                      min_num_mix_components=2,
                      max_num_mix_components=2,
                      mix_fname=None,
                      descriptors_transformation='num',
                      mix_type='abs',
                      opt_mix_ordered=False,
                      opt_verbose=False,
                      opt_noH=False,
                      frag_fname=fragments_ids,
                      per_atom_fragments=False,
                      self_association_mix=False,
                      reaction_diff=False,
                      quasimix=False,
                      id_field_name=id_field_name,
                      output_format=output_format,
                      ncores=n_cores)

    # filter sirms descriptors
    filter_descriptors.main_params(in_fname=x_fname,
                                   out_fname=x_fname,
                                   file_format=output_format)

def predict_properties(parameters: List, descriptors_fname: str, output_format: str) -> None:
    """
    Creates summarized file with predictions

    :parama parameters: list of dicts with parameters
    :parama fragmens_fname: path to file with calculated descriptors
    :parama output_format: svm/txt/...
    """

    for parameter in parameters:
        print("Prediction for {} started".format(parameter['name']))
        output_file_name = os.path.join(os.path.dirname(descriptors_fname),
                                        'predictions_{}.txt'.format(parameter['name']))

        if  "MPNN_fingerprint" in descriptors_fname:
            chemprop_predict.main_params(x_fname=descriptors_fname,
                             out_fname=output_file_name,
                             model_dir=parameter['path'],
                             model_type=parameter['type_of_model'],
                             # ad# uncertainty? bb?,
                             )
        else:
            predict.main_params(x_fname=descriptors_fname,
                            input_format=output_format,
                            out_fname=output_file_name,
                            model_names=parameter['types_of_alg'],
                            model_dir=parameter['path'],
                            model_type=parameter['type_of_model'],
                            ad=['bound_box'],
                            verbose=False,
                            )

def find_frags_rdkit(input_sdf_file: str, fragment_ids_file: str,
                     smarts_string: str, max_cuts: int,
                     keep_stereo: bool, error_fname: str,
                     verbose: bool=False) -> None:
    """
    Creates file with fragments from sdf file

    :param input_sdf_file: input file with compounds
    :param fragment_ids_file: name of output file with fragment ids
    :param smarts_string: ******NOT SURE******
    :param max_cuts: ******NOT SURE******
    :param keep_stereo: ******NOT SURE******
    :param: error_fname: path to log file from this function
    :param verbose: false default
    """

    print("Finding fragments has started")
    find_frags.main_params(in_sdf=input_sdf_file,
                                out_txt=fragment_ids_file,
                                query=smarts_string,
                                max_cuts=max_cuts,
                                radius = [0], # todo is it safe to hardcode this arg?
                                keep_stereo = keep_stereo,
                                verbose=verbose,
                                error_fname=error_fname)

def calc_frag_contrib(x_fname: str, parameters: List, types_of_alg: List,
                      models_dir: List, models_type: List,
                      properties_calc_contrib: List, in_format: str) -> None:
    """
    Calculate contributions of fragments. All records in list must be specified
    in same order.

    :param x_fname: input file with descriptors of fragments
    :param parameters: list with parameter names
    :param types_of_alg: list with types of alg used for predictions, e.g. [['rf', 'svm'], ['rf']]
    :param models_dir: list of paths to models directories
    :param models_type: list of types of models, e.g. ['reg', 'class']
    :param properties_calc_contrib: ******NOT SURE****** list, e.g.['overall']
    :param: in_format: ******NOT SURE****** 'svm'
    """

    for parameter, type_of_alg, model_dir, model_type in zip(parameters, types_of_alg, models_dir, models_type):
        # todo : need abiltiy of handling chunks in sirmsfile - for cases when too few frags were generated,  we need higher value

        print("Fragment contribution for {} started".format(parameter))

        if type_of_alg == ["MPNN"]:
            chemprop_frag_contrib.main_params(
                x_fname=x_fname,
                out_fname=os.path.join(os.path.dirname(x_fname),
                        'contrib_{}.txt'.format(parameter)),
                model_dir=model_dir,
                model_type=model_type,
                save_pred=False)

        else:
            frag_contrib.main_params(x_fname=x_fname,
                                 out_fname=os.path.join(os.path.dirname(x_fname),
                                                        'contrib_{}.txt'.format(parameter)),
                                 model_names=type_of_alg,
                                 model_dir=model_dir,
                                 prop_names=properties_calc_contrib,
                                 model_type=model_type,
                                 activity_file=None,
                                 verbose=False,
                                 save_pred=False,
                                 input_format=in_format,
                                 long_format=True,
                                 save_frag_ids=True)

def parse_threshold(thresholds: List) -> List:
    """
    Convert input thresholds to parsed 2D list

    :param thresholds: list of thresholds, e.g. ['more4', 'betwenn-0.5to1', less'-2']
    :return: list of parsed threshold, e.g. [['more',4], ['between', -0.5, 1], ['less', -2]]
    """

    threshold_match = []
    for threshold in thresholds:
        if 'less' in threshold:
            threshold_match.append(['less', float(threshold[4:])])
        elif 'more' in threshold:
            threshold_match.append(['more', float(threshold[4:])])
        elif 'between' in threshold:
            threshold_match.append(
                ['between', float(threshold[7:].split('to')[0]), float(threshold[7:].split('to')[1])])
    return threshold_match
