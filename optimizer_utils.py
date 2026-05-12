import os
import sys
import shutil
import re

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
from spci import descriptors
from spci import predict
from spci import find_frags_auto_rdkit as find_frags
from spci import filter_descriptors
from spci import calc_frag_contrib as frag_contrib

import chemprop_descr_and_predict
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
        if mol.GetProp('id') in output_poll.index:
            for col in output_poll.columns:
                mol.SetProp('predicted_{}'.format(col),
                            str(output_poll.loc[mol.GetProp('id'), col]))
            output.write(mol)

def create_database(working_dir: str, parameter_to_optimize: List, spci_models: bool = False) -> str:
    """
    Define new database and save it to working_dir.

    :param working_dir: path to working directory, new db will be stored there
    :param parameter_to_optimize: list of all parameters
    :param spci_models: bool specifying whether spci models are used (and thus bounding_box should be saved)
    :return: path_to_database
    """
    parameter_to_optimize = ["predicted_{}".format(parameter) for parameter in parameter_to_optimize]
    path_to_database = os.path.join(working_dir, 'output.db')

    if spci_models:
        table_str = ("""CREATE TABLE optimizer_table(
                        id TEXT NOT NULL,
                        smi TEXT NOT NULL UNIQUE,
                        mol_block TEXT NOT NULL UNIQUE,
                        protected_ids TEXT,
                        generation INTEGER NOT NULL,
                        parent TEXT,
                        transformation TEXT,
                        fit INTEGER, """ +
                     " REAL,".join(parameter_to_optimize) +
                     " REAL, bounding_box INTEGER)")
    else:
        table_str = ("""CREATE TABLE optimizer_table(
                        id TEXT NOT NULL,
                        smi TEXT NOT NULL UNIQUE,
                        mol_block TEXT NOT NULL UNIQUE,
                        protected_ids TEXT,
                        generation INTEGER NOT NULL,
                        parent TEXT,
                        transformation TEXT,
                        fit INTEGER, """ +
                     " REAL,".join(parameter_to_optimize) +
                     " REAL)")

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
    Read input sdf file, convert all molecules into smiles, check if they are in DB,
    and if not add them with all possible properties (fields), such as transformation rules,
    parents, number of generation and so on.

    :param input_sdf: path to input sdf file
    :param database: path to output database
    :param gen: actual generation of optimization
    :return: number of compounds added to database
    """

    # get generator of mols in sdf file
    supplier = Chem.SDMolSupplier(input_sdf, removeHs=False)

    with lite.connect(database) as con:

        cursor = con.cursor()

        max_rowid = cursor.execute("SELECT MAX(rowid) FROM optimizer_table").fetchone()[0]

        insert_query = []

        for i, mol in enumerate(supplier, 1):
            if not mol:
                continue

            mol.SetProp("_Name", "ID_{}_{}".format(gen, i))
            if gen == 0:
                mol.SetProp("parent", "None")
                mol.SetProp("transformation", "None")

            smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
            mol_block = Chem.MolToMolBlock(Chem.AddHs(mol))  # mol_block is always hydrogenized

            data = (mol.GetProp("_Name"),
                                 smi,
                                 mol_block,
                                 gen,
                                 mol.GetProp('parent'),
                                 mol.GetProp('transformation'))
            if 'protected_ids' in mol.GetPropNames():
                data += (mol.GetProp('protected_ids'), )
            else:
                data += (None, )
            insert_query.append(data)

        cursor.executemany("""INSERT OR IGNORE INTO optimizer_table (
                                     id, 
                                     smi,
                                     mol_block,
                                     generation, 
                                     parent, 
                                     transformation,
                                     protected_ids) 
                                  VALUES (?, ?, ?, ?, ?, ?, ?)""",
                           insert_query)

        con.commit()

        if max_rowid is None:  # the first iteration, empty DB
            num_inserted_compounds = cursor.execute("SELECT COUNT(rowid) FROM optimizer_table").fetchone()[0]
        else:
            num_inserted_compounds = cursor.execute("SELECT COUNT(rowid) FROM optimizer_table WHERE rowid > ?",
                                                    (max_rowid,)).fetchone()[0]

    return num_inserted_compounds


def get_mols(database: str, gen: int = None, fields: list = None) -> list:
    """
    param: database: path to database
    param: gen: the number of a generation, if None the last one is taken
    param: fields: list of fields to consider
    """
    mols = []
    with lite.connect(database) as con:
        cursor = con.cursor()
        if gen is None:
            gen = cursor.execute("""SELECT MAX(gen) FROM optimizer_table""").fetchone()[0]
        if fields:
            sql = f"""SELECT mol_block, {','.join(fields)}  
                      FROM optimizer_table
                      WHERE generation = ?"""
        else:
            sql = """SELECT mol_block  FROM optimizer_table WHERE generation = ?"""
        cursor.execute(sql, (gen,))
        for item in cursor.fetchall():
            mol = Chem.MolFromMolBlock(item[0], removeHs=False)
            if mol:
                for field, value in zip(fields, item[1:]):
                    if value is not None:
                        mol.SetProp(field, value)
                mols.append(mol)
    return mols


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



def calculate_fingerprints(input_sdf_file: str,
                            fingerprint_type: str,  model_path:str=None,parameter_name:str=None,
                            fragments_ids=None, id_field_name: str = 'id') -> None:
    """
    Create files with RDKIT fingerprints. Encoded as: ECFP4='MG2', atom pair fingerprint='AP', rdkit fingerprint: 'RDK',
    topological torsions: TT; binary (hashed) versions  are specified with 'b' prefix, e.g. 'bAP'.

    :param input_sdf_file: path to [optionally standardized] and labeled sdf file
    :param fingerprint_type: str  fingerprints to calculate e.g. 'bAP','MG2'
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
                          opt_noH=True, #  MPNN fingerprint with Hs  leads  to wrong predictions, regardless of how was built (Hs in data/model)
                          frag_fname=fragments_ids,
                          per_atom_fragments=False,
                          id_field_name=id_field_name,
                          model_path=model_path
                          )

    else: # rdkit fingerprint
        descriptors.main_params(in_fname=input_sdf_file,    # input
                                out_fname=x_fname,  # output
                                opt_verbose=False,
                                opt_noH=False,
                                frag_fname=fragments_ids,
                                per_atom_fragments=False,
                                id_field_name=id_field_name,
                                output_format="svm",
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

def filter_columns_by_keyword(df, keyword):
    """Returns a DataFrame with only columns containing the specified keyword."""
    filtered_df = df[[col for col in df.columns if keyword in col]]
    return filtered_df

def calculate_sirms_descriptors(input_sdf_file: str,
                                n_cores: int,
                                fragments_ids=None, id_field_name: str = 'id') -> None:
    """
    Create files with descriptors

    :param input_sdf_file: path to standardized and labeled sdf file
    :param n_cores: number of cores for computing
    :param fragments_ids: path to frag_ids file; if specified, use fragments ids
    :param id_field_name: specifies name of parameter in which is id of mol saved
    """

    print("Descriptors calculation started. Please wait it can take some time")


    # define output files
    if fragments_ids is not None:
        x_fname = os.path.join(os.path.dirname(input_sdf_file), 'new_x.txt')
    else:
        x_fname = os.path.join(os.path.dirname(input_sdf_file), 'x.txt')

    sirms.main_params(in_fname=input_sdf_file,    # input
                      out_fname=x_fname,        # output
                      opt_diff="elm",
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
                      output_format="svm",
                      ncores=n_cores)

    # filter sirms descriptors
    filter_descriptors.main_params(in_fname=x_fname,
                                   out_fname=x_fname,
                                   file_format="svm")

def predict_properties(parameters: List, descriptors_fname: str, multitask:bool=False) -> None:
    """
    Creates summarized file with predictions

    :parama parameters: list of dicts with parameters
    :parama fragmens_fname: path to file with calculated descriptors
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
                             multitask=multitask,
                             # ad# uncertainty? bb?,
                             )
        else:
            predict.main_params(x_fname=descriptors_fname,
                            input_format="svm",
                            out_fname=output_file_name,
                            model_names=parameter['types_of_alg'],
                            model_dir=parameter['path'],
                            model_type=parameter['type_of_model'],
                            ad=['bound_box'],
                            verbose=False,
                            )

def find_frags_rdkit(input_sdf_file: str, fragment_ids_file: str,
                     smarts_string: str, max_cuts: int, error_fname: str,
                     verbose: bool=False) -> None:
    """
    Creates file with fragments from sdf file

    :param input_sdf_file: input file with compounds
    :param fragment_ids_file: name of output file with fragment ids
    :param smarts_string:
    :param max_cuts:
    :param: error_fname: path to log file from this function
    :param verbose: false default
    """

    print("Finding fragments has started")

    find_frags.main_params(in_sdf=input_sdf_file,
                           out_txt=fragment_ids_file,
                           query=smarts_string,
                           max_cuts=max_cuts,
                           radius=[0],  # todo is it safe to hardcode this arg?
                           verbose=verbose,
                           keep_stereo=False,
                           error_fname=error_fname)


def calc_frag_contrib(x_fname: str, parameters: List, types_of_alg: List,
                      models_dir: List, models_type: List, multitask: bool=False) -> None:
    """
    Calculate contributions of fragments. All records in list must be specified
    in same order.

    :param x_fname: input file with descriptors of fragments
    :param parameters: list with parameter names
    :param types_of_alg: list with types of alg used for predictions, e.g. [['rf', 'svm'], ['rf']]
    :param models_dir: list of paths to models directories
    :param models_type: list of types of models, e.g. ['reg', 'class']
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
                multitask=multitask,
                save_pred=False)

        else:
            frag_contrib.main_params(x_fname=x_fname,
                                     out_fname=os.path.join(os.path.dirname(x_fname),
                                                            'contrib_{}.txt'.format(parameter)),
                                     model_names=type_of_alg,
                                     model_dir=model_dir,
                                     prop_names=['overall'],
                                     model_type=model_type,
                                     activity_file=None,
                                     verbose=False,
                                     save_pred=False,
                                     input_format="svm",
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

def retrieve_out_fname_param(out_fname):

    # retrieve param name from file name in order to index out by current column with prop
    out_name_param = re.sub("predictions_", "", os.path.basename(out_fname))  # if out fname is compoundspreds
    out_name_param = re.sub("contrib_", "", out_name_param)  # if out fname is frags contribs
    out_name_param = re.sub(".txt", "", out_name_param)  # final stripped  name of param
    return out_name_param