#!/usr/bin/env python

import argparse
import math
import random

from rdkit import Chem
import pandas as pd

from optimizer_utils import parse_threshold

from typing import List
from typing import NewType
pandas_table = NewType('Processed pandas table with id of compound and predicted properties',
                      pd.DataFrame
                      )
pandas_series_row = NewType('One row from pandas dataframe', pd.core.series.Series)


def  prepare_fragments_table(in_files: List, parameters: List, alg_types: List) -> pandas_table:
    """
    Collect all input files from fragment contributions into one big table.

    :param in_files: list of all contribution files
    :param parameters: list of all parameters
    :param alg_types: list of lists of all algs used for predictions
    :return pandas dataframe [frag_id, compound, fragment, parameters*]
    """

    tables = []
    for parameter, in_file,alg_type in zip(parameters, in_files, alg_types):  # over files

        table = pd.read_table(in_file)
        table.drop( 'Contribution_type', axis=1, inplace=True) # if only 'overall' exists, we can ignore it safely

        table = table.pivot(index=["Compound", "Frag_id","Fragment"], columns='Model', values='Contribution_value')
        # remove partial columns and compute average value
        table[parameter]  = table[alg_type].mean(axis=1)
        table.drop(alg_type, axis=1, inplace=True)

        tables.append(table)
        print(table.tail())
    return pd.concat(tables, axis=1, join='inner').reset_index(inplace=False) # move idx cols to become cols


def prepare_fragments_table_old(in_files: List, parameters: List, alg_types: List) -> pandas_table:
    """
    Connect all input files from fragment contributions into one big table.

    :param in_files: list of all contribution files
    :param parameters: list of all parameters
    :param alg_types: list of lists of all algs used for predictions
    :return pandas dataframe [frag_id, compound, fragment, parameters*]
    """

    tables = []

    #prepare number of lines for each parameter
    num_lines_files = [sum(1 for line in open(in_f)) for in_f in in_files]

    # without header and for every type
    num_lines_files = [(num - 1)/len(n_types) for num, n_types in zip(num_lines_files, alg_types)]

    for i, (parameter, in_file, num_lines) in enumerate(zip(parameters, in_files, num_lines_files)): # over files

        for i_alg, alg in enumerate(alg_types[i]): # over types of algs

            if i_alg == 0: # first type of alg
                table = pd.read_table(in_file, nrows=num_lines)
                columns = table.columns
                table.drop(['Model', 'Contribution_type'], axis=1, inplace=True)
                table.rename(columns={'Contribution_value': alg}, inplace=True)
                table.set_index('Frag_id', inplace=True)
            else:
                tmp_table = pd.read_table(in_file, header=None, skiprows=int(1+(i_alg*num_lines)), nrows=num_lines)
                tmp_table.columns = columns
                tmp_table.rename(columns={'Contribution_value': alg}, inplace=True)
                table[alg] = tmp_table[alg]

        # remove partial columns and compute average value
        table[parameter] = table.sum(axis=1)/len(alg_types[i])
        table.drop(alg_types[i], axis=1, inplace=True)
        if i > 0:
            table.drop(['Compound', 'Fragment'], axis=1, inplace=True)
        tables.append(table)
        print(table.tail())

    return pd.concat(tables, axis=1, join='inner')

def get_predicted_values_for_whole_compounds(in_file: str, parameters: List) -> pandas_table:
    """
    Creates pandas dataframe with predicted values for all parameters for all
    compounds.

    :param in_file: path to file with selected compounds
    :param parameters: list of all parameters
    :return: pandas dataframe [Compound(ID), parameters*]
    """

    predicted_values = []
    record = {}
    compounds = Chem.SDMolSupplier(in_file, sanitize=False)
    for mol in compounds:
        record['Compound'] = mol.GetProp('ID')
        for parameter in parameters:
            record[parameter] = float(mol.GetProp('pred_{}'.format(parameter)))
        predicted_values.append(record)
        record = {}
    return pd.DataFrame(predicted_values).set_index('Compound')

def compute_normalized_value(record: pandas_series_row, predictions: pandas_table,
                              parameter: str, threshold: List, range: int) -> float:
    """
    It is applied on pandas dataframe row (which is Series). Calculates normalized
    value according to threshold and predicted value of whole compound.

    :param record: pandas series, e.g. [Compound(ID), Fragment, parameters*]
    :param predictions: pandas dataframe with predicted values for compounds
    :param parameter: name of predicted parameter
    :param threshold: parsed threshold for one parameter
    :param range: range of possible prediction
    :return: normalized value from predicted contribution
    """
    # print("record", record)
    predicted_value = predictions.loc[record['Compound'], parameter]
    x = record[parameter]

    if threshold[0] == 'more':
        if threshold[1] <= predicted_value:
            x = abs(x)
            return 2 * ((1 / (1 + math.exp((7 / range) * - x))) - 1) + 1
        else:
            return 2 * ((1 / (1 + math.exp((7 / range) * - x))) - 1) + 1
    elif threshold[0] == 'less':
        if threshold[1] >= predicted_value:
            x = abs(x)
            return 2 * ((1 / (1 + math.exp((7 / range) * - x))) - 1) + 1
        else:
            return -(2 * ((1 / (1 + math.exp((7 / range) * - x))) - 1) + 1)
    # between
    else:
        if predicted_value <= threshold[2] and predicted_value >= threshold[1]:
            x = abs(x)
            return 2 * ((1 / (1 + math.exp((7 / range) * - x))) - 1) + 1
        elif predicted_value > threshold[2]:
            return -(2 * ((1 / (1 + math.exp((7 / range) * - x))) - 1) + 1)
        else:
            return 2 * ((1 / (1 + math.exp((7 / range) * - x))) - 1) + 1

def main(in_sdf_f, in_contrib_f, out_frag_f, out_worst_f, parameters, ranges,
         types_of_alg, thresholds, n_worst, random_ratio=0, brute_force=False):
    """
    :param types_of_alg: list of "_"separated alg types to be used with each optimized parameter, e.g. [rf_gbm, rf_gbm]
    """
    print('Processing contributions ...')

    n_random = math.floor(n_worst * random_ratio)

    types_of_alg = [types.split('_') for types in types_of_alg]

    thresholds = parse_threshold(thresholds)

    table = prepare_fragments_table(in_contrib_f, parameters, types_of_alg)
    print(table.tail())
    predictions = get_predicted_values_for_whole_compounds(in_sdf_f, parameters)

    for parameter, threshold, rng in zip(parameters, thresholds, ranges):
        print(table.loc[:,parameter])
        table.loc[:,parameter] = table.apply(compute_normalized_value, axis=1, predictions=predictions, parameter=parameter, threshold=threshold, range=rng)
    cols_to_ave = range(table.shape[1] - len(parameters),table.shape[1])
    table['Average'] = table.iloc[:,cols_to_ave].sum(axis=1) / len(parameters)
    print("after ave",table.tail())
    # prepare order of columns for fragment norm output
    order_cols = ['Compound', 'Frag_id', 'Fragment']
    order_cols.extend(parameters)
    order_cols.append('Average')

    # save to file normalized contribution
    table.reset_index()[order_cols].to_csv(out_frag_f, index=False, sep='\t')

    # prepare order of columns for worst fragments
    order_cols = ['Compound', 'Frag_id', 'Fragment', 'Average']

    if brute_force:
        table.reset_index()[order_cols].to_csv(out_worst_f, index=False, sep='\t')

    else:
        table = table.sort_values(['Compound', 'Average']).reset_index()

        out_indexes = []

        for id in predictions.index:
            fragment_indexes = table[table['Compound'] == id].index
            if n_random > 0:
                worst_fragments = fragment_indexes[:n_worst-n_random]
                worst_fragments = set(worst_fragments)
                while len(worst_fragments) != n_worst and len(worst_fragments) != len(fragment_indexes):
                    worst_fragments.add(random.choice(fragment_indexes))
                out_indexes.extend(list(worst_fragments))
            else:
                out_indexes.extend(fragment_indexes[:n_worst])

        # save to file worst compounds
        table.iloc[out_indexes].reset_index()[order_cols].to_csv(out_worst_f, index=False, sep='\t')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=
                                     'Normalize fragments contributions and pick the worst ones.')
    parser.add_argument('-is', '--in_sdf', required=True,
                        help='path to file which contains selected compounds from pareto/desirability/...')
    parser.add_argument('-in', '--in_contrib', required=True, nargs='*',
                        help='list of paths to calculated contributions of fragments for specific parameter')
    parser.add_argument('-of', '--out_frag', required=True,
                        help='path to file where you want to store normalized contributions for fragments')
    parser.add_argument('-ow', '--out_worst', required=True,
                        help='path to file where you want to store the worst fragments')
    parser.add_argument('-p', '--parameters', required=True, nargs='*',
                        help='list of predicted parameters')
    parser.add_argument('-r', '--ranges', required=True, nargs='*',
                        help='list of ranges of all parameters')
    parser.add_argument('-m', '--models', required=True, nargs='*',
                        help='types of algorithms used for predictions, e.g. [svm_gbm rf rf_gbm_svm]')
    parser.add_argument('-t', '--thresholds', required=True, nargs='*',
                        help='thresholds to be matched')
    parser.add_argument('-n', '--n_worst', action='store', type=int,
                        help='specifies number of worst fragments')
    parser.add_argument('-r', '--random_fragments', action='store', type=float,
                        help='percent of random selected compounds')
    parser.add_argument('-bf', '--brute_force', action='store_true', default=False,
                        help='use all compounds, no selections')

    args = vars(parser.parse_args())

    main(args['in_sdf'], args['in_contrib'], args['out_frag'], args['out_worst'],
         args['parameters'], args['ranges'], args['models'], args['thresholds'],
         args['n_worst'], args['random_fragments'], args['brute_force'])
