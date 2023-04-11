#!/usr/bin/env python

import argparse
import os
import math
import random

import sqlite3 as lite
import pandas as pd
pd.options.mode.chained_assignment = None

import pareto_simple_cull as pareto_alg
from optimizer_utils import parse_threshold

from sympy import symbols
from sympy.parsing.sympy_parser import parse_expr
import pickle as pkl
from typing import List
from typing import NewType
pandas_table = NewType('Processed pandas table with id of compound and predicted properties',
                      pd.DataFrame
                      )


def save_output_old(input_sdf: str, out_fname: str, output_poll: pandas_table) -> None:

    """
    Save output dictionary to file with predicted values. Can't use rdkit
    save_output_poll in optimizer_utils. Somehow it changes structure of compounds
    and during calculating fragment contribution it changes predicted values

    :param in_sdf: path to input sdf file
    :param out_fname: path to output sdf file
    :output_poll: pandas table with selected compounds
    """

    output_string = ''

    in_file = open(input_sdf, 'r')
    out_file = open(out_fname, 'w')
    iter_file = iter(in_file)
    not_find_beg = True
    not_find_end = True
    for id in output_poll.index:
        while not_find_beg:
            line = in_file.readline().rstrip()
            if line == id:
                output_string += line + '\n'
                while not_find_beg:
                    line = in_file.readline().rstrip()
                    if '$$$$' in line:
                        for record, parameter in zip(output_poll.loc[id], output_poll.columns):
                            output_string += '>  <pred_{}>\n {}\n\n'.format(parameter, record)
                        output_string += line + '\n'
                        not_find_end = False
                        not_find_beg = False
                    else:
                        output_string += line + '\n'
            else:
                while not '$$$$' in line:
                    line = in_file.readline().rstrip()
                pass
        in_file.seek(0)
        not_find_beg, not_find_end = True, True

    out_file.write(output_string)
    out_file.close()
    in_file.close()

def save_output(input_sdf: str, out_fname: str, output_poll: pandas_table) -> None:

    """
    Save output dictionary to file with predicted values. Can't use rdkit
    save_output_poll in optimizer_utils. Somehow it changes structure of compounds
    and during calculating fragment contribution it changes predicted values

    :param in_sdf: path to input sdf file
    :param out_fname: path to output sdf file
    :output_poll: pandas table with selected compounds
    """

    output_mols, mol_str = [], []

    in_file = open(input_sdf, 'r')
    out_file = open(out_fname, 'w')

    this_line_is_id = False
    found_id = 'NONE'

    for line in in_file:
        line = line.rstrip()

        if this_line_is_id:
            found_id = line
            this_line_is_id = False

        if line == '>  <ID>':
            this_line_is_id = True

        if line.rstrip() == '$$$$':
            if found_id in output_poll.index:
                for record, parameter in zip(output_poll.loc[found_id], output_poll.columns):
                    mol_str.append('>  <pred_{}>'.format(parameter))
                    mol_str.append(' {}'.format(record))
                    mol_str.append('')
                mol_str[0] = found_id
                mol_str.append(line)
                output_mols.append('\n'.join(mol_str))
                mol_str = []
                found_id = 'NONE'
            else:
                mol_str = []

        else:
            mol_str.append(line)

    out_file.write('\n'.join(output_mols))
    out_file.close()
    in_file.close()

def prepare_working_arr(in_pred: List, parameters: List, bounded_box: bool, proba_consensus:bool=True) -> pandas_table:
    #  todo param proba_consensus should go to config, so regression recalculation will be avoided+bettertracking of run
    """
    Reads file with predictions and process it into pandas table

    :param in_pred: list of paths to files with all predictions. Columns: 'Compounds', model0,model1,..,modeln,'consensus',	'bound_box'
    :param parameters: list of parameters to predict
    :param bounded_box: if True, then return only compounds within bounded box
    :param proba_consensus:bool=True. affects only classification (for regression will recalculate same value).
    If True - replace consensus with mean probability (flat mean over all models, except intermediate consensus models,
     i.e. svm_0, _1, rf_0, _1...).
    :return: pandas table with prepared predictions (consensus pred for each parameter). Columns:'id', param0,param1,..,paramn
    """

    tables = [pd.read_table(file) for file in in_pred]

    for table, parameter in zip(tables, parameters):
        # if ad
        if bounded_box:
            if table[table.bound_box == 1].shape[0] == 0: # no compounds in ad
                return None
            else:
                table.drop(table[table.bound_box==0].index, inplace=True)

        table.drop('bound_box', axis=1, inplace=True)
        if proba_consensus:
            table.drop(table.columns[-1], axis=1, inplace=True)# drop  consensus
            table['consensus']  = table.loc[:,['consensus' not in i for i in  table.columns]].mean(axis=1) # get new consensus

        cols_to_drop = list(range(1,table.shape[1]-1)) # drop all but (new) consensus
        table.drop(table.columns[cols_to_drop], axis=1, inplace=True)
        table.rename(columns={table.columns[0]: 'id', 'consensus': parameter}, inplace=True)
        table.set_index('id', inplace=True)

    tables = pd.concat(tables, axis=1, join='inner')

    # if ad
    if bounded_box:
        if tables.shape[0] == 0:
            return None

    return tables

def compute_distance_from_threshold(x: float, threshold: List) -> float:
    """
    Compute distance of predicted value from threshold value.
    This function is applied on pandas series.

    :param x: predicted value
    :param threshold: parsed threshold list, e.g. ['more', 7]
    :return: computed distance from one predicted value
    """

    if threshold[0] == 'more':
        if x > threshold[1]:
            return 0
        else:
            return threshold[1] - x

    elif threshold[0] == 'less':
        if x < threshold[1]:
            return 0
        else:
            return x - threshold[1]

    else:   # between
        if x >= threshold[1] and x <= threshold[2]:
            return 0
        else:
            return threshold[1] - x if x < threshold[1] else x - threshold[2]

def update_database(out_database: str, predictions: pandas_table,
                    output_filtering: pandas_table) -> None:
    """
    Updates predicted values for compounds in database.

    :param out_database: path to output database
    :param predictions: prepared pandas table with all data
    :param output_filtering: fitted compounds in table
    """

    columns = predictions.columns
    database_columns = ["predicted_{}".format(cols) for cols in columns]

    query = "UPDATE optimizer_table SET fit=?, "
    query += "=?, ".join(database_columns) + "=? WHERE id=?"

    con = lite.connect(out_database)
    with con:
        cursor = con.cursor()

        for index, row in predictions.iterrows():
            record = []
            if index in output_filtering.index:
                record.append(1)
            else:
                record.append(0)
            for col in columns:
                record.append(row[col])
            record.append(index)

            cursor.execute(query, tuple(record))
        con.commit()

def prepare_points_for_pareto(table: pandas_table) -> List:
    """
    Process pandas table into list of points which are distances from threshold.

    :param table: pandas table with distances from threshold of compounds
    :return: list of lists with distances, e.g. [[dist11, dist12], [dist21, dist22]]
    """

    points = []
    for index, row in table.iterrows():
        points.append(row.tolist())
    return points

def process_desirability_functions(desirabilities: List) -> List:
    """
    It parses list of desirabilities to special format

    :param desirabilities: list of desirabilities
    :return: special list of functions, e.g. [['0.45', 0], ['0.55', 10*x - 4.5], ['1000', 1]] - one desirability function
    """

    if len(desirabilities) == 1:   # if we have only one function
        function = desirabilities[0].split(",")
        for index, fun in enumerate(function):
            function[index] = [fun.split(":")[0]] + [parse_expr(fun.split(":")[1])]
        return [function]
    else:
        functions = []
        for fnc in desirabilities:
            fnc = fnc.split(',')
            for index, fun in enumerate(fnc):
                fnc[index] = [fun.split(":")[0]] + [parse_expr(fun.split(":")[1])]
            functions.append(fnc)
        return functions

def get_norm_value(x_input: float, function: List) -> float:
    """
    Compute norm value from predicted parameter. This function is applied on
    pandas series

    :param x: predicted value
    :param function: desirability function, e.g. [['0.45', 0], ['0.55', 10*x - 4.5], ['1000', 1]]
    :return: norm value
    """

    x = symbols("x")
    for index, bound in enumerate(function):
        if round(x_input, 5) <= float(bound[0]):
            return float(function[index][1].subs(x, x_input))
    return 0

def main(in_sdf, in_pred, out_database, out_fname, parameters,
         optimization_methods, thresholds, ad, desirabilities=[],
         n_compounds=0, random_compounds=0, brute_force=False):

    print('Processing predictions ...')

    selected_compounds_index = set()

    # process all predictions
    predictions = prepare_working_arr(in_pred, parameters, ad)

    if predictions is None:
        print('Compounds are not in ad. Calculating outside ad!')
        predictions = prepare_working_arr(in_pred, parameters, False)

    # check if brute_force is selected, then no selections
    if brute_force:
        save_output(in_sdf,
                    out_fname,
                    predictions)
        return predictions.shape[0]

    else:
        # filtering
        thresholds = parse_threshold(thresholds)

        # compute distances from thresholds and use it with pareto if specified
        distance_predictions = predictions.copy()
        for parameter, threshold in zip(parameters, thresholds):
            distance_predictions[parameter] = predictions[parameter].apply(compute_distance_from_threshold,
                                                                  threshold=threshold)

        # find compounds which are in threshold
        output_filtering = distance_predictions[distance_predictions.sum(axis=1) == 0]
        output_filtering = predictions.loc[output_filtering.index].copy()

        update_database(out_database, prepare_working_arr(in_pred, parameters, False), output_filtering)

        # set order of methods # todo:  unable using both methods squentiaally! allow only  one  (or change the logic)
        if 'desirability' in optimization_methods and 'pareto' in optimization_methods and len(optimization_methods) == 2:
            optimization_methods = ['pareto', 'desirability']

        for method in optimization_methods:
            if method == 'pareto':

                # use compounds which are not in threshold
                distance_predictions = distance_predictions[distance_predictions.sum(axis=1) > 0]
                print(distance_predictions.head())
                print(parameters)
                # get list of indexes from pareto frontier
                pareto = pareto_alg.is_pareto_efficient_simple(distance_predictions.loc[:,parameters].values)

                for index in predictions.loc[distance_predictions.iloc[pareto].index].index:
                    selected_compounds_index.add(index)

            elif method == 'desirability':

                num_of_pareto_selected = len(selected_compounds_index)
                if len(selected_compounds_index) >= n_compounds:
                    desirability_predictions = predictions.loc[selected_compounds_index].copy()
                else:
                    desirability_predictions = predictions.loc[~predictions.index.isin(
                        selected_compounds_index
                    )].copy()

                # use compounds which are not in threshold
                desirability_predictions = desirability_predictions.loc[distance_predictions[distance_predictions.sum(axis=1) > 0].index]

                # we have less or equal compounds in input sdf then we specified
                # that we need from this stage, so we use all of them
                if n_compounds >= desirability_predictions.shape[0]:
                    for index in predictions.loc[desirability_predictions.index].index:
                        selected_compounds_index.add(index)
                else:
                    functions = process_desirability_functions(desirabilities)

                    for parameter, function in zip(parameters, functions):
                        desirability_predictions[parameter] = desirability_predictions[parameter].\
                            apply(get_norm_value, function=function)

                    desirability_predictions['desirability'] = desirability_predictions.sum(axis=1)/(len(parameters))
                    desirability_predictions = desirability_predictions.sort_values(by='desirability', ascending=False)

                    for index in predictions.loc[desirability_predictions.head(n_compounds-num_of_pareto_selected).index].index:
                        selected_compounds_index.add(index)

            else:
                print('Unspecified optimization method!')

        if output_filtering.shape[0] > 0:
            save_output(in_sdf,
                        os.path.join(os.path.dirname(in_sdf), 'output_match.sdf'),
                        output_filtering)

        if random_compounds > 0:
            selected_indexes = not_random_selection(
                optimization_methods,
                random_compounds,
                selected_compounds_index,
                n_compounds
            )
        else:
            selected_indexes = set()

        # add to processed compounds also filtered compounds
        n_random = math.floor(n_compounds * random_compounds)
        filtered_indexes = list(output_filtering.index)

        selected_indexes.update(set(filtered_indexes))

        for i in selected_compounds_index:
            if i not in filtered_indexes:
                try:
                    selected_indexes.remove(i)
                except KeyError:
                    pass
        selected_indexes = set(list(selected_indexes)[:n_compounds-n_random])
        while len(selected_indexes) != n_compounds and len(selected_indexes) != predictions.shape[0]:
            selected_indexes.add(random.choice(predictions.index))

        # save selected compounds
        # every compounds were filtered
        if len(selected_indexes) == 0:
            selected_indexes = output_filtering.head(n_compounds).index
        save_output(in_sdf,
                    out_fname,
                    predictions.loc[list(selected_indexes)])


def not_random_selection(optimization_methods, random_ratio, selected_compounds, n_compounds):

    n_random = math.floor(n_compounds * random_ratio)
    if n_random == 0:
        return selected_compounds.copy()

    # desirability
    if len(optimization_methods) == 1 and 'desirability' in optimization_methods:
        if len(selected_compounds) == n_compounds:
            best_compounds = set(list(selected_compounds)[:-n_random])
            # while len(best_compounds) != n_compounds and len(best_compounds) != len(compounds):
            #     best_compounds.add(random.choice(compounds.index))
            return best_compounds
        else:
            return selected_compounds.copy()

    if len(optimization_methods) == 1 and 'pareto' in optimization_methods:
        if len(selected_compounds) == n_compounds:
            best_compounds = set(list(selected_compounds)[:-n_random])
            # while len(best_compounds) != n_compounds and len(best_compounds) != len(compounds):
            #     best_compounds.add(random.choice(compounds.index))
            return best_compounds
        else:
            best_compounds = set()
            while len(best_compounds) != (n_compounds - n_random) and len(best_compounds) != len(selected_compounds):
                best_compounds.add(random.choice(list(selected_compounds)))

            # while len(best_compounds) != n_compounds and len(best_compounds) != len(compounds):
            #     best_compounds.add(random.choice(compounds.index))
            return best_compounds

    # pareto then desirability
    if len(optimization_methods) == 2 and 'pareto' in optimization_methods and 'desirability' in optimization_methods:
        if len(selected_compounds) == n_compounds:
            selected_compounds = list(selected_compounds)
            random.shuffle(selected_compounds)
            best_compounds = set(selected_compounds[:-n_random])
            # while len(best_compounds) != n_compounds and len(best_compounds) != len(compounds):
            #     best_compounds.add(random.choice(compounds.index))
            return best_compounds
        else:
            return selected_compounds.copy()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Process predicted values with specified optimization method')
    parser.add_argument('-is', '--in_sdf', required=True,
                        help='path to file which contains standardized compounds')
    parser.add_argument('-ip', '--in_pred', required=True, nargs='*',
                        help='path to files which contains predicted values for properties')
    parser.add_argument('-od', '--output_database', required=True,
                        help='path to output database')
    parser.add_argument('-o', '--out', required=True,
                        help='processed predictions using specified opt. method')
    parser.add_argument('-p', '--parameters', required=True, nargs='*',
                        help='parameters for prediction')
    parser.add_argument('-m', '--methods', nargs='*',
                        help='method for processing predictions: pareto or desirability')
    parser.add_argument('-t', '--thresholds', nargs='*',
                        help='thresholds to be match, written in the same order as properties')
    parser.add_argument('-a', '--ad', action='store_true', default=False,
                        help='save to output file only if it is in the application domain')
    parser.add_argument('-d', '--desirabilities', nargs='*',
                        help='if desirability method specified, need to specify desirability string')
    parser.add_argument('-n', '--n_compounds', action='store', type=int,
                        help='if desirability method specified, need to specify number of selected compounds')
    parser.add_argument('-r', '--random_compounds', action='store', type=float,
                        help='percent of random selected compounds')
    parser.add_argument('-bf', '--brute_force', action='store_true', default=False,
                        help='use all compounds, no selections')

    args = vars(parser.parse_args())

    main(args['in_sdf'], args['in_pred'], args['output_database'], args['out'],
         args['parameters'], args['methods'], args['thresholds'], args['ad'],
         args['desirabilities'], args['n_compounds'], args['random_compounds'],
         args['brute_force'])
