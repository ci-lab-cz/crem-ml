#!/usr/bin/env python

import argparse
import os
import math
import random
import numpy as np
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

        if   '>  <id>' in line:
            this_line_is_id = True

        if line.rstrip() == '$$$$':
            if found_id in output_poll.index:
                for record, parameter in zip(output_poll.loc[found_id], output_poll.columns):
                    mol_str.append('>  <pred_{}>'.format(parameter))
                    mol_str.append('{}'.format(record))
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


def prepare_working_arr(in_pred: List, parameters: List, bounding_box: bool, proba_consensus:bool=True, spci_models:bool=False) -> pandas_table:
    #  todo param proba_consensus should go to config, so regression recalculation will be avoided+bettertracking of run
    """
    Reads file with predictions and process it into pandas table

    :param in_pred: list of paths to files with all predictions. Columns: 'Compounds', model0,model1,..,modeln,'consensus',	'bound_box'
    :param parameters: list of parameters to predict
    :param bounding_box: if True, then return only compounds within bounded box
    :param proba_consensus:bool=True. affects only classification (for regression will recalculate same value).
    If True - replace consensus with mean probability (flat mean over all models, except intermediate consensus models,
     i.e. svm_0, _1, rf_0, _1...).
    :return: pandas table with prepared predictions (consensus pred for each parameter). Columns:'id', param0,param1,..,paramn
    """

    tables = [pd.read_table(file) for file in in_pred]

    for table, parameter in zip(tables, parameters):
        # if ad
        if bounding_box:
            if table[table.bound_box == 1].shape[0] == 0: # no compounds in ad
                return None
            else:
                table.drop(table[table.bound_box==0].index, inplace=True)

        if proba_consensus:
            if 'consensus' in table.columns:
                table.drop('consensus', axis=1, inplace=True)
            model_cols = [c for c in table.columns if 'Compounds' not in str(c) and c != 'bound_box']
            table['consensus']  = table[model_cols].mean(axis=1)

        cols_to_keep = [table.columns[0], 'consensus']
        if spci_models and 'bound_box' in table.columns:
            cols_to_keep.append('bound_box')
            
        cols_to_drop = [c for c in table.columns if c not in cols_to_keep]
        table.drop(cols_to_drop, axis=1, inplace=True)
        
        rename_dict = {table.columns[0]: 'id', 'consensus': parameter}
        if spci_models and 'bound_box' in table.columns:
            rename_dict['bound_box'] = f'bound_box_{parameter}'
            
        table.rename(columns=rename_dict, inplace=True)
        table.set_index('id', inplace=True)

    tables = pd.concat(tables, axis=1, join='inner')
    
    if spci_models:
        bb_cols = [col for col in tables.columns if str(col).startswith('bound_box_')]
        if bb_cols:
            # Bounding box is evaluated as the minimum (AND operation) across all SPCI models
            tables['bounding_box'] = tables[bb_cols].min(axis=1).astype(int)
            tables.drop(bb_cols, axis=1, inplace=True)

    # if ad
    if bounding_box:   # TODO: PP, this condition is not needed
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
       return threshold[1] - x

    elif threshold[0] == 'less':
        return x - threshold[1]

    else:   # between
        return threshold[1] - x if x < threshold[1] else x - threshold[2]


def update_database(out_database: str, predictions: pandas_table,
                    output_filtering: pandas_table, spci_models: bool = False) -> None:
    """
    Updates predicted values for compounds in database.

    :param out_database: path to output database
    :param predictions: prepared pandas table with all data
    :param output_filtering: fitted compounds in table
    """

    columns = [col for col in predictions.columns if col != 'bounding_box']
    database_columns = ["predicted_{}".format(cols) for cols in columns]

    if spci_models and 'bounding_box' in predictions.columns:
        query = "UPDATE optimizer_table SET fit=?, "
        query += "=?, ".join(database_columns) + "=?, bounding_box=? WHERE id=?"
    else:
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
                
            if spci_models and 'bounding_box' in predictions.columns:
                record.append(int(row['bounding_box']))
                
            record.append(index)

            cursor.execute(query, tuple(record))
        con.commit()


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
         optimization_method, thresholds, ad, desirabilities=None,
         n_compounds=0, random_compounds=0, additive_agg = True, brute_force=False, spci_models=False):
    """
    Logic of algorithm:
    1. if brute force - save all compounds to out_fname.
    2. Otherwise:
      - calculate distances
      - filter out and save compounds with all(distance <= 0)- cmpds satisfying config criteria
      - add them to db
      - find out needed number of random compounds
      - filter out    any(dist>0) - cmpds not yet satisfying criteria.  Work with them
      - if less then  specified  n_compounds: take them all

      -  elif PARETO:
        - in a loop: run pareto on compounds "in  any(dist > 0) & not in previous pareto frontier" and
          select all compounds from pareto frontier of 1st, second...etc orders - until reach (possibly exceed)
          specified {n_compounds - random_compounds} OR until there's no more compounds to select from any(dist > 0)
         
      - elif DESIRABILITY:
            - compute desirability and rank compounds
            - if random_compounds: take only top {n_compounds - random compounds needed number}

      - add sample from any(dist > 0) & not in already selected; sample size = random compounds needed number
     (i.e. n_compounds * random_compounds)
      - save
    """
    print('Processing predictions ...')

    # process all predictions
    predictions = prepare_working_arr(in_pred, parameters, ad, spci_models=spci_models)

    if ad and predictions is None:
        print('Compounds are not in ad. Calculating outside ad!')
        predictions = prepare_working_arr(in_pred, parameters, False, spci_models=spci_models)

    # check if brute_force is selected, then no selections
    if brute_force:
        save_output(in_sdf,
                    out_fname,
                    predictions)
        return predictions.shape[0]

    else:
        # filtering
        thresholds = parse_threshold(thresholds)

        # compute distances from thresholds
        distance_predictions = predictions.copy()
        for parameter, threshold in zip(parameters, thresholds):
            distance_predictions[parameter] = predictions[parameter].apply(compute_distance_from_threshold,
                                                                  threshold=threshold)

        # find compounds which are in threshold
        output_filtering = distance_predictions[distance_predictions[parameters].apply(lambda x:  np.all(x<=0), axis=1)] # all parameteres within thres  
        output_filtering = predictions.loc[output_filtering.index].copy()
        if output_filtering.shape[0] > 0:
            save_output(
                in_sdf,
                os.path.join(os.path.dirname(in_sdf), 'output_match.sdf'),
                output_filtering
            )
        update_database(out_database, prepare_working_arr(in_pred, parameters, False, spci_models=spci_models), output_filtering, spci_models=spci_models)

        # calc random compounds needed number
        n_random = math.floor(n_compounds * random_compounds) #  will be used later
        ids_not_in_thr = distance_predictions[parameters].apply(lambda x: np.any(x > 0), axis=1) # ids of compounds not in threshold
        # print(ids_not_in_thr)

        # if we have less or equal compounds in input sdf then we specified
        # that we need in config, so we use all of them, regardless of optimization method:
        if n_compounds >= sum(ids_not_in_thr):
            selected_compounds_index = list(predictions.loc[distance_predictions[ids_not_in_thr].index].index)

        elif optimization_method == 'pareto':
                # use compounds which are not in threshold
                distance_predictions = distance_predictions[ids_not_in_thr]


                # get list of indexes from pareto frontier

                # repeat finding pareto frontier adding each time points from next  frontier - "second order", "third order"..
                selected_compounds_index = []
                pareto = np.zeros((distance_predictions.shape[0],), dtype=bool)# init with all false to use its inverse  each time
                # keep adding compounds from next order fronier until n_compounds is reached (reduced by n_random) or until
                # no more compounds left to select from
                while (sum(pareto) < (n_compounds-n_random)) and ( sum(pareto) < distance_predictions.shape[0]):
                    pareto_new = pareto_alg.is_pareto_efficient_simple(distance_predictions.loc[~pareto,parameters].values)
                #    change some of False points to True - new order frontier (by selecting "~" , i.e. False old True points remain
                    pareto[~pareto] = pareto_new
                selected_compounds_index = list(predictions.loc[distance_predictions.iloc[pareto].index].index)

        elif optimization_method == 'desirability':
                # use compounds which are not in threshold
                desirability_predictions = predictions.loc[distance_predictions[ids_not_in_thr].index]
                functions = process_desirability_functions(desirabilities)
                for parameter, function in zip(parameters, functions):
                    desirability_predictions[parameter] = desirability_predictions[parameter].\
                        apply(get_norm_value, function=function)

                if additive_agg: # additive
                    desirability_predictions['desirability'] = desirability_predictions[parameters].sum(axis=1)/(len(parameters))
                else: # multiplicative
                    desirability_predictions['desirability'] = desirability_predictions[parameters].product(axis=1)

                desirability_predictions = desirability_predictions.sort_values(by='desirability', ascending=False)
                selected_compounds_index = predictions.loc[desirability_predictions.head(n_compounds).index].index
                if random_compounds > 0:

                    selected_compounds_index = list(selected_compounds_index[:-n_random])
        else:
                print('Unspecified optimization method!')

        if random_compounds > 0:
            if len(selected_compounds_index) < n_compounds:
                predictions_diff = predictions.loc[predictions.index.difference(selected_compounds_index)]
                predictions_diff = predictions_diff.loc[predictions_diff.index.difference(output_filtering.index)]
                if predictions_diff.shape[0]>0: # any data available
                    selected_compounds_index.extend(
                        predictions_diff.sample( n=min(n_random,len(predictions_diff))).index ) # take min because may be not enough

        # save selected compounds

        save_output(
            in_sdf,
            out_fname,
            predictions.loc[selected_compounds_index]
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Process predicted values with specified optimization method')
    parser.add_argument('-is', '--in_sdf', required=True,
                        help='path to file which contains [optinally standardized] compounds')
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
                        help='number of selected compounds')
    parser.add_argument('-r', '--random_compounds', action='store', type=float,
                        help='percent of random selected compounds')
    parser.add_argument('-bf', '--brute_force', action='store_true', default=False,
                        help='use all compounds, no selections')

    parser.add_argument('-spci', '--spci_models', action='store_true', default=False,
                        help='if spci models are used')

    args = vars(parser.parse_args())

    main(args['in_sdf'], args['in_pred'], args['output_database'], args['out'],
         args['parameters'], args['methods'], args['thresholds'], args['ad'],
         args['desirabilities'], args['n_compounds'], args['random_compounds'],
         additive_agg=True, brute_force=args['brute_force'], spci_models=args['spci_models'])
