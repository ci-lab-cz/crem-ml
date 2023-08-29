#!/usr/bin/env python

import argparse
import os
import shutil
import sys
import re
from typing import Dict
from rdkit import Chem

import datetime

import process_config
import optimizer_utils
import process_predictions
import process_contributions
import frag_replacement_with_crem as frag_replacement


def optimize(settings: Dict, input_config: str, brute_force: bool, number_generations: int) -> None:
    """
    Run all optimization tasks

    :param settings: dictionary with input settings
    :param input_config: path to input config file
    :param brute_force: specifies if we want to use selections process
    :param number_generations: specifies number of generations, use with brute_force
    """

    parameters_list_dicts = [settings[parameter] for parameter in settings if "param_" in parameter]

    # create database
    settings['output_database'] = optimizer_utils.create_database(
        settings['working_dir'], [parameter['name'] for parameter in parameters_list_dicts])

    shutil.copyfile(input_config, os.path.join(settings['working_dir'], 'config.yaml'))

    # create output folder
    settings['working_dir'] = os.path.join(settings['working_dir'], 'out')
    if os.path.exists(settings['working_dir']):
        shutil.rmtree(settings['working_dir'])

    for gen in range(settings['num_of_generation']):

        if gen > number_generations and number_generations != 0:
            print("Optimizer reached number of specified generations.")
            sys.exit()

        # create generation dir
        generation_dir = os.path.join(settings['working_dir'], 'generation_{}'.format(gen))
        if os.path.exists(generation_dir):
           shutil.rmtree(generation_dir)
        os.makedirs(generation_dir)

        # for new IDs and check if something was created
        num_of_compounds = 0

        if gen == 0:
            # add new unique compounds into database
            num_of_compounds = optimizer_utils.add_mols_into_db(settings['seed_structure'],
                                                                settings['output_database'],
                                                                gen)

            # check if we have new compounds in new generation
            if num_of_compounds == 0:
                print("\nIn generation {} aren't new compounds".format(gen))
                sys.exit()

        # copy input_sdf_file into gen directory
        new_sdf = os.path.join(generation_dir, 'input_dataset.sdf')
        shutil.copyfile(settings['seed_structure'], new_sdf)

        # start generation
        start = datetime.datetime.now()
        print(50 * '_', '\nGeneration {}: {}'.format(gen, start))
        if "std_rules" in settings:
            # standardization
            settings['seed_structure'] = optimizer_utils.standardize_sdf(
            input_sdf_file = new_sdf,
            std_rules_path = settings['std_rules'],
             chemaxon_path = settings[ 'chemaxon']  )

        else:# only Add Hs
            new_sdf_Hs = Chem.SDWriter(os.path.join(os.path.dirname(new_sdf), 'input_dataset_Hs.sdf'))
            for mol in Chem.SDMolSupplier(new_sdf, removeHs=False):
                new_sdf_Hs.write(Chem.AddHs(mol))
            new_sdf_Hs.close()
            settings['seed_structure'] = os.path.join(os.path.dirname(new_sdf), 'input_dataset_Hs.sdf')


        if settings['descriptors_type'] == 'sirms':  # calculation of  sirms descriptors
            # calc atomic properties
            settings['seed_structure'] = optimizer_utils.calculate_atomic_prop(
                input_sdf_file=settings['seed_structure'],
                chemaxon_path=settings['chemaxon'],
                properties=settings['properties_chemaxon']
            )
            # calculation of sirms descriptors
            optimizer_utils.calculate_sirms_descriptors(settings['seed_structure'],
                                        settings['setup_file'],
                                        settings['properties_sirms'],
                                        settings['output_format'],
                                        settings['n_cores']
                                        )

        else:
            # calculation of  fingerprints

            if settings['descriptors_type'] == "MPNN_fingerprint":
                if not settings["multitask"]:
                    for i, dict in enumerate(parameters_list_dicts): # over parameters
                        # set path with mpnn model;
                        mpnn_path = parameters_list_dicts[i]['path']

                        optimizer_utils.calculate_fingerprints(settings['seed_structure'],
                                                               settings['descriptors_type'],
                                                               settings['output_format'],
                                                               mpnn_path,
                                                               str(parameters_list_dicts[i]['name'])
                                                               )

                        # predict properties based on different x.txt for each param
                        fragments_fname = os.path.join(generation_dir, str(
                        parameters_list_dicts[i]['name']) + '_MPNN_fingerprint_x.txt')
                        # predict properties based on MPNN FP (for specific parameter)
                        optimizer_utils.predict_properties([parameters_list_dicts[i]],  # take only current param in []
                                                           fragments_fname,
                                                           settings['output_format'],
                                                           settings['multitask']
                                                           )
                else: #multitask
                    mpnn_path = parameters_list_dicts[0]['path'] #  they allhave same path
                    optimizer_utils.calculate_fingerprints(settings['seed_structure'],
                                                   settings['descriptors_type'],
                                                   settings['output_format'],
                                                   mpnn_path,
                                                   str(parameters_list_dicts[0]['name'])
                                                        )
                    # predict properties based on  x.txt of 1 st param;  for all paramas ( because model predict all properties at once)
                    fragments_fname = os.path.join(generation_dir, str(parameters_list_dicts[0]['name'])+'_MPNN_fingerprint_x.txt')
                    optimizer_utils.predict_properties([ parameters_list_dicts[0]],# even though [0], model predict all tasks at once
                                                       fragments_fname,
                                                       settings['output_format'],
                                                       settings['multitask']
                                                       )
                    # copy descriptors for all params
                    for i,d in enumerate(parameters_list_dicts):
                        if i == 0:# all params except first
                            continue
                        # copy fingerprint for every fold (models can have 1 or more crossvaliadtaion folds)
                        n = 0
                        for k in os.listdir(mpnn_path): # iterate over model folder to fid out number of folds
                            if "fold" in k:
                                cur_fr_nm = os.path.join(generation_dir,
                                                     str(parameters_list_dicts[i][
                                                             'name']) + '_MPNN_fingerprint_x_' + str(n) + '.txt')
                                shutil.copyfile(re.sub(".txt", "_" + str(n) + ".txt", fragments_fname),  # copy fp for first param, n-th fold
                                            cur_fr_nm)
                                n += 1

                    # no need to copy predicted properties for all params - they are already calculated & written
            else: # non MPNN
                optimizer_utils.calculate_fingerprints(settings['seed_structure'],
                                                       settings['descriptors_type'],
                                                       settings['output_format'],
                                                       )

        # predict properties based on single x.txt for all params

        if  settings['descriptors_type'] != "MPNN_fingerprint":
            fragments_fname = os.path.join(generation_dir, 'x.txt')
            optimizer_utils.predict_properties(parameters_list_dicts,
                                                   fragments_fname,
                                                   settings['output_format']
                                                   )




        # process predictions
        list_of_prediction_files = []   # prepare list of file paths with predictions
        for parameter in parameters_list_dicts:
            list_of_prediction_files.append(
                os.path.join(generation_dir, 'predictions_{}.txt'.format(parameter['name'])))
        print(settings["seed_structure"])
        settings['processed_predictions_file'] = os.path.join(generation_dir, 'processed_predictions.sdf')
        process_predictions.main(settings['seed_structure'],
                                     list_of_prediction_files,
                                     settings['output_database'],
                                     settings['processed_predictions_file'],
                                     [parameter['name'] for parameter in parameters_list_dicts],
                                     settings['optimization_method'],
                                     [parameter['threshold'] for parameter in parameters_list_dicts],
                                     settings['bounded_box'],
                                     [parameter['desirability'] for parameter in parameters_list_dicts],
                                     settings['number_of_selected_compounds'],
                                     settings['random_compounds_selection'],
                                     brute_force
                                 )

        # get num of fitted compounds
        num_of_fitted_compounds = optimizer_utils.count_fitted_compounds(settings['output_database'])

        # update mols in database
        if num_of_fitted_compounds >= settings['num_output_compounds'] and not brute_force:
            print("Optimizer reached number of fitted compounds specified in config.")
            sys.exit()

        # find fragments

        settings['fragments_ids_file'] = os.path.join(generation_dir, 'fragments_ids.txt')
        error_fname_frag = os.path.join(generation_dir, 'fragments_log.log')
        optimizer_utils.find_frags_rdkit(settings['processed_predictions_file'],
                                         settings['fragments_ids_file'],
                                         settings['smart_string'],
                                         settings['max_cuts'],
                                         # settings['radius'], # todo is it safe to not to use it at all?
                                         settings['keep_stereo'],
                                         error_fname_frag)
        if settings['descriptors_type'] == 'sirms':
            # calculate sirms descriptors of fragments
            optimizer_utils.calculate_sirms_descriptors(settings['processed_predictions_file'],
                                        settings['setup_file'],
                                        settings['properties_sirms'],
                                        settings['output_format'],
                                        settings['n_cores'],
                                        fragments_ids=settings['fragments_ids_file']
                                        )

        else:
            if settings['descriptors_type'] == 'MPNN_fingerprint':
                if not settings['multitask']:
                    for i, dict in enumerate(parameters_list_dicts):
                        # set path with mpnn model
                        mpnn_path = parameters_list_dicts[i]['path']
                        param_name = str(parameters_list_dicts[i]['name'])
                        optimizer_utils.calculate_fingerprints(settings['seed_structure'],
                                                               settings['descriptors_type'],
                                                               settings['output_format'],
                                                               mpnn_path,
                                                               param_name,
                                                               fragments_ids=settings['fragments_ids_file']
                                                               )

                        new_fragments_fname = os.path.join(generation_dir, str(parameters_list_dicts[i]['name']) + \
                                                           '_MPNN_fingerprint_new_x.txt')
                        # calc contrib using different new_x.txt for different parameter
                        optimizer_utils.calc_frag_contrib(new_fragments_fname,
                                                          [param_name],
                                                          [parameters_list_dicts[i]['types_of_alg']],
                                                          [mpnn_path],
                                                          [parameters_list_dicts[i]['type_of_model']],
                                                          settings['output_format'],
                                                          settings['multitask']
                                                          )

                else: # multitask
                    mpnn_path = parameters_list_dicts[0]['path']
                    param_name = str(parameters_list_dicts[0]['name'])
                    optimizer_utils.calculate_fingerprints(settings['seed_structure'],
                                                       settings['descriptors_type'],
                                                       settings['output_format'],
                                                       mpnn_path,
                                                        param_name,
                                                       fragments_ids=settings['fragments_ids_file']
                                                       )

                    new_fragments_fname = os.path.join(generation_dir, str(parameters_list_dicts[0]['name'])+'_MPNN_fingerprint_new_x.txt')
                    # calc contrib using new_x.txt of 1st parameter;  for all paramas  (because model predicts all properties at once)
                    optimizer_utils.calc_frag_contrib(new_fragments_fname, # even though [0], model predict all tasks at once
                                                          [param_name],
                                                           [parameters_list_dicts[0]['types_of_alg']],
                                                          [mpnn_path],
                                                          [parameters_list_dicts[0]['type_of_model']],

                                                          settings['output_format'],
                                                          settings['multitask']
                                                      )
                    # copy fps
                    for i, dict in enumerate(parameters_list_dicts):
                        if i  == 0: continue
                        n = 0
                        for k in os.listdir(mpnn_path):
                            if "fold" in k:
                                cur_new_fr_nm = os.path.join(generation_dir,
                                                             str(parameters_list_dicts[i][
                                                                     'name']) + '_MPNN_fingerprint_new_x_' + str(
                                                                 n) + '.txt')
                                shutil.copyfile(re.sub(".txt", "_" + str(n) + ".txt", new_fragments_fname),
                                                cur_new_fr_nm) # copy fp for nth fold, 0th param to nth fold i th param
                                n += 1
                    # no need to copy contribs  - they are already created
            else:
                # calculation of  fingerprints  specified in config
                optimizer_utils.calculate_fingerprints(settings['seed_structure'],
                                                   settings['descriptors_type'],
                                                   settings['output_format'],
                                                   fragments_ids=settings['fragments_ids_file']
                                                   )


        # calculate fragments contributions

        if  settings['descriptors_type'] != 'MPNN_fingerprint': # calculate contribs using SINGLE new_x.txt for each param
            new_fragments_fname = os.path.join(generation_dir, 'new_x.txt')
            optimizer_utils.calc_frag_contrib(new_fragments_fname,
                                          [parameter['name'] for parameter in parameters_list_dicts],
                                          [parameter['types_of_alg'] for parameter in parameters_list_dicts],
                                          [parameter['path'] for parameter in parameters_list_dicts],
                                          [parameter['type_of_model'] for parameter in parameters_list_dicts],

                                          settings['output_format'])

        # find worst fragments
        settings['fragments_contrib_files'] = [os.path.join(generation_dir, 'contrib_{}.txt'.format(parameter['name']))
                                               for parameter in parameters_list_dicts]
        settings['worst_fragments_file'] = os.path.join(generation_dir, 'worst_fragments.txt')
        types_of_alg_contrib = ['_'.join(parameter['types_of_alg']) for parameter in parameters_list_dicts]

        process_contributions.main(settings['processed_predictions_file'],
                                   settings['fragments_contrib_files'],
                                   os.path.join(generation_dir, 'fragment_contrib_norm.txt'),
                                   settings['worst_fragments_file'],
                                   [parameter['name'] for parameter in parameters_list_dicts],
                                   [parameter['range'] for parameter in parameters_list_dicts],
                                   types_of_alg_contrib,
                                   [parameter['threshold'] for parameter in parameters_list_dicts],
                                   settings['number_of_worst_fragments'],
                                   settings['random_fragments_selection'],
                                   brute_force)

        # replace fragments
        new_compouds = os.path.join(generation_dir, '{}_gen_compounds.sdf'.format(gen))
        if 'protected_ids' not in settings:
            settings['protected_ids'] = None
        if 'min_inc' not in settings:
            settings['min_inc'] = -2
        if 'max_inc' not in settings:
            settings['max_inc'] = 2
        # print(settings['protected_ids'])
        frag_replacement.main(settings['processed_predictions_file'],
                                        settings['worst_fragments_file'],
                                        settings['fragments_ids_file'],
                                        settings['replacement_database'],
                                        settings['radius'],
                                        settings['min_inc'],
                                        settings['max_inc'],
                                        settings['max_frag_size'],
                                        new_compouds,
                                        settings['n_cores'],
                                        settings['protected_ids'])

        settings['seed_structure'] = new_compouds

        num_of_compounds = optimizer_utils.add_mols_into_db(settings['seed_structure'],
                                                            settings['output_database'],
                                                            gen+1)

        # check if we have new compounds in new generation
        if num_of_compounds == 0:
            print("\nAfter generation {} aren't new compounds".format(gen))
            sys.exit()


def main():
    parser = argparse.ArgumentParser(description='System for multiobjective optimization of small molecules properties')
    parser.add_argument('-i', '--input_config',
                        help='path to config with input settings')
    parser.add_argument('-bf', '--brute_force', action='store_true', default=False,
                        help='use all compounds and all fragments, no selections')
    parser.add_argument('-g', '--number_generations', action='store', type=int, default=0,
                        help='specifies number of generations, use with brute force') # todo  why we need it here?
    parser.add_argument('-n', '--n_params', action='store', type=int,
                        help='specifies number of parameters to optimize, use with definition of config structure')
    parser.add_argument('-d', '--define_config_structure', action='store_true', default=False,
                        help='define config structure and save it to output location')
    args = vars(parser.parse_args())

    if args['define_config_structure']:
        assert args['n_params'], "You have to specify number of parameters with -n or --n_params option"
        process_config.create_config(args['output_location'], args['n_params'])
    else:
        settings = process_config.test_config(args['input_config'])
        optimize(settings, args['input_config'], args['brute_force'], args['number_generations'])


if __name__ == '__main__':
    main()
