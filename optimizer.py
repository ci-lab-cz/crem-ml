#!/usr/bin/env python

import argparse
import os
import shutil
import sys

from typing import Dict

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
        settings['seed_structure'] = new_sdf

        # start generation
        start = datetime.datetime.now()
        print(50 * '_', '\nGeneration {}: {}'.format(gen, start))

        # standardization
        settings['seed_structure'] = optimizer_utils.standardize_sdf(
            input_sdf_file=settings['seed_structure'],
            std_rules_path=settings['std_rules'],
            chemaxon_path=settings['chemaxon']
        )
        if settings['descriptors_type'] == 'sirms':
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
            # calculation of  fingerprints  specified in config
            optimizer_utils.calculate_fingerprints(settings['seed_structure'],
                                                   settings['descriptors_type'],
                                                   settings['output_format'],

                                                        )
        fragments_fname = os.path.join(generation_dir, 'x.txt')

        # predict properties of std_lbl_sdf file
        optimizer_utils.predict_properties(parameters_list_dicts,
                                           fragments_fname,
                                           settings['output_format']
                                           )

        # process predictions
        list_of_prediction_files = []   # prepare list of file paths with predictions
        for parameter in parameters_list_dicts:
            list_of_prediction_files.append(
                os.path.join(generation_dir, 'predictions_{}.txt'.format(parameter['name'])))

        settings['processed_predictions_file'] = os.path.join(generation_dir, 'processed_predictions.sdf')
        process_predictions.main(settings['seed_structure'],
                                     list_of_prediction_files,
                                     settings['output_database'],
                                     settings['processed_predictions_file'],
                                     [parameter['name'] for parameter in parameters_list_dicts],
                                     settings['optimization_methods'],
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
            # calculation of  fingerprints  specified in config
            optimizer_utils.calculate_fingerprints(settings['seed_structure'],
                                                   settings['descriptors_type'],
                                                   settings['output_format'],
                                                   fragments_ids=settings['fragments_ids_file']
                                                   )
        new_fragments_fname = os.path.join(generation_dir, 'new_x.txt')

        # calculate fragments contributions
        optimizer_utils.calc_frag_contrib(new_fragments_fname,
                                          [parameter['name'] for parameter in parameters_list_dicts],
                                          [parameter['types_of_alg'] for parameter in parameters_list_dicts],
                                          [parameter['path'] for parameter in parameters_list_dicts],
                                          [parameter['type_of_model'] for parameter in parameters_list_dicts],
                                          settings['properties_calc_contrib'],
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
        print(settings['protected_ids'])
        frag_replacement.main(settings['processed_predictions_file'],
                                        settings['worst_fragments_file'],
                                        settings['fragments_ids_file'],
                                        settings['replacement_database'],
                                        settings['radius'],
                                        settings['max_frag_size'],  # todo: what frag? hac or all?
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
