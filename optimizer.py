#!/usr/bin/env python

import argparse
import os
import shutil
import sys
import re
from typing import Dict
from rdkit import Chem

import datetime

import chemprop_descr_and_predict
import chemprop_frag_contrib
import process_config
import optimizer_utils
import process_predictions
import process_contributions
import frag_replacement_with_crem as frag_replacement


def optimize(settings: Dict, input_config: str, brute_force: bool) -> None:
    """
    Run all optimization tasks

    :param settings: dictionary with input settings
    :param input_config: path to input config file
    :param brute_force: specifies if we want to use selections process
    """

    parameters_list_dicts = [settings[parameter] for parameter in settings if "param_" in parameter]

    use_spci_models = settings['descriptors_type'] != "MPNN_fingerprint"

    # create database
    settings['output_database'] = optimizer_utils.create_database(
        settings['working_dir'],
        [parameter['name'] for parameter in parameters_list_dicts],
        use_spci_models
    )

    shutil.copyfile(input_config, os.path.join(settings['working_dir'], 'config.yaml'))

    # create output folder
    settings['working_dir'] = os.path.join(settings['working_dir'], 'out')
    if os.path.exists(settings['working_dir']):
        shutil.rmtree(settings['working_dir'])

    for gen in range(settings['num_of_generations'] + 1):  # +1 to predict properties at the last iteration

        # create generation dir
        generation_dir = os.path.join(settings['working_dir'], 'generation_{}'.format(gen))
        if os.path.exists(generation_dir):  # TODO: PP, not necessary if the parent dir was already cleared
           shutil.rmtree(generation_dir)
        os.makedirs(generation_dir)

        if gen == 0:
            # add new unique compounds into database
            num_of_compounds = optimizer_utils.add_mols_into_db(
                settings['seed_structure'],
                settings['output_database'],
                gen
            )

            # check if we have new compounds in new generation
            if num_of_compounds == 0:
                print("\nIn generation {} aren't new compounds".format(gen))
                sys.exit()

        mols = optimizer_utils.get_mols(settings['output_database'],
                                        gen,
                                        ['id', 'parent', 'transformation', 'protected_ids'])
        settings['seed_structure'] = os.path.join(generation_dir,
                                                  'input_dataset_Hs.sdf')
        writer = Chem.SDWriter(settings['seed_structure'])
        for mol in mols:
            writer.write(mol)
        writer.close()

        # start generation
        start = datetime.datetime.now()
        print(50 * '_', '\nGeneration {}: {}'.format(gen, start))

        # calculation of descriptors and save to x.txt

        if settings['descriptors_type'] == 'sirms':  # calculation of  sirms descriptors
            optimizer_utils.calculate_sirms_descriptors(
                settings['seed_structure'],
                settings['n_cores']
            )

        elif settings['descriptors_type'] != "MPNN_fingerprint":
            optimizer_utils.calculate_fingerprints(
                settings['seed_structure'],
                settings['descriptors_type'],
            )

        # predict properties based on single x.txt for all params or using MPNN models directly

        if  settings['descriptors_type'] != "MPNN_fingerprint":
            fragments_fname = os.path.join(generation_dir, 'x.txt')
            optimizer_utils.predict_properties(
                parameters_list_dicts,
                fragments_fname
            )

        elif settings['descriptors_type'] == "MPNN_fingerprint":
            if not settings["multitask"]: # default multitask is False
                for i, dict in enumerate(parameters_list_dicts): # over parameters
                    # set path with mpnn model;
                    mpnn_path = parameters_list_dicts[i]['path']
                    param_name = parameters_list_dicts[i]['name']
                    chemprop_descr_and_predict.main_params(
                        in_fname=settings['seed_structure'],
                        out_fname=os.path.join(generation_dir,
                                               'predictions_{}.txt'.format(param_name)),
                        model_path=mpnn_path,
                        model_type=parameters_list_dicts[i]['type_of_model'],
                        variance_threshold=settings['variance_threshold'],
                        multitask=False
                    )

            else: #multitask
                mpnn_path = parameters_list_dicts[0]['path'] #  they allhave same path
                param_name = parameters_list_dicts[0]['name'] # just to init with , it will be changed for each param
                chemprop_descr_and_predict.main_params(
                    in_fname=settings['seed_structure'],
                    out_fname=os.path.join(generation_dir,
                                           'predictions_{}.txt'.format(param_name)),
                    model_path=mpnn_path,
                    model_type=parameters_list_dicts[0]['type_of_model'],
                    variance_threshold=settings['variance_threshold'],
                    multitask=True
                )

        # process predictions and save processed_predictions.sdf with selected compounds for CReM mutation
        # predicted activities will be saved to database at the same time

        list_of_prediction_files = []   # prepare list of file paths with predictions
        for parameter in parameters_list_dicts:
            list_of_prediction_files.append(
                os.path.join(generation_dir,
                             'predictions_{}.txt'.format(parameter['name'])))
        settings['processed_predictions_file'] = os.path.join(generation_dir, 'processed_predictions.sdf')
        if settings["optimization_method"] == "desirability":
            desirabilities  = [parameter['desirability'] for parameter in parameters_list_dicts]
        else:
            desirabilities = None

        process_predictions.main(
            settings['seed_structure'],
            list_of_prediction_files,
            settings['output_database'],
            settings['processed_predictions_file'],
            [parameter['name'] for parameter in parameters_list_dicts],
            settings['optimization_method'],
            [parameter['threshold'] for parameter in parameters_list_dicts],
            settings['bounding_box'],
            desirabilities,
            settings['number_of_selected_compounds'],
            settings['random_compounds_selection'],
            brute_force=brute_force,
            spci_models=use_spci_models
        )

        # get num of fitted compounds
        num_of_fitted_compounds = optimizer_utils.count_fitted_compounds(settings['output_database'])
        if num_of_fitted_compounds >= settings['num_output_compounds'] and not brute_force:
            print("Optimizer reached number of fitted compounds specified in config.")
            sys.exit()

        # exit of this iteration is the last one
        if gen >= settings['num_of_generations']:
            print(f"Optimizer reached the maximum number of iterations: {settings['num_of_generations']}.")
            sys.exit()

        # find fragments

        settings['fragments_ids_file'] = os.path.join(generation_dir, 'fragments_ids.txt')
        error_fname_frag = os.path.join(generation_dir, 'fragments_log.log')
        optimizer_utils.find_frags_rdkit(
            settings['processed_predictions_file'],
            settings['fragments_ids_file'],
            settings['smarts_string'],
            settings['max_cuts'],
            # settings['radius'], # todo is it safe to not to use it at all?
            error_fname_frag
        )

        # calculated descriptors/fingerprints to compute fragment contributions

        if settings['descriptors_type'] == 'sirms':
            # calculate sirms descriptors of fragments
            optimizer_utils.calculate_sirms_descriptors(
                settings['processed_predictions_file'],
                settings['n_cores'],
                fragments_ids=settings['fragments_ids_file']
            )

        elif settings['descriptors_type'] != 'MPNN_fingerprint':
            optimizer_utils.calculate_fingerprints(
                settings['processed_predictions_file'],
                settings['descriptors_type'],
                fragments_ids=settings['fragments_ids_file']
            )

        # calculate fragments contributions

        # sirms and other fingerprints
        if  settings['descriptors_type'] != 'MPNN_fingerprint': # calculate contribs using SINGLE new_x.txt for each param
            new_fragments_fname = os.path.join(generation_dir, 'new_x.txt')
            optimizer_utils.calc_frag_contrib(
                new_fragments_fname,
                [parameter['name'] for parameter in parameters_list_dicts],
                [parameter['types_of_alg'] for parameter in parameters_list_dicts],
                [parameter['path'] for parameter in parameters_list_dicts],
                [parameter['type_of_model'] for parameter in parameters_list_dicts]
            )

        elif settings['descriptors_type'] == 'MPNN_fingerprint':
            if "multitask" not in settings or not settings["multitask"]: # default multitask is False
                for i, dict in enumerate(parameters_list_dicts):
                    # set path with mpnn model
                    mpnn_path = parameters_list_dicts[i]['path']
                    param_name = str(parameters_list_dicts[i]['name'])

                    # calc contrib  for different parameters

                    chemprop_frag_contrib.main_params(
                        x_fname =settings['seed_structure'],   # TODO: PP, settings['processed_predictions_file'] instead of seed_structures
                        out_fname=os.path.join(generation_dir,
                                               'contrib_{}.txt'.format(param_name)),
                        model_dir=mpnn_path,
                        model_type=parameters_list_dicts[i]['type_of_model'],
                        frag_fname=settings['fragments_ids_file'],
                        per_atom_fragments=False,
                        id_field_name=None,
                        multitask=False,
                        variance_threshold=settings['variance_threshold'],
                        save_pred=True,
                        num_frag_id=True
                    )

            else: # multitask
                mpnn_path = parameters_list_dicts[0]['path']
                param_name = str(parameters_list_dicts[0]['name'])

                # calc contrib using name of 1st parameter;  for all paramas  (because model predicts all properties at once)
                chemprop_frag_contrib.main_params(
                    x_fname=settings['seed_structure'],   # TODO: PP, settings['processed_predictions_file'] instead of seed_structures
                    out_fname=os.path.join(generation_dir,
                                           'contrib_{}.txt'.format(param_name)),
                    model_dir=mpnn_path,
                    model_type=parameters_list_dicts[0]['type_of_model'],
                    frag_fname=settings['fragments_ids_file'],
                    per_atom_fragments=False,
                    id_field_name=None,
                    multitask=True,
                    variance_threshold=settings['variance_threshold'],
                    save_pred=True,
                    num_frag_id=True
                )

        # find worst fragments
        settings['fragments_contrib_files'] = [os.path.join(generation_dir,
                                                            'contrib_{}.txt'.format(parameter['name']))
                                                  for parameter in parameters_list_dicts]
        settings['worst_fragments_file'] = os.path.join(generation_dir, 'worst_fragments.txt')
        types_of_alg_contrib = ['_'.join(parameter['types_of_alg']) for parameter in parameters_list_dicts]

        process_contributions.main(
            settings['processed_predictions_file'],
            settings['fragments_contrib_files'],
            os.path.join(generation_dir, 'fragment_contrib_norm.txt'),
            settings['worst_fragments_file'],
            [parameter['name'] for parameter in parameters_list_dicts],
            [parameter['range'] for parameter in parameters_list_dicts],
            types_of_alg_contrib,
            [parameter['threshold'] for parameter in parameters_list_dicts],
            settings['number_of_worst_fragments'],
            settings['bounding_box'],
            settings['random_fragments_selection'],
            brute_force
        )

        # replace fragments
        new_compouds = os.path.join(generation_dir, '{}_gen_compounds.sdf'.format(gen))
        if 'protected_ids' not in settings:
            settings['protected_ids'] = None
        if 'min_inc' not in settings:
            settings['min_inc'] = -2
        if 'max_inc' not in settings:
            settings['max_inc'] = 2
        frag_replacement.main(
            settings['processed_predictions_file'],
            settings['worst_fragments_file'],
            settings['fragments_ids_file'],
            settings['replacement_database'],
            settings['radius'],
            settings['min_inc'],
            settings['max_inc'],
            settings['max_frag_size'],
            new_compouds,
            settings['n_cores'],
            settings['protected_ids']
        )

        settings['seed_structure'] = new_compouds

        num_of_compounds = optimizer_utils.add_mols_into_db(
            settings['seed_structure'],
            settings['output_database'],
            gen + 1
        )

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
    parser.add_argument('-n', '--n_params', action='store', type=int,
                        help='specifies number of parameters to optimize, use when running in the  config creation mode')
    parser.add_argument('-d', '--create_config_structure', action='store_true', default=False,
                        help='define config structure and save it to output location')
    parser.add_argument('-o', '--output_location', default=os.getcwd(),
                        help='output location. Specify when running in the  config creation mode. default: current dir')
    args = vars(parser.parse_args())

    if args['create_config_structure']:
        assert args['n_params'], "You have to specify number of parameters with -n or --n_params option"
        process_config.create_config(args['output_location'], args['n_params'])
    else:
        settings = process_config.test_config(args['input_config'])
        optimize(settings, args['input_config'], args['brute_force'])


if __name__ == '__main__':
    main()
