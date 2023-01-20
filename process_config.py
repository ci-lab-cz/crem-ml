#!/usr/bin/env python

import sys
import os
from os.path import exists, isfile
import yaml

from typing import Dict
from typing import TextIO
from typing import List


CONFIG_STRUCTURE = [['working_dir', 'path_to_output_dir'],
                    ['num_output_compounds', 'number of fitted compounds'],
                    ['setup_file', 'path_to_setup_file'],
                    ['std_rules', 'path_to_file_with_std_rules'],
                    ['chemaxon', 'path_to_chemaxon_bin_folder'],
                    ['seed_structure', 'path_to_seed_structure'],
                    ['number_of_selected_compounds', 'fill only if desirability is specified'],
                    ['random_compounds_selection', '0'],  # from 0 - 1, 0.2 means 20% of selected compounds are chosen randomly (floored)
                    ['descriptors_type','type of descriptors to use'],
                    ['bounded_box', 'True or False'],     # True
                    ['properties_chemaxon', 'fill'],      # 'charge logp acc don refractivity'
                    ['properties_sirms', 'fill'],         # 'CHARGE LOGP HB REFRACTIVITY'
                    ['properties_calc_contrib', 'fill'],  # 'overall'
                    ['smart_string', "'[#6+0;!$(*=,#[!#6])]!@!=!#[*]'"],
                    ['max_cuts', 'fill'],
                    ['radius', 'fill'],
                    ['keep_stereo', 'True or False'],
                    ['replacement_database', 'path_to_database_with_replacement'],
                    ['number_of_worst_fragments', 'fill'],
                    ['random_fragments_selection', '0'],  # from 0 - 1, 0.2 means 20% of selected fragments are chosen randomly (floored)
                    ['max_frag_size', 'fill'],
                    ['output_format', 'svm'],
                    ['num_of_generation', 'fill'],
                    ['n_cores', 'fill'],
                    ['optimization_methods', 'fill'],     # 'pareto desirability'
                    ['store_all_files', 'True or False']          # If false, it deletes all temp files, only db will be stored
                    ]

PARAMETER_STRUCTURE = [['name', 'name_of_parameter'],
                       ['path', 'path_to_parameter_model_folder'],
                       ['types_of_alg', 'space_separated_list_of_all_algs_used_for_predictions'],
                       ['type_of_model', 'class or reg'],
                       ['threshold', 'fill'],
                       ['range', 'fill'],
                       ['desirability', 'fill']
                       ]

def create_config(output_dir: str, num_of_parameters: int, file_name: str='config.yaml') -> None:
    """
    Create empty file for input settings for optimizer

    :param output_dir: path to output dir
    :param num_of_parametrs: number of parameters to optimize
    :param file_name: name of config file
    """

    assert exists(output_dir), "Output directory doesn't exist"

    # set indexes
    working_dir_index = 0
    num_parameters_index = 1

    # working dir
    CONFIG_STRUCTURE[working_dir_index][1] = os.path.abspath(output_dir)
    # num param
    CONFIG_STRUCTURE[num_parameters_index][1] = num_of_parameters

    conf_file = os.path.join(output_dir, file_name)
    with open(conf_file, 'w') as config:
        for item in CONFIG_STRUCTURE:
            config.write("{}: {}\n".format(item[0], item[1]))
        for i in range(num_of_parameters):
            config.write("param_{}: \n".format(str(i)))
            for item in PARAMETER_STRUCTURE:
                config.write("  {}: {}\n".format(item[0], item[1]))
    print("Structure of configuration file was created in {} directory. "
          "Fell free to rewrite default values.".format(CONFIG_STRUCTURE[working_dir_index][1]))

def test_config(input_config: str) -> Dict:
    """
    Test and process all settings in config file

    :param input_config: path to config file
    :return: dictionary with checked and processed settings
    """

    # check if config file exists
    assert isfile(input_config), "{} is not a file".format(input_config)

    with open(input_config, 'r') as stream:
        try:
            config = yaml.load(stream, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)

    # check non parameters settings
    for key, value in config.items():
        # check directories
        if (key == 'working_dir') or (key == 'chemaxon'):
            assert exists(config[key]), "{} doesn't exists".format(key)
            config[key] = value
        # check files
        elif (key == 'setup_file') \
            or (key == 'seed_structure') \
            or (key == 'std_rules') \
            or (key == 'replacement_database'):
                assert isfile(config[key]), "{} doesn't exists".format(key)
                config[key] = value
        # check param
        elif "param_" in key:
            print(key,value)
            # check if models dir exists
            assert exists(config[key]['path']), "{} doesn't exists".format(config[key]['path'])
            # check if types of models exists, e.g. model + type + ".pkl"
            config[key]['types_of_alg'] = config[key]['types_of_alg'].split(" ")
            for alg in config[key]['types_of_alg']:
                assert isfile(os.path.join(config[key]['path'], alg + ".pkl")), \
                        "{} doesn't exists".format(os.path.join(config[key]['path'], alg + ".pkl"))
        # check settings with numbers
        elif (key == 'n_cores') or (key == 'max_cuts') \
            or (key == 'radius') or (key == 'number_of_worst_fragments') \
            or (key == 'max_frag_size') or (key == 'num_of_generation') \
            or (key == 'num_output_compounds'):
                try:
                    num = int(value)
                except:
                    num = value
                assert isinstance(num, int), "{} is not defined properly".format(key)
                config[key] = num
        # check settings with float numbers
        elif (key == 'random_compounds_selection') or (key == 'random_fragments_selection'):
            try:
                num = float(value)
            except:
                num = value
            assert isinstance(num, float), "{} is not defined properly".format(key)
            config[key] = num
        # strip properties for chemaxon and sirms
        elif (key == 'properties_chemaxon') or (key == 'properties_sirms') \
            or (key == 'optimization_methods') or (key == 'properties_calc_contrib'):
            config[key] = value.split(" ")
    if 'sirms' not in config['descriptors_type']:
        print( "Note, 'properties_chemaxon' and 'properties_sirms' have no effect for descriptors specified, "
               "they are meaningful only for sirms")
    acceptable_descr = [ 'sirms','MG2','AP','RDK','TT','bMG2','bAP','bRDK']
    assert sum([config['descriptors_type'] == i for i in acceptable_descr]) == 1
    # if user specify desirabilty then user must specify number of selected of compounds
    if 'desirability' in config['optimization_methods']:
        try:
            num = int(config['number_of_selected_compounds'])
        except:
            num = config['number_of_selected_compounds']
        assert isinstance(num, int), "number_of_selected_compounds are not defined properly"
        config['number_of_selected_compounds'] = num


    return config
