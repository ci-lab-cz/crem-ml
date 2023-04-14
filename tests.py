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
import optimizer
from itertools import product


def main():
    """
    Integrative test.
    Test all combinations of input settings: descriptors_type, radius,optimization method, random selection for compounds.
    Use command line args to specify them, and they will override
    corresponding config settings. Config settings sholud be defined as usual ( to be partly overwritten).
    """
    parser = argparse.ArgumentParser(description='Integral test of optimizer')
    parser.add_argument('-i', '--input_config',
                        help='path to config with input settings')
    parser.add_argument('-bf', '--brute_force', action='store_true', default=False,
                        help='use all compounds and all fragments, no selections')
    parser.add_argument('-g', '--number_generations', action='store', type=int, default=0,
                        help='specifies number of generations, use with brute force')
    parser.add_argument('-dt', '--descriptors_type', nargs='*', default=['MG2', 'AP',],
                        help='descriptor types')
    parser.add_argument('-r', '--radius', nargs='*', default=[1],
                        help='radius for replacements')
    parser.add_argument('-o', '--opt_met', default=['desirability','pareto'],
                        help='optimization method')
    parser.add_argument('-rs', '--random_sele', default=[0,0.5,1],
                        help='random compounds selection')
    args = vars(parser.parse_args())



    settings = process_config.test_config(args['input_config'])
    # define "global" constants for each comb in upcoming loop
    seed_structure = settings['seed_structure'] # to start from scratch
    wd_pref = settings['working_dir']
    shutil.copyfile(args['input_config'], os.path.join(wd_pref, 'config.yaml'))  # config is same for all tested combs
    # todo replacee args in config with overwritten ones
    # over all tested combinations
    for comb in product(args['descriptors_type'], args['radius'],args['opt_met'], args['random_sele']):
        # re-set back initial!
        settings['seed_structure'] = seed_structure
        # create dir to save all output for current combination
        settings['working_dir'] = os.path.join(wd_pref, "_".join(map(str, comb)))
        os.makedirs(settings['working_dir'])
        # print(settings['working_dir'])
        # overwrite descriptors type
        settings['descriptors_type'] = comb[0]
        # overwrite model dirs  - to be able to read models from  correct subdir when running optimise()
        for parameter in settings:
            if "param_" in parameter:
                settings[parameter]['path']=  os.path.join(settings[parameter]['path'].rsplit("/", 1)[0], settings['descriptors_type'].lower())
        # overwrite radius
        settings['radius'] = comb[1]
        # ooerwrote opt method
        settings['optimization_method'] = [comb[2]]
        # overwrite random compound sele
        settings['random_compounds_selection'] = comb[3]

        try: # try - needed to handle sys.exits to enable loop continue
            optimizer.optimize(settings, args['input_config'], args['brute_force'], args['number_generations'])
            os.remove(os.path.join(wd_pref, "_".join(map(str, comb)), 'config.yaml'))  # config is same for all tested combs

        except SystemExit:
            os.remove(os.path.join(wd_pref, "_".join(map(str, comb)), 'config.yaml'))  # config is same for all tested combs
            continue


if __name__ == '__main__':
    main()
