CReM-ML: System for multiobjective optimization of small molecules properties
==========================================

Overview
----------
THe workflow consists of three parts:
+ Preparation part
+ Compound part
+ Fragment part

Prepartion part focuses on testing input data, configuration file, setting environment
and so on. Compound part works with entire compounds, it standardizes
compounds, selects the most promising ones, decides whether to finish
computation or not and so on. Fragment part works with fragments of compounds.
It tests their contribution against desired profile and decides whether
to replace/modify fragment or leave it as it is. 


Installation
-------------
This code requires the installation of the following packages:
+ joblib
+ numpy
+ pandas
+ scikit-learn
+ crem
+ spci
+ rdkit
+ sirms
+ sympy
+ sqlite
+ matplotlib
+ pyyaml
+ Optionally - Chemaxon command line tools: cxcalc, standardizer   

Packages  crem and  spci  can be installed using *pip* or downloaded from github:
https://github.com/DrrDom/sirms.git. 
https://github.com/DrrDom/spci.git

Chemaxon tools can be installed from https://chemaxon.com/

All other packages can be installed using *conda* or *pip*. 

Environment
-----------

`conda env create -f env.yml`

How To Use
-----------

Main script is called *optimizer.py*.
Create a config file, activate conda environment, and run

`$ cd ~/path_to_crem-ml/`

`$ python optimizer.py -i ~/path_to_config/config.yaml`

Config file structure
---------------------

+ working_dir - path to directory where all outputs are going to be stored
+ num_of_output_compounds - number of compounds with desired
properties
+ setup_file - path to file which contains rules for calculation of atomic
properties 
+ std_rules - path to file which contains rules for standardization of compounds
+ chemaxon - path to bin chemaxon’s bin directory
+ seed_structure - path to file with structures to be optimizied
+ number_of_selected_compounds - number of compounds used for
optimization in one generation
+ random_compounds_selection - ratio of randomly selected compounds
in one generation, 1 - completely random selection, 0 - all compounds are
selected using optimization technique
+ bounded_box - (True/False) use only compounds which are within applicability
domain 
+ properties_chemaxon - list of atomic properties using with Chemaxon
tools
+ properties_sirms - list of atomic properties using with SIRMS module
+ properties_calc_contrib - specifies which type of contribution is used for
calculation of fragments contribution, you can specify "overall" which uses all atomic labels (see SPCI docs).
+ smart_string - SMARTS pattern  for SIRMS module which defines how
to fragment compounds (bonds matched by SMARTS will be broken - for more details, see RDKit.Chem.rdMMPA docs)
+ max_cuts - number of maximum cuts used in fragmentation procedure (see RDKit.Chem.rdMMPA docs)
+ radius - how distant a context should be considered while making replacements using CReM module. 
+ keep_stereo - ***** (True/False) use information about stereochemistry
of compound
+ replacement_database - path to database with interchangeable fragments. Note: If invalid database is supplied, no 
compounds will be generated, and next steps can result in errors.
+ number_of_worst_fragments - number of fragments which have the
worst contribution per one compound  in one generation, candidates for replacements. Fragments will be selected using 
optimization technique and\or randomly (see random_fragments_selection) 
+ random_fragments_selection - ratio of randomly selected fragments
for one compound in one generation, 1 - completely random selection, 0 -
all fragments are selected using optimization technique
+ max_frag_size - maximum size of fragment  (hac) 
+ output_format - defines output format, svm or txt for the file with descriptors.
+ num_of_generation - maximum number of generations
+ n_cores - number of cores used for calculation of descriptors and  in CReM replacement.
+ optimization_method - list of selected optimization methods, e.g. desirability,
pareto
+ descriptors_type - 'sirms' or one of: MG2, bMG2 (Morgan radius 2),AP, bAP (atom-pair), RDK, bRDK (2-4 atoms RDK fingerprint),
TT (topological torsion). Prefix b means binary fingerprint of length 2048. Models should be built using same descriptors, 
+ on molecules with explicit hydrogens.
+ store_all_files - (True/False) specifies if you want to store or delete
intermediate files used for calculations within generations
+ Parameter(s)
– name - name of optimized parameter
– path - path to folder with models pkl files (sklearn models)
– types_of_alg - list of models to choose for prediction of both: whole compounds and fragment contributions
(from available pkl files), e.g. rf, svm, ...
– type_of_model - (reg/class) regression or classification (set according to your pickled models)
– threshold - desired value of parameter in special format, e.g. "less5",
"more8", "between-5to3"
– range - range of possible values of parameter (can be found in activity file 'activity.txt' in model folder)
– desirability - desirability string. Instruction to define a string:
The syntax is to use "," to separate continous pieces of desirability function and ':' to separate its domain from its value. 
Let’s consider an example string '0.5:0,0.8:(1/0.3)*x-(5/3),100:1'. Corresponding function 
consists of 3 pieces, separated by ",". The first one is defined on interval (-inf, .5) and its value is 0. 
The second one is defined on interval (0.5, 0.8) and its value is (1/0.3)*x-(5/3). The last one is defined on (.8,100)
with value equal 1. We need to define only right interval for each continuos piece. Values outside the intervals default to 0.


Citation
--------
