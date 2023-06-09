CReM-ML: System for multiobjective optimization of small molecules' properties
==========================================

Overview
----------
Name *CReM-ML* is composed of CReM: Chemically reasonlble mutations and ML: machine learning. THe method is an iterative 
compound optimizer based on ML models (QSAR) and Pareto-frontier optimization/desirability-function-based optimization.

THe workflow consists of three parts:
+ Preparation part
+ Compound part
+ Fragment part

Prepartion part focuses on testing input data, configuration file, setting environment
and so on. Compound part works with entire compounds, it optionally standardizes
compounds, selects the most promising ones, decides whether to finish
computation (if compounds satisfy specified values of properties and their number is sufficient) or not and so on.
Fragment part works with fragments of compounds.
It calculates their contributions to optimized parameteres (properties). Then, it tests these fragments' contributions
against desired profile and decides whether to replace each given fragment or leave it as it is. The result of this stage is 
pool of new compounds ("generation"). They proceed to preparation part -> compound part -> fragment part (loop closes).
Program stops when either: 
- yielded compound pool is >= to specified number;
- Maximum specified number of generations is reached.

Note on randomness
------------------

Randomness can be added to optimization process. The aim of that is to increase the  novelty of results (in other words, 
decrease the dependence on QSAR models, i.e. structure-activity trends from training  compounds used for modelling). 
To achieve that, random compounds and fragments can be "mixed into" selected ones. (Random compounds are compounds
randomly drawn from those  generated on current step; random fragments for a given compound are fragments randomly drawn 
from fragments resulting from breaking that compound.) Randomness can be  switched off for "safer" but less novel results.

Note on QSAR model qualiy
-------------------------
THe metod uses  QSAR models built in advance. The prformance critically depends on their quality

Installation
-------------
This code requires the installation of the following packages:
+ joblib
+ numpy
+ pandas
+ crem
+ spci
+ rdkit
+ sirms
+ sympy
+ sqlite
+ matplotlib
+ pyyaml
+ scikit-learn (required  in case of using SPCI QSAR models) 
+ Chemprop (https://github.com/chemprop/chemprop)  (required in case of using Chemprop QSAR models) 
+ Optionally - to use Chemprop with GPU -  cuda >= 8.0 ; cuDNN
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

Config file structure and description of input  parameters
----------------------------------------------------------

+ working_dir - path to directory where all outputs are going to be stored
+ num_of_output_compounds - number of compounds with desired
properties. The program will stop once this is satisfied (unless it will stop earlier upon another condition).
+ setup_file - (optional) path to file which contains rules for calculation of atomic
properties, needed only if SIRMS descriptors are used and  *properties_sirms != elm*
+ std_rules - (optional) path to file which contains rules for standardization of compounds.
+ If not provided - no standaardization will happen, only adding Hs at each generation. Note: Never use std_rules with 'protected_ids'. 
+ chemaxon - (optional) path to bin chemaxon’s bin directory
+ seed_structure - path to file with structures to be optimizied
+ number_of_selected_compounds - number of compounds selected for optimization in one generation. 
+ This has effect only when desirability optimization method is used. 
+ If pareto is used,  all  compounds from pareto frontier will be selected regardless of their number. 
+ Note, a  certain portion  of compounds will be selected randomly  if *1>=random_compounds_selection >0.*
+ *(See random_compounds_selection.)*
+ random_compounds_selection - Portion of randomly selected compounds in one generation.  
+ In case of using desirability the number of these random compounds  will be 
+ *number_of_selected_compounds * random_compounds_selection* (effectively, lowest-desirability compounds will be 
+ replaced by random ones to reach desired number). In
+ case of pareto this will be: *number_of_selected_compounds - number of pareto-selected* (effectively, some compounds will be randomly added 
+ to reach desired number, if necessary)
+ bounded_box - (True/False) use only compounds which are within applicability
domain 
+ properties_chemaxon - (optional) list of atomic properties using with Chemaxon
tools
+ properties_sirms - (optional) list of atomic properties using with SIRMS module
+ properties_calc_contrib - specifies which type of contribution is used for
calculation of fragments contribution, currently, you can specify "overall" which uses all atomic labels (see SPCI docs).
+ smart_string - SMARTS pattern  for SIRMS module which defines how
to fragment compounds.  (bonds matched by SMARTS will be broken - for more details, see RDKit.Chem.rdMMPA docs)
+ NOTE: Recommended to use default, if you decide to modify it - keep in mind, that breaking bonds with hydrogen atom  
+ while using fingerprints that ignore hydrogen (AtomPairs) will highly likely 
+ lead to these hydrogens being selected as worst fragments, because of 0 contributions.
+ max_cuts - number of maximum cuts used in fragmentation procedure (see RDKit.Chem.rdMMPA docs)
+ radius - how distant a context should be considered while making replacements using CReM module. 
+ keep_stereo - ***** (True/False) use information about stereochemistry
of compound
+ replacement_database - path to database with interchangeable fragments. Note: If invalid database is supplied, no 
compounds will be generated, and next steps can result in errors.
+ number_of_worst_fragments - number of fragments which have the
worst contribution per one compound  in one generation, they're  candidates for replacements. Fragments will be
+ selected with the following procedure: 1.for each property: normalize contribution usiing sigmoid-like function 
2. Average all normalized values. Note, certain portion of fragments will be selecteds randomly, 
3. if 1>=random_fragments_selection >0  *(See random_fragments_selection.)*
+ random_fragments_selection - ratio of randomly selected fragments
for one compound in one generation, 1 - completely random selection, 0 -
all fragments are selected based on normalized contributions.
+ max_frag_size - maximum size of both: fragment to be replaced and new fragment (heavy atom count) (this arg is used only by crem) 
+ output_format - defines output format, svm or txt for the file with descriptors.
+ num_of_generation - maximum number of generations. Program will stop, one this is satisfied (unless stopped earlier
+ upon another condition).
+ n_cores - number of cores used for calculation of descriptors and  in CReM replacement.
+ optimization_method -  optimization method to use, i.e. desirability,
or pareto (not both)
+ descriptors_type - 'sirms' or one of: MG2, bMG2 (Morgan radius 2), AP, bAP (atom-pair), RDK, bRDK (2-4 atoms RDK fingerprint)
TT (topological torsion); or MPNN_fingerprint. Prefix b means binary fingerprint of length 2048. Models  - except MPNN - 
+ should be built using same descriptors,  on molecules with explicit hydrogens (because in descriptor calculation 
+ and crem replacement explicit hydrogens are set to be on by default).
+ store_all_files - (True/False) specifies if you want to store or delete
intermediate files used for calculations within generations
+ Parameter(s)
– name - name of optimized parameter (biological or physico-chemical property)
– path - path to folder with models pkl files (sklearn models) or checkpoints (Chemprop model)
– types_of_alg - list of models to choose for prediction of  whole compounds and fragment contributions
(from available pkl/pt files), e.g. rf, svm, MPNN...
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

Building models before running CReM-ML
--------------------------------------

CReM-ML requires ready QSAR models  for propertes to be optimized. Compatible models are scikit-learn models and
Chemprop MPNN models (pytorch-based). 
Scikit-learn models can be built using SPCI package using SPCI GUI:

`$ spci`

or via command line as follows:

`$ spci_descriptors -i <path_to_sdf file with molecules>    -o <name of descriptors file to save> -b svm --fp_type <fingerprints from RDKIT>  -w  <field name in sdf file to use as molecule names>`

for more settings and detailed description, call

`$ spci_descriptors --help`

`spci_model -x <name of descriptors file from previous step> -f svm -y  <path to txt file with activities of molecules>` 
`-m <model types  from scikit-learn> -d <path to save models> -t <reg|class>`

for more settings and detailed description, call

`$ spci_model --help`


Chemprop models can be built using Chemprop package as follows:

`chemprop_train --data_path <path> --dataset_type <type> --save_dir <dir>`

More details: https://github.com/chemprop/chemprop

Citation
--------
