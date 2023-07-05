CReM-ML: System for multiobjective optimization of small molecules' properties
==========================================

Overview
----------
Name *CReM-ML* is composed of *CReM: Chemically reasonlble mutations* and *ML: machine learning*. The method is an iterative 
compound optimizer based on ML models (QSAR) and Pareto-frontier optimization/desirability-function-based optimization.
The concept of  *XAI: explainable AI* is employed. Model explanations allow to discover the influence of molecular 
fragments on their properties. Based on this, molecules are modified (optimized).


The workflow consists of three parts:
+ Preparation part
+ Compound part
+ Fragment part

Preparation part focuses on testing input data, configuration file, setting environment
and so on. Compound part works with entire compounds, it optionally standardizes
compounds, selects the most promising ones, decides whether to finish
computation (if there is sufficient number of compounds satisfying specified values of properties) and so on.
Fragment part works with fragments of compounds.
It calculates their contributions to optimized parameters (properties). Then, it tests these fragments' contributions
against desired profile and decides whether to replace each given fragment or leave it as it is.
The result of this stage is a pool of new compounds ("generation"). 
They proceed back to preparation part -> compound part -> fragment part (loop).
Program stops when either: 
- yielded compound pool with desired properties  is >= to specified number;
- Maximum specified number of generations is reached.

Note on randomness
------------------

Randomness can be added to optimization process. The aim of that is to increase the  novelty of results (in other words, 
decrease the dependence on QSAR models, i.e. structure-activity trends from training  compounds used for modelling). 
To achieve that, random compounds and fragments can be "mixed into" selected ones. (Random compounds are compounds
randomly drawn from those  generated on current step; random fragments for a given compound are fragments randomly drawn 
from fragments resulting from breaking that compound.) Randomness can be  switched off for "safer" but less novel results.

Note on QSAR model quality
-------------------------
The method uses  QSAR models built in advance. The performance critically depends on their quality

Installation
-------------
All the required packages can be installed by creating conda environment from a file provided

`conda env create -f env_no_builds.yml`

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

How To Use
-----------

Main script is called *optimizer.py*.
Create a config file. For quick-start template *config.yml* with default parameters can be used
(enter only task-specific parameters, i.e. working directory, model paths and so on). Example configs for some tasks 
are provided as well (exampples folder).
then activate conda environment

`$ conda activate crem_ml_env`

 and run

`$ cd ~/path_to_crem-ml/`



`$ python optimizer.py -i ~/path_to_config/config.yaml`

CReM replacement database
-------------------------
To use CReM-ML (and CReM) you need a database with interchangeable fragments, several version can be found at
http://www.qsar4u.com/pages/crem.php. (Currently recommended is replacements02_sa2.db.gz.)


Config file structure and description of input  parameters
----------------------------------------------------------

+ working_dir - path to directory where all outputs are going to be stored
+ num_of_output_compounds - number of compounds with desired
properties. The program will stop once this is satisfied (unless it will stop earlier upon another condition).
+ setup_file - (optional) path to file which contains rules for calculation of atomic
properties, required  only if SIRMS descriptors are used and  *properties_sirms != elm*
+ std_rules - (optional) path to file which contains rules for standardization of compounds. 
  If not provided - no standardization will happen, only adding Hydrogens. 
  Note: Do  not  use std_rules with 'protected_ids'. 
+ chemaxon - (optional) path to bin chemaxon’s bin directory. Required only if SIRMS 
  descriptors are used and  *properties_sirms != elm*
+ seed_structure - path to sdf  file with structures to be optimized
+ number_of_selected_compounds - number of compounds selected for optimization in one generation. 
  The larger this number, the more compounds will  be in the final pull of generated compounds (longer time).
  Desirability function-based  selection pressure at each generation will be lower.
  Conversely, the smaller - the less compounds will be generated, faster results. 
  Desirability function-based  selection pressure   at each generation  will be higher.

  NOTES:

  This has effect only when desirability optimization method is used. 
  If pareto is used,  all  compounds from pareto frontier will be selected regardless of their number. 
  A  certain portion  of compounds will be selected randomly  if *1>=random_compounds_selection >0.*
  *(See random_compounds_selection.)*
+ random_compounds_selection - Portion of randomly selected compounds in one generation.  
  In the case of using desirability the number of these random compounds  will equal 
  *number_of_selected_compounds * random_compounds_selection* (effectively, lowest-desirability compounds will be 
  replaced by random ones to reach desired number). In
  case of pareto this will be: *number_of_selected_compounds - number of pareto-selected* (effectively, some compounds will be randomly added 
  to reach desired number, if necessary)
+ descriptors_type - 'sirms' or one of: MG2, bMG2 (Morgan radius 2), AP, bAP (atom-pair), RDK, bRDK (2-4 atoms RDK fingerprint)
TT (topological torsion); or MPNN_fingerprint. Prefix b means binary fingerprint of length 2048. Models  - except MPNN - 
  should be built using same descriptors,  on molecules with explicit hydrogens (because at descriptor calculation 
  and crem replacement stages explicit hydrogens are set to be "on" by default). For MPNN fingerprints explicit hydrogens 
  are always removed from molecules (only this leads to correct result, no matter how models were built).
+ multitask -  (True/False) intended to be True with MPNN multitask models. Otherwise - has no effect: use False. 
  
  NOTE: if models are single task and multitask = True, or model is multitask and multitask=False - error will be thrown
+ bounded_box - (True/False) use only compounds which are within applicability
  domain
+ properties_chemaxon - (optional) list of atomic properties using with Chemaxon
  tools
+ properties_sirms - (optional) list of atomic properties using with SIRMS module
+ properties_calc_contrib - specifies which type of contribution is used for
  calculation of fragments contribution. Matters only for scikit-learn models currently, you can specify 
  only "overall" which uses all atomic labels (see SPCI docs).
+ smart_string - SMARTS pattern  which defines how
  to fragment compounds.  (bonds matched by SMARTS will be broken - for more details, see RDKit.Chem.rdMMPA docs)
  
  NOTE: Recommended to use default, if you decide to modify it - keep in mind, that breaking bonds with hydrogen atom  
  while using fingerprints that ignore hydrogen (AtomPairs) will highly likely 
 lead to these hydrogens being selected as worst fragments, because of 0 contributions.
+ protected_ids - field name in sdf file (with structure(s) to be optimized), which contains 0-based ids of atoms,
  that should not be affected by optimization. For instance, this could be an active scaffold which you  wish to preserve.
+ max_cuts - number of maximum cuts used in fragmentation procedure (see RDKit.Chem.rdMMPA docs)
+ radius - how distant a context should be considered while making replacements using CReM module. 
+ min_inc - minimal increase in size of fragment while making replacements using CReM module. If not specified,  
 the default is -2 heavy atoms. This means that new fragment can have 2 atoms less than the one replaced. 
+ max_inc - miaximal increase in size of fragment while making replacements using CReM module. The default is +2 heavy atoms.
  This means that new fragment can have 2 atoms more than the one replaced. 
+ keep_stereo -  (True/False) use information about stereochemistry
  of compound
+ replacement_database - path to database with interchangeable fragments.
  NOTE: If invalid database is supplied, no 
  compounds will be generated, and next steps can result in errors.
+ number_of_worst_fragments - number of fragments which have the
  worst contribution per one compound  in one generation, they're  candidates for replacements. Fragments will be
  selected with the following procedure: 1.for each property: normalize contribution using sigmoid-like function 
  2. Average all normalized values.  3. Rank them and choose the lowest. The higher this number - the less effect 
  *calculaated contributions* will have, that is, the less fragments will be excluded due to unfavorable contributions.
  NOTE,  if 1>=random_fragments_selection >0  *(See random_fragments_selection.)*, then 
  certain portion of fragments will be selected randomly
+ random_fragments_selection - ratio of randomly selected fragments
for one compound in one generation, 1 - completely random selection, 0 -
all fragments are selected based on normalized contributions.
+ max_frag_size - maximum size of both: fragment to be replaced and new fragment (heavy atom count) (this arg is used only by crem) 
+ output_format - defines output format, svm or txt for the file with descriptors (svm format is more compact).
+ num_of_generation - maximum number of generations. Program will stop, one this is satisfied (unless stopped earlier
+ upon another condition).
+ n_cores - number of cores used for calculation of descriptors and  in CReM replacement.
+ optimization_method -  optimization method to use, i.e. desirability,
or pareto (not both)
+ store_all_files - (True/False) specifies if you want to store or delete
intermediate files used for calculations within generations
+ Parameter(s)

– name - name of optimized parameter (biological or physico-chemical property)

– path - path to folder with models pkl files (sklearn models) or checkpoints (Chemprop model)

– types_of_alg - list of names of ML methods to use for predictions: rf, svm, gbm, MPNN.

   NOTE: MPNN can be used only alone (in one run of optimization).
   Other methods can be used in combination, in which case consensus predictions will be used) 

– type_of_model - (reg/class) regression or classification (set according to your pickled models)

– threshold - desired (target) nvalue of parameter in following format: "less5",
  "more8", "between-5to3". This means desired range of values, so, "more8" means you wish to obtain compounds
  with activity higher than 8.

– range - range of possible values of parameter (if you use SPCI models, it can be calculated from  activity
  file 'activity.txt' contained  in model folder. It is several orders of magnitude in log scale, e.g. 8...10 for regression 
  and always between 0...1 for classification.

– desirability - desirability string. Matters iff desirability optimization method is chosen. 
  Instruction to define a string:
  The syntax is to use "," to separate continous pieces of desirability function and ':' to separate its domain from
  its value. Domain is activity values of molecules, and values (desirabilities) are from 0 to 1. 

  Let’s consider an  example string '0.5:0,0.8:(1/0.3)*x-(5/3),100:1'. This si classification case, 
  so domain is [0...1] Corresponding function 
  consists of 3 pieces, separated by ",". The first one is defined on interval (-inf, .5) and its value is 0. 
  The second one is defined on interval (0.5, 0.8) and its value is (1/0.3)*x-(5/3). The last one is defined on (.8,100)
  with value equal 1 , meaning " starting from activity 0.5 and up to 0.8  desirability of molecules will linearly
  increase, after which will reach 1 and stabilize. "We need to define only right interval for each continuos piece. Values outside the intervals default 
  to 0. 
  The string is related directly to parameter threshold: for instance, if threshold is "more8", then desirability 
  can be defined for instance as "8:0,9:(x-8),100:1", meaning " starting from activity 8 and up to 9  desirability of molecules will linearly
  increase, after which will reach 1 and stabilize. "

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
