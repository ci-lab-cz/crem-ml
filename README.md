CReM-ML: System for multi-objective optimization of small molecules properties
==========================================

Overview
----------
Name *CReM-ML* is derived from  *CReM: Chemically reasonlble mutations* and *ML: machine learning*. The method is an iterative 
compound optimizer based on ML models (QSAR) and employing multiparameter optimization: MPO. We employ Pareto-frontier or desirability-functions for compound selection.
The concept of  *XAI (explainable Artificial Intelligence)* is employed. Model explanations (a.k.a. model interpretation) allow to determine the influence of molecular 
fragments on their properties. Based on these explanations molecules are modified by replacing certain fragments using pre-generated database. This (ideally) leads to molecules with improved properties.


The workflow consists of three parts, which  are repeated in a loop:
+ Preparation part
+ Compound part
+ Fragment part

Once the user has defined optimization goal - i.e. target values (ranges) of properties to be reached and desired number of compounds, the optimizer can be started.
Preparation part performs testing of the input data, configuration file, and  setting up the environment.
Compound part works with compounds: it optionally standardizes
compounds, checks whether some compounds match the goal criteria; then possibly the 
optimization loop stops (if there is enough compounds satisfying specified values of properties). Out of the rest of compounds, the most promising are selected for replacement stage using one of MPO approaches.
Fragment part modifies compounds by  working with their fragments.
It calculates fragments' contributions to target properties (called "parameters" in the configuration file) by means of XAI.
Then, the program tests these fragments' contributions
against desired profile and decides whether to replace each given fragment or leave it as it is.
The result of this replacement stage is a pool of new compounds: "generation". 
Those compounds  proceed again to the loop: *preparation part -> compound part -> fragment part*.
Program stops when either holds: 
- yielded compound pool with desired properties  is greater or equal to specified number (stored fore every generation in *output_match.sdf*);
- Maximum specified number of generations is reached;
- There are no new unique compounds after most recent  generation.

Note on randomness
------------------

Randomness can be added to optimization process. The aim of that is to increase the  novelty of results (by decreasing their
dependence on QSAR models, i.e. structure-activity trends from training  compounds used for modelling;
also, decreasing selection pressure of optimization method used: desirability or pareto frontier). 
To achieve that, random compounds and fragments can be "mixed into" selected ones. (Random compounds here  are the compounds
randomly drawn from those  generated at each given step; random fragments for a given compound  here are the fragments randomly drawn 
from fragments resulting from breaking that compound.) Randomness can be completely switched off for "safer" but less novel results.
Randomness can be  switched on (for fragments, compounds or both), partially or fully. In the  case of full (100%) randomness
for fragments, the optimization becomes  not guided by any QSAR.  In the  case of full (100%) randomness
for compound selection, the optimization becomes not guided by any selection pressure.      In the  case of full (100%) randomness
in both parts the optimization becomes fully stochastic.


Note on QSAR model quality
-------------------------
The method uses  QSAR models built in advance based on some available data on target properties. 
The performance of optimizer directly and critically depends on their quality. It is recommended to use highly accurate models.

Installation
-------------
All the required packages can be installed by creating conda environment from a file provided here: crem_ml_env_py370.yml.

`conda env create -f crem_ml_env_py370.yml`

Alternatively, installation of the following packages can be also  done manually:
+ joblib
+ numpy
+ pandas
+ crem
+ spci
+ rdkit
+ sympy
+ sqlite
+ matplotlib
+ pyyaml
+ scikit-learn (required  in case of using SPCI QSAR models) 
+ Chemprop (https://github.com/chemprop/chemprop)  (required in case of using Chemprop QSAR models) 
+ Optionally - to use Chemprop with GPU -  cuda >= 8.0 ; cuDNN
+ Optionally - Chemaxon command line tools: cxcalc, standardizer   
+ optionally - jupyter notebook, if you want run crem_ml_example.ipynb


Packages  crem and  spci  can be installed using *pip* or downloaded from github:
https://github.com/DrrDom/crem.git 
https://github.com/DrrDom/spci.git
Chemprop can be installed using *pip* or downloaded from github https://github.com/chemprop/chemprop
Rdkit can be installed using conda from rdkit channel:  conda install rdkit -c rdkit
Chemaxon tools (optional) can be installed from https://chemaxon.com/

All other packages can be installed using *conda* or *pip*. 

How To Use
-----------

Main script is called *optimizer.py*.
Create a config file. For quick-start template *config.yml* with default parameters can be used
User needs to enter only task-specific parameters, i.e. working directory, model paths, in case of using desirability for selection -  desirability strings
(instructons how to create a string - see *Config file structure and description of input  parameters.*) etc.
Example configs for  several use-cases are provided as well in 'examples/' folder, together with all the input data to run those cases.

`example_crem_ml.ipynb` contains an example of optimization and analysis of results.

Activate conda environment:

`$ conda activate crem_ml_env_py370`

 and run:

`$ cd ~/path_to_crem-ml/`



`$ python optimizer.py -i ~/path_to_config/config.yaml`

CReM replacement database
-------------------------
To use CReM-ML  you need a database with interchangeable fragments, several version can be found at
http://www.qsar4u.com/pages/crem.php. (Currently recommended is replacements02_sa2.db.gz.)


Config.yml file structure and description of input  parameters. 
----------------------------------------------------------

+ working_dir - path to directory where all outputs are going to be stored
+ num_output_compounds - number of compounds with desired
properties to be generated. The program will stop once this is satisfied (unless it will stop earlier upon satisfying another condition).
+ setup_file - (optional) path to file which contains rules for calculation of atomic
properties, required  only if SIRMS descriptors are used and  *properties_sirms != elm*
+ std_rules - (optional) path to file which contains rules for standardization of compounds. 
  If not provided - no standardization will happen, only adding Hydrogens. 
  Note: Do  not  use std_rules with 'protected_ids' (this will lead to wrong mapping of protected_ids to atoms). 
+ chemaxon - (optional) path to bin chemaxon’s bin directory. Required only if SIRMS 
  descriptors are used and  *properties_sirms != elm*
+ seed_structure - path to sdf  file with starting structures to be optimized
+ number_of_selected_compounds - number of compounds selected for optimization in one generation. 
  Larger  number will lead to: (1) more compounds in the final database of generated compounds; (2) longer overall running time;
  (3) lower  selection pressure at each generation.
  Smaller number will lead to opposite: (1) less compounds will be generated; (2) faster results;
  (3) selection pressure   at each generation  will be higher.
+ random_compounds_selection - Portion of compounds randomly selected for optimization  in each generation.  
   Number of these random compounds  will be equal  to
  *number_of_selected_compounds * random_compounds_selection* (Effectively, lowest-desirability  compounds or "less Pareto-optimal" compounds will be 
  replaced by random ones, retaining   *number_of_selected_compounds* as specified.  For precise definition of the procedure,  see *optimization_method*.)
+ descriptors_type - 'sirms' or one of: MG2, bMG2 (Morgan radius 2), AP, bAP (atom-pair), RDK, bRDK (2-4 atoms RDK fingerprint)
TT (topological torsion); or MPNN_fingerprint. Prefix b- means binary fingerprint of length 2048.
  NOTES: All models  - except MPNN - should have been built (using  descriptors specified here) on molecules with explicit hydrogens 
  (because at descriptor calculation 
  and CREM replacement stages explicit hydrogens are  "on" by default). For MPNN fingerprints it doesn't matter if the models
  were built with or without explicit hydrogens, they are ignored during fingerprint calculation).
+ multitask -  (optional) (True/False). Default:False. intended to be used with MPNN  models. Otherwise - has no effect. 
  
  NOTE: if models are single task and multitask = True, or model is multitask and multitask=False - error will be raised.
+ bounded_box - (True/False) use or not  only compounds which are inside bounding box applicability
  domain. Currently, has no effect for MPNN models. TODO: add MPNN too
+ properties_chemaxon - (optional) list of atomic properties using with Chemaxon
  tools.  Use only if SIRMS are used as *descriptors_type*.
+ properties_sirms - (optional) list of atomic properties using with SIRMS module. Use only if SIRMS are used as *descriptors_type*.
+ smart_string - SMARTS pattern  which defines how
  to fragment compounds.  (bonds matched by SMARTS will be broken - for more details, see RDKit.Chem.rdMMPA docs).
  Recommended to use "[!#1]!@!=!#[!#1]" (breaks any single bond between 2 non-hydrogen atoms)  or "[!#1]!@!=!#[*]" 
  (any single bond, except H-H)
  
  NOTE:  if you decide to modify recommended string - keep in mind, that breaking bonds with hydrogen atom  
  while using fingerprints that ignore hydrogen (AtomPairs) will highly likely 
 lead to these hydrogens being selected as worst fragments, because of 0 contributions.
+ protected_ids - field name in sdf file (with structure(s) to be optimized), which contains 0-based ids of atoms,
  that should not be affected by optimization. For instance, this could be an active scaffold which you  wish to retain.
+ max_cuts - number of maximum cuts used in fragmentation procedure (see RDKit.Chem.rdMMPA docs). Recommended is 1...3 cuts.
+ radius - how distant a context should be considered while making replacements using CReM module. REcommended: 1..3.
+ min_inc - (optional) minimal increase in size of fragment while making replacements using CReM module. If not specified,  
 the default is -2 heavy atoms. This means that new fragment can have 2 atoms less than the one replaced. 
+ max_inc - (optional)  maximum increase in size of fragment while making replacements using CReM module. The default is +2 heavy atoms.
  This means that new fragment can have 2 atoms more than the one replaced. 
+ keep_stereo -  (True/False) use information about stereochemistry
  of compound. For 2D models, e.g. models built on fingerprints use False.
+ replacement_database - path to database with interchangeable fragments.
  NOTE: If invalid database is supplied, no 
  compounds will be generated, and next steps can result in errors.
+ number_of_worst_fragments - number of fragments with the
  worst contribution in any given compound  at each generation to consider as  candidates for replacements. Fragments will be
  selected with the following procedure: (1) for each property: normalize contribution using sigmoid-like function.
  (2) Average all normalized values.  (3) Rank fragments and choose the lowest-valued ones. The higher this number - the less effect 
  *calculated contributions* will have, that is, larger portion of all generated fragments for a given compound will be replaced.
  NOTE,  if 1>=random_fragments_selection >0  *(See random_fragments_selection.)*, then 
  certain portion of fragments will be selected randomly
+ random_fragments_selection - ratio of randomly selected fragments
for one compound in one generation, 1 - completely random selection, 0 -
all fragments are selected based on normalized contributions.
+ max_frag_size - maximum size of both: fragment to be replaced and new fragment (heavy atom count) (this arg is used only by crem) 
+ output_format - defines output format, svm or txt for the file with descriptors (svm format is more compact). 
  Has no effect when using MPNN models.
+ num_of_generation - maximum number of generations. Program will stop, one this is satisfied (unless stopped earlier
+ upon another condition).
+ n_cores - number of cores used for calculation of descriptors and  in CReM replacement.
+ optimization_method -  optimization method to use, i.e. desirability,
or pareto (not both). Desirability ranks compounds according to desirability function computed from values of properties to be optimised.
 then it selects the top-ranked ones for the next (crem-replacement) stage. Pareto selects compounds lying on Pareto frontier
 in space of properties to be optimised. If there's not enough compounds (less than specified in *number_of_selected_compounds*)
 then next order frontier is employed, and next, continuing as long as needed. If *random_compound_selection* >0, *number_of_selected_compounds*
 is reduced accordingly. Random compounds are then added, so that final number is: (1) in case of desirability: strictly as specified, 
 if enough compounds are available; (2) in case of Pareto: at least as many as specified,
  (but can be more, since all compounds selected by "next order" frontier are retained), if enough compounds are available.
  NOTE: for either optimization method, if *not* enough compounds are available from previous generation -  
  *all* available compounds are selected.
+ store_all_files - (True/False) specifies if you want to store or delete
intermediate files used for calculations in generations
+ Parameter(s)

– name - name of optimized parameter (biological or physico-chemical property)

– path - path to folder with models pkl files (sklearn models) or checkpoints (Chemprop model)

– types_of_alg - list of names of ML methods to use for predictions: rf, svm, gbm, MPNN.

   NOTE: MPNN can be used only alone (in one run of optimization).
   Other methods can be used in combination, in which case consensus predictions will be used 

– type_of_model - (reg/class) regression or classification (set according to your pre-created models)

– threshold - desired (target) value of parameter in following format: "less5",
  "more8", "between-5to3". This means desired range of values, so, "more8" means you wish to obtain compounds
  with activity higher than 8.

– range - range of possible values of parameter (if you use SPCI models, it can be calculated from  activity
  file 'activity.txt' contained  in model folder. It is typically several orders of magnitude in log scale, e.g. 8...10 for regression 
  and always between 0...1 for classification.

– desirability - desirability string. Has effect iff desirability optimization method is chosen. 
  Instruction to define a string:
  The syntax is to use "," to separate continous pieces of desirability function and ':' to separate its domain from
  its value. Domain is activity values of molecules, and values (desirabilities) are from 0 to 1. 

  Let’s consider an  example string '0.5:0,0.8:(1/0.3)*x-(5/3),100:1'. This is classification case, 
  so domain is [0...1] Corresponding function 
  consists of 3 pieces, separated by ",". The first one is defined on interval (-inf, .5) and its value is 0. 
  The second one is defined on interval (0.5, 0.8) and its value is (1/0.3)*x-(5/3). The last one is defined on (.8,100)
  with value equal 1 , meaning "starting from activity 0.5 and up to 0.8  desirability of molecules will linearly
  increase, after which will reach 1 and stabilize." We need to define only right interval for each continuos piece. Values outside the intervals default 
  to 0. 
  The string is related directly to parameter threshold: for instance, if threshold is "more8", then desirability 
  can be defined for instance as "8:0,9:(x-8),100:1", meaning "starting from activity 8 and up to 9  desirability of molecules will linearly
  increase, after which will reach 1 and stabilize."

Building models before running CReM-ML
--------------------------------------

CReM-ML requires ready QSAR models  for properties to be optimized. Compatible models are scikit-learn models and
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
