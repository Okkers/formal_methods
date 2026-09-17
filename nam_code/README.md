# Provable Explanations for Neural Additive Models


## Folder Structure

	- `abcrown_dir/`: Directory of NN verifier alpha-beta-CROWN
	- `analyze/`: Directory to analyze scripts
	- `data/`: Directory for datasets
	- `dnnv_tools/`: Directory for NN verifier utils
	- `helper/`: Directory for auxiliary functions
	- `models/`: Directory for saved models
	- `results/`: Results will be saved here
	- `global_minimal_explanation_binary_abductive_parallel.py`: **main function** (see params)
	- `local_minima_explanation_sensitive_ordering.py`: main file for local subset-minimal approach (see params)
	- `requirements.txt`: additional dependencies to alpha-beta-CROWN


## Installation

Please create a conda environment according to the README.md file of alpha-beta-CROWN (located at `./abcrown_dir`),
and install all dependencies for alpha-beta-CROWN and auto-LiRPA.
Then, additionally install the dependencies specified by `requirements.txt` in this root directory.

Tested Python version: 3.11.11

## Run

After installing all dependencies, the explanation generation can be started using 

	python global_minimal_explanation_binary_abductive_parallel.py --dataset=credit --sample_index=0 --epsilon=0.5 --timeout=60 --num_processors=1
