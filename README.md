# Meta-learning with Latent Embedding Optimization (LEO)
CS-502: Deep learning in Biomedicine at EPFL. Project pption 2.
Implementation of the method called Latent Embedding Optimization (LEO). Original paper can be found [here](https://arxiv.org/abs/1807.05960).

### Files and Their Descriptions:
- `methods/leo.py`: Actual file that contains implementation of LEO. 
- `conf/method/leo.yaml`: File that contains configurations and fixed hyperparameters for LEO.
- `pretrained_weights/*.tar`: Files that are contain pretrained backbone weights for the algorithm.
- Other already existing files are updated such as `run.py` and `utils/io_utils.py` for compatibility with new algorithm and reproducibility of the results.
- Grid Search results for Swissprot can be found in branch `grid-search-swissprot` under folder `checkpoints\exp_name\results.txt` and for Tabula Muris in branch `boris\grid-search-tabula-muris` under folder `output\exp_name_results.txt`.
- In addition to this, detailed outputs for all experiments/training can be found under `output\exp_name.txt` in branches `grid-search-swissprot` and `boris\grid-search-tabula-muris`.

### How to Run
For the new method (LEO), execute  `run.py` with `method=leo` command line argument, and with the configuration wanted.
E.g. running command

`python run.py exp.name={exp_name} method=leo dataset=swissprot n_shot=1 method.latent_space_dim=64 method.enable_finetuning_loop=True method.optimize_backbone=True`

will train LEO model with latent space dimension set to 64, with inner finetuning loop enabled, with outer optimization of backbone enabled and with with random initialization for backbone (default). It will get the results on Swissprot for 5-way 1-shot case. 

## Authors
- Abdurrahman Said Gürbüz
- Boris Zhestiankin
- Sergi Blanco-Cuaresma
