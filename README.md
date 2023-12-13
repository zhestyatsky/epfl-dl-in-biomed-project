# epfl-dl-in-biomed-project
Project at CS-502: Deep learning in Biomedicine at EPFL. Option 2. \
Implementation of the method called Latent Embedding Optimization (LEO). Original paper can be found [here](https://arxiv.org/abs/1807.05960).

### Files and Their Descriptions:
`methods/leo.py`: Actual file that contains implementation of LEO. \
`conf/method/leo.yaml`: File that contains configurations and fixed hyperparameters for LEO. \
`pretrained_weights/*.tar`: Files that are used to pretrain the backbone for the algorithm. \
Other already existing files are updated such as `run.py` and `utils/io_utils.py` for compatibility with new algorithm and reproducibility of the results. \
Grid Search results for Swissprot can be found in branch `grid-search-swissprot` under folder `checkpoints\exp_name\results.txt` and for Tabula Muris in branch `boris\grid-search-tabula-muris` under folder `output\exp_name_results.txt` 

### How to Run
For the new method (LEO), execute the `run.py` with `method=leo` command line argument, and with the configuration wanted. \
e.g., `python run.py exp.name={exp_name} method=leo dataset=swissprot n_shot=1 method.latent_space_dim=64 method.enable_finetuning_loop=True method.optimize_backbone=True` command will train LEO model with latent space dimension 64, with random initialization for backbone, with inner finetuning loop enabled, with outer optimization of backbone enabled and get the results on Swissprot for 5-way 1-shot case. 




## Authors
- Abdurrahman Said Gürbüz
- Boris Zhestiankin
- Sergi Blanco-Cuaresma
