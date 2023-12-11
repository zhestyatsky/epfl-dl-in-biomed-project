import os
from itertools import product


# grid search configurations
hyperparameters = {
    'dataset': ['tabula_muris'],
    'n_shot': [1, 5],
    'method.optimize_backbone': [False, True],
    'method.enable_finetuning_loop': [False, True],
    'method.pretrained_backbone_weights_path': [
        'pretrained_weights/tabula_muris_baseline_model.tar',
        'pretrained_weights/tabula_muris_baseline_pp_model.tar',
    ],
    # Add more hyperparameters as needed
}

# Path to the run.py script (right now it should be in the same directory as this file)
run_script_path = './run.py'

# Output directory
output_dir = 'experiments/'

# Mapping for shorter names
short_name_mapping = {
    ('method.optimize_backbone', False): 'without_optimize_backbone',
    ('method.optimize_backbone', True): 'with_optimize_backbone',
    ('method.enable_finetuning_loop', False): 'finetuning_loop_disabled',
    ('method.enable_finetuning_loop', True): 'finetuning_loop_enabled',
    ('method.pretrained_backbone_weights_path', 'pretrained_weights/tabula_muris_baseline_model.tar'): 'pretrained_with_baseline',
    ('method.pretrained_backbone_weights_path', 'pretrained_weights/tabula_muris_baseline_pp_model.tar'): 'pretrained_with_baseline_pp',
    # Add more mappings as needed
}

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

configurations = [dict(zip(hyperparameters.keys(), values)) for values in product(*hyperparameters.values())]

for config in configurations:
    exp_name_parts = []
    for key, value in config.items():
        mapping_key = (key, value)
        if mapping_key in short_name_mapping:
            exp_name_parts.append(short_name_mapping[mapping_key])
        else:
            exp_name_parts.append(f"{key.split('.')[-1]}_{value}")
    exp_name = '_'.join(exp_name_parts)
    # Update exp.name according to other options
    #exp_name = ''.join([f"{key.split('.')[-1]}{value}" for key, value in config.items()])
    command = f"python {run_script_path} exp.name={exp_name} method=leo {' '.join([f'{key}={value}' for key, value in config.items()])}"

    # Create the output file name based on the configuration
    output_file = f"{output_dir}output_{exp_name}.txt"

    # Redirect output to the specified file
    command_with_output = f"{command} > {output_file} 2>&1"

    # Execute the command
    try:
        os.system(command_with_output)
    except Exception as e:
        print(f"Error running command: {e}")