import os
from itertools import product


# grid search configurations
hyperparameters = {
    'dataset': ['tabula_muris'],
    'n_shot': [1, 5],
    'lr': [1e-3, 3e-4, 1e-4],
    'method.latent_space_dim': [8, 16, 32],
    'method.weight_decay': [1e-6, 1e-8],
    'method.num_adaptation_steps': [5, 10],
    # Add more hyperparameters as needed
}

# Path to the run.py script (right now it should be in the same directory as this file)
run_script_path = './run.py'

# Output directory
output_dir = 'experiment/'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

configurations = [dict(zip(hyperparameters.keys(), values)) for values in product(*hyperparameters.values())]

for config in configurations:
    # Update exp.name according to other options
    exp_name = '_'.join([f"{key.split('.')[-1]}_{value}" for key, value in config.items()])
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