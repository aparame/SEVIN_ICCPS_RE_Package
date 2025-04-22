import os
import subprocess
import yaml
import glob
import torch
import re

# Custom YAML representer for lists to enforce flow style
def represent_list_flow_style(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

yaml.add_representer(list, represent_list_flow_style)

# Define constants

ABCROWN_CMD = "python /opt/alpha-beta-CROWN/complete_verifier/abcrown.py --config" if torch.cuda.is_available() \
    else "python /opt/alpha-beta-CROWN/complete_verifier/abcrown.py --device cpu --config"
RESULTS_FILE = "results/results_vanilla_formal_verification.txt"
CONFIG_FOLDER = "configs_vanilla"


VNNLIB_PATHS = [
    "vnnlibs"
]



ONNX_PATHS = ["saved_models/combined/NvidiaNet_GMVAE_vanilla.onnx",
	     "saved_models/combined/ResNet18_GMVAE_vanilla.onnx"]


# Create config folder if it doesn't exist
os.makedirs(CONFIG_FOLDER, exist_ok=True)

# Generate yaml files
for vnnlib_path in VNNLIB_PATHS:
    for onnx_path in ONNX_PATHS:
        for vnnlib_file in glob.glob(os.path.join(vnnlib_path, "*formal*.vnnlib")):
            config_data = {
                "model": {
                    "onnx_path": onnx_path,
                    "input_shape": [-1, 8]  # Will now be formatted as [-1, 8]
                },
                "specification": {
                    "vnnlib_path": vnnlib_file
                },
                "solver": {
                    "batch_size": 2000,
                    "alpha-crown": {
                        "iteration": 100,
                        "lr_alpha": 0.1
                    }
                },
                "bab": {
                    "timeout": 2000,
                    "branching": {
                        "reduceop": "min",
                        "method": "fsb",
                        "candidates": 50
                    }
                },
                "attack": {
                    "pgd_steps": 50000,
                    "pgd_restarts": 50
                }
            }

            # Create yaml file
            config_filename = f"{os.path.basename(vnnlib_file).split('.')[0]}_{os.path.basename(onnx_path).split('.')[0]}"
            # Define patterns to remove
            patterns_to_remove = [r'_GMVAE', r'_formal', r'_encodings_8']

            # Remove specified patterns
            for pattern in patterns_to_remove:
                config_filename = re.sub(pattern, '', config_filename)

            # Remove repeated character sequences (e.g., '__' -> '_')
            config_filename = re.sub(r'(_)\1+', r'\1', config_filename)
            config_path = os.path.join(CONFIG_FOLDER, f"{config_filename}.yaml")
            
            if not os.path.exists(config_path):
                with open(config_path, "w") as f:
                    yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

# Run verification process
results = []
for config_file in os.listdir(CONFIG_FOLDER):
    if config_file.endswith(".yaml"):
        config_path = os.path.join(CONFIG_FOLDER, config_file)
        cmd = f"{ABCROWN_CMD} {config_path}"
        try:
            if torch.cuda.is_available():
            	output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, timeout=10).decode("utf-8")
            else:
            	output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, timeout=40).decode("utf-8")
            last_two_lines = output.splitlines()[-2:]
        except subprocess.TimeoutExpired:
            last_two_lines = ["Result: unsat"]
        except Exception as e:
            last_two_lines = [f"ERROR - {str(e)}"]

        # Print to terminal
        print(f"Results for {config_file}:")
        print("\n".join(last_two_lines))
        print("-" * 50)  # Separator for readability

        # Save to results list
        results.append(f"{config_file}:\n" + "\n".join(last_two_lines) + "\n\n")

# Sort results alphabetically by config file name
results.sort()

# Write sorted results to the text file
with open(RESULTS_FILE, "w") as results_f:
    for result in results:
        results_f.write(result)
