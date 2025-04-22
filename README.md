 [![DOI](https://zenodo.org/badge/679296696.svg)](https://doi.org/10.5281/zenodo.14806735)
# Scalable and Explainable Verification of Image-based Neural Network Controllers

This repository contains the repeatability package for the codebase of the conference paper titled **"Scalable and Explainable Verification of Image-based Neural Network Controllers"** that is part of the proceedings for the **Internation Conference on Cyber-Physical Systems 2025**. The package includes scripts and configurations to reproduce the results presented in the paper, as well as tools for training and verifying neural network controllers.

---

## Installation Instructions

### Prerequisites
- **Operating System**: Ubuntu 20.04
- **Python Version**: 3.11.2

### PC Configuration
- **GPU**: Nvidia RTX 3090
- **CPU**: Intel Core-i9 13900K


## Docker Environment

If needed, we also provide a [Docker image](https://hub.docker.com/r/adi2440/iccps_sevin_rep). The docker container automatically runs 2 verification scripts (to reproduce results in Tables **1** and **3** from the paper). 
Alternatively, one can reproduce the results using the instructions below as well.
 
## Steps to Set Up the Environment

1.1 [Alpha-Beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN) **Installation and Setup**
 ----------------------

 α,β-CROWN is tested on Python 3.11 and PyTorch 2.3.1 (recent versions may also work).
 It can be installed easily into a conda environment. If you don't have conda, you can install
 [miniconda](https://docs.conda.io/en/latest/miniconda.html).
 
 Clone their verifier including the [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) submodule:
 ```bash
 git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
 ```
 
 Setup the conda environment from [`environment.yaml`](complete_verifier/environment.yaml)
 with pinned dependencies versions (CUDA>=12.1 is required):
 ```bash
 # Remove the old environment, if necessary.
 conda deactivate; conda env remove --name alpha-beta-crown
 # install all dependents into the alpha-beta-crown environment
 conda env create -f complete_verifier/environment.yaml --name alpha-beta-crown
 # activate the environment
 conda activate alpha-beta-crown
 # Move back to main directory
 cd ..
 ```
   
1.2 Clone the SEVIN_ICCPS_RE_Package Repository:
----------------------
Clone this repository in a new folder adjacent to the alpha-beta-CROWN repository.
   ```bash
   git clone https://github.com/aparame/ICCPS_SEVIN_REP
   cd ICCPR_SEVIN_REP
   ```

1.3 **Install Additional Packages**:
----------------------
Update the `alpha-beta-crown` conda environment with required packages listed in the `ICCPS_SEVIN_REP\environment.yml` file provided in the ICCPS_SEVIN_REP repository:
  ```bash
  conda env update --file environment.yml
  ```

---

## Reproducing Results from the Paper


2.1 Running Verification Scripts
 ----------------------
Execute the following Python scripts (to reproduce results in Tables **1 ** and **3** from the paper) after activating the conda environment `alpha-beta-crown` as mentioned above:
```bash
python vanilla_formal_verification.py
python robustness_verification.py
```

The results will be stored in the `results` folder as `.txt` files. These files can be used to generate the tables presented in the paper.


2.2 Understanding the results
 ----------------------
The generated result files `results_robustness_verification.txt` and `results_vanilla_formal_verification.txt` store the results in the format as below:
```
{type_of_specification}_{specification}_{NNC_architecture}_{augmentation_type}_{augmentation_value}.yaml

Result: **SAT/UNSAT**

Time: **t** (secs)
```
These results can be then used to generate Tables **1** and **3** from the paper.

---

## Supplementary Code: Training and Specification Generation Process

If you want to dive deeper into the training and specification process, move to the `util` folder and follow these steps:

### 3.1 Train the Gaussian Mixture Variational Autoencoder (GM-VAE)
- Use the `train_config.ini` file to configure and iterate over training parameters.
- Train the GM-VAE using the following scripts:
#### Vanilla GM-VAE:
 ```bash
 python vanilla_train.py
 ```
#### Robust GM-VAE (with image augmentations):
 ```bash
 python robust_train.py
 ```
 For the Robust GM-VAE training, you can set the following options in the Augmentation section of `train_config.ini` as:
 ```
 [Augmentation]
 type = mb 
 value = 4  
```
- where `type` can be `mb` for Motion Blur augmentation or `bright` for Image Brightness augmentation applied to the PyTorch Dataloader class during training. The `value` can be `2, 4 or 6 ` for Motion Blur and `0.2 or 0.4` for Image Brightness.

- The trained models will be saved in the `saved_models` directory.

### 3.2 Train the Neural Network Controller (NNC)
Train the neural network controller using the `nc_train.py` script. The training configuration can be controlled using the `config_NNC.ini` file.
```bash
python nc_train.py
```

- The trained NNC models will also be saved in the `saved_models` folder.

### 3.3 Combine the GM-VAE Decoder with the NNC
Combine the Decoder from the trained GM-VAE with the NNC trained earlier. Save the combined model in the `saved_models` directory.
```bash
python combined.py
```

### 3.4 Generate the VNNLIB specification files for Verification
Using the combined models from the previous step, generate the `.vnnlib` files to be used in the formal verification process later on. The `generate_vnnlib.py` code first pre-processes the latent encodings based on the action values and then generates the vnnlib file for verification.
```bash
python generate_vnnlib.py
```
- The `.vnnlib` files are saved in the `configs` folder.

### Conclusion
This concludes the supplementary tranining, and specification generation steps. You can now conduct formal verification using the `vanilla_formal_verification.py` and `robustness_verification.py` files from the home directory

---

## Directory Structure
```
ICCPS_SEVIN_REP/
├── configs/ # Contains all the .yaml configuration files to be run be alpha-beta-crown verifier
├── dataset/
  ├── processed_images/ # Contains the pre-processed front camera images
  ├── combined_file.csv # Action value information
├── encodings/processed # Contains information about the latent space encodings for each trained GM-VAE and specification to be verified
├── results #Saved verification results
├── saved_models/ # Stores trained models (GM-VAE, NNC, and combined models)
├── training_plots #Some plots generated in the training process as shown in the paper
├── util/
    ├── generate_vnnlib.py # Python code to vnnlib files from the encodings
    ├── train_config.ini # Configuration file for GM-VAE training
    ├── config_NNC.ini # Configuration file for neural network controller training
    ├── vanilla_train.py # Script to train vanilla GM-VAE
    ├── robust_train.py # Script to train robust GM-VAE
    ├── nc_train.py # Script to train the neural network controller   
├── vanilla_formal_verification.py # Script for vanilla formal verification
├── robustness_verification.py # Script for robustness verification
├── environment.yml # List of additional Python packages to install

└── README.md

```

## Author Information
You can contact aparame@clemson.edu for any queries or to report any bugs in the code.
- Authors:- Aditya Parameshwaran and Yue Wang



