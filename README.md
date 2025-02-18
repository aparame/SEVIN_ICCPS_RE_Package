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


### Steps to Set Up the Environment

1. **Set Up the Alpha-Beta-Crown Environment**:
- Clone the [alpha-beta-crown](https://github.com/Verified-Intelligence/alpha-beta-CROWN) repository and follow its instructions to create a Conda environment.
- Activate the Conda environment:
  ```
  conda activate alpha-beta-crown
  ```
- Return to home dir:
  ```
  cd ..
  ```

2. **Clone the SEVIN_ICCPS_RE_Package Repository**:
- Clone this repository in a new folder adjacent to the alpha-beta-CROWN repository
  ```
  git clone https://github.com/aparame/ICCPS_SEVIN_REP
  cd ICCPR_SEVIN_REP
  ```

3. **Install Additional Packages**:
- Update the `alpha-beta-crown` conda environment with required packages listed in the `ICCPS_SEVIN_REP\environment.yml` file provided in the ICCPS_SEVIN_REP repository:
  ```
  conda env update --file environment.yml
  ```

---

## Reproducing Results from the Paper

To recreate the results from **Table 1** and **Table 3** of the ICCPS paper:

1. Execute the following Python scripts after activating the conda environment `alpha-beta-crown` as mentioned above:
```
python vanilla_formal_verification.py
python robustness_verification.py
```

2. The results will be stored in the `results` folder as `.txt` files. These files can be used to generate the tables presented in the paper.

---

## Supplementary Code: Training and Specification Generation Process

If you want to dive deeper into the training and specification process, move to the `util` folder and follow these steps:

### 1. Train the Gaussian Mixture Variational Autoencoder (GM-VAE)
- Use the `train_config.ini` file to configure and iterate over training parameters.
- Train the GM-VAE using the following scripts:
#### Vanilla GM-VAE:
 ```
 python vanilla_train.py
 ```
#### Robust GM-VAE (with image augmentations):
 ```
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

### 2. Train the Neural Network Controller (NNC)
Train the neural network controller using the `nc_train.py` script. The training configuration can be controlled using the `config_NNC.ini` file.
```
python nc_train.py
```

- The trained NNC models will also be saved in the `saved_models` folder.

### 3. Combine the GM-VAE Decoder with the NNC
Combine the Decoder from the trained GM-VAE with the NNC trained earlier. Save the combined model in the `saved_models` directory.
```
python combined.py
```

### 4. Generate the VNNLIB specification files for Verification
Using the combined models from the previous step, generate the `.vnnlib` files to be used in the formal verification process later on. The `generate_vnnlib.py` code first pre-processes the latent encodings based on the action values and then generates the vnnlib file for verification.
```
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



