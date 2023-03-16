# DeepH-E3

This code is the implementation of the DeepH-E3 method described in the paper *General framework for E(3)-equivariant neural network representation of density functional theory Hamiltonian* ([arXiv:2210.13955](https://arxiv.org/abs/2210.13955)). 

You can find demo input files and instructions in these repositories: [Dataset 1](https://doi.org/10.5281/zenodo.7553640), [Dataset 2](https://doi.org/10.5281/zenodo.7553827) and [Dataset 3](https://doi.org/10.5281/zenodo.7553843). These are also the datasets used by the paper, so you can try to reproduce the results in the paper using those datasets.

The current code will be integrated into [DeepH-pack](https://github.com/mzjb/DeepH-pack) in the future.

## Installation

To use DeepH-E3, the following environment is required:

### Python

The python interpreter version needs to be at least 3.9. The following packages are also needed:

- Numpy
- PyTorch = 1.9.0
- PyTorch geometric = 1.7.2
- e3nn version 0.3.5
- h5py
- TensorBoard
- pathos
- pymatgen

### Julia

The installation of Julia is optional. If you want to parse openmx output and convert into the format used by DeepH-E3, or do sparse matrix diagonalization to obtain the band structure from DeepH-E3 output, then Julia is needed.

First prepare the Julia interpreter version 1.5.4. Then install the following packages:

- Arpack.jl
- HDF5.jl
- ArgParse.jl
- JLD.jl
- JSON.jl
- IterativeSolvers.jl
- DelimitedFiles.jl
- StaticArrays.jl

In order to use Julia with DeepH-E3, you have to add the julia executable to `PATH`.

## Usage

The usage of DeepH-E3 is similar to that of [DeepH-pack](https://github.com/mzjb/DeepH-pack). Although not mandatory, it is recommended that you learn the usage of DeepH-pack first.

### Preprocess

First, you have to convert the output of DFT codes to the format that can be directly read by DeepH-E3. It is recommended that you directly use the preprocess utility of DeepH-pack. You must set `local_coordinate = False` in the DeepH preprocess config.

DeepH-E3 also has its own input conversion utility. You can use the command

```
/path/to/deephe3-preprocess.py preprocess.ini
```

to preprocess DFT data. The default config file for `preprocess.ini` can be found in `DeepH-E3/deephe3/default_configs/base_default.ini`. This default config also includes a description of each input variables.

### Train your model

Based on the data, one can train the DeepH-E3 model. Training can be done by the command

```
/path/to/deephe3-train.py train.ini
```

The default config file for `train.ini` can be found in `DeepH-E3/deephe3/default_configs/train_default.ini`. This default config also includes a description of each input variables.

### Model inference

Once you have a trained model, you can use that model to predict the Hamiltonian of some material structures and get `hamiltonians_pred.h5`. Inference can be done by the command

```
/path/to/deephe3-eval.py eval.ini
```

The default config file for `eval.ini` can be found in `DeepH-E3/deephe3/default_configs/eval_default.ini`. This default config also includes a description of each input variables.

For large structures that are impossible to calculate with DFT, one needs the modified OpenMX code that only generates the overlap matrix: [Overlap-only-OpenMX](https://github.com/mzjb/overlap-only-OpenMX). In this case, DeepH-E3 needs the processed `overlaps.h5` instead of `hamiltonians.h5` to generate the crystal graph. You have to set `inference=True` in `eval.ini`.
