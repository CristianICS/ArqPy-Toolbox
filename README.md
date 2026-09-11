# ArqPy Toolbox

This toolbox generates a series of products derived from remote sensing images for archaeological analysis, with a particular focus on crop mark detection. It includes Masked Autoencoder (MAE) saliency analysis and CNN-based pansharpening using methods from the [Z-PNN repository](https://github.com/matciotola/Z-PNN).

The downloaded application includes the preprocessing, MAE, and Z-PNN environments. [Meta's SAM 3 model](https://github.com/facebookresearch/sam3), launched through `sam3_detection.bat`, is the only component that requires a separate environment and model access.

## Introduction

The toolbox is distributed as a self-contained package and **does not require Conda, Miniforge, or manual environment setup by the end user**.

The available operations should be executed in the following recommended order:

1. `atmcorr` - atmospheric correction
2. `pansharpening` - spatial-spectral fusion
3. `pca` - principal component analysis
4. `spectral_indices` - spectral index computation
5. `highpass` - spatial filtering
6. `mae` - Masked Autoencoder-based analysis

The additional operations are:

7. `pansharpening_cnn` - bundled CNN-based spatial-spectral fusion
8. `sam3_detection` - optional SAM 3 crop-mark segmentation requiring external setup

Each operation provides a graphical user interface for configuring the parameters required for execution.

## Tool instructions

The [ArqPy tool guides](docs/tools/README.md) provide a separate, archaeologist-oriented manual for every launcher. Each guide explains all GUI options and defaults, input requirements, processing behaviour, output filenames, interpretation, and important limitations.

## Available sensors

The `atmcorr`, `pca`, `pansharpening`, and `spectral_indices` operations are currently supported for the following sensors:

* WorldView-3 (WV3)
* WorldView LEGION-06

The `highpass` and `mae` operations are available for any image provided in GeoTIFF format.

GeoEye1 (GE1) is only supported for `pansharpening_cnn` tool.

## Data

This repository includes a reduced WorldView-3 image for testing and demonstration purposes. The dataset covers the Zar Tepe archaeological site in Uzbekistan.

## Installation

No installation is required beyond downloading and extracting the toolbox.

The Python environments included in the downloaded application are compressed as `env.7z`, `env_mae.7z`, and `env_pnn.7z`; they must be decompressed before the corresponding toolbox operations are launched.

Together, these compressed environments keep the downloadable application below the 2 GB release-asset limit while providing all components except SAM 3.

1. Download the toolbox release.
2. Extract the archive to a local directory, for example `C:\Toolbox`.
3. Inside the extracted `Toolbox` directory, decompress `env.7z`, `env_mae.7z`, and `env_pnn.7z`.
4. Ensure that each archive creates exactly one environment directory directly inside `Toolbox`.

When using 7-Zip, choose **Extract Here** if the archives already contain their top-level `env`, `env_mae`, and `env_pnn` directories. Do not choose an option that creates another directory with the same name around them.

The resulting paths must resemble:

```txt
Toolbox\env\Scripts\activate.bat
Toolbox\env_mae\Scripts\activate.bat
Toolbox\env_pnn\Scripts\activate.bat
```

Paths such as `Toolbox\env\env\Scripts\activate.bat` or `Toolbox\env_pnn\env_pnn\Scripts\activate.bat` are incorrect. If this nested structure is created, move the inner environment directory contents up one level before running the application.

The toolbox includes all runtime environments except the optional SAM 3 environment.

## Directory structure

After extraction, the toolbox directory must have the following structure:

```txt
Toolbox
|
|- app        (Python scripts used to perform the operations)
|- env        (packed Python environment for preprocessing)
|  |
|  \- OTB-9.1.1-Win64  (Orfeo ToolBox binaries)
|- env_mae    (packed Python environment for MAE analysis)
|- env_pnn    (bundled Z-PNN Python environment)
|  |
|  \- Z-PNN   (modified Z-PNN repository)
|- atmcorr.bat
|- ...
\- README.md
```

## Bundled components and external dependencies

The default toolbox relies on **Orfeo ToolBox (OTB) version 9.1.1**, which is already included in the release. No additional downloads are required.

### MAE model weights

The pretrained MAE weights are not stored inside `env_mae`. Leaving the local checkpoint field blank makes the tool use cached weights or download them on first use. For offline use, download the compatible checkpoint in advance:

```bat
Toolbox\env_mae\Scripts\activate.bat
hf download timm/vit_base_patch16_224.mae model.safetensors --local-dir C:\models\mae
```

Select `C:\models\mae\model.safetensors` in the MAE interface when running without internet access.

### Z-PNN pansharpening

The release includes `env_pnn` and the compatible modified Z-PNN repository. After extracting `env_pnn.7z`, `pansharpening_cnn.bat` can be launched without installing Conda or downloading another environment. The expected layout is:

```txt
Toolbox\env_pnn\Z-PNN\main.py
```

### Optional external dependency: SAM 3

`Toolbox/sam3_detection.bat` launches a folder-level crop-mark segmentation tool based on [Meta's SAM 3 image processor](https://github.com/facebookresearch/sam3).

The first step is to create and pack a separate environment from the project root:

```bat
mamba env create -f requirements_sam3.yml
conda-pack -n sam3_cropmarks -o Toolbox/env_sam3.zip
```

SAM 3 currently needs Python 3.12, a recent PyTorch version, and a CUDA-compatible GPU. Extract `env_sam3.zip` as `Toolbox/env_sam3`.

Before the first run, request access to the `facebook/sam3` checkpoint on Hugging Face and authenticate from the environment:

```bat
Toolbox\env_sam3\Scripts\activate.bat
hf auth login
```

Choose browser login or paste a read-access token. Then download the checkpoint once:

```bash
hf download facebook/sam3 sam3.pt --local-dir C:\models\sam3
```

## Known limitations

KML files are not currently supported as clipping layers because the GDAL environment packaged with the application does not include the `LIBKML` driver. Convert KML data to another supported vector format, such as GeoJSON, GeoPackage, or ESRI Shapefile, before using it for clipping.

## Running the toolbox

To launch the toolbox operations:

1. Double-click the desired operation, for example `atmcorr.bat`; or
2. Run it from a Windows command prompt.

Each batch file automatically activates the appropriate environment and routes the operation to the correct backend.

## Supported systems

* Operating system: **Windows 11**
* No administrator privileges required

## Setting up the development environment

The release includes the complete application directory and the preprocessing, MAE, and Z-PNN environments. The optional SAM 3 environment remains external.

To reproduce the same structure, follow the steps below in a Miniforge or Miniconda terminal.

Create the environments using the provided `.yml` files:

```bash
mamba env create -f requirements_otb.yml
mamba env create -f requirements_mae.yml
mamba env create -f requirements_pnn.yml
```

The default MAE environment is CPU-only to keep the downloadable toolbox small. To build the larger CUDA-capable alternative instead, use:

```bash
mamba env create -f requirements_mae_gpu.yml
```

Install `conda-pack` in your base Python environment if it is not already installed:

```bash
conda install -c conda-forge conda-pack
```

Package the environments and place the resulting archives in the `Toolbox` directory. These archives will need to be unpacked later:

```bash
conda-pack -n arcpy_otb -o Toolbox/env.zip
conda-pack -n mae -o Toolbox/env_mae.zip --format zip --compress-level 9 --n-threads -1 --exclude "*.pyc" --exclude "*/__pycache__/*" --exclude "Library/include/*" --exclude "Library/lib/*.lib" --exclude "Library/bin/*.pdb" --exclude "include/*"
conda-pack -n pansharpen_dl -o Toolbox/env_pnn.zip
```

The MAE exclusions remove Python caches, development headers, import libraries, and debug symbols that are not needed to run the packaged application.

For the optional GPU environment, package it under the same release folder name expected by `mae.bat`:

```bash
conda-pack -n mae_gpu -o Toolbox/env_mae.zip --format zip --compress-level 9 --n-threads -1 --exclude "*.pyc" --exclude "*/__pycache__/*" --exclude "Library/include/*" --exclude "Library/lib/*.lib" --exclude "Library/bin/*.pdb" --exclude "include/*"
```

Finally, add [Orfeo ToolBox](https://www.orfeo-toolbox.org/) to the `env` directory.

Download **OTB version 9.1.1** and place it in the toolbox directory using the following structure:

```txt
Toolbox
|
|- env
|  |
|  |- ...
|  |- OTB-9.1.1-Win64
|  |- ...
|_ ...
```

After reproducing the complete toolbox structure, compress the final application from a Bash terminal with:

```bash
tar -czf ArqPy-1.2.1.tar.gz -C "C:/ArqPy-Toolbox-1.2.1" .
```

## Citation

If you use **ArqPy** in your work, please consider citing it:

```bibtex
@article{s26175478,
  author         = {Iranzo, Cristian and Uribe, Paula and Angás, Jorge and Pérez-Cabello, Fernando},
  title          = {ArqPy: A Python Toolbox for Remote Sensing Image Preprocessing and AI-Assisted Interpretation of Derived Products for Archaeological Prospection},
  journal        = {Sensors},
  volume         = {26},
  year           = {2026},
  number         = {17},
  article-number = {5478},
  url            = {https://www.mdpi.com/1424-8220/26/17/5478},
  issn           = {1424-8220},
  doi            = {10.3390/s26175478}
}
```

