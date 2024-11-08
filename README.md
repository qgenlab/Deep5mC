# Deep5mC: Predicting 5-methylcytosine (5mC) Methylation Status by a Deep Learning Transformer Approach

Deep5mC is a high-performing and advanced deep learning model designed to predict the probability of 5mC methylation within a given DNA sequence. Deep5mC uses CNNs and transformer layers to have a contextual understanding of the given DNA sequence. The input is 2561bp sequences. The output is a methylation percentage for the central position of the input sequence.

To run Deep5mC and/or reproduce the results presented in the Deep5mC paper, users need to install necessary packages per the installation instructions below. Also, training or using Deep5mC needs GPU with >24GB memory. We recommend to use A100 80GB or better GPU with minimize the training and testing time. 

## Installation

### Conda Environment 

Install the `conda` environment using the `environment.yml` file:
```
conda env create -f environment.yml
conda activate Deep5mC
```

### Dataset

The dataset can be installed and preprocessed using

```
src/preprocessing.sh [REFERENCE_GENOME] [THREADS]
```

- The reference genome argument is a path to the hg-19 reference genome and is required.
- Threads is an optional integer argument.
- The results will be stored in `.fasta` files located at `src/data/chr*.fasta`.

## Example Usage

Look at `doc/Example.ipynb` for examples of using Deep5mC. </br>

All command options can be seen in `src/README.md`.

### Test Command Example
```
python src/main.py test \
    test_input.fasta \ # Input file(s)
    output/ \ # Output directory
    --model pretrained_model.pt \ # Model file path
```
### Train Command Example
```
python src/main.py train \
    test_input.fasta \ # Input file(s)
    output/ \ # Output directory
```


## Input File Format

All input is in the following `.fasta` format:
```
> Percentage
2561bp Sequence
```
A short example file can be seen in `doc/test_input.fasta`.

## Pretrained Model Availability

A pretrained model is available in the [releases](https://github.com/qgenlab/Deep5mC/releases) page. This model has been trained on human 5mC data from chromosomes 1 through 20.
