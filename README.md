# Deep5mC

Deep5mC is a high-performing and advanced deep learning model designed to predict the probability of 5mC methylation within a given DNA sequence. The model's architecture includes CNNs, transformer layers, and relative position embedding, which allows Deep5mC to have a great contextual understanding of the given DNA sequence. 

## Installation

Install the `conda` environment using the `environment.yml` file:
```
conda env create -f environment.yml
conda activate Deep5mC
```

## Input File Example

All input is in the following `.fasta` format:
```
> Percentage
2561bp Sequence
```
A short example file can be seen in `doc/test_input.fasta`.

## Example Usage

Look at `doc/Example.ipynb` for examples of using Deep5mC. </br>

All command options can be seen in `src/README.md` or by running the help command:
```
python src/main.py [train | test] -h
```


## Pretrained Model Availability

A pretrained model is available in the [releases](https://github.com/qgenlab/Deep5mC/releases) page. This model has been trained on human 5mC data from chromosomes 1 through 20.
