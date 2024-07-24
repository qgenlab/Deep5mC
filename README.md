# Deep5mC

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
A short example file can be seen in `test_input.fasta`.

## Example Usage

Look at `doc/Example.ipynb` for examples of using Deep5mC. </br>

All command options can be seen in `src/README.md` or by running the help command:
```
python src/main.py [train | test] -h
```


## Pretrained Model Availability

A pretrained model is available in the [releases](https://github.com/qgenlab/Deep5mC/releases) page. This model has been trained on human 5mC data from chromosomes 1 through 20.
