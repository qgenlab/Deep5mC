# Deep5mC

## Installation

Install the `conda` environment using the `environment.yml` file:
```
conda env create -f environment.yml
conda activate Deep5mC
```

## Data Set Availability

There is a [dataset](https://www.bcgsc.ca/downloads/111_reference_epigenomes/112epigenomes/5mC/SBS_Removed_E027_E064_Fixed_E012/) that can be downloaded through `doc/download.sh`, which contains 5mC methylation percentages, hg19 locations, and read coverage information. </br>
</br>
This dataset is made available through Canada's Michael Smith Genome Sciences Centre. </br>

## Input File Example

All input is in the following `.fasta` format:
```
> Percentage
2561bp Sequence
```
A short example file can be seen in `test_input.fasta`.

## Example Usage

Look at `Example.ipynb` for examples of using Deep5mC. </br>

All command options can be seen in `src/README.md` or by running the help command:
```
python src/main.py [train | test] -h
```


## Pretrained Model Availability

A pretrained model is available in the [releases](https://github.com/qgenlab/Deep5mC/releases) page. This model has been trained on human 5mC data from chromosomes 1 through 20.