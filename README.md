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

<<<<<<< HEAD
Look at `Example.ipynb` for examples of using Deep5mC. </br>

All command options can be seen in `src/README.md` or by running the help command:
=======
```
python src/main.py [train | test] \
    <input files> \
    <output directory> \
    --model <optional pretrained model> \
    -d <optional gpu device>
```

### Example
```
python src/main.py test \
    input1.fasta input2.fasta input3.fasta \
    output_dir/ \
    --model pretrained_model.pt \
    -d 7
```

After running, the `output_dir/` will hold the expected values, predicted values, and some statistical measures to evaluate the model's performance. </br>
</br>
More command line options can be seen in `src/README.md` or by running the help command:
>>>>>>> 6685bb6fa6f08c4c4da4e52fd2ed8971090d5c56
```
python src/main.py [train | test] -h
```


## Pretrained Model Availability

A pretrained model is available in the [releases](https://github.com/qgenlab/Deep5mC/releases) page. This model has been trained on human 5mC data from chromosomes 1 through 20.
