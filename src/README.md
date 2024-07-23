# Deep5mC Help Pages

## Train

```
usage: python main.py train [input_files ...] output_directory [-h]
                            [-d DEVICE] [--model MODEL] [--positive POSITIVE] [--negative NEGATIVE]
                            [--batch_size BATCH_SIZE] [--batch_count BATCH_COUNT]
                            [--statistics STATISTICS] [--bar_length BAR_LENGTH]
                            [-e EPOCHS] [--percent_unmethylated PERCENT_UNMETHYLATED]

positional arguments:
  input_files           .fasta format, where description contains expected percentage.

  output_directory      Output directory, which is used to hold results

optional arguments:
  -h, --help            Show this help message and exit.

  -d DEVICE, --device DEVICE
                        Zero-indexed GPU device to use. Defaults to CPU if argument is not used.

  --model MODEL         A saved version of the model to load.
                        File uses the *.pt extension.

  --positive POSITIVE   The positive threshold for generating class-based statistics.
                        Decimal between 0 and 1.

  --negative NEGATIVE   The negative threshold for generating class-based statistics.
                        Decimal between 0 and 1.

  --batch_size BATCH_SIZE
                        The batch size per iteration. Preferably a number that is 2^n. Decrease if out of memory.

  --batch_count BATCH_COUNT
                        The number of batches before the backpropogation algorithm runs.
                        This allows the use of batch sizes that may be "too large" for the allocated memory, while still giving similar results.
                        Preferably a number that is 2^n.

  --statistics STATISTICS
                        The number of batches between statistic calculations during the train/test loops.
                        A higher number can help improve performance.

  --bar_length BAR_LENGTH
                        The length of the entire loading bar string.

  -e EPOCHS, --epochs EPOCHS
                        The number of full iterations across the data set.

  --percent_unmethylated PERCENT_UNMETHYLATED
                        The ratio of negative to positive data samples. Positive samples are randomly removed from train set each epoch to match ratio.
                        Must be greater or equal to 0 and less than 1. If the value is set to zero, that signifies no ratio is enforced.
```

## Test

```
usage: python main.py test [input_files ...] output_directory [-h]
                           [-d DEVICE] [--model MODEL] [--positive POSITIVE] [--negative NEGATIVE]
                           [--batch_size BATCH_SIZE] [--batch_count BATCH_COUNT]
                           [--statistics STATISTICS] [--bar_length BAR_LENGTH]

positional arguments:
  input_files           .fasta format, where description contains expected percentage.

  output_directory      Output directory, which is used to hold results

optional arguments:
  -h, --help            Show this help message and exit.

  -d DEVICE, --device DEVICE
                        Zero-indexed GPU device to use. Defaults to CPU if argument is not used.

  --model MODEL         A saved version of the model to load.
                        File uses the *.pt extension.

  --positive POSITIVE   The positive threshold for generating class-based statistics.
                        Decimal between 0 and 1.

  --negative NEGATIVE   The negative threshold for generating class-based statistics.
                        Decimal between 0 and 1.

  --batch_size BATCH_SIZE
                        The batch size per iteration. Preferably a number that is 2^n. Decrease if out of memory.

  --batch_count BATCH_COUNT
                        The number of batches before the backpropogation algorithm runs.
                        This allows the use of batch sizes that may be "too large" for the allocated memory, while still giving similar results.
                        Preferably a number that is 2^n.

  --statistics STATISTICS
                        The number of batches between statistic calculations during the train/test loops.
                        A higher number can help improve performance.

  --bar_length BAR_LENGTH
                        The length of the entire loading bar string.
```