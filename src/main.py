import argparse, textwrap
import contextlib
import os, sys

import torch
from utils import GlobalParameters, train_test_loop

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser(prog=f"{os.path.basename(__file__)}",
                        description='DNA Methylation Regression Model', formatter_class=argparse.RawTextHelpFormatter, add_help=False, 
                        usage=f"""\
python {os.path.basename(__file__)} {{train, test}} [-h]
""")

    # train/test commands
    subparsers = parser.add_subparsers(dest='command')

    train_parser = subparsers.add_parser("train", help=f"Train the model. Run `python {os.path.basename(__file__)} train -h` for specific command options.", 
                                         formatter_class=argparse.RawTextHelpFormatter, add_help=False, usage=f"""\
python {os.path.basename(__file__)} train [input_files ...] output_directory [-h] 
{" " * (len(f"usage: python {os.path.basename(__file__)} train "))}[-d DEVICE] [--model MODEL] [--positive POSITIVE] [--negative NEGATIVE]
{" " * (len(f"usage: python {os.path.basename(__file__)} train "))}[--batch_size BATCH_SIZE] [--batch_count BATCH_COUNT]
{" " * (len(f"usage: python {os.path.basename(__file__)} train "))}[--statistics STATISTICS] [--bar_length BAR_LENGTH]
{" " * (len(f"usage: python {os.path.basename(__file__)} train "))}[-e EPOCHS] [--percent_unmethylated PERCENT_UNMETHYLATED]
""")
    test_parser = subparsers.add_parser("test", help=f"Test the model. Run `python {os.path.basename(__file__)} test -h` for specific command options.", 
                                         formatter_class=argparse.RawTextHelpFormatter, add_help=False, usage=f"""\
python {os.path.basename(__file__)} test [input_files ...] output_directory [-h] 
{" " * (len(f"usage: python {os.path.basename(__file__)} test "))}[-d DEVICE] [--model MODEL] [--positive POSITIVE] [--negative NEGATIVE]
{" " * (len(f"usage: python {os.path.basename(__file__)} test "))}[--batch_size BATCH_SIZE] [--batch_count BATCH_COUNT]
{" " * (len(f"usage: python {os.path.basename(__file__)} test "))}[--statistics STATISTICS] [--bar_length BAR_LENGTH] 
""")

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help='Show this help message and exit.\n ')

    for temp_parser in [train_parser, test_parser]:
        # required
        temp_parser.add_argument('input_files', type=argparse.FileType('r'), nargs='+', help=".fasta format, where description contains expected percentage.\n ")
        temp_parser.add_argument('output_directory', nargs=1, help="Output directory, which is used to hold results")
    
        # optional
        temp_parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help='Show this help message and exit.\n ')
        temp_parser.add_argument('-d', '--device', nargs=1, required=False, default="cpu", 
                            help="Zero-indexed GPU device to use. Defaults to CPU if argument is not used.\n ")
        temp_parser.add_argument('--model', nargs=1, type=argparse.FileType('r'), required=False, default=[None], 
                            help="A saved version of the model to load.\nFile uses the *.pt extension.\n ")
        temp_parser.add_argument('--positive', nargs=1, type=float, required=False, default=[0.8], 
                            help="The positive threshold for generating class-based statistics.\nDecimal between 0 and 1.\n ")
        temp_parser.add_argument('--negative', nargs=1, type=float, required=False, default=[0.2], 
                            help="The negative threshold for generating class-based statistics.\nDecimal between 0 and 1.\n ")
        temp_parser.add_argument('--batch_size', nargs=1, type=int, required=False, default=[16], 
                            help="The batch size per iteration. Preferably a number that is 2^n. Decrease if out of memory.\n ")
        temp_parser.add_argument('--batch_count', nargs=1, type=int, required=False, default=[32], 
                            help="The number of batches before the backpropogation algorithm runs.\nThis allows the use of batch sizes that may be \"too large\" for the allocated memory, while still giving similar results.\nPreferably a number that is 2^n.\n ")
        temp_parser.add_argument('--statistics', nargs=1, type=int, required=False, default=[10], 
                            help="The number of batches between statistic calculations during the train/test loops.\nA higher number can help improve performance.\n ")
        temp_parser.add_argument('--bar_length', nargs=1, type=int, required=False, default=[200], help="The length of the entire loading bar string.\n ")

    # training-specific arguments

    
    train_parser.add_argument('-e', '--epochs', nargs=1, type=int, required=False, default=[1], 
                              help='The number of full iterations across the data set.\n ') 
    train_parser.add_argument('--percent_unmethylated', nargs=1, type=float, required=False, default=[0.5], 
                              help="The ratio of negative to positive data samples. Positive samples are randomly removed from train set each epoch to match ratio.\nMust be greater or equal to 0 and less than 1. If the value is set to zero, that signifies no ratio is enforced.\n ")

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        quit()

    ##
    ## Parse device input
    ##
    
    if args.device != 'cpu':
        try:
            devices = torch.cuda.device_count()
            device_selected = int(args.device[0])
            if devices < device_selected+1 or device_selected < 0:
                parser.error(f"Device option must be in range. Input is a zero-indexed gpu number. You have {devices} GPUs.")
            else:
                # update global variable
                GlobalParameters.device = torch.device(f"cuda:{device_selected}")
        except ValueError:
            parser.error("Device option must be either ""cpu"" or an integer.")
    else:
        # update global variable
        GlobalParameters.device = torch.device("cpu")


    ##
    ## Prepare output directory
    ##
        
    # create directory if doesnt exist
    os.makedirs(args.output_directory[0], exist_ok=True)

    # update global variable
    GlobalParameters.output_directory = (args.output_directory[0] if args.output_directory[0][-1] == "/" else args.output_directory[0] + "/")

    ##
    ## prepare positive/negative class thresholds
    ##

    if args.negative[0] > args.positive[0]:
        parser.error(f"Negative class threshold ({args.negative[0]*100}%) can not be greater than the positive class threshold ({args.positive[0]*100}%).")
    elif args.negative[0] < 0 or args.negative[0] > 1:
        parser.error(f"Negative class threshold ({args.negative[0]*100}%) out of bounds. Must be a float between 0 and 1, inclusive.")
    elif args.positive[0] < 0 or args.positive[0] > 1:
        parser.error(f"Positive class threshold ({args.positive[0]*100}%) out of bounds. Must be a float between 0 and 1, inclusive.")
    
    GlobalParameters.positive_threshold = args.positive[0]
    GlobalParameters.negative_threshold = args.negative[0]

    ##
    ## Prepare other variables
    ##

    if args.batch_size[0] <= 0:
        parser.error("Batch size must be greater than zero.")
    elif args.batch_count[0] <= 0:
        parser.error("Batch count must be greater than zero.")
    elif args.statistics[0] <= 0:
        parser.error("Batches per statistic must be greater than zero.")
    elif args.bar_length[0] <= 0:
        parser.error("Progress bar length must be greater than zero.")

    GlobalParameters.batch_size = args.batch_size[0]
    GlobalParameters.batch_count = args.batch_count[0]
    GlobalParameters.batches_per_statistic = args.statistics[0]
    GlobalParameters.ncols = args.bar_length[0]

    if args.command == "train":
        ##
        ## prepare epochs
        ##
        if args.epochs[0] <= 0:
            parser.error("Epochs must be greater than zero.")
        GlobalParameters.epochs = args.epochs[0]
        
        ##
        ## prepare percent_unmethylated
        ##
    
        if args.percent_unmethylated[0] >= 1 or args.percent_unmethylated[0] < 0:
            parser.error(f"Option --percent_unmethylated out of bounds. Option must be greater or equal to 0 and less than 1. If the value is set to zero, that signifies no ratio is enforced.")

        GlobalParameters.percent_unmethylated = args.percent_unmethylated[0]

    
    # display parameters used
    with open(f'{GlobalParameters.output_directory}log.txt', 'a') as file, contextlib.redirect_stdout(file), contextlib.redirect_stderr(file):
        print("python " + " ".join(sys.argv)) # log the python command used to run program
        
        GlobalParameters.display([str(f.name) for f in args.input_files], (args.model[0].name if args.model[0] != None else None))
    GlobalParameters.display([str(f.name) for f in args.input_files], (args.model[0].name if args.model[0] != None else None))

    # start program
    train_test_loop.main_program([str(f.name) for f in args.input_files], (args.model[0].name if args.model[0] != None else None), training=(args.command == "train"))

