import torch

########################
## Model Architecture ##
########################

d_model = 1024
nhead = 16
num_layers = 16

learning_rate = 1e-7 #5e-6

dropout = 0.2

#####################
## Data Parameters ##
#####################

epochs = 1

## batch_size is the number of input sequences per iteration
## batch_count is the number of iterations per backpropogation algorithm call
#
## batch_size * batch_count is the total number of input sequences per backpropogation algorithm call
#
## This allows for the illusion of having a larger batch size, without needing a lot of memory 
batch_size = 16
batch_count = 32 

# Data set
positive_threshold = 0.80
negative_threshold = 0.20
percent_unmethylated = 0.50

# How often to calculate statistics.
# A larger number increases performance
batches_per_statistic = 10

# Number of bases on each side of the methylated base
n = 1280

device = torch.device('cuda:7')

output_directory = f"results/"

# Length of loading bar
ncols = 200


def display(input_files, model):
    print(f"""
{"-"*(ncols//2)}

########################
## Model Architecture ##
########################

    hidden layer size = {d_model}
    number of attention heads = {nhead}
    number of layers in TransformerEncoder module = {num_layers}
    learning rate = {learning_rate}
    dropout rate = {dropout * 100}%

#####################
## Other Parameters ##
#####################

    epochs = {epochs}
    batch size = {batch_size}
    number of batches per backpropagation algorithm call = {batch_count}
    
    positive class threshold = {positive_threshold * 100}%
    negative class threshold = {negative_threshold * 100}%
    unmethylated data percent = {percent_unmethylated * 100}%
    
    batches per statistic calculation = {batches_per_statistic}
    number of bases per side of methylated base = {n}
    cpu / gpu used = {device}
    model file = {model}
    input files = {input_files}
    output directory = {output_directory}
    number of columns in progress bars = {ncols}
    
{"-"*(ncols//2)}
""")