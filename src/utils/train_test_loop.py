import argparse

import tools.model as BERT
import tools.statistics as statistics
import tools.CustomDataLoader as CustomDataLoader
import tools.GlobalParameters as GlobalParameters

import torch
import torch.nn as nn

import pandas as pd
import numpy as np
from collections import deque

import contextlib

import glob

import traceback
from tqdm import tqdm

##########################
## Train/Test Functions ##
##########################

def test(percents, sequences, batches_left, model, loss, device, run_stats):
    # Use model
    with torch.autocast('cuda'):
        sequences = sequences.type(torch.float32)

        out = model(sequences)  

        # is true if output is a 0-dim scalar
        if out.shape == ():
            out = out.unsqueeze(0)
        
        l = loss(out, percents)


    # Generate statistics on model output
    if run_stats:
        pearson, accuracy, roc_auc, tn, fp, fn, tp, sensitivity, specificity, MCC = statistics.generate(percents, out, images=False, matrix=False)
    else:
        pearson = accuracy = roc_auc = tn = fp = fn = tp = sensitivity = specificity = MCC = -100

    return l, out.detach().cpu(), pearson, accuracy, roc_auc, tn, fp, fn, tp, sensitivity, specificity, MCC

current_batch_count = 0

def train(percents, sequences, batches_left, model, loss, opt, device, run_stats, scaler):
    global current_batch_count
    # Use model
    with torch.autocast('cuda'):
        sequences = sequences.type(torch.float32)

        out = model(sequences)    

        # is true if output is a 0-dim scalar
        if out.shape == ():
            out = out.unsqueeze(0)

        l = loss(out, percents)

    scaled = scaler.scale(l/GlobalParameters.batch_count).backward()
    
    if current_batch_count % GlobalParameters.batch_count == 0 or batches_left == 0:
        scaler.step(opt)
        scaler.update()

    current_batch_count += 1

    
    # Generate statistics on model output
    if run_stats:
        pearson, accuracy, roc_auc, tn, fp, fn, tp, sensitivity, specificity, MCC = statistics.generate(percents, out, images=False, matrix=False)
    else:
        pearson = accuracy = roc_auc = tn = fp = fn = tp = sensitivity = specificity = MCC = -100

    return l, out.detach().cpu(), pearson, accuracy, roc_auc, tn, fp, fn, tp, sensitivity, specificity, MCC

def update_pbar(pbar, file, loss, accuracy, pearson, roc_auc, sensitivity, specificity, MCC, count, training):
    # Update progress bar
    if count % 20 == 0 and count != 0:
        pbar.set_postfix(status=("Training" if training else "Testing"), file=file, loss=(loss/count).item(), sensitivity=sensitivity, 
                         specificity=specificity, accuracy=accuracy, pearson=pearson, MCC=MCC, roc_auc=roc_auc)
        
def loop(data, len_data, function, file_name, opt, model, loss_func, training):
    total_loss = count = accuracy = pearson = TN = FN = TP = FP = roc_auc = sensitivity = specificity = MCC = 0
    last_expected = deque()
    last_out = deque()

    statistic_separation = len_data // GlobalParameters.batches_per_statistic
    statistic_separation = (statistic_separation if statistic_separation != 0 else 1)
    with tqdm(total=len_data, ncols=GlobalParameters.ncols) as pbar:
        for i, batch in enumerate(data): 
            percents, sequences = batch
            del batch
            batches_left = len_data-i

            update_pbar(pbar, file_name, total_loss, accuracy, pearson, roc_auc, sensitivity, specificity, MCC, count, training)

            l, out, temp_pearson, temp_accuracy, \
            temp_roc_auc, temp_tn, temp_tp, temp_fn, temp_fp, \
            temp_sensitivity, temp_specificity,  temp_MCC = function(percents, sequences, batches_left, (i % statistic_separation == 0 or batches_left == 0))
            
            if temp_accuracy != -100:
                accuracy = temp_accuracy
                pearson = temp_pearson
                roc_auc = temp_roc_auc
                tn = temp_tn 
                tp = temp_tp
                fn = temp_fn
                fp = temp_fp
                sensitivity = temp_sensitivity
                specificity = temp_specificity
                MCC = temp_MCC 
            
            total_loss = torch.add(total_loss, l.detach().cpu())
            count += 1

            last_expected.extend(percents)
            last_out.extend(out)

            pbar.n = i*GlobalParameters.batch_size
            pbar.refresh()

    return torch.tensor(last_expected).flatten(), torch.tensor(last_out).flatten(), total_loss.item(), count
            
##################
## Main Program ##
##################

def main_program(input_files, model_file, training):
    
    model = BERT.BERT().to(GlobalParameters.device) 
    opt = torch.optim.Adam(model.parameters(), lr=GlobalParameters.learning_rate)  
    loss = nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler()

    # Load model
    if model_file is not None:
        model.load_state_dict(torch.load(model_file))
        model.eval()
    
    progress = []
    losses = []                                     
    accuracies = []
    pearsons = []
    roc_aucs = []
    tns = []
    tps = []
    fns = []
    fps = []
    sensitivities = []
    specificities = []
    MCCs = []
    
    try:
        for e in range(GlobalParameters.epochs):
            for file in input_files:
                file_name = file[file.rfind("/")+1:]

                print('Loading data set...')
                data, len_data = CustomDataLoader.getDataLoader(file, training=training) 
                print('Done loading.')

                if training:
                    # Train model
                    train_test_function = (lambda percents, sequences, run_stats, batches_left, model=model, loss_func=loss, opt=opt, scaler=scaler, device=GlobalParameters.device: \
                                      train(percents, sequences, batches_left, model, loss_func, opt, device, run_stats, scaler))
                else:
                    # Test model
                    train_test_function = (lambda percents, sequences, run_stats, batches_left, model=model, loss_func=loss, opt=opt, device=GlobalParameters.device: \
                                      test(percents, sequences, batches_left, model, loss_func, device, run_stats))

                
                expected, out, l, count = loop(data, len_data, train_test_function, file_name, opt, model, loss, training=training)
                
                del data
                
                pearson, accuracy, roc_auc, tn, fp, fn, tp, sensitivity, specificity, MCC, matrix = statistics.generate(expected, out, images=True, matrix=True, model_file=f"{file_name}_e_{e+1}")

                # End of Training/Testing loop. Print Results
                
                results = ""
                
                results += "-"*75 + "\n" 
                if training:
                    results += f"Epoch {e+1}/{GlobalParameters.epochs}, {file_name}:\n\n"
                else:
                    results += f"{file_name}:\n\n"

                if count != 0:
                    if training:
                        progress.append(f"Epoch {e+1} {file_name}")
                    else:
                        progress.append(file_name)
                    losses.append(l/count)
                    accuracies.append(accuracy)
                    pearsons.append(pearson)
                    roc_aucs.append(roc_auc)
                    tns.append(tn)
                    tps.append(tp)
                    fns.append(fn)
                    fps.append(fp)
                    sensitivities.append(sensitivity)
                    specificities.append(specificity)
                    MCCs.append(MCC)
                    
                    results += f"""Loss = {l/count}
Accuracy = {accuracy}
Pearson = {pearson}
MCC = {MCC}
ROC AUC = {roc_auc}
TN, TP, FN, FP = {tn}, {tp}, {fn}, {fp}
Sensitivity = {sensitivity}  
Specificity = {specificity}

Confusion Matrix:
Predicted(x) vs. True(y)
- [{matrix[0]}
?  {matrix[1]}
+  {matrix[2]}]
    -     ?     +
    
"""

                if GlobalParameters.batch_size <= 20:
                    results += "Last results:\n"
                else:
                    results += "Last results (shortened):\n"
                results += f"Expected: {expected[:20]}\n"
                results += f"Output:   {out[:20]}\n"

            
                results += "-"*75 + "\n"


                with open(f'{GlobalParameters.output_directory}log.txt', 'a') as file, contextlib.redirect_stdout(file), contextlib.redirect_stderr(file):
                    print(results, end="") 
                print(results, end="")

                # Save results
                data = pd.DataFrame({
                                     'Progress':     progress,
                                     'Loss':         losses,
                                     'Accuracy':     accuracies,
                                     'Pearson':      pearsons,
                                     'MCC':          MCCs,
                                     'ROC AUC':      roc_aucs,
                                     'TN':           tns,
                                     'TP':           tps,
                                     'FN':           fns,
                                     'FP':           fps,
                                     'Sensitivity':       sensitivities,
                                     'Specificity':    specificities
                                    }) 

                data.to_csv(f"{GlobalParameters.output_directory}/out.csv", index=False)
                
                try:
                    torch.save(expected, f"{GlobalParameters.output_directory}/expected_{file_name}.pt")
                except:
                    np.save(expected, f"{GlobalParameters.output_directory}/expected_{file_name}.npy")

                try:
                    torch.save(out, f"{GlobalParameters.output_directory}/out_{file_name}.pt")
                except:
                    np.save(out, f"{GlobalParameters.output_directory}/out_{file_name}.npy")

                if training:
                    # Save model
                    torch.save(model.state_dict(), f'{GlobalParameters.output_directory}/model_epoch_{e+1}_{file_name}.pt')        
   
    except Exception:
        with open(f'{GlobalParameters.output_directory}log.txt', 'a') as file, contextlib.redirect_stdout(file), contextlib.redirect_stderr(file):
            print(traceback.format_exc())
        print(traceback.format_exc())
        torch.save(model.state_dict(), f'{GlobalParameters.output_directory}/before_error.pt')
        