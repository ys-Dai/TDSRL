import os, json
import numpy as np
import torch
import argparse

import utils.config as config
from compute_F1 import *



# Estimate anomaly scores.
def estimate(test_data, model, post_activation, out_dim, batch_size, window_sliding, divisions,
             check_count=None, device='cpu'):
    # Estimation settings
    window_size = model.max_seq_len * model.patch_size
    assert window_size % window_sliding == 0
    
    n_column = out_dim
    n_batch = batch_size
    batch_sliding = n_batch * window_size
    _batch_sliding = n_batch * window_sliding

    output_values = torch.zeros(len(test_data), n_column, device=device)
    count = 0
    checked_index = np.inf if check_count == None else check_count
    
    # Record output values.
    for division in divisions:
        data_len = division[1] - division[0]
        last_window = data_len - window_size + 1
        _test_data = test_data[division[0]:division[1]]
        _output_values = torch.zeros(data_len, n_column, device=device)
        n_overlap = torch.zeros(data_len, device=device)
    
        with torch.no_grad():
            _first = -batch_sliding
            for first in range(0, last_window-batch_sliding+1, batch_sliding):
                for i in range(first, first+window_size, window_sliding):
                    # Call mini-batch data.
                    x = torch.Tensor(_test_data[i:i+batch_sliding].copy()).reshape(n_batch, window_size, -1).to(device)
                    
                    # Evaludate and record errors.
                    y = post_activation(model(x))
                    _output_values[i:i+batch_sliding] += y.view(-1, n_column)
                    n_overlap[i:i+batch_sliding] += 1

                    count += n_batch

                    if count > checked_index:
                        print(count, 'windows are computed.')
                        checked_index += check_count

                _first = first

            _first += batch_sliding

            for first, last in zip(range(_first, last_window, _batch_sliding),
                                   list(range(_first+_batch_sliding, last_window, _batch_sliding)) + [last_window]):
                # Call mini-batch data.
                x = []
                for i in list(range(first, last-1, window_sliding)) + [last-1]:
                    x.append(torch.Tensor(_test_data[i:i+window_size].copy()))

                # Reconstruct data.
                x = torch.stack(x).to(device)

                # Evaludate and record errors.
                y = post_activation(model(x))
                for i, j in enumerate(list(range(first, last-1, window_sliding)) + [last-1]):
                    _output_values[j:j+window_size] += y[i]
                    n_overlap[j:j+window_size] += 1

                count += n_batch

                if count > checked_index:
                    print(count, 'windows are computed.')
                    checked_index += check_count

            # Compute mean values.
            _output_values = _output_values / n_overlap.unsqueeze(-1)
            
            # Record values for the division.
            output_values[division[0]:division[1]] = _output_values
            
    return output_values


def main(options):
    # Load test data.
    test_data = np.load(config.TEST_DATASET[options.dataset]).copy().astype(np.float32)
    
    # Ignore the specific columns.
    if options.dataset in config.IGNORED_COLUMNS.keys():
        ignored_column = np.array(config.IGNORED_COLUMNS[options.dataset])
        remaining_column = [col for col in range(len(test_data[0])) if col not in ignored_column]
        test_data = test_data[:, remaining_column]
    
    # Load model.
    device = torch.device('cuda:{}'.format(options.gpu_id))
    # device = torch.device('mps')  # ios system
    model = torch.load(options.model_dict, map_location=device)
    if options.parameters_dict != None:
        model.load_state_dict(torch.load(options.parameters_dict, map_location='cpu'))
    model.eval()
    
    # Data division
    data_division = config.DEFAULT_DIVISION[options.dataset] if options.data_division == None else options.data_division 
    if data_division == 'total':
        divisions = [[0, len(test_data)]]
    else:
        with open(config.DATA_DIVISION[options.dataset][data_division], 'r') as f:
            divisions = json.load(f)
        if isinstance(divisions, dict):
            divisions = divisions.values()
            
    n_column = len(test_data[0]) if options.reconstruction_output else 1
    post_activation = torch.nn.Identity().to(device) if options.reconstruction_output\
                      else torch.nn.Sigmoid().to(device)
            
    # Estimate scores.
    output_values = estimate(test_data, model, post_activation, n_column, options.batch_size,
                             options.window_sliding, divisions, options.check_count, device)
    
    # Save predicted anomaly scores.
    output_values = output_values.cpu().numpy()
    if options.save_pre_score:
        outfile_scores = options.parameters_dict[:-3] + '_results.npy' if options.pre_score_outfile == None else options.pre_score_outfile
        np.save(outfile_scores, output_values)

    
    # Compute evaluation metrics results.
    compute(options, output_values)




    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--dataset", default='SWaT', type=str, help='SMAP/SMD/SWaT')
    
    parser.add_argument("--model_dict", default=r'logs\Example_SWaT\Example_SWaT_model.pt', type=str, help='model file (.pt) to estimate')
    parser.add_argument("--parameters_dict", default=r'logs\Example_SWaT\state\Example_SWaT.pt', type=str, help='model parameters file (.pt) to estimate')
    
    parser.add_argument('--save_pre_score', default=False, action='store_true', help='whether save the predicted anomaly scores')
    parser.add_argument("--pre_score_outfile", default=None, type=str, help='output file name (.npy) to predicted save anomaly scores')
    
    parser.add_argument("--data_division", default='total', type=str, help='data division; None(defualt)/channel/class/total')
    parser.add_argument("--check_count", default=5000, type=int, help='check count of window computing')
    
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--window_sliding", default=16, type=int, help='sliding steps of windows; window size should be divisible by this value')
    parser.add_argument('--reconstruction_output', default=False, action='store_true', help='option for reconstruction model (deprecated)')
    
    ## compute evaluation metrics results
    parser.add_argument("--matrics_outfile", default=None, type=str, help='output file name (.txt) to save computation metrics')
    parser.add_argument('--smooth_scores', default=False, action='store_true', help='option for smoothing scores (ewma)')
    parser.add_argument("--smoothing_weight", default=0.9, type=float, help='ewma weight when smoothing socres')
    parser.add_argument('--modified_f1', default=False, action='store_true', help='modified f1 scores (not used now)')
    
    parser.add_argument("--min_anomaly_rate", default=0.001, type=float, help='minimum threshold rate')
    parser.add_argument("--max_anomaly_rate", default=0.3, type=float, help='maximum threshold rate')



    options = parser.parse_args()
    main(options)
