import json, os
import numpy as np
import argparse

import utils.config as config



# Exponential weighted moving average
def ewma(series, weighting_factor=0.9):
    current_factor = 1 - weighting_factor
    _ewma = series.copy()
    for i in range(1, len(_ewma)):
        _ewma[i] = _ewma[i-1] * weighting_factor + _ewma[i] * current_factor
    return _ewma


# Get anomaly sequences.
def anomaly_sequence(label):
    anomaly_args = np.argwhere(label).flatten()  # Indices for abnormal points.
    
    # Terms between abnormal invervals
    terms = anomaly_args[1:] - anomaly_args[:-1]
    terms = terms > 1

    # Extract anomaly sequences.
    sequence_args = np.argwhere(terms).flatten() + 1
    sequence_length = list(sequence_args[1:] - sequence_args[:-1])
    sequence_args = list(sequence_args)

    sequence_args.insert(0, 0)
    if len(sequence_args) > 1:
        sequence_length.insert(0, sequence_args[1])
    sequence_length.append(len(anomaly_args) - sequence_args[-1])

    # Get anomaly sequence arguments.
    sequence_args = anomaly_args[sequence_args]
    anomaly_label_seq = np.transpose(np.array((sequence_args, sequence_args + np.array(sequence_length))))
    return anomaly_label_seq, sequence_length


# Interval-dependent point
def interval_dependent_point(sequences, lengths):
    n_intervals = len(sequences)
    n_steps = np.sum(lengths)
    return (n_steps / n_intervals) / lengths


def f1_score(gt, pr, anomaly_rate=0.05, adjust=True, modify=False):
    # get anomaly intervals
    gt_aug = np.concatenate([np.zeros(1), gt, np.zeros(1)]).astype(np.int32)
    gt_diff = gt_aug[1:] - gt_aug[:-1]

    begin = np.where(gt_diff == 1)[0]
    end = np.where(gt_diff == -1)[0]

    intervals = np.stack([begin, end], axis=1)

    # quantile cut
    pa = pr.copy()
    q = np.quantile(pa, 1-anomaly_rate)
    pa = (pa > q).astype(np.int32)
    
    # Modified F1
    if modify:
        gt_seq_args, gt_seq_lens = anomaly_sequence(gt)  # gt anomaly sequence args
        ind_p = interval_dependent_point(gt_seq_args, gt_seq_lens)  # interval-dependent point
        
        # Compute TP and FN.
        TP = 0
        FN = 0
        for _seq, _len, _p in zip(gt_seq_args, gt_seq_lens, ind_p):
            n_tp = pa[_seq[0]:_seq[1]].sum()
            n_fn = _len - n_tp
            TP += n_tp * _p
            FN += n_fn * _p
            
        # Compute TN and FP.
        TN = ((1 - gt) * (1 - pa)).sum()
        FP = ((1 - gt) * pa).sum()

    else:
        # point adjustment
        if adjust:
            for s, e in intervals:
                interval = slice(s, e)
                if pa[interval].sum() > 0:
                    pa[interval] = 1

        # confusion matrix
        TP = (gt * pa).sum()
        TN = ((1 - gt) * (1 - pa)).sum()
        FP = ((1 - gt) * pa).sum()
        FN = (gt * (1 - pa)).sum()

        assert (TP + TN + FP + FN) == len(gt)

    # Compute p, r, f1.
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2*precision*recall/(precision+recall)

    return precision, recall, f1_score




# Compute evaluation metrics results.
def compute(options, result):
    # Load test data, estimation results, and label.
    test_data = np.load(config.TEST_DATASET[options.dataset])
    test_label = np.load(config.TEST_LABEL[options.dataset]).copy().astype(np.int32)
    data_dim = len(test_data[0])

    if options.data_division == 'total':
        divisions = [[0, len(test_data)]]
    else:
        with open(config.DATA_DIVISION[options.dataset][options.data_division], 'r') as f:
            divisions = json.load(f)
        if isinstance(divisions, dict):
            divisions = divisions.values()
        
    # output_values = np.load(result)
    output_values = result
    if output_values.ndim == 2:
        output_values = output_values[:, 0]
    
    if options.smooth_scores:
        smoothed_values = ewma(output_values, options.smoothing_weight)
        
    # Result text file
    if options.matrics_outfile == None:
        prefix = options.parameters_dict[:-3]
        result_file = prefix + '_metrics.txt'
    else:
        prefix = options.matrics_outfile[:-4]
        result_file = options.matrics_outfile
    result_file = open(result_file, 'w')
        
    

        
    # Compute F1-scores.
    
    if not options.modified_f1:
        
        if options.data_division == 'total':
            best_eval = (0, 0, 0)
            best_rate = 0
            for rate in np.arange(options.min_anomaly_rate, options.max_anomaly_rate+0.001, 0.001):
                evaluation = f1_score(test_label, output_values, rate, True)
                # result_file.write(f'anomaly rate: {rate:.3f} | precision: {evaluation[0]:.5f} | recall: {evaluation[1]:.5f} | F1-score: {evaluation[2]:.5f}\n')
                if evaluation[2] > best_eval[2]:
                    best_eval = evaluation
                    best_rate = rate
            result_file.write('Evaluation results\n')
            result_file.write(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | F1-score: {best_eval[2]:.5f}\n\n\n')
            print('Evaluation results')
            print(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | F1-score: {best_eval[2]:.5f}\n')
            
        else:
            average_eval = np.zeros(3)
            for division in divisions:
                _test_label = test_label[division[0]:division[1]]
                _output_values = output_values[division[0]:division[1]]
                best_eval = (0, 0, 0)
                for rate in np.arange(options.min_anomaly_rate, options.max_anomaly_rate+0.001, 0.001):
                    evaluation = f1_score(_test_label, _output_values, rate, True)
                    if evaluation[2] > best_eval[2]:
                        best_eval = evaluation
                average_eval += np.array(best_eval)
            average_eval /= len(divisions)
            result_file.write('Evaluation results\n')
            result_file.write(f'precision: {average_eval[0]:.5f} | recall: {average_eval[1]:.5f} | F1-score: {average_eval[2]:.5f}\n\n\n')
            print('Evaluation results')
            print(f'precision: {average_eval[0]:.5f} | recall: {average_eval[1]:.5f} | F1-score: {average_eval[2]:.5f}\n')
    

    if options.smooth_scores:
                                    
        if not options.modified_f1:
            result_file.write('<Evaluation results with smoothed scores>\n\n')
            best_eval = (0, 0, 0)
            best_rate = 0
            for rate in np.arange(options.min_anomaly_rate, options.max_anomaly_rate+0.001, 0.001):
                evaluation = f1_score(test_label, smoothed_values, rate, True)
                result_file.write(f'anomaly rate: {rate:.3f} | precision: {evaluation[0]:.5f} | recall: {evaluation[1]:.5f} | F1-score: {evaluation[2]:.5f}\n')
                if evaluation[2] > best_eval[2]:
                    best_eval = evaluation
                    best_rate = rate
            result_file.write('Evaluation results with smoothed scores\n')
            result_file.write(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | F1-score: {best_eval[2]:.5f}\n\n\n')
            print('Evaluation results with smoothed scores')
            print(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | F1-score: {best_eval[2]:.5f}\n')
    
    # Close file.
    result_file.close()
        
