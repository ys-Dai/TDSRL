import torch
import torch.nn.functional as F

import numpy as np
import torch.fft as fft


def F_banks(x, channels, bank, num_perturbation_position, enlarge_min, enlarge_max, device):
    augmented_signals = x.clone()  

    for channel in channels:
        original_signal_tensor = x[channel]  
        original_signal = original_signal_tensor.cpu().numpy()

        spectrum_ori = np.fft.fft(original_signal) 

        num_additional_frequencies = num_perturbation_position
        positive_fre = np.arange(len(original_signal)//2)   

        if bank == 'low':
            freq_indices = positive_fre[:len(positive_fre)//2] 
        elif bank == 'high':
            freq_indices = positive_fre[len(positive_fre)//2:] 
        elif bank == 'mixture':
            freq_indices = positive_fre

        spectrum_new = spectrum_ori.copy()

        magnitude = np.abs(spectrum_ori)    
        max_magnitude = magnitude.max()
        max_index = np.argmax(magnitude)  
        symmetrical_index = len(spectrum_ori) - max_index  

        selected_indices = np.random.choice(freq_indices, size=num_additional_frequencies, replace=False)  

        for random_index in selected_indices:
            enlarge_factor  = np.random.uniform(enlarge_min, enlarge_max)  
            component = max_magnitude * enlarge_factor

            if random_index != max_index and random_index != symmetrical_index:
                phase_pos = np.angle(spectrum_new[random_index])
                phase_neg = np.angle(spectrum_new[-random_index])

                spectrum_new[random_index] = component * np.exp(1j * phase_pos)
                spectrum_new[-random_index] = component * np.exp(1j * phase_neg)  

        new_signal = np.fft.ifft(spectrum_new).real

        new_signal_tensor = torch.tensor(new_signal).to(device)

        augmented_signals[channel] = new_signal_tensor


    return augmented_signals


def F_Components_Scaling(x, channels, device):
    augmented_signals = x.clone()  

    for channel in channels:
        original_signal_tensor = x[channel]  
        original_signal = original_signal_tensor.cpu().numpy()

        spectrum_ori = np.fft.fft(original_signal) 
   
        magnitude = np.abs(spectrum_ori)
        threshold = 1e-5  
        significant_indices = np.where(magnitude > threshold)[0]  
        significant_pos_indices = significant_indices[:len(significant_indices)//2] 

        spectrum_new = spectrum_ori.copy()

        for random_index in significant_pos_indices:
 
            scaling_factor = np.random.uniform(0, 2)

            spectrum_new[random_index] *= scaling_factor
            spectrum_new[-random_index] *= scaling_factor

        new_signal = np.fft.ifft(spectrum_new).real

        new_signal_tensor = torch.tensor(new_signal).to(device)

        augmented_signals[channel] = new_signal_tensor


    return augmented_signals