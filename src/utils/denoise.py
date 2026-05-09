import pywt
import numpy as np
import torch

from typing import Literal

class Improved_thresholding:
    def __init__(self,
                 signals,
                 wavelet="db6",
                 level=5):
        self.signals = signals
        self.wavelet = wavelet
        self.level = level

    def run(self):
        """
        Denoise the ECG signals using the Improved Thresholding technique.
        This technique is presented in the paper:
        https://www.researchgate.net/publication/255626313_ECG_DeNoising_using_improved_thresholding_based_on_Wavelet_transforms
        """
        denoised_signals = torch.zeros_like(self.signals)
        for i in range(self.signals.shape[1]):
            base_signal = self.signals[:, i].numpy()
            coeffs = self.__wavelet_decomposition(base_signal)

            d_hat = [coeffs[0]]
            for j in range(1, len(coeffs)):
                sigma = np.median(np.abs(coeffs[j])) / 0.6745
                threshold = sigma * np.sqrt(2 * np.log(np.linalg.norm(coeffs[j], ord=2)))
                threshold /= np.log(len(coeffs) - j + 1)
                filtered_coeff = np.where(np.abs(coeffs[j]) > threshold, np.sign(coeffs[j]) * (np.abs(coeffs[j]) - threshold), 0)
                d_hat.append(filtered_coeff)
            denoised_signal = pywt.waverec(d_hat, wavelet=self.wavelet)
            denoised_signals[:, i] = torch.tensor(denoised_signal[:self.signals.shape[0]])
        return denoised_signals

    def __wavelet_decomposition(self, base_signal):
        coeffs = pywt.wavedec(data=base_signal,
                              wavelet=self.wavelet,
                              level=self.level)
        return coeffs


def signal_to_noise_ratio(signal, noise):
    signal_power = torch.mean(signal ** 2)
    noise_power = torch.mean(noise ** 2)
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr

def mean_squared_error(signal, noise):
    mse = torch.mean((signal - noise) ** 2)
    return mse

def root_mean_squared_error(signal, noise):
    rmse = torch.sqrt(mean_squared_error(signal, noise))
    return rmse

def percentage_root_mean_squared_difference(signal, noise):
    rmse = root_mean_squared_error(signal, noise)
    prd = (rmse / torch.sqrt(torch.mean(signal ** 2))) * 100
    return prd

class WaveletDenoising:
    def __init__(self,
                 wavelet,
                 threshold: Literal["soft", "hard", "improved"] = "soft",
                 level=5,
                 threshold_selection: Literal["sqtwolog", "heursure", "minimaxi", "rigrsure"] = "sqtwolog",
                 rescaling_approach: Literal["one", "sln", "mln"] = "one"):
        self.wavelet = wavelet
        self.threshold = threshold
        self.level = level
        self.threshold_selection = threshold_selection
        self.rescaling_approach = rescaling_approach

    def run(self, signal):
        coeffs = self.__decompose(signal)
        thresholded_coeffs = self.__thresholding(coeffs)
        denoised_signal = self.__reconstruct(thresholded_coeffs)
        return denoised_signal

    def __decompose(self, signal):
        coeffs = pywt.wavedec(data=signal,
                              wavelet=self.wavelet,
                              level=self.level)
        return coeffs

    def __thresholding(self, coeffs):
        cA = coeffs[0]
        cD = coeffs[1:]
        
        new_coeffs = [cA]
        
        # Rescaling factor based on the selected approach
        cD1 = cD[-1]
        sigma_global = np.median(np.abs(cD1)) / 0.6745

        for i, d_i in enumerate(cD):
            if self.rescaling_approach == "mln":
                sigma = np.median(np.abs(d_i)) / 0.6745
            elif self.rescaling_approach == "sln":
                sigma = sigma_global
            else:
                sigma = 1

            # Determine the threshold based on the selected method
            N = len(d_i)
            if self.threshold_selection == "sqtwolog":
                threshold = sigma * np.sqrt(2 * np.log(N))
            elif self.threshold_selection == "rigrsure":
                sorted_coeffs = np.sort(np.abs(d_i))
                risks = (N - np.arange(1, N + 1)) * sorted_coeffs**2 + np.cumsum(sorted_coeffs**2) - 2 * np.arange(1, N + 1) * sorted_coeffs**2
                threshold = sorted_coeffs[np.argmin(risks)]
            elif self.threshold_selection == "heursure":
                threshold = sigma * np.sqrt(2 * np.log(N))
                if np.sum(d_i**2) < N * sigma**2:
                    threshold = 0
            elif self.threshold_selection == "minimaxi":
                threshold = sigma * 0.3936 + 0.1829 * np.log(N)
            else:
                threshold = sigma * np.sqrt(2 * np.log(N))

            # Apply the selected thresholding method
            if self.threshold == "hard":
                d_thresholded = np.where(np.abs(d_i) > threshold, d_i, 0)
            elif self.threshold == "soft":
                d_thresholded = np.sign(d_i) * np.maximum(np.abs(d_i) - threshold, 0)
            elif self.threshold == "improved":
                d_thresholded = np.where(np.abs(d_i) > threshold, 
                                         np.sign(d_i) * (np.abs(d_i) - threshold),
                                         0)

            new_coeffs.append(d_thresholded)

        return new_coeffs

    def __reconstruct(self, coeffs):
        denoised_signal = pywt.waverec(coeffs, wavelet=self.wavelet)
        return denoised_signal

class Particle_Swarm_Optimization:
    def __init__(self,
                 c1=2.0,
                 c2=2.0,
                 N_particles=5,
                 max_iter=50,
                 population_size=100):
        self.c1 = c1
        self.c2 = c2
        self.N_particles = N_particles
        self.max_iter = max_iter
        self.population_size = population_size

    def optimize(self, signal, noise):
        pass
