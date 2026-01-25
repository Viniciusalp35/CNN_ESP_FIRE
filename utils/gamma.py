import numpy as np


def gamma_correction(image, gamma=1.0):
    invGamma = 1.0 / gamma
    gamma_corrected = np.array(((image / 255) ** invGamma) * 255).astype(np.uint8)
    return gamma_corrected
