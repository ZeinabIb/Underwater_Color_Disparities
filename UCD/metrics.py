from skimage import metrics
import numpy as np


# Function to compute PSNR
def compute_psnr(original, processed):
    return metrics.peak_signal_noise_ratio(original, processed)


# Function to compute SSIM
def compute_ssim(original, processed):
    return metrics.structural_similarity(original, processed, multichannel=True)


# Function to compute standard deviation
def compute_std(image):
    return np.std(image)


# Function to compute UQIM
def compute_uqim(original, processed):
    # UQIM computation can be complex;
    # placeholder for the actual implementation.
    # You can look for existing libraries for UQIM.
    return np.mean(np.abs(original - processed))  # A placeholder calculation


# Assuming you want to compare `i0` (original) and `i1` (processed)
original_image = "./data_lr_2x/im_xb_6.jpg"  # Change this to the actual original image you want to compare
processed_image = "./results/resultsim_xb_6_.jpg"  # Your processed image

# Calculate and print the metrics
# psnr_value = compute_psnr(original_image, processed_image)
# ssim_value = compute_ssim(original_image, processed_image)
std_value = compute_std(processed_image)
# uqim_value = compute_uqim(original_image, processed_image)

# print(f"PSNR: {psnr_value:.2f} dB")
# print(f"SSIM: {ssim_value:.4f}")
print(f"Standard Deviation: {std_value:.4f}")
# print(f"UQIM: {uqim_value:.4f}")
