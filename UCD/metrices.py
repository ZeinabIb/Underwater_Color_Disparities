import cv2
import numpy as np
import os

def getPSNR(I1, I2):
    s1 = cv2.absdiff(I1, I2)
    s1 = np.float32(s1)
    s1 = s1 * s1
    sse = s1.sum()
    if sse <= 1e-10:
        return 0
    else:
        if len(I1.shape) == 2:  # Grayscale image
            mse = sse / (I1.shape[0] * I1.shape[1])
        else:  # Color image
            mse = sse / (I1.shape[0] * I1.shape[1] * I1.shape[2])

        psnr = 10.0 * np.log10((255 * 255) / mse)
        return psnr

def getSSISM(I1, I2):
    C1 = 6.5025
    C2 = 58.5225

    I1 = np.float32(I1)
    I2 = np.float32(I2)
    I2_2 = I2 * I2
    I1_2 = I1 * I1
    I1_I2 = I1 * I2

    mu1 = cv2.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(I2, (11, 11), 1.5)
    mu1_2 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_2 = cv2.GaussianBlur(I1_2, (11, 11), 1.5) - mu1_2
    sigma2_2 = cv2.GaussianBlur(I2_2, (11, 11), 1.5) - mu2_2
    sigma12 = cv2.GaussianBlur(I1_I2, (11, 11), 1.5) - mu1_mu2

    t1 = 2 * mu1_mu2 + C1
    t2 = 2 * sigma12 + C2
    t3 = t1 * t2
    t1 = mu1_2 + mu2_2 + C1
    t2 = sigma1_2 + sigma2_2 + C2
    ssim_map = cv2.divide(t3, t1 * t2)
    ssim = cv2.mean(ssim_map)
    return ssim[0]

def getUQI(I1, I2):
    # Ensure I1 and I2 are float32
    I1 = np.float32(I1)
    I2 = np.float32(I2)
    
    # Compute means
    mu_X = np.mean(I1)
    mu_Y = np.mean(I2)
    
    # Compute variances
    sigma_X_sq = np.var(I1)
    sigma_Y_sq = np.var(I2)
    
    # Compute covariance
    sigma_XY = np.cov(I1.flatten(), I2.flatten())[0][1]
    
    # Compute UQI
    uqi = (4 * mu_X * mu_Y * sigma_XY) / ((mu_X**2 + mu_Y**2) * (sigma_X_sq + sigma_Y_sq))
    
    return uqi

def process_images_in_folder(original_folder, result_folder):
    original_images = [os.path.join(original_folder, f) for f in os.listdir(original_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    result_images = [os.path.join(result_folder, f) for f in os.listdir(result_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    num_images = len(original_images)
    if num_images == 0:
        print("No images found in the original folder.")
        return
    
    total_psnr = 0
    total_ssim = 0
    total_uqi = 0
    num_pairs = 0
    
    for img_path in original_images:
        img_name = os.path.basename(img_path)
        result_img_path = os.path.join(result_folder, img_name)  # Find corresponding result image
        
        if result_img_path not in result_images:
            print(f"Result image for {img_name} not found.")
            continue
        
        img1 = cv2.imread(img_path)
        img2 = cv2.imread(result_img_path)

        if img1 is None or img2 is None:
            print(f"Could not read images: {img_path} or {result_img_path}")
            continue

        if img1.shape != img2.shape:
            print(f"Image size mismatch between {img_path} and {result_img_path}")
            continue

        if len(img1.shape) == 3:  # Convert color images to grayscale for SSIM, UQI
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if len(img2.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        psnr = getPSNR(img1, img2)
        ssim = getSSISM(img1, img2)
        uqi = getUQI(img1, img2)

        print(f"PSNR between {img_name}: {psnr}")
        print(f"SSIM between {img_name}: {ssim}")
        print(f"UQI between {img_name}: {uqi}")

        total_psnr += psnr
        total_ssim += ssim
        total_uqi += uqi
        num_pairs += 1

    if num_pairs > 0:
        mean_psnr = total_psnr / num_pairs
        mean_ssim = total_ssim / num_pairs
        mean_uqi = total_uqi / num_pairs
        print(f"\nMean PSNR: {mean_psnr}")
        print(f"Mean SSIM: {mean_ssim}")
        print(f"Mean UQI: {mean_uqi}")
    else:
        print("No valid image pairs found.")

# Path to your original and result folders
original_folder = "./data_lr_2x/"
result_folder = "./results/"
process_images_in_folder(original_folder, result_folder)
