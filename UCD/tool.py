import numpy as np
from PIL import Image
from skimage import exposure
import cv2
from skimage.filters import gaussian
from skimage.color import rgb2lab,rgb2gray,lab2rgb
import math
def load_image(img_fname1):
    img = np.array(Image.open(img_fname1))
    img = np.array(img).astype(np.float32)
    return np.float32(img) / 255.0

def cal_lab(img):
    img = (img *255).astype(np.uint8)
    lab = cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
    A= np.mean(np.mean(lab[:,:,1]))/255.0
    B= np.mean(np.mean(lab[:,:,2]))/255.0
    return A,B

def scale(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def img_scale(img):
    out = np.zeros_like(img)
    for i in range(3):
        out[:,:,i] = scale(img[:,:,i])
    return out

def ACES(img,k):
    img = img * k
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    img[:,:,0] = (img[:,:,0]*(a*img[:,:,0]+b))/(img[:,:,0]*(c*img[:,:,0]+d)+e)
    img[:, :, 1] = (img[:, :, 1] * (a * img[:, :, 1] + b)) / (img[:, :, 1] * (c * img[:, :, 1] + d) + e)
    img[:,:,2] = (img[:,:,2]*(a*img[:,:,2]+b))/(img[:,:,2]*(c*img[:,:,2]+d)+e)
    return img

def Rescale_intensity(image_rgb,x):
    p2, p98 = np.percentile(image_rgb[:, :, 0], (x, 100-x))
    image_rgb[:, :, 0] = exposure.rescale_intensity(image_rgb[:, :, 0], in_range=(p2, p98))
    p2, p98 = np.percentile(image_rgb[:, :, 1], (x, 100-x))
    image_rgb[:, :, 1] = exposure.rescale_intensity(image_rgb[:, :, 1], in_range=(p2, p98))
    p2, p98 = np.percentile(image_rgb[:, :, 2], (x, 100-x))
    image_rgb[:, :, 2] = exposure.rescale_intensity(image_rgb[:, :, 2], in_range=(p2, p98))
    return  image_rgb

def color_balance(img,a,b,c):
    midtones_coef = [a, b, c]
    mid_coef_self = np.array([
        [-1, 1, 1],
        [1, -1, 1],
        [1, 1, -1]])
    mid_coef_self1 = np.array(
        [1, 1, 1]
        )
    tmp = np.sort(midtones_coef)
    midtones_coef = midtones_coef - tmp[1]
    midtones_gamma = np.exp(np.dot(mid_coef_self1,np.dot(np.diag(1 * midtones_coef), mid_coef_self)))
    for i in range(3):
        img[:,:,i] = pow(img[:,:,i],midtones_gamma[i])
    i1 = np.minimum(np.maximum(img, 0), 1)
    return i1

def High_pass(img,sigma):
    gauss_out = gaussian(img, sigma=sigma, multichannel=True)
    img_out = (img - gauss_out) + 0.5
    mask_1 = img_out  < 0
    mask_2 = img_out  > 1
    img_out = img_out * (1-mask_1)
    img_out = img_out * (1-mask_2) + mask_2
    return img_out

def Overlay(img_1, img_2):
    mask = img_2 < 0.5
    img = 2 * img_1 * img_2 * mask + (1-mask) * (1- 2 * (1-img_1)*(1-img_2))
    return img


def newbluegreen(i1,a,k):
    ab=0.92
    gb = cv2.compareHist(i1[:, :, 1], i1[:, :, 2], 0)
    b_ = np.mean(i1[:, :, 2])
    g_ = np.mean(i1[:, :, 1])
    n=0
    while gb<ab:
        if k==0:
            i1[:,:,1] = scale(i1[:,:,1])
            i1[:,:,2] = i1[:,:,2] + (n==0)*a*(1-i1[:,:,2])*(g_-b_)*i1[:,:,1] +(n!=0)*(a+ab-gb)*(1-i1[:,:,2])*(g_-b_)*i1[:,:,1]
        else:
            i1[:, :, 2] = scale(i1[:,:,2])
            i1[:, :, 1] = i1[:, :, 1] + (n==0)*a*(1 - i1[:, :, 1]) * (b_ - g_) * i1[:, :, 2]+(n!=0)*(a+ab-gb)*(1 - i1[:, :, 1]) * (b_ - g_) * i1[:, :, 2]
        n+=1
        gb_ = cv2.compareHist(i1[:, :, 1], i1[:, :, 2], 0)
        if gb_<gb or gb_==gb:
            break
        else:
            gb = gb_
    return i1

def newred(i1,a):
    rg = cv2.compareHist(i1[:, :, 0], i1[:, :, 1], 0)
    r_ = np.mean(i1[:, :, 0])
    g_ = np.mean(i1[:, :, 1])
    n=0
    ab = 0.99
    while rg < ab:
        i1[:,:,1] = scale(i1[:,:,1])
        i1[:,:,0] = i1[:,:,0] + (n==0)*a*(1-i1[:,:,0])*(g_-r_)*i1[:,:,1] + (n!=0)*(a+ab-rg)*(1-i1[:,:,0])*(g_-r_)*i1[:,:,1]
        n+=1
        rg_ = cv2.compareHist(i1[:, :, 1], i1[:,:,0], 0)
        if rg_ < rg or rg_ == rg or r_>g_:
            break
        else:
            rg = rg_
    return i1

def ada_color(img):
    A,B = cal_lab(img)
    As = [126/255.0,134/255.0]
    Bs = [128/255.0,140/255.0] 
    n = 0
    while A<As[0] or A>As[1] or B<Bs[0] or B>Bs[1]:
        if A < As[0] :
            if B < Bs[0]:
                a = (As[0] - A) + Bs[0] -B
                a = a/0.2
                img = color_balance(img, a, 0, 0)
                A, B = cal_lab(img)
            elif B > Bs[1]:
                    b = A - As[0]
                    b = b / 0.2
                    c = B - Bs[1]
                    c = c / 0.2
                    img = color_balance(img, 0, b, c)
                    A, B = cal_lab(img)
            else:
                a = As[0] - A
                a = a / 0.2
                b = A - As[0]
                b = b / 0.2
                img = color_balance(img, a, b, 0)
                A, B = cal_lab(img)
        if A > As[1]:
            if B < Bs[0]:
                    b = A-As[1]
                    b = b / 0.2
                    c = B - Bs[0]
                    c = c / 0.2
                    img = color_balance(img, 0, b, c)
                    A, B = cal_lab(img)
            elif B > Bs[1]:
                a = (As[1] - A) + Bs[1]-B
                a = a / 0.2
                img = color_balance(img, a, 0, 0)
                A, B = cal_lab(img)
            else:
                a = As[1] - A
                a = a / 0.2
                b = A - As[1]
                b = b / 0.2
                img = color_balance(img, a, b, 0)
                A, B = cal_lab(img)

        if B < Bs[0] :
            if A < As[0]:
                a = (Bs[0]-B) + As[0]-A
                a = a / 0.2
                img = color_balance(img, a, 0, 0)
                A, B = cal_lab(img)
            elif A > As[1]:
                    b = A - As[1]
                    b = b / 0.2
                    c = B - Bs[0]
                    c = c / 0.2
                    img = color_balance(img, 0, b, c)
                    A, B = cal_lab(img)
            else:
                c = B - Bs[0]
                c = c / 0.2
                img = color_balance(img, 0, 0, c)
                A, B = cal_lab(img)
        if B > Bs[1]:
            if A < As[0]:
                    b = A - As[0]
                    b = b / 0.2
                    c = B - Bs[1]
                    c = c / 0.2
                    img = color_balance(img, 0, b, c)
                    A, B = cal_lab(img)
            elif A > As[1]:
                a = Bs[1] - B + As[1]-A
                a = a / 0.2
                img = color_balance(img, a, 0, 0)
                A, B = cal_lab(img)
            else:
                c = B - Bs[1]
                c = c / 0.2
                img = color_balance(img, 0, 0, c)
                A, B = cal_lab(img)
        n+=1
        if n > 30:
            break
    return img

def maxmap(img,CT,gamma):
    Amax = np.array([1 - img[:, :, 0] ** gamma, 1 - img[:, :, 1] ** gamma,1 - img[:, :, 2] ** gamma]).max(axis=0)
    D = np.zeros_like(img)
    for i in range(3):
        D[:,:,i] = img[:,:,i] - gaussian(img[:,:,i])
        img[:, :, i] = D[:,:,i] + Amax*CT[:, :, i]+ (1-Amax)* img[:, :, i]
    img = Rescale_intensity(img,2)
    return img


