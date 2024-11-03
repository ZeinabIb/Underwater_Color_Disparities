import numpy as np
import math
import scipy as sp
import sys
from skimage.restoration import denoise_bilateral
import collections
from skimage.morphology import closing, square,disk
import matplotlib
from PIL import Image
import time
from scipy.optimize import differential_evolution
from matplotlib import pyplot as plt
import scipy
import cv2
from skimage import exposure
matplotlib.use('Agg')
np.random.seed(0)

def find_backscatter_estimation_points(img, depths, num_bins=10, fraction=0.01, max_vals=np.inf, min_depth_percent=0.1):
    z_max, z_min = np.max(depths), np.min(depths)
    min_depth = z_min + (min_depth_percent * (z_max - z_min))
    z_ranges = np.linspace(z_min, z_max, num_bins + 1)
    img_norms = np.stack([img], axis=2)
    img_norms = np.mean(img_norms, axis=2)
    points_r = []
    for i in range(len(z_ranges) - 1):
        a, b = z_ranges[i], z_ranges[i+1]
        locs = np.where(np.logical_and(depths > min_depth, np.logical_and(depths >= a, depths <= b)))
        norms_in_range, px_in_range, depths_in_range = img_norms[locs], img[locs], depths[locs]
        arr = sorted(zip(norms_in_range, px_in_range, depths_in_range), key=lambda x: x[0])
        points = arr[:min(math.ceil(fraction * len(arr)), max_vals)]
        points_r.extend([(z, p) for n, p, z in points])
    return np.array(points_r)

def find_backscatter_values(B_pts, depths, rawdepth,restarts=5):
    np.random.seed(0)
    B_vals, B_depths = B_pts[:, 1], B_pts[:, 0]
    coefs = None
    best_loss = np.inf

    def estimate(depths, B_inf, beta_B, J_prime, beta_D_prime):
        val = (B_inf * (1 - np.exp(-1 * beta_B * depths))) + (J_prime * np.exp(-1 * beta_D_prime * depths))
        return val
    def loss(B_inf, beta_B, J_prime, beta_D_prime):
        val = np.mean(np.abs(B_vals - estimate(B_depths, B_inf, beta_B, J_prime, beta_D_prime)))
        return val

    bounds_lower = [0,0,0,0]
    bounds_upper = [1,5,1,5]
    for _ in range(restarts):
        try:
            optp, pcov = sp.optimize.curve_fit(
                f=estimate,
                xdata=B_depths,
                ydata=B_vals,
                p0=np.random.random(4) * bounds_upper,
                bounds=(bounds_lower, bounds_upper),
            )
            l = loss(*optp)
            if l < best_loss:
                best_loss = l
                coefs = optp

        except:
            slope, intercept, r_value, p_value, std_err = sp.stats.linregress(B_depths, B_vals)
            BD = (slope * depths) + intercept
            return BD,(slope * rawdepth) + intercept, np.array([slope, intercept])
    return estimate(depths, *coefs),estimate(rawdepth, *coefs), coefs

def estimate_illumination(img, B, neighborhood_map, num_neighborhoods, p=0.5, f=2.0, max_iters=1000, tol=1E-5):
    D = img - B
    D_min = np.min(D)
    D_max = np.max(D)
    if D_min < 0:
        D = scale(D)*(D_max-D_min)
    avg_cs = np.zeros_like(img)
    avg_cs_prime = np.copy(avg_cs)
    sizes = np.zeros(num_neighborhoods)
    locs_list = [None] * num_neighborhoods
    for label in range(1, num_neighborhoods + 1):
        locs_list[label - 1] = np.where(neighborhood_map == label)
        sizes[label - 1] = np.size(locs_list[label - 1][0])
    for i in range(max_iters):
        for label in range(1, num_neighborhoods + 1):
            locs = locs_list[label - 1]
            size = sizes[label - 1] - 1
            avg_cs_prime[locs] = (1 / size) * (np.sum(avg_cs[locs]) - avg_cs[locs])
        new_avg_cs = (D * p) + (avg_cs_prime * (1 - p))
        if(np.max(np.abs(avg_cs - new_avg_cs)) < tol):
            break
        avg_cs = new_avg_cs
    return f * denoise_bilateral(np.maximum(0, avg_cs))

def calculate_beta_D(depths, a, b, c, d):
    return (a * np.exp(b * depths)) + (c * np.exp(d * depths))

def refine_wideband_attentuation(depths,rawdepth, illum,estimation,  restarts=5, min_depth_fraction = 0.1, l=1.0):
    np.random.seed(0)
    global rawBD
    eps = 1E-8
    z_max, z_min = np.max(depths), np.min(depths)
    min_depth = z_min + (min_depth_fraction * (z_max - z_min))
    bound = 10
    coefs = None
    best_loss = np.inf
    locs = np.where(np.logical_and(illum > 0, np.logical_and(depths > min_depth, estimation > eps)))
    if np.array(locs).shape[1] ==0:
        coefs = [1,1,1,1]
        BD = np.ones_like(depths)
    else:
        def calculate_reconstructed_depths(depths, a, b, c, d):
            res = np.exp(-depths*(calculate_beta_D(depths, a, b, c, d)))
            return res
        def loss(a, b, c, d):
            return np.mean(np.abs(illum[locs] - calculate_reconstructed_depths(depths[locs], a, b, c, d)))

        for _ in range(restarts):
            try:
                optp, pcov = sp.optimize.curve_fit(
                    f=calculate_reconstructed_depths,
                    xdata=depths[locs],
                    ydata=illum[locs],
                    p0= np.abs(np.random.random(4)) * np.array([1., -1., 1., -1.]),
                    bounds=([0, -bound, 0, -bound], [bound, 0, bound, 0]))
                L = loss(*optp)
                if L < best_loss:
                    best_loss = L
                    coefs = optp
            except:
                slope, intercept, r_value, p_value, std_err = sp.stats.linregress(depths[locs],estimation[locs])
                BD = (slope * depths + intercept)
                rawBD = (slope * rawdepth + intercept)
                return l * BD,l *rawBD, np.array([slope, intercept])
        BD = l * calculate_beta_D(depths, *coefs)
        rawBD = l * calculate_beta_D(rawdepth, *coefs)
    return BD,rawBD, coefs
def ACES1(img,k):
    img = img * k
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    img = (img*(a*img+b))/(img*(c*img+d)+e)
    return img
def scale(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def recover_image(img, depths, B, beta_D):
    D = img - B[:,:,0]
    D_min = np.min(D)
    D_max = np.max(D)
    if D_min < 0:
        D = scale(D) * (D_max - D_min)
    res = D * np.exp(beta_D[:,:,0] * depths)
    return res

def wbalance_gw(img):
    dr = np.average(img[:, :, 0])
    dg = np.average(img[:, :, 1])
    db = np.average(img[:, :, 2])
    dsum = (dr + dg + db)/3
    img[:, :, 0] *= (dsum/dr)
    img[:, :, 1] *= (dsum/dg)
    img[:, :, 2] *= (dsum/db)
    return img


def construct_neighborhood_map(depths, epsilon):
    eps = (np.max(depths) - np.min(depths)) * epsilon
    nmap = np.zeros_like(depths).astype(np.int32)
    n_neighborhoods = 1
    while np.any(nmap == 0):
        locs_x, locs_y = np.where(nmap == 0)
        start_index = np.random.randint(0, len(locs_x))
        start_x, start_y = locs_x[start_index], locs_y[start_index]
        q = collections.deque()
        q.append((start_x, start_y))
        while not len(q) == 0:
            x, y = q.pop()
            if np.abs(depths[x, y] - depths[start_x, start_y]) <= eps:
                nmap[x, y] = n_neighborhoods
                if 0 <= x < depths.shape[0] - 1:
                    x2, y2 = x + 1, y
                    if nmap[x2, y2] == 0:
                        q.append((x2, y2))
                if 1 <= x < depths.shape[0]:
                    x2, y2 = x - 1, y
                    if nmap[x2, y2] == 0:
                        q.append((x2, y2))
                if 0 <= y < depths.shape[1] - 1:
                    x2, y2 = x, y + 1
                    if nmap[x2, y2] == 0:
                        q.append((x2, y2))
                if 1 <= y < depths.shape[1]:
                    x2, y2 = x, y - 1
                    if nmap[x2, y2] == 0:
                        q.append((x2, y2))
        n_neighborhoods += 1
    zeros_size_arr = sorted(zip(*np.unique(nmap[depths == 0], return_counts=True)), key=lambda x: x[1], reverse=True)
    if len(zeros_size_arr) > 0:
        nmap[nmap == zeros_size_arr[0][0]] = 0
    return nmap, n_neighborhoods - 1

def find_closest_label(nmap, start_x, start_y):
    mask = np.zeros_like(nmap).astype(np.bool)
    q = collections.deque()
    q.append((start_x, start_y))
    while not len(q) == 0:
        x, y = q.pop()
        if 0 <= x < nmap.shape[0] and 0 <= y < nmap.shape[1]:
            if nmap[x, y] != 0:
                return nmap[x, y]
            mask[x, y] = True
            if 0 <= x < nmap.shape[0] - 1:
                x2, y2 = x + 1, y
                if not mask[x2, y2]:
                    q.append((x2, y2))
            if 1 <= x < nmap.shape[0]:
                x2, y2 = x - 1, y
                if not mask[x2, y2]:
                    q.append((x2, y2))
            if 0 <= y < nmap.shape[1] - 1:
                x2, y2 = x, y + 1
                if not mask[x2, y2]:
                    q.append((x2, y2))
            if 1 <= y < nmap.shape[1]:
                x2, y2 = x, y - 1
                if not mask[x2, y2]:
                    q.append((x2, y2))



def refine_neighborhood_map(nmap, min_size = 50, radius = 3):
    refined_nmap = np.zeros_like(nmap)
    vals, counts = np.unique(nmap, return_counts=True)
    neighborhood_sizes = sorted(zip(vals, counts), key=lambda x: x[1], reverse=True)
    num_labels = 1
    for label, size in neighborhood_sizes:
        if size >= min_size and label != 0:
            refined_nmap[nmap == label] = num_labels
            num_labels += 1
    for label, size in neighborhood_sizes:
        if size < min_size and label != 0:
            for x, y in zip(*np.where(nmap == label)):
                try:
                    refined_nmap[x, y] = find_closest_label(refined_nmap, x, y)
                except:
                    refined_nmap[x, y] = 0
    refined_nmap = closing(refined_nmap, square(radius))
    return refined_nmap, num_labels - 1
def estimate_wideband_attentuation(depths, illum, radius = 6, max_val = 10.0):
    eps = 1E-8
    BD = np.minimum(max_val, -np.log(illum + eps) / (np.maximum(0, depths) + eps))
    mask = np.where(np.logical_and(depths > eps, illum > eps), 1, 0)
    refined_attenuations = denoise_bilateral(closing(np.maximum(0, BD * mask), disk(radius)))
    return refined_attenuations, []
def run_pipeline(rawimg, rawdepth,p,f,l,ep):
    img = np.array(Image.fromarray(rawimg).resize((32,32)))
    depths = np.resize(rawdepth,(32,32))
    ptsR = find_backscatter_estimation_points(img, depths, fraction=0.01, min_depth_percent=0.1)
    Br,rawBr, coefsR1 = find_backscatter_values(ptsR, depths, rawdepth,restarts=5)
    nmap, _ = construct_neighborhood_map(depths,epsilon=ep)
    nmap, n = refine_neighborhood_map(nmap)
    illR = estimate_illumination(img, Br, nmap, n, p=p, max_iters=1000, tol=1E-5, f=f)
    beta_D_r, _ = estimate_wideband_attentuation(depths, illR)
    refined_beta_D_r, rawBD,coefsR2 = refine_wideband_attentuation(depths,rawdepth, illR,beta_D_r,min_depth_fraction=0, l=l)
    B = np.stack([rawBr], axis=2)
    beta_D = np.stack([rawBD], axis=2)
    res = recover_image(rawimg, rawdepth, B, beta_D)
    return res,coefsR1,coefsR2

