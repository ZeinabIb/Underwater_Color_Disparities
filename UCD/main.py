import numpy as np
import cv2
import os
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use("Agg")
from tool import (
    load_image,
    newbluegreen,
    newred,
    ACES,
    ada_color,
    High_pass,
    Overlay,
    maxmap,
)
from seathru import run_pipeline

# from contrast import integral_contrast

import os

img_path = os.path.abspath("./data_lr_2x/")
depth_path = os.path.abspath("./depth_data/")
result_path = os.path.abspath("./results")
data = os.listdir(img_path)
print("data lr 2x loaded")


for i in range(len(data)):
    path = img_path + data[i]
    print(f"Processing {path}")
    print(data[i])
    depth_file = depth_path + "\\" + data[i][:-4] + ".npy"
    print(f"depth_file : {depth_file}")
    depth = np.array(np.load(depth_file)).astype(np.float32)
    print(f"depth  : {depth}")
    i0 = load_image(path)
    (win, hei, _) = i0.shape
    depth = np.resize(depth, (win, hei))
    g = np.mean(i0[:, :, 1])
    b = np.mean(i0[:, :, 2])
    if g > b:
        i0[:, :, 1], coefsR1, coefsR2 = run_pipeline(
            i0[:, :, 1], depth, 0.01, 2, 1, 0.1
        )
        k = 0
    else:
        i0[:, :, 2], coefsR1, coefsR2 = run_pipeline(
            i0[:, :, 2], depth, 0.01, 2, 1, 0.1
        )
        k = 1
    i0 = ACES(i0, 0.4)
    i1 = newbluegreen(i0, 1, k)
    i1 = newred(i1, 1)
    i1 = np.maximum(i1, 0)
    i0 = np.maximum(i0, 0)
    i1 = maxmap(i0, i1, 1.2)
    i1 = np.float32(integral_contrast(i1, 20, 20))
    i1 = np.float32(np.minimum(np.maximum(i1, 0), 1))
    i1 = ada_color(i1)
    i2 = High_pass(i1, 5)
    i1 = Overlay(i1, i2)
    i1 = np.float32(np.minimum(np.maximum(i1, 0), 1))
    plt.imsave(result_path + "%s.jpg" % (str(data[i][:-4])), i1)
