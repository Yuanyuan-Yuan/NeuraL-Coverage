# source: https://github.com/ARiSE-Lab/deepTest/blob/master/testgen/epoch_testgen_coverage.py

import numpy as np
import cv2

# def pad(src, dst):
#     H_src, W_src, C_src = src.shape
#     H_dst, W_dst, C_dst = dst.shape
#     top = np.max((H_src - H_dst) // 2, 0)
#     bottom = top
#     left = np.max((W_src - W_dst) // 2, 0)
#     right = left
#     COLOR = [255, 255, 255] # white
#     padded = cv2.copyMakeBorder(dst, top, bottom, left, right, cv2.BORDER_CONSTANT, value=COLOR)
#     return padded

def pad(src, dst):
    H_src, W_src, C = src.shape
    H_dst, W_dst, C = dst.shape
    COLOR = (255, 255, 255)
    padded = np.full((H_src, W_src, C), COLOR, dtype=np.uint8)
    top = np.max((H_src - H_dst) // 2, 0)
    left = np.max((W_src - W_dst) // 2, 0)
    padded[top:top+H_dst, left:left+W_dst] = dst
    return padded


def image_translation(img, params):
    rows, cols, ch = img.shape
    M = np.float32([[1, 0, params[0]], [0, 1, params[1]]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def image_scale(img, params):
    res = cv2.resize(img, None, fx=params, fy=params, interpolation=cv2.INTER_CUBIC)
    res = pad(img, res)
    return res

def image_shear(img, params):
    rows, cols, ch = img.shape
    factor = params*(-1.0)
    M = np.float32([[1, factor, 0], [0, 1, 0]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    dst = pad(img, dst)
    return dst

def image_rotation(img, params):
    rows, cols, ch = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), params, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def image_contrast(img, params):
    alpha = params
    # new_img = cv2.multiply(img, np.array([alpha]))                    # mul_img = img*alpha
    new_img = cv2.convertScaleAbs(img, beta=0, alpha=alpha)
    return new_img

def image_brightness(img, params):
    beta = params
    # new_img = cv2.add(img, beta)                                  # new_img = img + beta
    new_img = cv2.convertScaleAbs(img, beta=beta, alpha=1)
    return new_img

def image_blur(img, params):
    img_type = img.dtype
    if(np.issubdtype(img_type, np.integer)):
        img = np.uint8(img)
    else:
        img = np.float32(img)
    
    blur = []
    if params == 1:
        blur = cv2.blur(img, (3, 3))
    if params == 2:
        blur = cv2.blur(img, (4, 4))
    if params == 3:
        blur = cv2.blur(img, (5, 5))
    if params == 4:
        blur = cv2.GaussianBlur(img, (3, 3), 0)
    if params == 5:
        blur = cv2.GaussianBlur(img, (5, 5), 0)
    if params == 6:
        blur = cv2.GaussianBlur(img, (7, 7), 0)
    if params == 7:
        blur = cv2.medianBlur(img, 3)
    if params == 8:
        blur = cv2.medianBlur(img, 5)
    if params == 9:
        blur = cv2.blur(img, (6, 6))
    if params == 10:
        blur = cv2.bilateralFilter(img, 9, 75, 75)
    
    blur = blur.astype(img_type)
    return blur
