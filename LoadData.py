#!/usr/bin/python
#encoding:utf-8

import SimpleITK as sitk
import numpy as np
import os
import cv2


# def histequ(gray, nlevels=256):
#     # Compute histogram
#     histogram = np.bincount(gray.flatten(), minlength=nlevels)
#     # print("histogram: ", histogram)
#
#     # Mapping function
#     uniform_hist = (nlevels - 1) * (np.cumsum(histogram)/(gray.size * 1.0))
#     uniform_hist = uniform_hist.astype('uint8')
#     # print ("uniform hist: ", uniform_hist)
#
#     # Set the intensity of the pixel in the raw gray to its corresponding new intensity
#     height, width = gray.shape
#     uniform_gray = np.zeros(gray.shape, dtype='uint8')  # Note the type of elements
#     for i in range(height):
#         for j in range(width):
#             uniform_gray[i, j] = uniform_hist[gray[i,j]]
#
#     return uniform_gray


def clahe_equalized(imgs, start, end):
    assert (len(imgs.shape) == 3)  #3D arrays
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(start, end+1):
        imgs_equalized[i, :, :] = clahe.apply(np.array(imgs[i, :, :], dtype=np.uint8))
    return imgs_equalized


def read_img(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    return data


def load_nii_position(filename):
    img = sitk.ReadImage(filename)
    z = img.GetOrigin()[2]
    spacing = img.GetSpacing()[2]
    return z, spacing


def data_generator_xd(paths, hsize, wsize):
    x = []
    y = []
    # maxx = 0
    # minx = 1000
    # maxy = 0
    # miny = 1000
    # maxz = 0
    # minz = 1000
    file = os.listdir(paths)
    for filename in file:
        rpath = os.path.join(filename, 'data.nii.gz')
        spath = os.path.join(filename, 'label.nii.gz')
        r_nda3D = read_img(os.path.join(paths, rpath))  # 获取每个病例的3D数组
        s_nda3D = read_img(os.path.join(paths, spath))

        # t = np.nonzero(s_nda3D)
        # # print(s_nda3D.shape)
        # x1 = np.max(t[0])
        # y1 = np.max(t[1])
        # z1 = np.max(t[2])
        # x2 = np.min(t[0])
        # y2 = np.min(t[1])
        # z2 = np.min(t[2])
        # print("({}_{},{}_{},{}_{})".format(x2, x1, y2, y1, z2, z1))
        # if x1 > maxx:
        #     maxx = x1
        # if x2 < minx:
        #     minx = x2
        # if y1 > maxy:
        #     maxy = y1
        # if y2 < miny:
        #     miny = y2
        # if z1 > maxz:
        #     maxz = z1
        # if z2 < minz:
        #     minz = z2

        # s_nda3D = s_nda3D[np.ix_(range(0, s_nda3D.shape[0]), range(150, 374), range(170, 362))]  # 切取器官所在区域
        # r_nda3D = r_nda3D[np.ix_(range(0, r_nda3D.shape[0]), range(150, 374), range(170, 362))]
        s_nda3D = s_nda3D[np.ix_(range(0, s_nda3D.shape[0]), range(112, 432), range(90, 410))]  # 切取器官所在区域
        r_nda3D = r_nda3D[np.ix_(range(0, r_nda3D.shape[0]), range(112, 432), range(90, 410))]
        # r_nda3D = clahe_equalized(r_nda3D, 0, r_nda3D.shape[0] - 1)
        for j in range(r_nda3D.shape[0]):
            s_nda3D_new = cv2.resize(s_nda3D[j], (wsize, hsize))    # 获取每张切片的2D数组
            r_nda3D_new = cv2.resize(r_nda3D[j], (wsize, hsize))
            # r_nda3D_new = cv2.resize(histequ(r_nda3D[j]), (hsize, wsize))
            x.append(r_nda3D_new)
            y.append(s_nda3D_new)

    print(np.array(x).shape)
    print(np.array(y).shape)
    # print("max x:{}".format(maxx))  # 118
    # print("min x:{}".format(minx))  # 0
    # print("max y:{}".format(maxy))  # 367
    # print("min y:{}".format(miny))  # 156
    # print("max z:{}".format(maxz))  # 359
    # print("min z:{}".format(minz))  # 174

    return np.array(x), np.array(y)


def test_data(paths):
    file = os.listdir(paths)
    for filename in file:
        rpath = os.path.join(filename, 'data.nii.gz')
        spath = os.path.join(filename, 'label.nii.gz')
        r_nda3D = read_img(os.path.join(paths, rpath))  # 获取每个病例的3D数组
        s_nda3D = read_img(os.path.join(paths, spath))


if __name__ == '__main__':
    path = "./data/Thoracic_OAR"
    data, label = data_generator_xd(path, 224, 320)
    print(data, label)

