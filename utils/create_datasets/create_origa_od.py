# *coding:utf-8 *

import cv2
import shutil
import os
import scipy.io as scio
import numpy as np

image_folder = '/data/zaiwang/Dataset/ORIGA_OD/650image'
mask_folder = '/data/zaiwang/Dataset/ORIGA_OD/650mask'

crop_image_folder = '/data/zaiwang/Dataset/ORIGA_OD/crop_image'
crop_mask_folder = '/data/zaiwang/Dataset/ORIGA_OD/crop_mask'

if not os.path.exists(crop_image_folder):
    os.mkdir(crop_image_folder)

if not os.path.exists(crop_mask_folder):
    os.mkdir(crop_mask_folder)


def find_OD_center(mask_matric):
    x_cord, y_cord = np.where(mask_matric == 2)
    x_center = int((np.min(x_cord) + np.max(x_cord)) / 2.)
    y_center = int((np.min(y_cord) + np.max(y_cord)) / 2.)

    return x_center, y_center


def crop_origa_od_file():
    for mat_file_name in os.listdir(mask_folder)[:]:
        mat_file_path = os.path.join(mask_folder, mat_file_name)
        print(image_folder)
        img_file_path = os.path.join(image_folder, mat_file_name.split('.')[0] + '.jpg')
        print("img_file_path is ", img_file_path)
        img = cv2.imread(img_file_path)
        print(mat_file_path)
        mat_file = scio.loadmat(mat_file_path)
        mask_file = mat_file['maskFull']
        print("mask_file shape is {}".format(np.shape(mask_file)))
        print(np.unique(mask_file))
        x_center, y_center = find_OD_center(mask_file)
        new_img = img[x_center - 400:x_center + 400, y_center - 400:y_center + 400, :]
        save_path = os.path.join(crop_image_folder, mat_file_name.split('.')[0] + '.jpg')
        cv2.imwrite(save_path, new_img)

        mask = mask_file[x_center - 400:x_center + 400, y_center - 400:y_center + 400]
        crop_mask = np.zeros(shape=np.shape(mask))
        crop_mask[mask == 2] = 255
        save_crop_path = os.path.join(crop_mask_folder, mat_file_name.split('.')[0] + '.png')
        cv2.imwrite(save_crop_path, crop_mask)


if __name__ == '__main__':
    crop_origa_od_file()
