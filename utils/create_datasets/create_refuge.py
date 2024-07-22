# *coding:utf-8 *

"""
This code is to create cropped optic disc dataset from the original refuge dataset.
We crop the disc image with the size of 800 * 800.
"""

import os
import cv2
import shutil

image_folder =