import os
import random
import shutil
import cv2
import numpy as np

org_img_folder = '/data/baisr/bsisr/octnew/label'
def getFileList(dir, Filelist, ext=None):
    """
    获取文件夹及其子文件夹中文件列表
    输入 dir：文件夹根目录
    输入 ext: 扩展名
    返回： 文件路径列表
    """
    newDir = dir
    for s in os.listdir(dir):
            newDir = os.path.join(org_img_folder, s)
            Filelist.append(newDir)

    return Filelist

# 把下面改成你的参数设置
org_img_folder = '/data/baisr/bsisr/octnew/label'  # 待处理源文件夹路径
name = '/data/baisr/bsisr/octnew/test/labelold'
tar_img_folder1 = '/data/baisr/bsisr/octnew/test/labelill1'     # 移动到新文件夹路径
tar_img_folder2 = "/data/baisr/bsisr/octnew/test/labelill2"
tar_img_folder3 = "/data/baisr/bsisr/octnew/test/labelill3"


org_lab_folder = '/data/baisr/bsisr/octnew/test/label'  # 待处理源文件夹路径
tar_lab_folder1 = '/data/baisr/bsisr/octnew/test/labelill1'     # 移动到新文件夹路径
tar_lab_folder2 = "/data/baisr/bsisr/octnew/test/labelill2"
tar_lab_folder3 = "/data/baisr/bsisr/octnew/test/labelill3"

pickpercent = 1  # 需要从源文件夹中抽取的图片比例
img_format = 'png' # 需要处理的图片后缀
i = 1  # 选取后的图片从1开始命名

# 检索源文件夹并随机选择图片
imglist = getFileList(name, [], img_format)  # 获取源文件夹及其子文件夹中图片列表
# imglist1 = getFileList(tar_img_folder1, [], img_format)
picknumber = int(len(imglist)*pickpercent)
#samplelist = random.sample(imglist, picknumber)  # 获取随机抽样后的图片列表
# labellist = getFileList(org_lab_folder, [], img_format)


print('本次执行检索到 ' + str(len(imglist)) + ' 张图像\n')
print('本次共随机抽取百分之 ' + str(pickpercent*100) + '% 的图像\n')
# print(str(len(imglist1)))
# 复制选取好的图片到新文件夹中，并重新命名
#new_img_folder = tar_img_folder1
i = 0;
# for imgpath in imglist:
#     #name = str(i).zfill(5)  # 设置图片名为5位数，即从00001开始重新命名
#     #new_img_folder = os.path.join(tar_img_folder1, name + '.' + img_format)
#     #i = i + 1
#     # 如果不需要重命名就把上面三行注释掉
#     #labpath = imgpath.replace("img", "label")
#     #imglist.remove(imgpath)
#     shutil.copy(imgpath, tar_img_folder1)# 复制图片到新文件夹
#     shutil.copy(imgpath, tar_img_folder2)
#     shutil.copy(imgpath, tar_img_folder3)
#     i += 1

for labpath in imglist:
    #labellist.remove(labpath)
    mask = cv2.imread(labpath, cv2.IMREAD_GRAYSCALE)
    #print(mask)
    #masktrue = cv2.imread('/data/baisr/bsisr/oct/cirrus/masks/Cirrus_TRAIN002_109.png', cv2.IMREAD_GRAYSCALE)
    #print(masktrue)
    parent_dir, file_name = labpath.rsplit('/', 1)
    mask = np.array(mask, np.float32)
    #masktrue = np.array(masktrue, np.float32)
    mask1 = np.zeros_like(mask)
    mask2 = np.zeros_like(mask)
    mask3 = np.zeros_like(mask)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] == 1:
                mask1[i][j] = 255
            elif mask[i][j] == 2:
                mask2[i][j] = 255
            elif mask[i][j] == 3:
                mask3[i][j] = 255

    cv2.imwrite(tar_img_folder1 + '/' + file_name, mask1)
    cv2.imwrite(tar_img_folder2 + '/' + file_name, mask2)
    cv2.imwrite(tar_img_folder3 + '/' + file_name, mask3)


labellist1 = getFileList(tar_lab_folder1, [], img_format)
print(str(len(labellist1)))
# for imgpath in imglist:
#     #name = str(i).zfill(5)  # 设置图片名为5位数，即从00001开始重新命名
#     #new_img_folder = os.path.join(tar_img_folder1, name + '.' + img_format)
#     #i = i + 1
#     # 如果不需要重命名就把上面三行注释
#     labpath = imgpath.replace("img", "label")
#     shutil.copy(imgpath, tar_img_folder3)# 复制图片到新文件夹
#     shutil.copy(labpath, tar_lab_folder3)