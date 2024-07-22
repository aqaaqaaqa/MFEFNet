"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation
"""
import torch
import torch.utils.data as data
from torch.autograd import Variable as V
from PIL import Image
import clip
import cv2
import numpy as np
import os
import scipy.misc as misc
import matplotlib.pyplot as plt

prompt = ['a batch of medical image need to segment Hard Exudates out in DDR dataset',
          'a batch of medical image need to segment Haemorrhages out in DDR dataset, please pay more attention to the part',
          'a batch of medical image need to segment Microaneurysms out in DDR dataset, please pay more attention to the part',
          'a batch of medical image need to segment Soft Exudates out in DDR dataset',
          'a batch of medical image need to segment Haemorrhages out in idrid dataset',
          'a batch of medical image need to segment Hard Exudates out in idrid dataset',
          'a batch of medical image need to segment Microaneurysms out in idrid dataset',
          'a batch of medical image need to segment Soft Exudates out in idrid dataset',
          'a batch of medical image need to segment Intraretinal Fluid out in OCT dataset',
          'a batch of medical image need to segment Subretinal Fluid out in OCT dataset',
          'a batch of medical image need to segment Pigment Epithelium Detachment out in OCT dataset'
          ]
device = "cuda:2"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
text = clip.tokenize(prompt).to(device)
with torch.no_grad():
    text_features = model.encode_text(text).to(device)
text_features = text_features.float()

dataid = 0


def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        # image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0),
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask


def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.rot90(image)
        mask = np.rot90(mask)

    return image, mask


def default_loader(img_path, mask_path):
    img = cv2.imread(img_path)
    # print("img:{}".format(np.shape(img)))
    img = cv2.resize(img, (448, 448))

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    mask = 255. - cv2.resize(mask, (448, 448))

    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    img, mask = randomShiftScaleRotate(img, mask,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)

    mask = np.expand_dims(mask, axis=2)
    #
    # print(np.shape(img))
    # print(np.shape(mask))

    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    # mask = abs(mask-1)
    return img, mask


def default_DRIVE_loader(img_path, mask_pathEX, mask_pathHE, mask_pathMA, mask_pathSE):
    img = cv2.imread(img_path)
    #img = cv2.resize(img, (448, 448))
    # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    maskEX = np.array(Image.open(mask_pathEX))
    maskHE = np.array(Image.open(mask_pathHE))
    maskMA = np.array(Image.open(mask_pathMA))
    maskSE = np.array(Image.open(mask_pathSE))

    #mask = cv2.resize(mask, (448, 448))



    maskEX = np.array(maskEX, np.float32) / 255.0
    maskHE = np.array(maskHE, np.float32) / 255.0
    maskMA = np.array(maskMA, np.float32) / 255.0
    maskSE = np.array(maskSE, np.float32) / 255.0

    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    # mask = abs(mask-1)
    return img, mask


def default_DRIVE_loaderidrid(img_path, mask_path):
    img = cv2.imread(img_path)
    # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = np.array(Image.open(mask_path))

    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    img, mask = randomShiftScaleRotate(img, mask,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)

    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2, 0, 1)
    # mask = abs(mask-1)
    return img, mask


def read_DDR_person(root_path= '', mode='train'):
    images = []
    masksEX = []
    masksHE = []
    masksMA = []
    masksSE = []

    if mode == 'train':
        image_root = '/data/baisr/bsisr/DDR_lesion_segmentation/train/image'
        gt_rootEX = '/data/baisr/bsisr/DDR_lesion_segmentation/train/label/EX'
        gt_rootHE = '/data/baisr/bsisr/DDR_lesion_segmentation/train/label/HE'
        gt_rootMA = '/data/baisr/bsisr/DDR_lesion_segmentation/train/label/MA'
        gt_rootSE = '/data/baisr/bsisr/DDR_lesion_segmentation/train/label/SE'


    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name)
        label_pathEX = os.path.join(gt_rootEX, image_name.replace('jpg', 'tif'))
        label_pathHE = os.path.join(gt_rootHE, image_name.replace('jpg', 'tif'))
        label_pathMA = os.path.join(gt_rootMA, image_name.replace('jpg', 'tif'))
        label_pathSE = os.path.join(gt_rootSE, image_name.replace('jpg', 'tif'))

        if cv2.imread(image_path) is not None:
            # if os.path.exists(image_path) and os.path.exists(label_path):
            images.append(image_path)
            masksEX.append(label_pathEX)
            masksHE.append(label_pathHE)
            masksMA.append(label_pathMA)
            masksSE.append(label_pathSE)


    return images, masksEX, masksHE,masksMA, masksSE


def read_DDRHE_person(root_path, mode='train'):
    images = []
    masks = []
    dataid = []

    if mode == 'train':
        image_root = '/data/baisr/bsisr/DDR_lesion_segmentation/train/image'
        gt_root = '/data/baisr/bsisr/DDR_lesion_segmentation/train/label/HE'
    else:
        image_root = '../eyesegment/DDR_lesion_segmentation/test/image'
        gt_root = '../eyesegment/DDR_lesion_segmentation/test/label/HE'

    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name)
        label_path = os.path.join(gt_root, image_name.replace('jpg', 'tif'))

        if cv2.imread(image_path) is not None:
            # if os.path.exists(image_path) and os.path.exists(label_path):
            images.append(image_path)
            masks.append(label_path)
            dataid.append(1)

    return images, masks, dataid


def read_DDRMA_person(root_path, mode='train'):
    images = []
    masks = []
    dataid = []
    if mode == 'train':
        image_root = '/data/baisr/bsisr/DDR_lesion_segmentation/train/image'
        gt_root = '/data/baisr/bsisr/DDR_lesion_segmentation/train/label/MA'
    else:
        image_root = '../eyesegment/DDR_lesion_segmentation/test/image'
        gt_root = '../eyesegment/DDR_lesion_segmentation/test/label/MA'

    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name)
        label_path = os.path.join(gt_root, image_name.replace('jpg', 'tif'))

        if cv2.imread(image_path) is not None:
            # if os.path.exists(image_path) and os.path.exists(label_path):
            images.append(image_path)
            masks.append(label_path)
            dataid.append(2)

    return images, masks, dataid


def read_DDRSE_person(root_path, mode='train'):
    images = []
    masks = []
    dataid = []

    if mode == 'train':
        image_root = '/data/baisr/bsisr/DDR_lesion_segmentation/train/image'
        gt_root = '/data/baisr/bsisr/DDR_lesion_segmentation/train/label/SE'
    else:
        image_root = '../eyesegment/DDR_lesion_segmentation/test/image'
        gt_root = '../eyesegment/DDR_lesion_segmentation/test/label/SE'

    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name)
        label_path = os.path.join(gt_root, image_name.replace('jpg', 'tif'))

        if cv2.imread(image_path) is not None:
            # if os.path.exists(image_path) and os.path.exists(label_path):
            images.append(image_path)
            masks.append(label_path)
            dataid.append(3)

    return images, masks, dataid


def read_idrid_person(root_path, mode='train'):
    images = []
    masks = []
    dataid = []

    if mode == 'train':
        image_root = '/data/baisr/bsisr/idrid_Segmentation/1. Original Images/a. Training Set'
        gt_root = '/data/baisr/bsisr/idrid_Segmentation/2. All Segmentation Groundtruths/a. Training Set/Hard Exudates'
    for label_name in os.listdir(gt_root):
        image_path = os.path.join(image_root, label_name.replace('_EX.tif', '.jpg'))
        label_path = os.path.join(gt_root, label_name)

        if cv2.imread(image_path) is not None:
            # if os.path.exists(image_path) and os.path.exists(label_path):
            images.append(image_path)
            masks.append(label_path)
            dataid.append(0)

    return images, masks, dataid


def read_idridHE_person(root_path, mode='train'):
    images = []
    masks = []
    dataid = []

    if mode == 'train':
        image_root = '/data/baisr/bsisr/idrid_Segmentation/1. Original Images/a. Training Set'
        gt_root = '/data/baisr/bsisr/idrid_Segmentation/2. All Segmentation Groundtruths/a. Training Set/Haemorrhages'
    else:
        image_root = '/data/baisr/bsisr/idrid_Segmentation/1. Original Images/b. Testing Set'
        gt_root = '/data/baisr/bsisr/idrid_Segmentation/2. All Segmentation Groundtruths/b. Testing Set/Haemorrhages'

    for label_name in os.listdir(gt_root):
        image_path = os.path.join(image_root, label_name.replace('_HE.tif', '.jpg'))
        label_path = os.path.join(gt_root, label_name)

        if cv2.imread(image_path) is not None:
            # if os.path.exists(image_path) and os.path.exists(label_path):
            images.append(image_path)
            masks.append(label_path)
            dataid.append(1)

    return images, masks, dataid


def read_idridMA_person(root_path, mode='train'):
    images = []
    masks = []
    dataid = []

    if mode == 'train':
        image_root = '/data/baisr/bsisr/idrid_Segmentation/1. Original Images/a. Training Set'
        gt_root = '/data/baisr/bsisr/idrid_Segmentation/2. All Segmentation Groundtruths/a. Training Set/Microaneurysms'
    else:
        image_root = '/data/baisr/bsisr/idrid_Segmentation/1. Original Images/b. Testing Set'
        gt_root = '/data/baisr/bsisr/idrid_Segmentation/2. All Segmentation Groundtruths/b. Testing Set/Microaneurysms'

    for label_name in os.listdir(gt_root):
        image_path = os.path.join(image_root, label_name.replace('_MA.tif', '.jpg'))
        label_path = os.path.join(gt_root, label_name)

        if cv2.imread(image_path) is not None:
            # if os.path.exists(image_path) and os.path.exists(label_path):
            images.append(image_path)
            masks.append(label_path)
            dataid.append(2)

    return images, masks, dataid


# def read_idridOD_person(root_path, mode='train'):
#     images = []
#     masks = []
#     dataid = []
#
#     if mode == 'train':
#         image_root = '/data/baisr/bsisr/idrid_Segmentation/1. Original Images/a. Training Set'.replace('/data/baisr/', '/vepfs/d_bsr/')
#         gt_root = '/data/baisr/bsisr/idrid_Segmentation/2. All Segmentation Groundtruths/a. Training Set/Optic Disc'.replace('/data/baisr/', '/vepfs/d_bsr/')
#     else:
#         image_root = '/data/baisr/bsisr/idrid_Segmentation/1. Original Images/b. Testing Set'.replace('/data/baisr/', '/vepfs/d_bsr/')
#         gt_root = '/data/baisr/bsisr/idrid_Segmentation/2. All Segmentation Groundtruths/b. Testing Set/Optic Disc'.replace('/data/baisr/', '/vepfs/d_bsr/')
#
#     for label_name in os.listdir(gt_root):
#         image_path = os.path.join(image_root, label_name.replace('_OD.tif', '.jpg'))
#         label_path = os.path.join(gt_root, label_name)
#
#         if cv2.imread(image_path) is not None:
#
#             #if os.path.exists(image_path) and os.path.exists(label_path):
#                 images.append(image_path)
#                 masks.append(label_path)
#                 dataid.append(7)
#
#     return images, masks,dataid

def read_idridSE_person(root_path, mode='train'):
    images = []
    masks = []
    dataid = []

    if mode == 'train':
        image_root = '/data/baisr/bsisr/idrid_Segmentation/1. Original Images/a. Training Set'
        gt_root = '/data/baisr/bsisr/idrid_Segmentation/2. All Segmentation Groundtruths/a. Training Set/Soft Exudates'
    else:
        image_root = '/data/baisr/bsisr/idrid_Segmentation/1. Original Images/b. Testing Set'
        gt_root = '/data/baisr/bsisr/idrid_Segmentation/2. All Segmentation Groundtruths/b. Testing Set/Soft Exudates'

    for label_name in os.listdir(gt_root):
        image_path = os.path.join(image_root, label_name.replace('_SE.tif', '.jpg'))
        label_path = os.path.join(gt_root, label_name)

        if cv2.imread(image_path) is not None:
            # if os.path.exists(image_path) and os.path.exists(label_path):
            images.append(image_path)
            masks.append(label_path)
            dataid.append(3)

    return images, masks, dataid


def read_OCTill1_person(root_path, mode='train'):
    images = []
    masks = []
    dataid = []

    if mode == 'train':
        image_root = '/data/baisr/bsisr/octnew/train/imgill1'
        gt_root = '/data/baisr/bsisr/octnew/train/labelill1'
    else:
        image_root = '/data/baisr/bsisr/octnew/test/imgill1'
        gt_root = '/data/baisr/bsisr/octnew/test/labelill1'

    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name)
        label_path = os.path.join(gt_root, image_name)

        if cv2.imread(image_path) is not None:
            # if os.path.exists(image_path) and os.path.exists(label_path):
            images.append(image_path)
            masks.append(label_path)
            dataid.append(0)
    # dataid = 8
    return images, masks, dataid


def read_OCTill2_person(root_path, mode='train'):
    images = []
    masks = []
    dataid = []

    if mode == 'train':
        image_root = '/data/baisr/bsisr/octnew/train/imgill2'
        gt_root = '/data/baisr/bsisr/octnew/train/labelill2'
    else:
        image_root = '/data/baisr/bsisr/octnew/test/imgill2'
        gt_root = '/data/baisr/bsisr/octnew/test/labelill2'

    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name)
        label_path = os.path.join(gt_root, image_name)

        if cv2.imread(image_path) is not None:
            # if os.path.exists(image_path) and os.path.exists(label_path):
            images.append(image_path)
            masks.append(label_path)
            dataid.append(1)
    # dataid = 9
    return images, masks, dataid


def read_OCTill3_person(root_path, mode='train'):
    images = []
    masks = []
    dataid = []

    if mode == 'train':
        image_root = '/data/baisr/bsisr/octnew/train/imgill3'
        gt_root = '/data/baisr/bsisr/octnew/train/labelill3'
    else:
        image_root = '/data/baisr/bsisr/octnew/test/imgill3'
        gt_root = '/data/baisr/bsisr/octnew/test/labelill3'

    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name)
        label_path = os.path.join(gt_root, image_name)

        if cv2.imread(image_path) is not None:
            # if os.path.exists(image_path) and os.path.exists(label_path):
            images.append(image_path)
            masks.append(label_path)
            dataid.append(2)
    # dataid = 10
    return images, masks, dataid


class ImageFolder(data.Dataset):

    def __init__(self, root_path, datasets='Messidor', mode='train'):
        self.root = root_path
        self.mode = mode
        self.dataset = datasets
        self.images = []
        self.labels = []
        self.datasetid = []
        # assert self.dataset in ['RIM-ONE', 'Messidor', 'ORIGA', 'DRIVE', 'Cell', 'Vessel', 'ORIGA_OD', 'humanseg','DDREX'], \
        #    "the dataset should be in 'Messidor', 'ORIGA', 'RIM-ONE', 'Vessel' "

        # imagesa, labelsa, id = read_DDREX_person(self.root, self.mode)
        # self.images = self.images + imagesa
        # self.labels = self.labels + labelsa
        # self.datasetid = self.datasetid + id
        # imagesa, labelsa, id = read_DDRHE_person(self.root, self.mode)
        # self.images = self.images + imagesa
        # self.labels = self.labels + labelsa
        # self.datasetid = self.datasetid + id
        # imagesa, labelsa, id = read_DDRMA_person(self.root, self.mode)
        # self.images = self.images + imagesa
        # self.labels = self.labels + labelsa
        # self.datasetid = self.datasetid + id
        # imagesa, labelsa, id = read_DDRSE_person(self.root, self.mode)
        # self.images = self.images + imagesa
        # self.labels = self.labels + labelsa
        # self.datasetid = self.datasetid + id

        imagesa, labelsa, id = read_idridEX_person(self.root, self.mode)
        self.images = self.images + imagesa
        self.labels = self.labels + labelsa
        self.datasetid = self.datasetid + id
        # imagesa, labelsa, id = read_idridHE_person(self.root, self.mode)
        # self.images = self.images + imagesa
        # self.labels = self.labels + labelsa
        # self.datasetid = self.datasetid + id
        # imagesa, labelsa, id = read_idridMA_person(self.root, self.mode)
        # self.images = self.images + imagesa
        # self.labels = self.labels + labelsa
        # self.datasetid = self.datasetid + id
        # imagesa, labelsa, id = read_idridSE_person(self.root, self.mode)
        # self.images = self.images + imagesa
        # self.labels = self.labels + labelsa
        # self.datasetid = self.datasetid + id

        # imagesa, labelsa, id = read_OCTill1_person(self.root, self.mode)
        # self.images = self.images + imagesa
        # self.labels = self.labels + labelsa
        # self.datasetid = self.datasetid + id
        # imagesa, labelsa, id = read_OCTill2_person(self.root, self.mode)
        # self.images = self.images + imagesa
        # self.labels = self.labels + labelsa
        # self.datasetid = self.datasetid + id
        # imagesa, labelsa, id = read_OCTill3_person(self.root, self.mode)
        # self.images = self.images + imagesa
        # self.labels = self.labels + labelsa
        # self.datasetid = self.datasetid + id

    def __getitem__(self, index):
        # img, mask = default_DRIVE_loader(self.images[index], self.labels[index])
        # img = torch.Tensor(img)
        # mask = torch.Tensor(mask)
        # datasetid1 = self.datasetid[index]

        img, mask = default_DRIVE_loaderidrid(self.images[index], self.labels[index])
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        datasetid1 = self.datasetid[index]
        # if index >= 2128:
        #     datasetid1 = int((index - 2128) / 54)
        # else:
        #     datasetid1 = int(index / 532)

        # if index // 5549 == 0:
        #     datasetid1 = 8
        # elif index // 5549 == 1:
        #     datasetid1 = 9
        # elif index // 5549 == 2:
        #     datasetid1 = 10
        return img, mask, datasetid1

    def __len__(self):
        assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
        return len(self.images)

if __name__=='__main__':
    DDR, DDREX, DDRHE, DDRMA, DDRSE = read_DDR_person()
    ddrex = []
    ddrhe = []
    ddrma = []
    ddrse = []
    for img_name in DDREX:
        img = np.array(Image.open(img_name))
        elements_255 = (img == 255)
        count_255 = np.sum(elements_255)
        if count_255 != 0:
            ddrex.append(count_255)
    for img_name in DDRHE:
        img = np.array(Image.open(img_name))
        elements_255 = (img == 255)
        count_255 = np.sum(elements_255)
        if count_255 != 0:
            ddrhe.append(count_255)
    for img_name in DDRMA:
        img = np.array(Image.open(img_name))
        elements_255 = (img == 255)
        count_255 = np.sum(elements_255)
        if count_255 != 0:
            ddrma.append(count_255)
    for img_name in DDRSE:
        img = np.array(Image.open(img_name))
        elements_255 = (img == 255)
        count_255 = np.sum(elements_255)
        if count_255 != 0:
            ddrse.append(count_255)

    ddrex = np.sort(np.array(ddrex))
    ddrhe = np.sort(np.array(ddrhe))
    ddrma = np.sort(np.array(ddrma))
    ddrse = np.sort(np.array(ddrse))
    ex10 = ddrex[int(len(ddrex) * 0.1)]
    ex90 = ddrex[int(len(ddrex) * 0.9)]
    he10 = ddrhe[int(len(ddrhe) * 0.1)]
    he90 = ddrhe[int(len(ddrhe) * 0.9)]
    ma10 = ddrma[int(len(ddrma) * 0.1)]
    ma90 = ddrma[int(len(ddrma) * 0.9)]
    se10 = ddrse[int(len(ddrse) * 0.1)]
    se90 = ddrse[int(len(ddrse) * 0.9)]



    plt.figure()

    yticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticks)


    unique_vals, val_counts = np.unique(ddrex, return_counts=True)
    cumcount = np.cumsum(val_counts)
    cdf = cumcount/float(np.sum(val_counts))
    plt.subplot(2, 2, 1)
    plt.plot(unique_vals, cdf, color = 'red')
    #plt.xlim(0, 1000000)
    plt.xscale('log')
    plt.xlabel('EXs size(pixel)')
    plt.ylabel('Cumulative Probability')

    unique_vals, val_counts = np.unique(ddrhe, return_counts=True)
    cumcount = np.cumsum(val_counts)
    cdf = cumcount / float(np.sum(val_counts))
    plt.subplot(2, 2, 2)
    plt.plot(unique_vals, cdf, color = 'blue')
    # plt.xlim(0, 1000000)
    plt.xscale('log')
    plt.xlabel('HEs size(pixel)')
    plt.ylabel('Cumulative Probability')

    unique_vals, val_counts = np.unique(ddrma, return_counts=True)
    cumcount = np.cumsum(val_counts)
    cdf = cumcount / float(np.sum(val_counts))
    plt.subplot(2, 2, 3)
    plt.plot(unique_vals, cdf, color = 'green')
    # plt.xlim(0, 1000000)
    plt.xscale('log')
    plt.xlabel('MAs size(pixel)')
    plt.ylabel('Cumulative Probability')

    unique_vals, val_counts = np.unique(ddrse, return_counts=True)
    cumcount = np.cumsum(val_counts)
    cdf = cumcount / float(np.sum(val_counts))
    plt.subplot(2, 2, 4)
    plt.plot(unique_vals, cdf, color = 'yellow')
    # plt.xlim(0, 1000000)
    plt.xscale('log')
    plt.xlabel('SEs size(pixel)')
    plt.ylabel('Cumulative Probability')


    plt.show()