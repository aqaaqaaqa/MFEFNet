import clip
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import sklearn.metrics as metrics
import cv2
import os
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, f1_score

from time import time
from PIL import Image
from Metrics import dice, mask_iou, mask_to_boundary
from loss import dice_loss
# from models import ResUnet
# from vision_transformer import SwinUnet
# from attentionunet import AttU_Net
# from unetpp import NestedUNet
# from deeplabv3 import DeepLabV3
# from networks.cenet import CE_Net_NEW, UNet, CE_Net_NEW22, CE_Net_NEWOCT, CE_Net_NEW11, CE_Net_true
from networks.cenet import CE_Net_NEW11

prompt = ['a batch of medical image need to segment Hard Exudates out in DDR dataset',#ex
          'a batch of medical image need to segment Haemorrhages out in DDR dataset, please pay more attention to the part',#he
          'a batch of medical image need to segment Microaneurysms out in DDR dataset, please pay more attention to the part',#ma
          'a batch of medical image need to segment Soft Exudates out in DDR dataset',#se
          # 'a batch of medical image need to segment Haemorrhages out in idrid dataset',
          # 'a batch of medical image need to segment Hard Exudates out in idrid dataset',
          # 'a batch of medical image need to segment Microaneurysms out in idrid dataset',
          # 'a batch of medical image need to segment Soft Exudates out in idrid dataset',
          # 'a batch of medical image need to segment Intraretinal Fluid out in OCT dataset',
          # 'a batch of medical image need to segment Subretinal Fluid out in OCT dataset',
          # 'a batch of medical image need to segment Pigment Epithelium Detachment out in OCT dataset'
          ]
device = "cuda:2"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
text = clip.tokenize(prompt).to(device)
with torch.no_grad():
    text_features = model.encode_text(text).to(device)
text_features = text_features.float()#4 512
text_featuresori = text_features.unsqueeze(0).repeat(1, 1, 1)#1 4 512
text_features = text_features.unsqueeze(2).repeat(1, 1, 512)


class TTAFrame():
    def __init__(self, net, id):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.text_features = text_features
        self.datasetid = id

    def textprepare1(self):
        text_features = self.text_features[self.datasetid, :].cuda()
        new_tensor = torch.zeros((512, 512), dtype=torch.float).cuda()
        new_tensor[:512, :512] = text_features
        text_features = new_tensor.reshape(1, 1, 512, 512).cuda()
        textnew_features = text_features
        # for i in range(4):
        #     text_features = torch.stack([text_features, text_features], dim = 0)
        # print(text_features.shape)
        new_tensor1 = torch.zeros((4, 1, 512, 512), dtype=torch.float).cuda()
        new_tensor1[:4, :1, :512, :512] = text_features
        return new_tensor1

    def textprepare(self):
        text_features1 = self.text_features.unsqueeze(0)
        #text_features1 = text_features1.repeat(12, 1 , 1, 1)


        # #text_features1 = self.text_features[self.datasetid[0], :, :].unsqueeze(0)
        # #for i in range(len(self.datasetid) - 1):
        #     #text_features1 = torch.cat((text_features1, self.text_features[self.datasetid[i + 1], :, :].unsqueeze(0)), dim = 0)
        # text_features = self.text_features[self.datasetid, :].cuda()
        # new_tensor = torch.zeros((512, 512), dtype=torch.float).cuda()
        # new_tensor[:512, :512] = text_features
        # text_features = new_tensor.reshape(1, 1, 512, 512).cuda()
        # textnew_features = text_features
        # for i in range(4):
        #     text_features = torch.stack([text_features, text_features], dim = 0)
        # print(text_features.shape)
        # new_tensor1 = torch.zeros((12, 1, 512, 512), dtype=torch.float).cuda()
        # new_tensor1[:12, :1, :512, :512] = text_features
        return text_features1

    def test_one_img_from_path(self, path, evalmode=True, without_TTA=False):
        if evalmode:
            self.net.eval()
        batchsize = 12
        if without_TTA:
            return self.test_one_img_without_test_aug(path)
        elif batchsize >= 8:
            return self.test_one_img_from_path_1(path)
        elif batchsize >= 4:
            return self.test_one_img_from_path_2(path)
        elif batchsize >= 2:
            return self.test_one_img_from_path_4(path)

    def test_one_img_without_test_aug(self, path):
        text = self.textprepare()
        img = cv2.imread(path)
        img = cv2.resize(img, (448, 448))
        img = np.expand_dims(img, 0)
        img = img.transpose(0, 3, 1, 2)
        img = V(torch.Tensor(np.array(img, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        imglittle = self.slide(56, img) # 12 192 56 56
        imgmid = self.slide(112, img) # 12 48 112 112
        imgmid = imgmid.repeat(1, 2, 1, 1)
        #img = V(torch.Tensor(np.array(img, np.float32)).cuda())
        mask = self.net.forward(img, imglittle, imgmid, text, text_featuresori).squeeze().cpu().data.numpy()
        mask = mask[self.datasetid]
        #mask = self.net.forward(img).squeeze().cpu().data.numpy()
        return mask #448 448

    def slide(self, size, img):
        times = img.shape[2] / size
        chunks = torch.chunk(img, int(times), dim=2)
        result = []
        i = -1
        for chunk in chunks:
            chunks2 = torch.chunk(chunk, int(times), dim=3)
            for chunk22 in chunks2:
                if i == -1:
                    result = chunk22
                    i = 0
                else:
                    result = torch.cat((result, chunk22), dim = 1)
        return result

    def test_one_img_from_path_8(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        return mask2

    def test_one_img_from_path_4(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        return mask2

    def test_one_img_from_path_2(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = img3.transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).cuda())
        img6 = img4.transpose(0, 3, 1, 2)
        img6 = np.array(img6, np.float32) / 255.0 * 3.2 - 1.6
        img6 = V(torch.Tensor(img6).cuda())

        maska = self.net.forward(img5).squeeze().cpu().data.numpy()  # .squeeze(1)
        maskb = self.net.forward(img6).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask3

    def test_one_img_from_path_1(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img = cv2.resize(img, (448, 448))

        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = np.concatenate([img3, img4]).transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).cuda())

        mask = self.net.forward(img5).squeeze().cpu().data.numpy()  # .squeeze(1)
        mask1 = mask[:4] + mask[4:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask3

    def load(self, path):
        model = torch.load(path)
        self.net.load_state_dict(model)


image_root = '/data/baisr/bsisr/DDR_lesion_segmentation/test/image'#.replace('/data/baisr/', '/vepfs/d_bsr/')
gt_root = '/data/baisr/bsisr/DDR_lesion_segmentation/test/label/EX'#.replace('/data/baisr/', '/vepfs/d_bsr/')
dataid = 0
images_listEX = []
masks_listEX = []
for image_name in os.listdir(image_root):
    image_path = os.path.join(image_root, image_name)
    label_path = os.path.join(gt_root, image_name.replace("jpg", "tif"))

    if cv2.imread(image_path) is not None:

        if os.path.exists(image_path) and os.path.exists(label_path):
            images_listEX.append(image_path)
            masks_listEX.append(label_path)
image_root = '/data/baisr/bsisr/DDR_lesion_segmentation/test/image'#.replace('/data/baisr/', '/vepfs/d_bsr/')
gt_root = '/data/baisr/bsisr/DDR_lesion_segmentation/test/label/HE'#.replace('/data/baisr/', '/vepfs/d_bsr/')
dataid = 1
images_listHE = []
masks_listHE = []
for image_name in os.listdir(image_root):
    image_path = os.path.join(image_root, image_name)
    label_path = os.path.join(gt_root, image_name.replace("jpg", "tif"))

    if cv2.imread(image_path) is not None:

        if os.path.exists(image_path) and os.path.exists(label_path):
            images_listHE.append(image_path)
            masks_listHE.append(label_path)
image_root = '/data/baisr/bsisr/DDR_lesion_segmentation/test/image'#.replace('/data/baisr/', '/vepfs/d_bsr/')
gt_root = '/data/baisr/bsisr/DDR_lesion_segmentation/test/label/MA'#.replace('/data/baisr/', '/vepfs/d_bsr/')
dataid = 2
images_listMA = []
masks_listMA = []
for image_name in os.listdir(image_root):
    image_path = os.path.join(image_root, image_name)
    label_path = os.path.join(gt_root, image_name.replace("jpg", "tif"))

    if cv2.imread(image_path) is not None:

        if os.path.exists(image_path) and os.path.exists(label_path):
            images_listMA.append(image_path)
            masks_listMA.append(label_path)
image_root = '/data/baisr/bsisr/DDR_lesion_segmentation/test/image'#.replace('/data/baisr/', '/vepfs/d_bsr/')
gt_root = '/data/baisr/bsisr/DDR_lesion_segmentation/test/label/SE'#.replace('/data/baisr/', '/vepfs/d_bsr/')
dataid = 3
images_listSE = []
masks_listSE = []
for image_name in os.listdir(image_root):
    image_path = os.path.join(image_root, image_name)
    label_path = os.path.join(gt_root, image_name.replace("jpg", "tif"))

    if cv2.imread(image_path) is not None:

        if os.path.exists(image_path) and os.path.exists(label_path):
            images_listSE.append(image_path)
            masks_listSE.append(label_path)
solver = TTAFrame(CE_Net_NEW11, dataid)