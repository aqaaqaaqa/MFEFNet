from networks.cenet import CE_Net_NEW, UNet, CE_Net_NEW22, CE_Net_NEWOCT, CE_Net_NEW11
from framework import MyFrame
from loss import dice_bce_loss, boundary_dice_bce_loss, dice_loss
from networks.cenetma import CE_Net_NEW11 as CE_Net_NEW11MA
from PIL import Image
from data import ImageFolder
from models import  ResUnet
from vision_transformer import SwinUnet
from attentionunet import AttU_Net
from unetpp import NestedUNet
from deeplabv3 import DeepLabV3
import cv2
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from Metrics import calculate_auc_test, accuracy
from test_cenet import test_ce_net_ORIGA

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
from vision_transformer import SwinUnet
# from attentionunet import AttU_Net
from unetpp import NestedUNet
from deeplabv3 import DeepLabV3
from networks.cenet import CE_Net_NEW, UNet, CE_Net_NEW22, CE_Net_NEWOCT, CE_Net_NEW11, CE_Net_true
from networks.cenet import CE_Net_NEW11, CE_Net_true

prompt = [#'a batch of medical image need to segment Hard Exudates out in DDR dataset',#ex
          #'a batch of medical image need to segment Haemorrhages out in DDR dataset, please pay more attention to the part',#he
          #'a batch of medical image need to segment Microaneurysms out in DDR dataset, please pay more attention to the part',#ma
          #'a batch of medical image need to segment Soft Exudates out in DDR dataset',#se
          'a batch of medical image need to segment Haemorrhages out in idrid dataset',
          'a batch of medical image need to segment Hard Exudates out in idrid dataset',
          'a batch of medical image need to segment Microaneurysms out in idrid dataset',
          'a batch of medical image need to segment Soft Exudates out in idrid dataset',
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
        #mask = self.net.forward(img, imglittle, imgmid, text, text_featuresori).squeeze().cpu().data.numpy()
        #mask = mask[self.datasetid]
        mask = self.net.forward(img).squeeze().cpu().data.numpy()
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

def test_ce_net_ORIGA(root_path, weight_path1, weight_path2, weight_path3, weight_path4, dataset):
    # root_path = '/data/zaiwang/Dataset/ORIGA_OD'
    without_TTA = True
    test_dataset_category_name = dataset
    # image_root = '/data/baisr/bsisr/DDR_lesion_segmentation/test/image'#.replace('/data/baisr/', '/vepfs/d_bsr/')
    # gt_root = '/data/baisr/bsisr/DDR_lesion_segmentation/test/label/EX'#.replace('/data/baisr/', '/vepfs/d_bsr/')
    #
    # dataid = 0
    # images_listex = []
    # masks_listex = []
    # for image_name in os.listdir(image_root):
    #     image_path = os.path.join(image_root, image_name)
    #     label_path = os.path.join(gt_root, image_name.replace("jpg", "tif"))
    #
    #     if cv2.imread(image_path) is not None:
    #
    #         if os.path.exists(image_path) and os.path.exists(label_path):
    #             images_listex.append(image_path)
    #             masks_listex.append(label_path)
    #
    # image_root = '/data/baisr/bsisr/DDR_lesion_segmentation/test/image'#.replace('/data/baisr/', '/vepfs/d_bsr/')
    # gt_root = '/data/baisr/bsisr/DDR_lesion_segmentation/test/label/HE'#.replace('/data/baisr/', '/vepfs/d_bsr/')
    # dataid = 1
    # images_listhe = []
    # masks_listhe = []
    # for image_name in os.listdir(image_root):
    #     image_path = os.path.join(image_root, image_name)
    #     label_path = os.path.join(gt_root, image_name.replace("jpg", "tif"))
    #
    #     if cv2.imread(image_path) is not None:
    #
    #         if os.path.exists(image_path) and os.path.exists(label_path):
    #             images_listhe.append(image_path)
    #             masks_listhe.append(label_path)
    #
    # image_root = '/data/baisr/bsisr/DDR_lesion_segmentation/test/image'#.replace('/data/baisr/', '/vepfs/d_bsr/')
    # gt_root = '/data/baisr/bsisr/DDR_lesion_segmentation/test/label/MA'#.replace('/data/baisr/', '/vepfs/d_bsr/')
    # dataid = 2
    # images_listma = []
    # masks_listma = []
    # for image_name in os.listdir(image_root):
    #     image_path = os.path.join(image_root, image_name)
    #     label_path = os.path.join(gt_root, image_name.replace("jpg", "tif"))
    #
    #     if cv2.imread(image_path) is not None:
    #
    #         if os.path.exists(image_path) and os.path.exists(label_path):
    #             images_listma.append(image_path)
    #             masks_listma.append(label_path)
    #
    #
    #
    # image_root = '/data/baisr/bsisr/DDR_lesion_segmentation/test/image'#.replace('/data/baisr/', '/vepfs/d_bsr/')
    # gt_root = '/data/baisr/bsisr/DDR_lesion_segmentation/test/label/SE'#.replace('/data/baisr/', '/vepfs/d_bsr/')
    # dataid = 3
    # images_listse = []
    # masks_listse = []
    # for image_name in os.listdir(image_root):
    #     image_path = os.path.join(image_root, image_name)
    #     label_path = os.path.join(gt_root, image_name.replace("jpg", "tif"))
    #
    #     if cv2.imread(image_path) is not None:
    #
    #         if os.path.exists(image_path) and os.path.exists(label_path):
    #             images_listse.append(image_path)
    #             masks_listse.append(label_path)


    image_root = '/data/baisr/bsisr/idrid_Segmentation/1. Original Images/b. Testing Set'#.replace('/data/baisr/', '/vepfs/d_bsr/')
    gt_root = '/data/baisr/bsisr/idrid_Segmentation/2. All Segmentation Groundtruths/b. Testing Set/Hard Exudates'#.replace('/data/baisr/', '/vepfs/d_bsr/')
    dataid = 0
    images_listex = []
    masks_listex = []
    for label_name in os.listdir(gt_root):
        image_path = os.path.join(image_root, label_name.replace('_EX.tif', '.jpg'))
        label_path = os.path.join(gt_root, label_name)

        if cv2.imread(image_path) is not None:

            if os.path.exists(image_path) and os.path.exists(label_path):
                images_listex.append(image_path)
                masks_listex.append(label_path)

    image_root = '/data/baisr/bsisr/idrid_Segmentation/1. Original Images/b. Testing Set'#.replace('/data/baisr/', '/vepfs/d_bsr/')
    gt_root = '/data/baisr/bsisr/idrid_Segmentation/2. All Segmentation Groundtruths/b. Testing Set/Haemorrhages'#.replace('/data/baisr/', '/vepfs/d_bsr/')
    dataid = 1
    images_listhe = []
    masks_listhe = []
    for label_name in os.listdir(gt_root):
        image_path = os.path.join(image_root, label_name.replace('_HE.tif', '.jpg'))
        label_path = os.path.join(gt_root, label_name)

        if cv2.imread(image_path) is not None:

            if os.path.exists(image_path) and os.path.exists(label_path):
                images_listhe.append(image_path)
                masks_listhe.append(label_path)


    image_root = '/data/baisr/bsisr/idrid_Segmentation/1. Original Images/b. Testing Set'#.replace('/data/baisr/', '/vepfs/d_bsr/')
    gt_root = '/data/baisr/bsisr/idrid_Segmentation/2. All Segmentation Groundtruths/b. Testing Set/Microaneurysms'#.replace('/data/baisr/', '/vepfs/d_bsr/')
    dataid = 2
    images_listma = []
    masks_listma = []
    for label_name in os.listdir(gt_root):
        image_path = os.path.join(image_root, label_name.replace('_MA.tif', '.jpg'))
        label_path = os.path.join(gt_root, label_name)

        if cv2.imread(image_path) is not None:

            if os.path.exists(image_path) and os.path.exists(label_path):
                images_listma.append(image_path)
                masks_listma.append(label_path)

    image_root = '/data/baisr/bsisr/idrid_Segmentation/1. Original Images/b. Testing Set'#.replace('/data/baisr/', '/vepfs/d_bsr/')
    gt_root = '/data/baisr/bsisr/idrid_Segmentation/2. All Segmentation Groundtruths/b. Testing Set/Soft Exudates'#.replace('/data/baisr/', '/vepfs/d_bsr/')
    dataid = 3
    images_listse = []
    masks_listse = []
    for label_name in os.listdir(gt_root):
        image_path = os.path.join(image_root, label_name.replace('_SE.tif', '.jpg'))
        label_path = os.path.join(gt_root, label_name)

        if cv2.imread(image_path) is not None:

            if os.path.exists(image_path) and os.path.exists(label_path):
                images_listse.append(image_path)
                masks_listse.append(label_path)
    solverex = TTAFrame(DeepLabV3, 0)
    solverex.load(weight_path1)
    solverhe = TTAFrame(DeepLabV3, 1)
    solverhe.load(weight_path2)
    solverma = TTAFrame(DeepLabV3, 2)
    solverma.load(weight_path3)
    solverse = TTAFrame(DeepLabV3, 3)
    solverse.load(weight_path4)

    for i in range(len(masks_listse)):

        image_pathse = images_listse[i]
        image_pathex = image_pathse[i]
        image_pathhe = images_listhe[i]
        image_pathma = images_listma[i]

        # ground_truth_pathse = masks_listse[i]
        # ground_truth_pathex = ground_truth_pathse.replace('Soft Exudates','Hard Exudates').replace('SE','EX')
        # ground_truth_pathhe = ground_truth_pathse.replace('Soft Exudates','Haemorrhages').replace('SE','HE')
        # ground_truth_pathma = ground_truth_pathse.replace('Soft Exudates','Microaneurysms').replace('SE','MA')
        #
        #
        #
        # ground_truthex = np.array(Image.open(ground_truth_pathex))
        # ground_truthex = cv2.resize(ground_truthex, dsize=(448, 448))
        # ground_truthhe = np.array(Image.open(ground_truth_pathhe))
        # ground_truthhe = cv2.resize(ground_truthhe, dsize=(448, 448))
        # ground_truthma = np.array(Image.open(ground_truth_pathma))
        # ground_truthma = cv2.resize(ground_truthma, dsize=(448, 448))
        # ground_truthse = np.array(Image.open(ground_truth_pathse))
        # ground_truthse = cv2.resize(ground_truthse, dsize=(448, 448))

        # mask = np.zeros((448, 448))
        # mask[ground_truthex > 0.5] = 1
        # mask[ground_truthhe > 0.5] = 2
        # mask[ground_truthma > 0.5] = 3
        # mask[ground_truthse > 0.5] = 4


        mask = np.zeros((448,448))
        maskex = solverex.test_one_img_from_path(image_pathse, without_TTA=without_TTA)
        mask[maskex > 0.5] = 1
        maskhe = solverhe.test_one_img_from_path(image_pathse, without_TTA=without_TTA)
        mask[maskhe > 0.5] = 2
        maskma = solverma.test_one_img_from_path(image_pathse, without_TTA=without_TTA)
        mask[maskma > 0.5] = 3
        maskse = solverse.test_one_img_from_path(image_pathse, without_TTA=without_TTA)
        mask[maskse > 0.5] = 4
        colors = ['black', 'red', 'green', 'blue', 'yellow']
        cmap = ListedColormap(colors)
        norm = plt.Normalize(vmin=0, vmax=4)
        fig, ax = plt.subplots()
        ax.imshow(mask, cmap=cmap, norm=norm)
        ax.axis('off')
        folder_path = '/data/baisr/cenet/deeplabv3res'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        image_filename = os.path.join(folder_path, 'colored_image'+ str(i) +'.png',)
        plt.savefig(image_filename, bbox_inches='tight', pad_inches=0, dpi = 400)
        plt.close(fig)
        #DDR的mask是255，idrid是1
        # ground_truth = np.array(ground_truth, np.float32) / 255.0
        # ground_truth[ground_truth > 0.5] = 1
        # ground_truth[ground_truth <= 0.5] = 0
        #name = image_path.split('/')[-1]
        #cv2.imwrite(target + name.split('.')[0] + '-mask.png', mask.astype(np.uint8))



if __name__=='__main__':
    root_path = '/data/zaiwang/Dataset/humanseg'
    #weight_path = './weights/CE_Net_NEW11FUSIONidridMABCELOSS.th'
    weight_pathex = '/data/baisr/cenet/deeplabv3idridEX/CENet_plus/weights/CE_Net_NEW11FUSIONidridMABCELOSS.th'
    weight_pathhe = '/data/baisr/cenet/deeplabv3idridHE/CENet_plus/weights/CE_Net_NEW11FUSIONidridMABCELOSS.th'
    weight_pathma = '/data/baisr/cenet/deeplabv3idridMA/CENet_plus/weights/CE_Net_NEW11FUSIONidridMABCELOSS.th'
    weight_pathse = '/data/baisr/cenet/deeplabv3idridSE/CENet_plus/weights/CE_Net_NEW11FUSIONidridMABCELOSS.th'
    test_ce_net_ORIGA(root_path, weight_pathex, weight_pathhe, weight_pathma, weight_pathse, 'idridEX')