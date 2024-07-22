# *coding:utf-8 *

#CLIP出来应该是512
import torch
torch.cuda.current_device()
torch.cuda._initialized = True
import torch.utils.data as data
from torch.autograd import Variable as V

import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import warnings

warnings.filterwarnings('ignore')

from time import time
#from Visualizer import Visualizer
from networks.cenet import CE_Net_NEW, UNet, CE_Net_NEW22, CE_Net_NEWOCT, CE_Net_NEW11, CE_Net_true
from framework import MyFrame
from loss import dice_bce_loss, boundary_dice_bce_loss, dice_loss
from PIL import Image
from data import ImageFolder
from models import  ResUnet
from vision_transformer import SwinUnet
from attentionunet import AttU_Net
from unetpp import NestedUNet
from deeplabv3 import DeepLabV3
import cv2
import numpy as np
from Metrics import calculate_auc_test, accuracy
from test_cenet import test_ce_net_ORIGA


# ROOT = '/data/zaiwang/Dataset/ORIGA'
# ROOT = '/data/zaiwang/Dataset/Messidor'
# ROOT = '/data/zaiwang/Dataset/ORIGA_OD'
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
ROOT = '../eyesegment/'
NETWORK = CE_Net_NEW11
LOSS_TYPE = dice_bce_loss
#LOSS_TYPE = dice_bce_loss
Dataset_name = ["DDREX", 'DDRHE', 'DDRMA', 'DDRSE', 'idridEX', 'idridHE', 'idridMA', 'idridSE', 'octIRF', 'octSRF', 'octPED']
# 20210826 NAME = 'Unet-origin-' + ROOT.split('/')[-1]
# 20210827 NAME = 'boundary_iou-' + ROOT.split('/')[-1] + '-v1'
# V1: weighted boundary_dice_bce_loss weight = 1
# V2: weighted boundary_dice_bce_loss weight = 0.5
# V3: weighted boundary_dice_bce_loss weight = 0.25
# V4: weighted boundary_dice_bce_loss weight = 2
NAME = 'MFEFNet'
print(NAME)

def train_CE_Net_Vessel():
    save_weight_path = './weights/' + NAME + Dataset_name[0] + Dataset_name[1] +'.th'
    # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[0])
    # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[8])
    # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[9])
    # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[10])
    # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[0])
    # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[1])
    # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[2])
    # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[3])
    # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[4])
    # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[5])
    # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[6])
    # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[7])
    # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[8])
    #test_ce_net_ORIGA(ROOT, save_weight_path)
    IMAGE_SHAPE = (448, 448)
    Use_Test = False
    BATCH_SIZE_PER_CARD = 12
    #viz = Visualizer(env=NAME)

    # 20210826 dice_bce_loss
    # 20210827 boundary_dice_bce_loss
    loss_type = boundary_dice_bce_loss
    solver = MyFrame(NETWORK, LOSS_TYPE, 2e-4)
    #print(solver.net)
    batch_size = 12

    # Preparing the dataloader

    dataset0 = ImageFolder(root_path=ROOT, datasets=Dataset_name[0])
    save_weight_path = './weights/' + NAME + 'FUSIONidridMABCELOSS' + '.th'
    print(save_weight_path)
    # dataset1 = ImageFolder(root_path=ROOT, datasets=Dataset_name[1])
    # dataset2 = ImageFolder(root_path=ROOT, datasets=Dataset_name[2])
    # dataset3 = ImageFolder(root_path=ROOT, datasets=Dataset_name[3])
    TEST_RESULT = False
    data_loader0 = torch.utils.data.DataLoader(
        dataset0,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True)
    # data_loader1 = torch.utils.data.DataLoader(
    #     dataset1,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=0,
    #     drop_last=True)
    #
    # data_loader2 = torch.utils.data.DataLoader(
    #     dataset2,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=0,
    #     drop_last=True)
    #
    # data_loader3 = torch.utils.data.DataLoader(
    #     dataset3,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=0,
    #     drop_last=True)

    mylog = open('logs/' + NAME + '.log', 'w')

    tic = time()
    no_optim = 0
    total_epoch = 1000
    train_epoch_best_loss = 10000.

    for epoch in range(0, total_epoch + 1):
        data_loader_iter0 = iter(data_loader0)
        train_epoch_loss = 0

        index = 0

        for img, mask, datasetid in data_loader_iter0:
            solver.set_input(img, mask, 0, datasetid)

            train_loss, pred = solver.optimize()
            #print(train_loss)

            train_epoch_loss += train_loss
            index = index + 1

        # data_loader_iter1 = iter(data_loader1)
        #
        # for img, mask in data_loader_iter1:
        #     solver.set_input(img, mask, datasetid=1)
        #
        #     train_loss, pred = solver.optimize()
        #     #print(train_loss)
        #     train_epoch_loss1 += train_loss
        #     train_epoch_loss += train_loss
        #
        #     index = index + 1
        # data_loader_iter2 = iter(data_loader2)
        #
        # for img, mask in data_loader_iter2:
        #     solver.set_input(img, mask, datasetid=2)
        #
        #     train_loss, pred = solver.optimize()
        #     #print(train_loss)
        #     train_epoch_loss2 += train_loss
        #     train_epoch_loss += train_loss
        #
        #     index = index + 1
        #
        # data_loader_iter3 = iter(data_loader3)
        #
        # for img, mask in data_loader_iter3:
        #     solver.set_input(img, mask, datasetid=3)
        #
        #     train_loss, pred = solver.optimize()
        #     #print(train_loss)
        #     train_epoch_loss3 += train_loss
        #     train_epoch_loss += train_loss
        #
        #     index = index + 1
        #     if index % 40 == 0:
        #         print(train_epoch_loss0)
        #         print(train_epoch_loss1)
        #         print(train_epoch_loss2)
        #         print(train_epoch_loss3)
                # train_epoch_loss /= index
                # viz.plot(name='loss', y=train_epoch_loss)
                #show_image = (img + 1.6) / 3.2 * 255.
                #viz.img(name='images', img_=show_image[0, :, :, :])
                #viz.img(name='labels', img_=mask[0, :, :, :])
                #viz.img(name='prediction', img_=pred[0, :, :, :])

        #show_image = (img + 1.6) / 3.2 * 255.
        #viz.img(name='images', img_=show_image[0, :, :, :])
        #viz.img(name='labels', img_=mask[0, :, :, :])
        #viz.img(name='prediction', img_=pred[0, :, :, :])


        train_epoch_loss = train_epoch_loss / len(data_loader_iter0)
        print('********')
        print('epoch:', epoch, '    time:', int(time() - tic), file=mylog)
        print( 'train_loss:', train_epoch_loss , file=mylog)
        print('SHAPE:', IMAGE_SHAPE,file=mylog)
        print('********')
        print('epoch:', epoch, '    time:', int(time() - tic))
        print('train_loss:', train_epoch_loss)
        print('SHAPE:', IMAGE_SHAPE)

        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            solver.save(save_weight_path)
        if no_optim > 50:
            # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[0])
            # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[1])
            # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[2])
            # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[3])
            test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[4])
            test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[5])
            test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[6])
            test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[7])
            # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[8])
            # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[9])
            # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[10])
            #test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[11])
            TEST_RESULT = True
            print(mylog, 'early stop at %d epoch' % epoch)
            print('early stop at %d epoch' % epoch)
            break
        if no_optim > 20:
            if solver.old_lr < 5e-7:
                # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[0])
                # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[1])
                # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[2])
                # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[3])
                test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[4])
                test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[5])
                test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[6])
                test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[7])
                # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[8])
                # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[9])
                # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[10])
                # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[11])
                TEST_RESULT = True
                print("after 10 epochs, the loss did not decrease and lr is smaller than 5e-7")
                break
            # solver.load('./weights/' + NAME + '.th')
            solver.update_lr(2.0, factor=True, mylog=mylog)
            no_optim = 0
        mylog.flush()
    if TEST_RESULT:
        print("The training process has finished")
    else:
        # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[0])
        # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[1])
        # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[2])
        # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[3])
        test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[4])
        test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[5])
        test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[6])
        test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[7])
        # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[8])
        # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[9])
        # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[10])
        # test_ce_net_ORIGA(ROOT, save_weight_path, Dataset_name[11])
    print(mylog, 'Finish!')
    print('Finish!')
    mylog.close()


if __name__ == '__main__':
    # step 1 : python -m visdom.server
    print(torch.cuda.device_count())
    device = torch.device("cuda:2")
    train_CE_Net_Vessel()
