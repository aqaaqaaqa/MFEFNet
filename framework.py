# *coding:utf-8 *

import torch
import torch.nn as nn
from torch.autograd import Variable as V
import clip
import cv2
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler

prompt = [#'a batch of medical image need to segment Hard Exudates out in DDR dataset',#ex
          #'a batch of medical image need to segment Haemorrhages out in DDR dataset, please pay more attention to the part',#he
          #'a batch of medical image need to segment Microaneurysms out in DDR dataset, please pay more attention to the part',#ma
          #'a batch of medical image need to segment Soft Exudates out in DDR dataset',#se
          'a batch of medical image need to segment Hard Exudates out in idrid dataset',
          'a batch of medical image need to segment Haemorrhages out in idrid dataset, please pay more attention to the part',
          'a batch of medical image need to segment Microaneurysms out in idrid dataset, please pay more attention to the part',
          'a batch of medical image need to segment Soft Exudates out in idrid dataset',
          #'a batch of medical image need to segment Intraretinal Fluid out in OCT dataset',
          #'a batch of medical image need to segment Subretinal Fluid out in OCT dataset',
          #'a batch of medical image need to segment Pigment Epithelium Detachment out in OCT dataset'
          ]
device = "cuda:2"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
text = clip.tokenize(prompt).to(device)
with torch.no_grad():
    text_features = model.encode_text(text).to(device)
text_features = text_features.float()#4 512
text_featuresori = text_features.unsqueeze(0).repeat(12, 1, 1)#12 4 512
text_features = text_features.unsqueeze(2).repeat(1, 1, 512)
#print(text_features)
#text_features = text_features.unsqueeze(1)


class MyFrame():
    def __init__(self, net, loss, lr=1e-1, evalmode=False):
        self.net = net().cuda()

        self.net = torch.nn.DataParallel(self.net)
        self.load('./weights/CE_Net_NEW11FUSIONidridMABCELOSS.th')
        self.optimizer = torch.optim.AdamW(params=self.net.parameters(), lr=lr)
        #self.optimizer = torch.optim.SGD(params=self.net.parameters(), lr=lr)
        self.loss = loss()
        self.old_lr = lr
        self.text_features = text_features
        self.datasetid = -1
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()


    def set_input(self, img_batch, mask_batch=None, img_id=None, datasetid = 0):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id
        self.datasetid = datasetid

    def test_one_img(self, img):
        pred = self.net.forward(img)

        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0

        mask = pred.squeeze().cpu().data.numpy()
        return mask

    def test_batch(self):
        self.forward(volatile=True)
        mask = self.net.forward(self.img).cpu().data.numpy().squeeze(1)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        return mask, self.img_id

    def test_one_img_from_path(self, path):
        img = cv2.imread(path)
        img = np.array(img, np.float32) / 255.0 * 3.2 - 1.6
        img = V(torch.Tensor(img).cuda())

        mask = self.net.forward(img).squeeze().cpu().data.numpy()  # .squeeze(1)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        return mask

    def forward(self, volatile=False):
        self.img = V(self.img.cuda(), volatile=volatile)
        if self.mask is not None:
            self.mask = V(self.mask.cuda(), volatile=volatile)

    def optimize(self):
        self.forward()

        text = self.textprepare()
        imglittle = self.slide(56) # 12 192 56 56
        imgmid = self.slide(112) # 12 48 112 112
        imgmid = imgmid.repeat(1, 2, 1, 1)
        #print('forward')
        model = self.net
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, 'requires gradient and is being updated')
        #     else:
        #         print(name, 'does not require gradient and is not being updated')
        pred = self.net.forward(self.img, imglittle, imgmid, text, text_featuresori)#12 4 448 448
        #pred = self.net.forward(self.img, text, text_featuresori)
        #prednew = self.net.forward(self.img)
        prednew = pred[0, self.datasetid[0]].unsqueeze_(0).unsqueeze_(0)
        for i in range(11):
            prednew = torch.cat((prednew, pred[i + 1, self.datasetid[i]].unsqueeze_(0).unsqueeze_(0)), dim = 0)
        loss = self.loss(self.mask, prednew)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
        self.optimizer.step()
        self.optimizer.zero_grad()
        #print('done')
        return loss.data, prednew

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def update_lr(self, new_lr, mylog, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print(mylog, 'update learning rate: %f -> %f' % (self.old_lr, new_lr))
        print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr
    def textprepare(self):
        text_features1 = self.text_features.unsqueeze(0)
        text_features1 = text_features1.repeat(12, 1 , 1, 1)


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
    def slide(self, size):
        times = self.img.shape[2] / size
        chunks = torch.chunk(self.img, int(times), dim=2)
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


