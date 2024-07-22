import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import clip
#import convnextv2

import convnextv2 as cn
import swin_transformerori
from functools import partial

import Constants

nonlinearity = partial(F.relu, inplace=True)
def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)
# prompt = ['a batch of medical image need to segment Hard Exudates out', 'a batch of medical image need to segment Hard Exudates out ']
# device = "cuda:0"
# model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
# text = clip.tokenize(prompt[0]).to(device).float()
# with torch.no_grad():
#     text_features = model.encode_text(text).to(device)
# text_features = text_features.float()


# def tensorreshape(oritensor = prompt[0], datasetid = 0):
#     new_tensor = torch.zeros((448, 448), dtype=torch.float)
#     new_tensor[:448, :448] = oritensor
#     text_features = new_tensor.reshape(1, 448, 448)
#     #print(text_features.shape)
#     return text_features
#
# def texttensor(datasetid = 0):
#     device = "cuda:0"
#     model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
#     text = clip.tokenize(prompt[datasetid]).to(device)
#     # [1, 512]
#     with torch.no_grad():
#         text_features = model.encode_text(text).to(device)
#     return text_features.float()

class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class DACblock_without_atrous(nn.Module):
    def __init__(self, channel):
        super(DACblock_without_atrous, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out

        return out
class fusion_feature(nn.Module):
    def __init__(self):
        super(fusion_feature, self).__init__()
        self.GAP1 = nn.Sequential(
            #nn.GroupNorm(4, 772),
            #nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(768, 768, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.GAP2 = nn.Sequential(
            # nn.GroupNorm(4, 772),
            # nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(768, 768, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
    def forward(self, x, imglittle, imgmid):
        imglittle1 = self.GAP1(imglittle)
        imgmid1 = self.GAP1(imgmid)
        imglittle = torch.mul(imglittle1, imglittle)
        imgmid = torch.mul(imgmid1, imgmid)
        x = x + imgmid + imglittle
        return x

class DACblock_with_inception(nn.Module):
    def __init__(self, channel):
        super(DACblock_with_inception, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)

        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.conv1x1 = nn.Conv2d(2 * channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate3(self.dilate1(x)))
        dilate_concat = nonlinearity(self.conv1x1(torch.cat([dilate1_out, dilate2_out], 1)))
        dilate3_out = nonlinearity(self.dilate1(dilate_concat))
        out = x + dilate3_out
        return out


class DACblock_with_inception_blocks(nn.Module):
    def __init__(self, channel):
        super(DACblock_with_inception_blocks, self).__init__()
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.conv3x3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.conv5x5 = nn.Conv2d(channel, channel, kernel_size=5, dilation=1, padding=2)
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.conv1x1(x))
        dilate2_out = nonlinearity(self.conv3x3(self.conv1x1(x)))
        dilate3_out = nonlinearity(self.conv5x5(self.conv1x1(x)))
        dilate4_out = self.pooling(x)
        out = dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out



class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(2, 3, 6, 14)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class textBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max1(x)
        x = self.conv3(x)
        x = self.relu3(x)
        return x

# class Task_specific_controller(nn.Module):
#     def __init__(self):
#
#
#     def forward(self, input):




class CE_Net_(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(CE_Net_, self).__init__()
        self.textconv1 = nn.Conv2d(4, 4, kernel_size=7, stride=1, padding=0, dilation= 3)
        self.textrelu1 = nonlinearity
        self.textconv2 = nn.Conv2d(4, 4, kernel_size=7, stride=1, padding=0, dilation= 3)
        self.textrelu2 = nonlinearity
        self.textconv3 = nn.Conv2d(4, 4, kernel_size=7, stride=1, padding=0, dilation= 3)
        self.textrelu3 = nonlinearity
        self.textconv4 = nn.Conv2d(4, 4, kernel_size=4, stride=1, padding=0, dilation=2)
        self.textrelu4 = nonlinearity
        self.textconv5 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=0, dilation=2)
        self.textrelu5 = nonlinearity

        self.beforefirstcov = nn.Conv2d(7, 3, kernel_size=3, stride=1, padding=1)
        self.beforefirstrelu = nonlinearity

        weight_nums, bias_nums = [], []
        weight_nums.append(8 * 8)
        weight_nums.append(8 * 8)
        weight_nums.append(8 * 1)
        bias_nums.append(8)
        bias_nums.append(8)
        bias_nums.append(1)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums


        self.GAP1 = nn.Sequential(
            nn.GroupNorm(12, 516),
            nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            # nn.Conv3d(256, 256, kernel_size=1, stride=1, padding=0)
        )
        self.GAP2 = nn.Sequential(
            nn.GroupNorm(16, 64),
            nn.ReLU(inplace =True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            # nn.Conv3d(256, 256, kernel_size=1, stride=1, padding=0)
        )
        self.controller = nn.Conv2d(1028, sum(weight_nums+bias_nums), kernel_size=1, stride=1, padding=0)

        self.upsample = nn.Upsample(scale_factor=(2, 2), mode='bilinear')



        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)


        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.textlatterpart = textBlock()
        self.beforemid = nn.Conv2d(580, 516, kernel_size=3, stride=1, padding=1)
        self.beforemidrelu = nonlinearity
        self.dblock = DACblock(512)
        self.spp = SPPblock(512)

        self.decoder4 = DecoderBlock(516, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, 8, 3, padding=1)

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)
        # params为[N, 169]的张量，N为预测到的mask数量
        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 1)
            # print(weight_splits[l].shape, bias_splits[l].shape)

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            # print(i, x.shape, w.shape)
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def forward(self, x, text = '', textori = ''):
        batchsize, _, h, w = x.shape
        #textencoder
        #x = x.unsqueeze(1).repeat(1, 4, 1, 1, 1)
        #text = text.unsqueeze(1).repeat(1, 4, 1, 1, 1)
        text = self.textconv1(text)
        text = self.textrelu1(text)
        text = self.textconv2(text)
        text = self.textrelu2(text)
        text = self.textconv3(text)
        text = self.textrelu3(text)
        text = self.textconv4(text)
        text = self.textrelu4(text)
        text = self.textconv5(text)
        text = self.textrelu5(text) # 3 4 448 448
        #print(text)

        # Encoder
        #print(x.shape)
        #print(text.shape)
        x = torch.cat((x, text), dim=1) # 3 7 448 448
        x = self.beforefirstcov(x)
        x = self.beforefirstrelu(x)
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)










        # Center
        # 不再把cat后的数据输回去也不过textlatterpart
        #textori  3 4 512
        textorinew = textori[0].unsqueeze(2).unsqueeze(2)
        textnew = self.textlatterpart(text)#3 64 14 14
        e4 = self.dblock(e4)
        e4 = self.spp(e4)#3 516 14 14
        e5 = torch.cat((e4, textnew), dim=1)
        e5 = self.beforemid(e5)
        e5 = self.beforemidrelu(e5)
        prompt1 = self.GAP1(e4) #3 516 1 1
        #prompt2 = self.GAP2(textnew) #3 64 1 1
        #prompt = torch.cat((prompt1, prompt2), dim=1)#3 580 1 1
        #prompt = prompt.view(batchsize, 580)#3 580
        #prompt = prompt.unsqueeze(1).repeat(1, 4, 1)
        #params = self.controller(prompt)#3 153
        # print(params.size(1))
        # print(sum(self.weight_nums))
        # print(sum(self.bias_nums))
        #weights, biases = self.parse_dynamic_params(params, 8, self.weight_nums, self.bias_nums)


        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)



        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)#3 8 448 448
        # out = out.reshape(1, -1, 448, 448)# 1 24 448 448
        # logits = self.heads_forward(out, weights, biases, batchsize)
        # #logits = self.upsample(logits)
        # logits = self.upsample(logits)
        # logits = logits.view(batchsize, 4, 1, 448, 448)

        logits_array = []
        for i in range(batchsize):
            x_cond = torch.cat((prompt1[i].unsqueeze(0).repeat(4, 1, 1, 1), textorinew), dim=1) #580 1 1
            params = self.controller(x_cond)#4 153 1 1
            params.squeeze_(-1).squeeze_(-1)#4 153
            head_inputs = out[i].unsqueeze(0) #1 8 448 448
            head_inputs = head_inputs.repeat(4, 1, 1, 1) #4 8 448 448
            N, _, H, W = head_inputs.size()
            head_inputs = head_inputs.reshape(1, -1, H, W) # 1 32 448 448
            # print(head_inputs.shape, params.shape)
            weights, biases = self.parse_dynamic_params(params, 8, self.weight_nums, self.bias_nums)

            logits = self.heads_forward(head_inputs, weights, biases, N)
            logits_array.append(logits.reshape(1, -1, H, W))

        out = torch.cat(logits_array, dim=0)

        return F.sigmoid(out) #3 4 448 448


class CE_Net_NEW(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(CE_Net_NEW, self).__init__()
        self.textconv1 = nn.Conv2d(4, 4, kernel_size=7, stride=1, padding=0, dilation=3)
        self.textrelu1 = nonlinearity
        self.textconv2 = nn.Conv2d(4, 4, kernel_size=7, stride=1, padding=0, dilation=3)
        self.textrelu2 = nonlinearity
        self.textconv3 = nn.Conv2d(4, 4, kernel_size=7, stride=1, padding=0, dilation=3)
        self.textrelu3 = nonlinearity
        self.textconv4 = nn.Conv2d(4, 4, kernel_size=4, stride=1, padding=0, dilation=2)
        self.textrelu4 = nonlinearity
        self.textconv5 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=0, dilation=2)
        self.textrelu5 = nonlinearity

        self.beforefirstcov = nn.Conv2d(7, 3, kernel_size=3, stride=1, padding=1)
        self.beforefirstrelu = nonlinearity

        weight_nums, bias_nums = [], []
        weight_nums.append(8 * 8)
        weight_nums.append(8 * 8)
        weight_nums.append(8 * 1)
        bias_nums.append(8)
        bias_nums.append(8)
        bias_nums.append(1)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums

        self.GAP1 = nn.Sequential(
            nn.GroupNorm(4, 772),
            nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            # nn.Conv3d(256, 256, kernel_size=1, stride=1, padding=0)
        )
        self.GAP2 = nn.Sequential(
            nn.GroupNorm(16, 64),
            nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            # nn.Conv3d(256, 256, kernel_size=1, stride=1, padding=0)
        )
        self.controller = nn.Conv2d(1284, sum(weight_nums + bias_nums), kernel_size=1, stride=1, padding=0)

        self.upsample = nn.Upsample(scale_factor=(2, 2), mode='bilinear')

        filters = [96, 192, 384, 772]
        resnet = models.resnet34(pretrained=True)

        cnforward = cn.convnextv2_tiny()
        swimtrforward = swin_transformerori.SwinTransformer()

        model_params_swim = torch.load('swin_tiny_patch4_window7_224_22k.pth', map_location="cuda:0")['model']
        model_params_conv = torch.load('convnextv2_tiny_22k_224_ema.pt', map_location="cuda:0")['model']

        cnforward.load_state_dict(model_params_conv)
        swimtrforward.load_state_dict(model_params_swim, False)


        self.downsample1cn = cnforward.downsample_layers[0]
        self.stage1cn = cnforward.stages[0]
        self.downsample2cn = cnforward.downsample_layers[1]
        self.stage2cn = cnforward.stages[1]
        self.downsample3cn = cnforward.downsample_layers[2]
        self.stage3cn = cnforward.stages[2]
        self.downsample4cn = cnforward.downsample_layers[3]
        self.stage4cn = cnforward.stages[3]

        cnforward1 = cn.convnextv2_tiny()
        cnforward1.load_state_dict(model_params_conv)
        self.downsample1cn1 = cnforward1.downsample_layers[0]
        self.stage1cn1 = cnforward1.stages[0]
        self.downsample2cn1 = cnforward1.downsample_layers[1]
        self.stage2cn1 = cnforward1.stages[1]
        self.downsample3cn1 = cnforward1.downsample_layers[2]
        self.stage3cn1 = cnforward1.stages[2]
        self.downsample4cn1 = cnforward1.downsample_layers[3]
        self.stage4cn1 = cnforward1.stages[3]


        self.bfswin1 = swimtrforward.patch_embed
        self.bfswin2 = swimtrforward.pos_drop
        self.swimstage1 = swimtrforward.layers[0]
        self.swimstage2 = swimtrforward.layers[1]
        self.swimstage3 = swimtrforward.layers[2]
        self.swimstage4 = swimtrforward.layers[3]

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.convmid = nn.Conv2d(2304, 768, kernel_size=1, stride=1, padding=0)


        self.midconv1 = nn.Conv2d(192, 96, kernel_size=1, stride=1, padding=0)
        self.midconv2 = nn.Conv2d(384, 192, kernel_size=1, stride=1, padding=0)
        self.midconv3 = nn.Conv2d(1152, 384, kernel_size=1, stride=1, padding=0)



        self.dblock = DACblock(768)
        self.spp = SPPblock(768)

        self.decoder4 = DecoderBlock(772, filters[2]) #384
        self.decoder3 = DecoderBlock(filters[2], filters[1]) #384 192
        self.decoder2 = DecoderBlock(filters[1], filters[0]) #192 96
        self.decoder1 = DecoderBlock(filters[0], filters[0]) #96 96

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, 8, 3, padding=1)

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)
        # params为[N, 169]的张量，N为预测到的mask数量
        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 1)
            # print(weight_splits[l].shape, bias_splits[l].shape)

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            # print(i, x.shape, w.shape)
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def forward(self, x, imglittle, imgmid, text='', textori=''):
        batchsize, _, h, w = x.shape

        # Encoder
        # imgmid 12 48 112 112
        # imglittle 12 192 56 56

        x = self.bfswin1(x)
        x = self.bfswin2(x)# 3 12544 96
        e1 = self.swimstage1(x)#3 3136 192
        e2 = self.swimstage2(e1)#3 784 384
        e3 = self.swimstage3(e2)#3 196 768
        e4 = self.swimstage4(e3)#3 196 768


        # e2a = self.downsample2cn(imgmid)
        # e2a = self.stage2cn(e2a) #3 192 56 56
        # e3a = self.downsample3cn(e2a)
        # e3a = self.stage3cn(e3a)
        # e4a = self.downsample4cn(e3a)
        # e4a = self.stage4cn(e4a)
        #
        # e3b = self.downsample3cn1(imglittle)
        # e3b = self.stage3cn1(e3b)
        # e4b = self.downsample4cn1(e3b)
        # e4b = self.stage4cn1(e4b)

        e1 = e1.transpose(1, 2)
        e1 = e1.view(batchsize, 192, 56, 56)
        #e1 = torch.cat((e1, e2a), dim = 1)
        #e1 = self.midconv2(e1)


        e2 = e2.transpose(1, 2)
        e2 = e2.view(batchsize, 384, 28, 28)
        #e2 = torch.cat((e2, e3a, e3b), dim = 1)
        #e2 = self.midconv3(e2)

        e3 = e3.transpose(1, 2)
        e3 = e3.view(batchsize, 768, 14, 14)
        e4 = e4.transpose(1, 2)
        e4 = e4.view(batchsize, 768, 14, 14)
        #e4 = torch.cat((e4, e4a, e4b), dim = 1)
        #e4 = self.convmid(e4)




        # Center
        # textori  3 4 512
        textorinew = textori[0].unsqueeze(2).unsqueeze(2)
        e4 = self.dblock(e4)
        e4 = self.spp(e4)  # 3 772 14 14
        prompt1 = self.GAP1(e4)  # 3 772 1 1

        # Decoder
        d4 = self.decoder4(e4) + e2 #
        d3 = self.decoder3(d4) + e1 #
        d2 = self.decoder2(d3)
        d1 = self.decoder1(d2) #

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)  # 3 8 448 448
        # out = out.reshape(1, -1, 448, 448)# 1 24 448 448
        # logits = self.heads_forward(out, weights, biases, batchsize)
        # #logits = self.upsample(logits)
        # logits = self.upsample(logits)
        # logits = logits.view(batchsize, 4, 1, 448, 448)

        logits_array = []
        for i in range(batchsize):
            x_cond = torch.cat((prompt1[i].unsqueeze(0).repeat(4, 1, 1, 1), textorinew), dim=1)  # 580 1 1
            params = self.controller(x_cond)  # 4 153 1 1
            params.squeeze_(-1).squeeze_(-1)  # 4 153
            head_inputs = out[i].unsqueeze(0)  # 1 8 448 448
            head_inputs = head_inputs.repeat(4, 1, 1, 1)  # 4 8 448 448
            N, _, H, W = head_inputs.size()
            head_inputs = head_inputs.reshape(1, -1, H, W)  # 1 32 448 448
            # print(head_inputs.shape, params.shape)
            weights, biases = self.parse_dynamic_params(params, 8, self.weight_nums, self.bias_nums)

            logits = self.heads_forward(head_inputs, weights, biases, N)
            logits_array.append(logits.reshape(1, -1, H, W))

        out = torch.cat(logits_array, dim=0)

        return F.sigmoid(out)  # 3 4 448 448







class CE_Net_backbone_DAC_without_atrous(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(CE_Net_backbone_DAC_without_atrous, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = DACblock_without_atrous(512)


        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)
        # e4 = self.spp(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)

class CE_Net_backbone_DAC_with_inception(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(CE_Net_backbone_DAC_with_inception, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = DACblock_with_inception(512)


        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)
        # e4 = self.spp(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)

class CE_Net_backbone_inception_blocks(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(CE_Net_backbone_inception_blocks, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = DACblock_with_inception_blocks(512)


        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)
        # e4 = self.spp(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class CE_Net_OCT(nn.Module):
    def __init__(self, num_classes=12, num_channels=3):
        super(CE_Net_OCT, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = DACblock(512)
        self.spp = SPPblock(512)

        self.decoder4 = DecoderBlock(516, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)
        e4 = self.spp(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return out



class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.max_pool_conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        #x = self.relu(x)
        return F.sigmoid(x)


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x




class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=3, out_ch=1):
        super(U_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        # d1 = self.active(out)

        return out



class CE_Net_NEW11(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(CE_Net_NEW11, self).__init__()
        weight_nums, bias_nums = [], []
        weight_nums.append(8 * 8)
        weight_nums.append(8 * 8)
        weight_nums.append(8 * 1)
        bias_nums.append(8)
        bias_nums.append(8)
        bias_nums.append(1)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums

        self.GAP1 = nn.Sequential(
            nn.GroupNorm(4, 772),
            nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            # nn.Conv3d(256, 256, kernel_size=1, stride=1, padding=0)
        )
        self.GAP2 = nn.Sequential(
            nn.GroupNorm(16, 64),
            nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            # nn.Conv3d(256, 256, kernel_size=1, stride=1, padding=0)
        )
        self.controller = nn.Conv2d(1284, sum(weight_nums + bias_nums), kernel_size=1, stride=1, padding=0)

        self.upsample = nn.Upsample(scale_factor=(2, 2), mode='bilinear')

        filters = [96, 192, 384, 772]

        cnforward = cn.convnextv2_tiny()
        swimtrforward = swin_transformerori.SwinTransformer()

        model_params_swim = torch.load('swin_tiny_patch4_window7_224_22k.pth', map_location="cuda:0")['model']
        model_params_conv = torch.load('convnextv2_tiny_22k_224_ema.pt', map_location="cuda:0")['model']

        cnforward.load_state_dict(model_params_conv)
        swimtrforward.load_state_dict(model_params_swim, False)

        self.convfn1 = nn.Conv2d(96, 96, kernel_size=1, stride=1, padding=0)
        self.convfn2 = nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0)

        self.downsample1cn = cnforward.downsample_layers[0]
        self.stage1cn = cnforward.stages[0]
        self.downsample2cn = cnforward.downsample_layers[1]
        self.stage2cn = cnforward.stages[1]
        self.downsample3cn = cnforward.downsample_layers[2]
        self.stage3cn = cnforward.stages[2]
        self.downsample4cn = cnforward.downsample_layers[3]
        self.stage4cn = cnforward.stages[3]
        self.fusionfeature = fusion_feature()
        cnforward1 = cn.convnextv2_tiny()
        cnforward1.load_state_dict(model_params_conv)
        self.downsample1cn1 = cnforward1.downsample_layers[0]
        self.stage1cn1 = cnforward1.stages[0]
        self.downsample2cn1 = cnforward1.downsample_layers[1]
        self.stage2cn1 = cnforward1.stages[1]
        self.downsample3cn1 = cnforward1.downsample_layers[2]
        self.stage3cn1 = cnforward1.stages[2]
        self.downsample4cn1 = cnforward1.downsample_layers[3]
        self.stage4cn1 = cnforward1.stages[3]


        self.bfswin1 = swimtrforward.patch_embed
        self.bfswin2 = swimtrforward.pos_drop
        self.swimstage1 = swimtrforward.layers[0]
        self.swimstage2 = swimtrforward.layers[1]
        self.swimstage3 = swimtrforward.layers[2]
        self.swimstage4 = swimtrforward.layers[3]

        self.convmid = nn.Conv2d(2304, 768, kernel_size=1, stride=1, padding=0)


        self.midconv1 = nn.Conv2d(192, 96, kernel_size=1, stride=1, padding=0)
        self.midconv2 = nn.Conv2d(384, 192, kernel_size=1, stride=1, padding=0)
        self.midconv3 = nn.Conv2d(1152, 384, kernel_size=1, stride=1, padding=0)



        self.dblock = DACblock(768)
        self.spp = SPPblock(768)

        self.decoder4 = DecoderBlock(772, filters[2]) #384
        self.decoder3 = DecoderBlock(filters[2], filters[1]) #384 192
        self.decoder2 = DecoderBlock(filters[1], filters[0]) #192 96
        self.decoder1 = DecoderBlock(filters[0], filters[0]) #96 96

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, 8, 3, padding=1)

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)
        # params为[N, 169]的张量，N为预测到的mask数量
        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 1)
            # print(weight_splits[l].shape, bias_splits[l].shape)

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            # print(i, x.shape, w.shape)
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def forward(self, x, imglittle, imgmid, text='', textori=''):
        batchsize, _, h, w = x.shape

        # Encoder
        # imgmid 12 48 112 112
        # imglittle 12 192 56 56

        x = self.bfswin1(x)
        x = self.bfswin2(x)# 3 12544 96
        e1 = self.swimstage1(x)#3 3136 192
        e2 = self.swimstage2(e1)#3 784 384
        e3 = self.swimstage3(e2)#3 196 768
        e4 = self.swimstage4(e3)#3 196 768

        imgmid = self.convfn1(imgmid)
        e2a = self.downsample2cn(imgmid)
        e2a = self.stage2cn(e2a) #3 192 56 56
        e3a = self.downsample3cn(e2a)
        e3a = self.stage3cn(e3a)
        e4a = self.downsample4cn(e3a)
        e4a = self.stage4cn(e4a)

        imglittle = self.convfn2(imglittle)
        e3b = self.downsample3cn1(imglittle)
        e3b = self.stage3cn1(e3b)
        e4b = self.downsample4cn1(e3b)
        e4b = self.stage4cn1(e4b)

        e1 = e1.transpose(1, 2)
        e1 = e1.view(batchsize, 192, 56, 56)
        e1 = torch.cat((e1, e2a), dim = 1)
        e1 = self.midconv2(e1)


        e2 = e2.transpose(1, 2)
        e2 = e2.view(batchsize, 384, 28, 28)
        e2 = torch.cat((e2, e3a, e3b), dim = 1)
        e2 = self.midconv3(e2)

        e3 = e3.transpose(1, 2)
        e3 = e3.view(batchsize, 768, 14, 14)
        e4 = e4.transpose(1, 2)
        e4 = e4.view(batchsize, 768, 14, 14)
        e4 = self.fusionfeature(e4, e4b, e4a)        
        e4 = torch.cat((e4, e4a, e4b), dim = 1)
        e4 = self.convmid(e4)

        
        #e4 = torch.cat((e4, e4a, e4b), dim = 1)
        #e4 = self.convmid(e4)




        # Center
        # textori  3 4 512
        textorinew = textori[0].unsqueeze(2).unsqueeze(2)
        e4 = self.dblock(e4)
        e4 = self.spp(e4)  # 3 772 14 14
        prompt1 = self.GAP1(e4)  # 3 772 1 1

        # Decoder
        d4 = self.decoder4(e4) + e2 #
        d3 = self.decoder3(d4) + e1 #
        d2 = self.decoder2(d3)
        d1 = self.decoder1(d2) #

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)  # 3 8 448 448

        logits_array = []
        for i in range(batchsize):
            x_cond = torch.cat((prompt1[i].unsqueeze(0).repeat(4, 1, 1, 1), textorinew), dim=1)  # 580 1 1
            params = self.controller(x_cond)  # 4 153 1 1
            params.squeeze_(-1).squeeze_(-1)  # 4 153
            head_inputs = out[i].unsqueeze(0)  # 1 8 448 448
            head_inputs = head_inputs.repeat(4, 1, 1, 1)  # 4 8 448 448
            N, _, H, W = head_inputs.size()
            head_inputs = head_inputs.reshape(1, -1, H, W)  # 1 32 448 448
            weights, biases = self.parse_dynamic_params(params, 8, self.weight_nums, self.bias_nums)

            logits = self.heads_forward(head_inputs, weights, biases, N)
            logits_array.append(logits.reshape(1, -1, H, W))

        out = torch.cat(logits_array, dim=0)

        return F.sigmoid(out)  # 3 4 448 448




class CE_Net_NEW22(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(CE_Net_NEW22, self).__init__()
        weight_nums, bias_nums = [], []
        weight_nums.append(8 * 8)
        weight_nums.append(8 * 8)
        weight_nums.append(8 * 1)
        bias_nums.append(8)
        bias_nums.append(8)
        bias_nums.append(1)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums

        self.GAP1 = nn.Sequential(
            nn.GroupNorm(4, 772),
            nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            # nn.Conv3d(256, 256, kernel_size=1, stride=1, padding=0)
        )
        self.GAP2 = nn.Sequential(
            nn.GroupNorm(16, 64),
            nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            # nn.Conv3d(256, 256, kernel_size=1, stride=1, padding=0)
        )
        self.controller = nn.Conv2d(1284, sum(weight_nums + bias_nums), kernel_size=1, stride=1, padding=0)

        self.upsample = nn.Upsample(scale_factor=(2, 2), mode='bilinear')

        filters = [96, 192, 384, 772]

        cnforward = cn.convnextv2_tiny()
        swimtrforward = swin_transformerori.SwinTransformer()

        model_params_swim = torch.load('swin_tiny_patch4_window7_224_22k.pth', map_location="cuda:0")['model']
        model_params_conv = torch.load('convnextv2_tiny_22k_224_ema.pt', map_location="cuda:0")['model']

        cnforward.load_state_dict(model_params_conv)
        swimtrforward.load_state_dict(model_params_swim, False)


        self.downsample1cn = cnforward.downsample_layers[0]
        self.stage1cn = cnforward.stages[0]
        self.downsample2cn = cnforward.downsample_layers[1]
        self.stage2cn = cnforward.stages[1]
        self.downsample3cn = cnforward.downsample_layers[2]
        self.stage3cn = cnforward.stages[2]
        self.downsample4cn = cnforward.downsample_layers[3]
        self.stage4cn = cnforward.stages[3]

        cnforward1 = cn.convnextv2_tiny()
        cnforward1.load_state_dict(model_params_conv)
        self.downsample1cn1 = cnforward1.downsample_layers[0]
        self.stage1cn1 = cnforward1.stages[0]
        self.downsample2cn1 = cnforward1.downsample_layers[1]
        self.stage2cn1 = cnforward1.stages[1]
        self.downsample3cn1 = cnforward1.downsample_layers[2]
        self.stage3cn1 = cnforward1.stages[2]
        self.downsample4cn1 = cnforward1.downsample_layers[3]
        self.stage4cn1 = cnforward1.stages[3]


        self.bfswin1 = swimtrforward.patch_embed
        self.bfswin2 = swimtrforward.pos_drop
        self.swimstage1 = swimtrforward.layers[0]
        self.swimstage2 = swimtrforward.layers[1]
        self.swimstage3 = swimtrforward.layers[2]
        self.swimstage4 = swimtrforward.layers[3]

        self.convmid = nn.Conv2d(2304, 768, kernel_size=1, stride=1, padding=0)


        self.midconv1 = nn.Conv2d(192, 96, kernel_size=1, stride=1, padding=0)
        self.midconv2 = nn.Conv2d(384, 192, kernel_size=1, stride=1, padding=0)
        self.midconv3 = nn.Conv2d(1152, 384, kernel_size=1, stride=1, padding=0)



        self.dblock = DACblock(768)
        self.spp = SPPblock(768)

        self.decoder4 = DecoderBlock(772, filters[2]) #384
        self.decoder3 = DecoderBlock(filters[2], filters[1]) #384 192
        self.decoder2 = DecoderBlock(filters[1], filters[0]) #192 96
        self.decoder1 = DecoderBlock(filters[0], filters[0]) #96 96

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, 8, 3, padding=1)

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)
        # params为[N, 169]的张量，N为预测到的mask数量
        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 1)
            # print(weight_splits[l].shape, bias_splits[l].shape)

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            # print(i, x.shape, w.shape)
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def forward(self, x, imglittle, imgmid, text='', textori=''):
        batchsize, _, h, w = x.shape

        # Encoder
        # imgmid 12 48 112 112
        # imglittle 12 192 56 56

        x = self.downsample1cn(x)
        e1 = self.stage1cn(x)
        e2 = self.downsample2cn(e1)
        e2 = self.stage2cn(e2)
        e3 = self.downsample3cn(e2)
        e3 = self.stage3cn(e3)
        e4 = self.downsample4cn(e3)
        e4 = self.stage4cn(e4)


        # Center
        # textori  3 4 512
        textorinew = textori[0].unsqueeze(2).unsqueeze(2)
        e4 = self.dblock(e4)
        e4 = self.spp(e4)  # 3 772 14 14
        prompt1 = self.GAP1(e4)  # 3 772 1 1

        # Decoder
        d4 = self.decoder4(e4) + e3 #
        d3 = self.decoder3(d4) + e2 #
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2) #

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)  # 3 8 448 448

        logits_array = []
        for i in range(batchsize):
            x_cond = torch.cat((prompt1[i].unsqueeze(0).repeat(4, 1, 1, 1), textorinew), dim=1)  # 580 1 1
            params = self.controller(x_cond)  # 4 153 1 1
            params.squeeze_(-1).squeeze_(-1)  # 4 153
            head_inputs = out[i].unsqueeze(0)  # 1 8 448 448
            head_inputs = head_inputs.repeat(4, 1, 1, 1)  # 4 8 448 448
            N, _, H, W = head_inputs.size()
            head_inputs = head_inputs.reshape(1, -1, H, W)  # 1 32 448 448
            weights, biases = self.parse_dynamic_params(params, 8, self.weight_nums, self.bias_nums)

            logits = self.heads_forward(head_inputs, weights, biases, N)
            logits_array.append(logits.reshape(1, -1, H, W))

        out = torch.cat(logits_array, dim=0)

        return F.sigmoid(out)  # 3 4 448 448










class CE_Net_NEWOCT(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(CE_Net_NEWOCT, self).__init__()

        weight_nums, bias_nums = [], []
        weight_nums.append(8 * 8)
        weight_nums.append(8 * 8)
        weight_nums.append(8 * 1)
        bias_nums.append(8)
        bias_nums.append(8)
        bias_nums.append(1)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums

        self.GAP1 = nn.Sequential(
            nn.GroupNorm(4, 772),
            nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            # nn.Conv3d(256, 256, kernel_size=1, stride=1, padding=0)
        )
        self.GAP2 = nn.Sequential(
            nn.GroupNorm(16, 64),
            nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            # nn.Conv3d(256, 256, kernel_size=1, stride=1, padding=0)
        )
        self.controller = nn.Conv2d(1284, sum(weight_nums + bias_nums), kernel_size=1, stride=1, padding=0)

        self.upsample = nn.Upsample(scale_factor=(2, 2), mode='bilinear')

        filters = [96, 192, 384, 772]
        resnet = models.resnet34(pretrained=True)

        cnforward = cn.convnextv2_tiny()
        swimtrforward = swin_transformerori.SwinTransformer()

        model_params_swim = torch.load('swin_tiny_patch4_window7_224_22k.pth', map_location="cuda:0")['model']
        model_params_conv = torch.load('convnextv2_tiny_22k_224_ema.pt', map_location="cuda:0")['model']

        cnforward.load_state_dict(model_params_conv)
        swimtrforward.load_state_dict(model_params_swim, False)


        self.downsample1cn = cnforward.downsample_layers[0]
        self.stage1cn = cnforward.stages[0]
        self.downsample2cn = cnforward.downsample_layers[1]
        self.stage2cn = cnforward.stages[1]
        self.downsample3cn = cnforward.downsample_layers[2]
        self.stage3cn = cnforward.stages[2]
        self.downsample4cn = cnforward.downsample_layers[3]
        self.stage4cn = cnforward.stages[3]

        cnforward1 = cn.convnextv2_tiny()
        cnforward1.load_state_dict(model_params_conv)
        self.downsample1cn1 = cnforward1.downsample_layers[0]
        self.stage1cn1 = cnforward1.stages[0]
        self.downsample2cn1 = cnforward1.downsample_layers[1]
        self.stage2cn1 = cnforward1.stages[1]
        self.downsample3cn1 = cnforward1.downsample_layers[2]
        self.stage3cn1 = cnforward1.stages[2]
        self.downsample4cn1 = cnforward1.downsample_layers[3]
        self.stage4cn1 = cnforward1.stages[3]


        self.bfswin1 = swimtrforward.patch_embed
        self.bfswin2 = swimtrforward.pos_drop
        self.swimstage1 = swimtrforward.layers[0]
        self.swimstage2 = swimtrforward.layers[1]
        self.swimstage3 = swimtrforward.layers[2]
        self.swimstage4 = swimtrforward.layers[3]

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.convmid = nn.Conv2d(2304, 768, kernel_size=1, stride=1, padding=0)


        self.midconv1 = nn.Conv2d(192, 96, kernel_size=1, stride=1, padding=0)
        self.midconv2 = nn.Conv2d(384, 192, kernel_size=1, stride=1, padding=0)
        self.midconv3 = nn.Conv2d(1152, 384, kernel_size=1, stride=1, padding=0)



        self.dblock = DACblock(768)
        self.spp = SPPblock(768)

        self.decoder4 = DecoderBlock(772, filters[2]) #384
        self.decoder3 = DecoderBlock(filters[2], filters[1]) #384 192
        self.decoder2 = DecoderBlock(filters[1], filters[0]) #192 96
        self.decoder1 = DecoderBlock(filters[0], filters[0]) #96 96

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, 8, 3, padding=1)

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)
        # params为[N, 169]的张量，N为预测到的mask数量
        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 1)
            # print(weight_splits[l].shape, bias_splits[l].shape)

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            # print(i, x.shape, w.shape)
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def forward(self, x, imglittle, imgmid, text='', textori=''):
        batchsize, _, h, w = x.shape

        # Encoder
        # imgmid 12 48 112 112
        # imglittle 12 192 56 56

        x = self.bfswin1(x)
        x = self.bfswin2(x)# 3 12544 96
        e1 = self.swimstage1(x)#3 3136 192
        e2 = self.swimstage2(e1)#3 784 384
        e3 = self.swimstage3(e2)#3 196 768
        e4 = self.swimstage4(e3)#3 196 768


        # e2a = self.downsample2cn(imgmid)
        # e2a = self.stage2cn(e2a) #3 192 56 56
        # e3a = self.downsample3cn(e2a)
        # e3a = self.stage3cn(e3a)
        # e4a = self.downsample4cn(e3a)
        # e4a = self.stage4cn(e4a)
        #
        # e3b = self.downsample3cn1(imglittle)
        # e3b = self.stage3cn1(e3b)
        # e4b = self.downsample4cn1(e3b)
        # e4b = self.stage4cn1(e4b)

        e1 = e1.transpose(1, 2)
        e1 = e1.view(batchsize, 192, 56, 56)
        #e1 = torch.cat((e1, e2a), dim = 1)
        #e1 = self.midconv2(e1)


        e2 = e2.transpose(1, 2)
        e2 = e2.view(batchsize, 384, 28, 28)
        #e2 = torch.cat((e2, e3a, e3b), dim = 1)
        #e2 = self.midconv3(e2)

        e3 = e3.transpose(1, 2)
        e3 = e3.view(batchsize, 768, 14, 14)
        e4 = e4.transpose(1, 2)
        e4 = e4.view(batchsize, 768, 14, 14)
        #e4 = torch.cat((e4, e4a, e4b), dim = 1)
        #e4 = self.convmid(e4)




        # Center
        # textori  3 4 512
        textorinew = textori[0].unsqueeze(2).unsqueeze(2)
        e4 = self.dblock(e4)
        e4 = self.spp(e4)  # 3 772 14 14
        prompt1 = self.GAP1(e4)  # 3 772 1 1

        # Decoder
        d4 = self.decoder4(e4) + e2 #
        d3 = self.decoder3(d4) + e1 #
        d2 = self.decoder2(d3)
        d1 = self.decoder1(d2) #

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)  # 3 8 448 448
        # out = out.reshape(1, -1, 448, 448)# 1 24 448 448
        # logits = self.heads_forward(out, weights, biases, batchsize)
        # #logits = self.upsample(logits)
        # logits = self.upsample(logits)
        # logits = logits.view(batchsize, 4, 1, 448, 448)

        logits_array = []
        for i in range(batchsize):
            x_cond = torch.cat((prompt1[i].unsqueeze(0).repeat(3, 1, 1, 1), textorinew), dim=1)  # 580 1 1
            params = self.controller(x_cond)  # 4 153 1 1
            params.squeeze_(-1).squeeze_(-1)  # 4 153
            head_inputs = out[i].unsqueeze(0)  # 1 8 448 448
            head_inputs = head_inputs.repeat(3, 1, 1, 1)  # 4 8 448 448
            N, _, H, W = head_inputs.size()
            head_inputs = head_inputs.reshape(1, -1, H, W)  # 1 32 448 448
            # print(head_inputs.shape, params.shape)
            weights, biases = self.parse_dynamic_params(params, 8, self.weight_nums, self.bias_nums)

            logits = self.heads_forward(head_inputs, weights, biases, N)
            logits_array.append(logits.reshape(1, -1, H, W))

        out = torch.cat(logits_array, dim=0)

        return F.sigmoid(out)  # 3 4 448 448