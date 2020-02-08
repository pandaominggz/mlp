import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNet(nn.Module):
    def __init__(self, height, width, channels, maxdisp):
        super(GCNet, self).__init__()

        self.height = height
        self.width = width
        self.channels = channels
        self.maxdisp = int(maxdisp / 2)
        self.build_module()

    def build_module(self):
        self.layer_dict = nn.ModuleDict()
        self.layer_dict['section_1'] = SectionOne(self.height, self.width, self.channels)
        self.layer_dict['section_2'] = SectionTwo(self.height, self.width,self.maxdisp)
        self.layer_dict['section_3'] = SectionThree(self.height, self.width, self.maxdisp)
        self.layer_dict['section_4'] = SectionFour()

    def forward(self, imgl, imgr):
        imgl, imgr = self.layer_dict['section_1'].forward(imgl,imgr)
        cost_volume = self.layer_dict['section_2'].forward(imgl,imgr)
        out = self.layer_dict['section_3'].forward(cost_volume)
        out = self.layer_dict['section_4'].forward(out)
        return out


class SectionOne(nn.Module):
    def __init__(self, height, width, channels):
        super(SectionOne, self).__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.build_module()

    def build_module(self):
        self.layer_dict = nn.ModuleDict()

        self.layer_dict['conv2d_0'] = nn.Conv2d(self.channels, 32, (5, 5), 2, 2)  # 1
        self.layer_dict['bn2d_0'] = nn.BatchNorm2d(32)
        self.layer_dict['relu2d_0'] = nn.ReLU()
        self.resNetBlock()  # 2-16
        self.layer_dict['conv2d_1'] = nn.Conv2d(32, 32, (3, 3), 1, 1)  # 18

    def resNetBlock(self):
        for i in range(8 * 2):
            self.layer_dict['res_conv2d_' + str(i)] = nn.Conv2d(32, 32, (3, 3), 1, 1)
            self.layer_dict['res_bn2d_' + str(i)] = nn.BatchNorm2d(32)

            if i % 2 == 0:
                self.layer_dict['res_relu2d_' + str(i)] = nn.ReLU()
            else:
                self.layer_dict['res_relu2d_' + str(i)] = nn.ReLU()

    def forward(self, imgl, imgr):
        imgl = self.layer_dict['conv2d_0'].forward(imgl)
        imgr = self.layer_dict['conv2d_0'].forward(imgr)
        imgl = self.layer_dict['bn2d_0'].forward(imgl)
        imgr = self.layer_dict['bn2d_0'].forward(imgr)
        imgl = self.layer_dict['relu2d_0'].forward(imgl)
        imgr = self.layer_dict['relu2d_0'].forward(imgr)

        outputl, outputr = imgl, imgr
        for i in range(8 * 2):
            outputl = self.layer_dict['res_conv2d_' + str(i)].forward(outputl)
            outputr = self.layer_dict['res_conv2d_' + str(i)].forward(outputr)
            outputl = self.layer_dict['res_bn2d_' + str(i)].forward(outputl)
            outputr = self.layer_dict['res_bn2d_' + str(i)].forward(outputr)

            if i % 2 == 0:
                outputl = self.layer_dict['res_relu2d_' + str(i)].forward(outputl)
                outputr = self.layer_dict['res_relu2d_' + str(i)].forward(outputr)
            else:
                outputl = outputl + imgl
                outputr = outputl + imgr
                outputl = self.layer_dict['res_relu2d_' + str(i)].forward(outputl)
                outputr = self.layer_dict['res_relu2d_' + str(i)].forward(outputr)
                imgl = outputl
                imgr = outputr

        outputl = self.layer_dict['conv2d_1'].forward(outputl)
        outputr = self.layer_dict['conv2d_1'].forward(outputr)
        return outputl, outputr


class SectionTwo(nn.Module):
    def __init__(self, height, width,maxdisp):
        super(SectionTwo, self).__init__()
        self.height = height
        self.width = width
        self.maxdisp = maxdisp

    def cost_volume(self, imgl, imgr):
        xx_list = []
        pad_opr1 = nn.ZeroPad2d((0, self.maxdisp, 0, 0))
        xleft = pad_opr1(imgl)
        for d in range(self.maxdisp):
            pad_opr2 = nn.ZeroPad2d((d, self.maxdisp - d, 0, 0))
            xright = pad_opr2(imgr)
            xx_temp = torch.cat((xleft, xright), 1)
            xx_list.append(xx_temp)
        xx = torch.cat(xx_list, 1)
        xx = xx.view(1, self.maxdisp, 64, int(self.height / 2), int(self.width / 2) + self.maxdisp)
        xx0 = xx.permute(0, 2, 1, 3, 4)
        xx0 = xx0[:, :, :, :, :int(self.width / 2)]
        return xx0

    def forward(self, imgl, imgr):
        x = self.cost_volume(imgl, imgr)
        return x


class SectionThree(nn.Module):
    def __init__(self, height, width, maxdisp):
        super(SectionThree, self).__init__()
        self.height = height
        self.width = width
        self.maxdisp = maxdisp
        self.build_module()

    def build_module(self):
        self.layer_dict = nn.ModuleDict()

        self.layer_dict['conv3d_0'] = nn.Conv3d(64, 32, (3, 3, 3), 1, 1)  # 19
        self.layer_dict['bn3d_0'] = nn.BatchNorm3d(32)
        self.layer_dict['relu3d_0'] = nn.ReLU()

        self.layer_dict['conv3d_1'] = nn.Conv3d(32, 32, (3, 3, 3), 1, 1)  # 20
        self.layer_dict['bn3d_1'] = nn.BatchNorm3d(32)
        self.layer_dict['relu3d_1'] = nn.ReLU()

        # 1
        self.layer_dict['conv3d_2'] = nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1)
        self.layer_dict['bn3d_2'] = nn.BatchNorm3d(64)
        self.layer_dict['relu3d_2'] = nn.ReLU()

        self.layer_dict['conv3d_3'] = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.layer_dict['bn3d_3'] = nn.BatchNorm3d(64)
        self.layer_dict['relu3d_3'] = nn.ReLU()

        self.layer_dict['conv3d_4'] = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.layer_dict['bn3d_4'] = nn.BatchNorm3d(64)
        self.layer_dict['relu3d_4'] = nn.ReLU()

        # 2
        self.layer_dict['conv3d_5'] = nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1)
        self.layer_dict['bn3d_5'] = nn.BatchNorm3d(64)
        self.layer_dict['relu3d_5'] = nn.ReLU()

        self.layer_dict['conv3d_6'] = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.layer_dict['bn3d_6'] = nn.BatchNorm3d(64)
        self.layer_dict['relu3d_6'] = nn.ReLU()

        self.layer_dict['conv3d_7'] = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.layer_dict['bn3d_7'] = nn.BatchNorm3d(64)
        self.layer_dict['relu3d_7'] = nn.ReLU()

        # 3
        self.layer_dict['conv3d_8'] = nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1)
        self.layer_dict['bn3d_8'] = nn.BatchNorm3d(64)
        self.layer_dict['relu3d_8'] = nn.ReLU()

        self.layer_dict['conv3d_9'] = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.layer_dict['bn3d_9'] = nn.BatchNorm3d(64)
        self.layer_dict['relu3d_9'] = nn.ReLU()

        self.layer_dict['conv3d_10'] = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.layer_dict['bn3d_10'] = nn.BatchNorm3d(64)
        self.layer_dict['relu3d_10'] = nn.ReLU()

        # 4
        self.layer_dict['conv3d_11'] = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.layer_dict['bn3d_11'] = nn.BatchNorm3d(128)
        self.layer_dict['relu3d_11'] = nn.ReLU()

        self.layer_dict['conv3d_12'] = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1)
        self.layer_dict['bn3d_12'] = nn.BatchNorm3d(128)
        self.layer_dict['relu3d_12'] = nn.ReLU()

        self.layer_dict['conv3d_13'] = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1)
        self.layer_dict['bn3d_13'] = nn.BatchNorm3d(128)
        self.layer_dict['relu3d_13'] = nn.ReLU()

        self.layer_dict['Tconv3d_0'] = nn.ConvTranspose3d(128, 64, 3, 2, 1, 1)
        self.layer_dict['Tbn3d_0'] = nn.BatchNorm3d(64)
        self.layer_dict['Trelu3d_0'] = nn.ReLU()

        self.layer_dict['Tconv3d_1'] = nn.ConvTranspose3d(64, 64, 3, 2, 1, 1)
        self.layer_dict['Tbn3d_1'] = nn.BatchNorm3d(64)
        self.layer_dict['Trelu3d_1'] = nn.ReLU()

        self.layer_dict['Tconv3d_2'] = nn.ConvTranspose3d(64, 64, 3, 2, 1, 1)
        self.layer_dict['Tbn3d_2'] = nn.BatchNorm3d(64)
        self.layer_dict['Trelu3d_2'] = nn.ReLU()

        self.layer_dict['Tconv3d_3'] = nn.ConvTranspose3d(64, 32, 3, 2, 1, 1)
        self.layer_dict['Tbn3d_3'] = nn.BatchNorm3d(32)
        self.layer_dict['Trelu3d_3'] = nn.ReLU()

        self.layer_dict['Tconv3d_4'] = nn.ConvTranspose3d(32, 1, 3, 2, 1, 1)

        self.layer_dict['conv3d_14'] = nn.Conv3d(64, 64, 3, 2, 1)
        self.layer_dict['bn3d_14'] = nn.BatchNorm3d(64)
        self.layer_dict['relu3d_14'] = nn.ReLU()
        self.layer_dict['conv3d_15'] = nn.Conv3d(64, 64, 3, 2, 1)
        self.layer_dict['bn3d_15'] = nn.BatchNorm3d(64)
        self.layer_dict['relu3d_15'] = nn.ReLU()
        self.layer_dict['conv3d_16'] = nn.Conv3d(64, 64, 3, 2, 1)
        self.layer_dict['bn3d_16'] = nn.BatchNorm3d(64)
        self.layer_dict['relu3d_16'] = nn.ReLU()

    def forward(self, cost_volum):
        x = cost_volum
        x = self.layer_dict['conv3d_0'].forward(x)
        x = self.layer_dict['bn3d_0'].forward(x)
        x = self.layer_dict['relu3d_0'].forward(x)
        x = self.layer_dict['conv3d_1'].forward(x)
        x = self.layer_dict['bn3d_1'].forward(x)
        conv3d_20 = self.layer_dict['relu3d_1'].forward(x)

        conv3d_block_1 = self.layer_dict['conv3d_2'].forward(cost_volum)
        conv3d_block_1 = self.layer_dict['bn3d_2'].forward(conv3d_block_1)
        conv3d_block_1 = self.layer_dict['relu3d_2'].forward(conv3d_block_1)
        conv3d_block_1 = self.layer_dict['conv3d_3'].forward(conv3d_block_1)
        conv3d_block_1 = self.layer_dict['bn3d_3'].forward(conv3d_block_1)
        conv3d_block_1 = self.layer_dict['relu3d_3'].forward(conv3d_block_1)
        conv3d_block_1 = self.layer_dict['conv3d_4'].forward(conv3d_block_1)
        conv3d_block_1 = self.layer_dict['bn3d_4'].forward(conv3d_block_1)
        conv3d_block_1 = self.layer_dict['relu3d_4'].forward(conv3d_block_1)

        # conv3d block
        conv3d_21 = self.layer_dict['conv3d_14'].forward(cost_volum)
        conv3d_21 = self.layer_dict['bn3d_14'].forward(conv3d_21)
        conv3d_21 = self.layer_dict['relu3d_14'].forward(conv3d_21)

        conv3d_block_2 = self.layer_dict['conv3d_5'].forward(conv3d_21)
        conv3d_block_2 = self.layer_dict['bn3d_5'].forward(conv3d_block_2)
        conv3d_block_2 = self.layer_dict['relu3d_5'].forward(conv3d_block_2)
        conv3d_block_2 = self.layer_dict['conv3d_6'].forward(conv3d_block_2)
        conv3d_block_2 = self.layer_dict['bn3d_6'].forward(conv3d_block_2)
        conv3d_block_2 = self.layer_dict['relu3d_6'].forward(conv3d_block_2)
        conv3d_block_2 = self.layer_dict['conv3d_7'].forward(conv3d_block_2)
        conv3d_block_2 = self.layer_dict['bn3d_7'].forward(conv3d_block_2)
        conv3d_block_2 = self.layer_dict['relu3d_7'].forward(conv3d_block_2)

        conv3d_24 = self.layer_dict['conv3d_15'].forward(conv3d_21)
        conv3d_24 = self.layer_dict['bn3d_15'].forward(conv3d_24)
        conv3d_24 = self.layer_dict['relu3d_15'].forward(conv3d_24)

        conv3d_block_3 = self.layer_dict['conv3d_8'].forward(conv3d_24)
        conv3d_block_3 = self.layer_dict['bn3d_8'].forward(conv3d_block_3)
        conv3d_block_3 = self.layer_dict['relu3d_8'].forward(conv3d_block_3)
        conv3d_block_3 = self.layer_dict['conv3d_9'].forward(conv3d_block_3)
        conv3d_block_3 = self.layer_dict['bn3d_9'].forward(conv3d_block_3)
        conv3d_block_3 = self.layer_dict['relu3d_9'].forward(conv3d_block_3)
        conv3d_block_3 = self.layer_dict['conv3d_10'].forward(conv3d_block_3)
        conv3d_block_3 = self.layer_dict['bn3d_10'].forward(conv3d_block_3)
        conv3d_block_3 = self.layer_dict['relu3d_10'].forward(conv3d_block_3)

        conv3d_27 = self.layer_dict['conv3d_15'].forward(conv3d_24)
        conv3d_27 = self.layer_dict['bn3d_15'].forward(conv3d_27)
        conv3d_27 = self.layer_dict['relu3d_15'].forward(conv3d_27)

        conv3d_block_4 = self.layer_dict['conv3d_11'].forward(conv3d_27)
        conv3d_block_4 = self.layer_dict['bn3d_11'].forward(conv3d_block_4)
        conv3d_block_4 = self.layer_dict['relu3d_11'].forward(conv3d_block_4)
        conv3d_block_4 = self.layer_dict['conv3d_12'].forward(conv3d_block_4)
        conv3d_block_4 = self.layer_dict['bn3d_12'].forward(conv3d_block_4)
        conv3d_block_4 = self.layer_dict['relu3d_12'].forward(conv3d_block_4)
        conv3d_block_4 = self.layer_dict['conv3d_13'].forward(conv3d_block_4)
        conv3d_block_4 = self.layer_dict['bn3d_13'].forward(conv3d_block_4)
        conv3d_block_4 = self.layer_dict['relu3d_13'].forward(conv3d_block_4)

        # deconv
        deconv3d = self.layer_dict['Tconv3d_0'].forward(conv3d_block_4)
        deconv3d = self.layer_dict['Tbn3d_0'].forward(deconv3d + conv3d_block_3)
        deconv3d = self.layer_dict['Trelu3d_0'].forward(deconv3d)

        deconv3d = self.layer_dict['Tconv3d_1'].forward(deconv3d)
        deconv3d = self.layer_dict['Tbn3d_1'].forward(deconv3d + conv3d_block_2)
        deconv3d = self.layer_dict['Trelu3d_1'].forward(deconv3d)

        deconv3d = self.layer_dict['Tconv3d_2'].forward(deconv3d)
        deconv3d = self.layer_dict['Tbn3d_2'].forward(deconv3d + conv3d_block_1)
        deconv3d = self.layer_dict['Trelu3d_2'].forward(deconv3d)

        deconv3d = self.layer_dict['Tconv3d_3'].forward(deconv3d)
        deconv3d = self.layer_dict['Tbn3d_3'].forward(deconv3d + conv3d_20)
        deconv3d = self.layer_dict['Trelu3d_3'].forward(deconv3d)
        print(deconv3d.shape)

        deconv3d = self.layer_dict['Tconv3d_4'].forward(deconv3d)
        out = deconv3d.view(1, self.maxdisp * 2, self.height, self.width)

        return out


class SectionFour(nn.Module):
    def __init__(self):
        super(SectionFour, self).__init__()
        self.build_module()

    def build_module(self):
        self.layer_dict = nn.ModuleDict()
        self.layer_dict['softMax'] = nn.Softmax()

    def forward(self, x):
        # x = -x
        # x = self.layer_dict['softMax'].forward(x)
        prob = F.softmax(-x, 1)
        return x
