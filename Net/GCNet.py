import torch
import torch.nn as nn


class GCNet(nn.Module):
    def __init__(self, height, width, channels):
        super(GCNet, self).__init__()

        # initial something
        self.height = height
        self.width = width
        self.channels = channels
        self.build_module()

    def build_module(self):
        self.layer_dict = nn.ModuleDict()
        section1 = SectionOne(self.height, self.width, self.channels)
        # print(section1)
        section2 = SectionTwo(self.height, self.width, self.channels)
        # print(section2)

    def forward(self, x):
        return x


class SectionOne(nn.Module):
    def __init__(self, height, width, channels):
        super(SectionOne, self).__init__()

        # initial something
        self.height = height
        self.width = width
        self.channels = channels
        self.features = 32
        self.num_resBlock = 8
        self.build_module()

    def build_module(self):
        self.layer_dict = nn.ModuleDict()
        x = torch.zeros((self.height, self.width))

        self.layer_dict['conv_0'] = nn.Conv2d(self.channels, self.features, (5, 5), 2, 2)
        x = self.layer_dict['conv_0'].forward(x)

        self.layer_dict['bn_0'] = nn.BatchNorm2d(32)
        x = self.layer_dict['bn_0'].forward(x)

        self.layer_dict['relu_0'] = nn.ReLU()
        x = self.layer_dict['relu_0'].forward(x)

        x = self.resNetBlock(self.num_resBlock, x)

        self.layer_dict['conv_1'] = nn.Conv2d(self.features, self.features, (3, 3), 1, 1)
        x = self.layer_dict['conv_1'].forward(x)

        return x

    def resNetBlock(self, input):
        output = input
        for i in range(self.num_resBlock * 2):

            self.layer_dict['res_conv_' + i] = nn.Conv2d(self.features, self.features, (3, 3), 1, 1)
            output = self.layer_dict['res_conv_' + i].forward(output)

            self.layer_dict['res_bn_' + i] = nn.BatchNorm2d(32)
            output = self.layer_dict['res_bn_' + i].forward(output)

            if i % 2 == 0:
                self.layer_dict['res_relu_' + i] = nn.ReLU()
                output = self.layer_dict['res_relu_' + i].forward(output)

            else:
                output = output + input
                self.layer_dict['res_relu_' + i] = nn.ReLU()
                output = self.layer_dict['res_relu_' + i].forward(output)
                input = output

        return output

    def forward(self, x):
        x = self.layer_dict['conv_0'].forward(x)
        x = self.layer_dict['bn_0'].forward(x)
        x = self.layer_dict['relu_0'].forward(x)
        output = x
        for i in range(self.num_resBlock * 2):
            output = self.layer_dict['res_conv_' + i].forward(output)
            output = self.layer_dict['res_bn_' + i].forward(output)
            if i % 2 == 0:
                output = self.layer_dict['res_relu_' + i].forward(output)
            else:
                output = output + x
                output = self.layer_dict['res_relu_' + i].forward(output)
                x = output

        output = self.layer_dict['conv_1'].forward(output)
        return output


class SectionTwo(nn.Module):
    def __init__(self):
        super(SectionTwo, self).__init__()
        # initial something
        self.build_module()

    def build_module(self):
        self.layer_dict = nn.ModuleDict()
        x = torch.zeros((self.height, self.width))
        return x

    def forward(self, x):
        return 0
