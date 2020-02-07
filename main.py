from dataloader.data_loader import DataLoader as dL
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from Net.GCNet import GCNet
from torch import optim
import torch.nn as nn
import numpy as np
import torch
import os


def main(setType, transform):
    if setType == 'train':
        height = 540
        width = 960
        channels = 3
        maxdisp = 4
        batch = 1
        epoch_total = 20
        with torch.no_grad():
            GcNet = GCNet(height, width, channels, maxdisp)
            net = GcNet.to(0)
            data = DataLoader(dL(setType, transform), batch_size=batch, shuffle=True, num_workers=1)
            # net = torch.nn.DataParallel(GcNet).cuda()
            train(net, data, height, width,maxdisp, batch, epoch_total)
    elif setType == 'test':
        test()


def train(net, dataloader, height, width, maxdisp, batch_size, epoch_total):
    loss_mul_list = []
    for d in range(maxdisp):
        loss_mul_temp = Variable(torch.Tensor(np.ones([batch_size, 1, height, width]) * d)).cuda()
        loss_mul_list.append(loss_mul_temp)
    loss_mul = torch.cat(loss_mul_list, 1)
    optimizer = optim.RMSprop(net.parameters(), lr=0.001, alpha=0.9)
    loss_fn = nn.L1Loss()
    imL = Variable(torch.FloatTensor(1).cuda())
    imR = Variable(torch.FloatTensor(1).cuda())
    dispL = Variable(torch.FloatTensor(1).cuda())
    start_epoch = 0
    for epoch in range(start_epoch, epoch_total):
        net.train()
        data_iter = iter(dataloader)

        print('\nEpoch: %d' % epoch)
        train_loss = 0
        acc_total = 0
        for step in range(len(dataloader)):
        # for step in range(len(dataloader) - 1):
            print('----epoch:%d------step:%d------' % (epoch, step))
            data = next(data_iter)
            # randomH = np.random.randint(0, 160)
            # randomW = np.random.randint(0, 400)
            # imageL = data['imgL'][:, :, randomH:(randomH + height), randomW:(randomW + width)]
            # imageR = data['imgR'][:, :, randomH:(randomH + height), randomW:(randomW + width)]
            # disL = data['dispL'][:, :, randomH:(randomH + height), randomW:(randomW + width)]
            imageL = data['imgL']
            imageR = data['imgR']
            disL = data['dispL']
            # imL.data.resize_(imageL.size()).copy_(imageL)
            # imR.data.resize_(imageR.size()).copy_(imageR)
            # dispL.data.resize_(disL.size()).copy_(disL)
            imL.resize_(imageL.size()).copy_(imageL)
            imR.resize_(imageR.size()).copy_(imageR)
            dispL.resize_(disL.size()).copy_(disL)
            # normalize
            # imgL=normalizeRGB(imL)
            # imgR=normalizeRGB(imR)

            net.zero_grad()
            optimizer.zero_grad()

            x = net(imL, imR)
            # print(x.shape)
            # print(loss_mul.shape)
            # print(net)
            result = torch.sum(x.mul(loss_mul), 1)
            # print(result.shape)
            tt = loss_fn(result, dispL)
            train_loss += tt.data
            # tt = loss(x, loss_mul, dispL)
            tt.backward()
            optimizer.step()
            print('=======loss value for every step=======:%f' % (tt.data))
            print('=======average loss value for every step=======:%f' % (train_loss / (step + 1)))
            result = result.view(batch_size, 1, height, width)
            diff = torch.abs(result.data.cpu() - dispL.data.cpu())
            print(diff.shape)
            accuracy = torch.sum(diff < 3) / float(height * width * batch_size)
            acc_total += accuracy
            print('====accuracy for the result less than 3 pixels===:%f' % accuracy)
            print('====average accuracy for the result less than 3 pixels===:%f' % (acc_total / (step + 1)))


def test():
    pass


def checkpoint(path):
    if not os.path.exists('./checkpoint/ckpt.t7'):
        checkpoint = torch.load('./checkpoint/ckpt.t7')
        # net.load_state_dict(checkpoint['net'])
        # start_epoch = checkpoint['epoch']
        # accu = checkpoint['accur']


if __name__ == '__main__':
    transform_train = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    main('train', transform_train)
