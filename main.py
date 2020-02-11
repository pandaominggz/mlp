from dataloader.data_loader import DataLoader as dL
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from net.GCNet import GCNet
from torch import optim
import torch.nn as nn
import numpy as np
import datetime
import torch
import time
import cv2


# import os


def main(setType, transform):
    if setType == 'train':
        load_state = False
        height = 288
        width = 512
        channels = 3
        maxdisp = 32
        batch = 4
        epoch_total = 50
        # with torch.no_grad():
        #     GcNet = GCNet(height, width, channels, maxdisp)
        #     net = GcNet.to(0)
        #     data = DataLoader(dL(setType, transform), batch_size=batch, shuffle=True, num_workers=1)
        # train(net, data, height, width,maxdisp, batch, epoch_total)
        # GcNet = GCNet(height, width, channels, maxdisp)
        # net = GcNet.to(0)
        # data = DataLoader(dL(setType, transform), batch_size=batch, shuffle=True, num_workers=1)
        # net = torch.nn.DataParallel(GcNet).cuda()
        # train(net, data, height, width,maxdisp, batch, epoch_total)
        start = time.clock()
        try:
            GcNet = GCNet(height, width, channels, maxdisp, batch)
            net = GcNet.to(0)
            data = DataLoader(dL(setType, transform), batch_size=batch, shuffle=True, num_workers=1)
            # net = torch.nn.DataParallel(GcNet).cuda()
            # train(load_state, net, data, height, width, maxdisp, batch, epoch_total)
            train(net, data, height, width, maxdisp, batch, epoch_total)
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                    print(exception)
            else:
                raise exception
        end = time.clock()
        print("running time:",(end - start))

    elif setType == 'test':
        test()


# def train(loadstate, net, dataloader, height, width, maxdisp, batch_size, epoch_total):
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
    loss_list = []
    start_epoch = 0
    # accu = 0
    # if loadstate==True:
    #     checkpoint = torch.load('./checkpoint/ckpt.t7')
    #     net.load_state_dict(checkpoint['net'])
    #     start_epoch = checkpoint['epoch']
    #     accu=checkpoint['accur']
    # print('startepoch:%d accuracy:%f' % (start_epoch, accu))
    for epoch in range(start_epoch, epoch_total):
        net.train()
        data_iter = iter(dataloader)

        # print('\nEpoch: %d' % epoch)
        train_loss = 0
        acc_total = 0
        for step in range(len(data_iter)):
            # for step in range(len(dataloader) - 1):
            print('----epoch:%d------step:%d------' % (epoch, step))
            data = next(data_iter)
            randomH = np.random.randint(0, 252)
            randomW = np.random.randint(0, 448)
            imageL = data['imgL'][:, :, randomH:(randomH + height), randomW:(randomW + width)]
            imageR = data['imgR'][:, :, randomH:(randomH + height), randomW:(randomW + width)]
            disL = data['dispL'][:, :, randomH:(randomH + height), randomW:(randomW + width)]
            # imageL = data['imgL']
            # imageR = data['imgR']
            # disL = data['dispL']
            # imL.data.resize_(imageL.size()).copy_(imageL)
            # imR.data.resize_(imageR.size()).copy_(imageR)
            # dispL.data.resize_(disL.size()).copy_(disL)
            imL.resize_(imageL.size()).copy_(imageL)
            imR.resize_(imageR.size()).copy_(imageR)
            dispL.resize_(disL.size()).copy_(disL)
            # print(imageL.shape)
            # print(disL.shape)
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

            # save
            # if step % 100 == 0:
            #     loss_list.append(train_loss / (step + 1))
            if step == len(dataloader) - 1:
                # print('=======>saving model......')
                # state={'net':net.state_dict(),'step':step,
                #        'loss_list':loss_list,'epoch':epoch,'accur':acc_total}
                # torch.save(state,'checkpoint/ckpt.t7')
                im = result[0, :, :, :].data.cpu().numpy().astype('uint8')
                im = np.transpose(im, (1, 2, 0))
                cv2.imwrite('train_result'+ str(epoch)+'.png', im, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
                gt = np.transpose(dispL[0, :, :, :].data.cpu().numpy(), (1, 2, 0))
                cv2.imwrite('train_gt'+ str(epoch)+'.png', gt, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
                if epoch == epoch_total-1:
                    print('=======>saving model......')
                    date = str(datetime.datetime.now())
                    state = {'net': net.state_dict()}
                    fileName = './checkpoint/ckpt_'+date[0:19]+'.t7'
                    torch.save(state, fileName)


def test():
    pass


# def checkpoint(path):
#     if not os.path.exists('./checkpoint/ckpt.t7'):
#         checkpoint = torch.load('./checkpoint/ckpt.t7')
#         # net.load_state_dict(checkpoint['net'])
#         # start_epoch = checkpoint['epoch']
#         # accu = checkpoint['accur']


if __name__ == '__main__':
    transform_train = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    main('train', transform_train)