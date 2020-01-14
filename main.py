from dataloader.data_loader import DataLoader as dL
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import os


def main(setType, transform):
    data = DataLoader(dL(setType, transform), batch_size=4, shuffle=True, num_workers=1)

    if setType == 'train':
        train(setType, data)
    elif setType == 'test':
        test(setType, data)


def train(setType, data):
    pass


def test(setType, data):
    pass


def checkpoint(path):
    if not os.path.exists('./checkpoint/ckpt.t7'):
        checkpoint = torch.load('./checkpoint/ckpt.t7')
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        accu = checkpoint['accur']


if __name__ == '__main__':
    transform_train = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    main('train', transform_train)
