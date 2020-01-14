from dataloader.file_finder import FileFinder
from dataloader.pfm_reader import PFMReader
from torch.utils.data import Dataset
import pickle
import os
import cv2


class DataLoader(Dataset):

    def __init__(self, setType, transform):
        if not os.path.exists('./pkl/originals_left_' + setType + '.pkl'):
            finder = FileFinder(setType)
            finder.do()

        self.setType = setType
        self.transform = transform
        originals_left = open('./pkl/originals_left_' + setType + '.pkl', 'rb')
        originals_right = open('./pkl/originals_right_' + setType + '.pkl', 'rb')
        disparity_left = open('./pkl/disparity_left_' + setType + '.pkl', 'rb')
        disparity_right = open('./pkl/disparity_right_' + setType + '.pkl', 'rb')
        self.paths_originals_left = pickle.load(originals_left)
        self.paths_originals_right = pickle.load(originals_right)
        self.paths_disparity_left = pickle.load(disparity_left)
        self.paths_disparity_right = pickle.load(disparity_right)

        # 关闭流
        originals_left.close()
        originals_right.close()
        disparity_left.close()
        disparity_right.close()

    def __getitem__(self, index):
        imageL = cv2.imread(self.paths_originals_left[index]).reshape(540, 960, 3)
        imageR = cv2.imread(self.paths_originals_right[index]).reshape(540, 960, 3)

        disparityL = PFMReader(self.paths_disparity_left[index]).load().get_img()
        disparityR = PFMReader(self.paths_disparity_left[index]).load().get_img()

        imageL = self.transform(imageL)
        imageR = self.transform(imageR)

        return {'imgL': imageL, 'imgR': imageR, 'dispL': disparityL, 'dispR': disparityR}

    def __len__(self):
        return len(self.paths_originals_left)
