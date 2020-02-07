import numpy as np


class PFMReader(object):

    def __init__(self, fileName):
        self.fileName = fileName
        self.channels = 0
        self.width = 0
        self.height = 0
        self.scale = 0
        self.endian = 0
        self.img = 0

    def load(self):
        with open(self.fileName, "rb") as file:
            # line 1
            header = file.readline().rstrip().decode('utf-8')
            if header == 'PF':
                self.channels = 3
            elif header == 'Pf':
                self.channels = 1
            else:
                raise Exception('Not a PFM file.')

            # line 2
            size = file.readline().rstrip().decode('utf-8')
            self.width = int(size.split(' ')[0])
            self.height = int(size.split(' ')[1])

            # line 3
            self.scale = float(file.readline().rstrip().decode('utf-8'))
            if self.scale < 0:
                self.endian = '<'
            else:
                self.endian = '>'

            # loading the rest of lines
            self.img = np.fromfile(file, self.endian + 'f').reshape((self.height, self.width, self.channels))    # not sure what (self.endian + 'f') means
            # self.img = np.flipud(img)   # not sure why flipud
            return self.img

    def get_img(self):
        return self.img
