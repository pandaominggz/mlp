import os
import pickle


class FileFinder(object):

    def __init__(self, Type):

        self.Type = Type

        # self.dir_scene_flow_originals = '/media/daoming/GF/SSceneFlow/originals/' + Type
        # self.dir_scene_flow_disparity = '/media/daoming/GF/SSceneFlow/disparity/' + Type

        self.dir_scene_flow_originals = '/home/daoming/Documents/SSceneFlow/originals/' + Type
        self.dir_scene_flow_disparity = '/home/daoming/Documents/SSceneFlow/disparity/' + Type

        self.paths_originals = []
        self.paths_originals_left = []
        self.paths_originals_right = []

        self.paths_disparity = []
        self.paths_disparity_left = []
        self.paths_disparity_right = []

    def search_files(self):
        for root, dirs, files in os.walk(self.dir_scene_flow_originals):
            for file in files:
                self.paths_originals.append(os.path.join(root, file))

        for root, dirs, files in os.walk(self.dir_scene_flow_disparity):
            for file in files:
                self.paths_disparity.append(os.path.join(root, file))

        for i in range(len(self.paths_originals)):
            file_name = self.paths_originals[i].split('/')[-1].split('.')[0]
            fileName_with_suffix = self.paths_disparity[i].split('/')[-1]
            if self.paths_originals[i].find('left') > -1:
                self.paths_originals_left.append(self.paths_originals[i])
                self.paths_disparity_left.append(
                    self.paths_disparity[i].replace(fileName_with_suffix, file_name + '.pfm'))
            elif self.paths_originals[i].find('right') > -1:
                self.paths_originals_right.append(self.paths_originals[i])
                self.paths_disparity_right.append(
                    self.paths_disparity[i].replace(fileName_with_suffix, file_name + '.pfm'))

    def write_pkl(self):
        originals_left = open('./pkl/originals_left_' + self.Type + '.pkl', 'wb')
        originals_right = open('./pkl/originals_right_' + self.Type + '.pkl', 'wb')
        disparity_left = open('./pkl/disparity_left_' + self.Type + '.pkl', 'wb')
        disparity_right = open('./pkl/disparity_right_' + self.Type + '.pkl', 'wb')
        # print(len(self.paths_originals_left))
        # print(len(self.paths_originals_right))
        # print(len(self.paths_disparity_left))
        # print(len(self.paths_disparity_right))

        pickle.dump(self.paths_originals_left, originals_left)
        pickle.dump(self.paths_originals_right, originals_right)
        pickle.dump(self.paths_disparity_left, disparity_left)
        pickle.dump(self.paths_disparity_right, disparity_right)

        originals_left.close()
        originals_right.close()
        disparity_left.close()
        disparity_right.close()

    def do(self):
        self.search_files()
        self.write_pkl()
