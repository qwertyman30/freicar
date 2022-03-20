import torch.utils.data as data
import os
import os.path
import cv2
import torch
import yaml
from pathlib import Path
import requests
from zipfile import ZipFile
import sys
import numpy as np


########################################################################
# Demo freicar dataloader for semantic segmentation and regression
# Author: Johan Vertens (vertensj@informatik.uni-freiburg.de)
########################################################################

def searchForFiles(name, search_path):
    paths = []
    for path in Path(search_path).rglob(name):
        paths.append(path)
    return paths


def loadRGB(paths, size=None, pad_info=None):
    rgb_im = [cv2.imread(p) for p in paths]
    if size is not None:
        rgb_im = [cv2.resize(im, size) for im in rgb_im]
    rgb_im = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in rgb_im]
    rgb_im = [(torch.from_numpy(im)).permute(2, 0, 1) / 255. for im in rgb_im]

    if pad_info is not None:
        rgb_im = [torch.nn.functional.pad(d, pad_info, 'constant', 0) for d in rgb_im]

    return rgb_im


def loadRegressionOrSegmentation(paths, size=None, pad_info=None):
    rgb_im = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths]
    if size is not None:
        rgb_im = [cv2.resize(im, size, interpolation=cv2.INTER_NEAREST) for im in rgb_im]
    rgb_im = [(torch.from_numpy(im)).unsqueeze(0) for im in rgb_im]

    if pad_info is not None:
        rgb_im = [torch.nn.functional.pad(d, pad_info, 'constant', 0) for d in rgb_im]

    return rgb_im


def transform(image):
    prob = np.random.uniform(0, 1)
    # change brightness
    if prob <= 0.3:
        brightness_factor = np.random.uniform(0.2, 0.5)
        image = image[0].cpu().data.numpy() * brightness_factor
        image = [torch.from_numpy(image)]

    prob = np.random.uniform(0, 1)
    # gauss blur
    if prob <= 0.3:
        image = image[0].permute(1, 2, 0)
        image = cv2.GaussianBlur(image.cpu().data.numpy(), ksize=(5, 5), sigmaX=0.1, sigmaY=2.0,
                                 borderType=cv2.BORDER_DEFAULT)
        image = [torch.from_numpy(image).permute(2, 0, 1)]

    return image


def generatePaths(fnames, replace, type):
    npaths = []
    for fn in fnames:
        npaths.append(fn.replace(replace, type))
    return npaths


class FreiCarLoader(data.Dataset):
    def downloadData(self, data_dir, link, zip_name):

        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        file_name = data_dir + zip_name

        with open(file_name, "wb") as f:
            print("Downloading %s" % file_name)
            response = requests.get(link, stream=True)
            total_length = response.headers.get('content-length')

            if total_length is None:  # no content length header
                f.write(response.content)
            else:
                dl = 0
                total_length = int(total_length)
                for data in response.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    done = int(50 * dl / total_length)
                    sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                    sys.stdout.flush()

        print('Unzipping data...')
        with ZipFile(data_dir + '/' + zip_name, 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(data_dir)

        print('Removing zipped file ....')
        os.remove(data_dir + '/' + zip_name)

    def filterEvalFiles(self, all_paths, eval_paths):
        out = []
        for p in all_paths:
            if os.path.basename(p) not in eval_paths:
                out.append(p)
        return out

    def __init__(self, data_dir, padding, split='training', load_real=False, transform=True):
        db_path = data_dir

        if not os.path.exists(data_dir + '/detection_data/'):
            print(data_dir + '/detection_data/')
            self.downloadData(data_dir, 'http://aisdatasets.informatik.uni-freiburg.de/freicar/detection_data.zip',
                              'detection_data.zip')

        if not os.path.exists(data_dir + "/seg_reg_data/"):
            self.downloadData(data_dir, 'http://aisdatasets.informatik.uni-freiburg.de/freicar/seg_reg_data.zip',
                              'seg_reg_data.zip')

        self.db_path = db_path
        self.load_real = load_real

        eval_name_file = open(db_path + '/detection_data/eval_names.txt', 'r')
        self.eval_files = eval_name_file.readlines()
        self.eval_files = [os.path.basename(f.rstrip()) for f in self.eval_files]

        if split == 'training':
            self.rgb_path_files = sorted(searchForFiles('*.png', db_path + '/detection_data/synth/'),
                                         key=lambda y: float(y.name.split('.')[0].replace('_', '.')))
            self.rgb_path_files = [str(p) for p in self.rgb_path_files]
            self.rgb_path_files = self.filterEvalFiles(self.rgb_path_files, self.eval_files)
        elif split == 'validation':
            self.rgb_path_files = [db_path + '/detection_data/synth/' + ef for ef in self.eval_files]
        else:
            print('Split: %s not known...' % split)

        self.seg_paths = generatePaths(self.rgb_path_files, '/detection_data/synth/',
                                       '/seg_reg_data/segmentation_labels/')
        self.reg_paths = generatePaths(self.rgb_path_files, '/detection_data/synth/', '/seg_reg_data/lane_dt/')

        if self.load_real:
            self.real_rgb_paths = generatePaths(self.rgb_path_files, 'synth', 'real')
            self.rgb_path_files.extend(self.real_rgb_paths)
            self.seg_paths.extend(self.seg_paths)
            self.reg_paths.extend(self.reg_paths)

        self.transform = transform
        self.padding = padding
        self.info_file = searchForFiles('info.yaml', db_path + '/detection_data/info/')[0]

        assert (self.rgb_path_files.__len__() > 0)

        self.length = len(self.rgb_path_files)

        self.filenames = [os.path.basename(p).replace('.png', '') for p in self.rgb_path_files]

        print('Length of FreiCAR detection dataset: %d ' % self.length)

    def __getitem__(self, index):
        syn_rgb_path = self.rgb_path_files[index]
        seg_path = self.seg_paths[index]
        reg_path = self.reg_paths[index]

        out = {}

        image = loadRGB([syn_rgb_path], pad_info=self.padding)

        seg = loadRegressionOrSegmentation([seg_path], pad_info=self.padding)
        reg = loadRegressionOrSegmentation([reg_path], pad_info=self.padding)

        if self.transform:
            image = transform(image)

        out['rgb'] = image[0]
        out['seg'] = seg[0]
        out['reg'] = reg[0]

        return out

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'
