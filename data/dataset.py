import os, sys
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset

# LaneClsDataset(data_root,
                # os.path.join(data_root, 'train_gt.txt'),
                # img_transform=img_transform, target_transform=target_transform,
                # simu_transform = simu_transform,
                # griding_num=griding_num, 
                # row_anchor = tusimple_row_anchor,
                # segment_transform=segment_transform,use_aux=use_aux, num_lanes = num_lanes)

class LaneClsDataset(Dataset):
    def __init__(self, base_dir, cfg,
                img_transform=None, target_transform=None, seg_transform=None, load_name : bool = False):
        super().__init__()
        self.base_dir = base_dir # ./datasets/tusimple/train/
        self.txtfile = base_dir + 'train_gt.txt'
        self.cfg = cfg
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.seg_transform = seg_transform
        self.griding_num = int(cfg['griding_num'])
        self.load_name = load_name
        self.use_aux = cfg['use_aux']
        self.num_lanes = int(cfg['num_lanes'])
        self.row_anchor = list(cfg['row_anchor'])
        self.row_anchor.sort()

        with open(self.txtfile, 'r') as f:
            self.list = f.readlines()

    def __getitem__(self, idx):
        l = self.list[idx]
        l_info = l.split() # .../jpg , .../png , 1 , 1 , 1 , 1 
        img_name, label_name = l_info[0], l_info[1]
        if img_name[0] == '/':
            img_name = img_name[1:]
            label_name = label_name[1:]
        
        label_path = os.path.join(self.base_dir, label_name)
        label = Image.open(label_path)

        img_path = os.path.join(self.base_dir, img_name)
        img = Image.open(img_path)

        lane_pts = self.get_index(label) # get the coordinates of lanes at row anchors



    def get_index(self, label):
        w, h = label.size

        if h != 288: # row_anchor는 288 기준인데, label이 288이 아니라면, label 크기에 맞게 index를 얻어야 하기에
            scale_f = lambda x : int((x * 1.0/288) * h)
            scale_row_anchor = list(map(scale_f,self.row_anchor))
        
        all_idx = np.zeros((self.num_lanes, len(scale_row_anchor), 2)) # 4 x 56 x 2
        for idx, anchor in enumerate(scale_row_anchor):
            
