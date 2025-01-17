from torch.utils.data import Dataset
import glob
import os
import pickle
import torch
from PIL import Image
import numpy as np
import csv
import torch
import random
import scipy.linalg as la
from torch import nn


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class ndataset(Dataset):
    
    def __init__(self, split='train', path='/media/foz/41bc5ab6-c5ae-4fe8-8bf7-9ed053ace67a/data/SRN/data/SRN/cars_train', picklefile='/media/foz/41bc5ab6-c5ae-4fe8-8bf7-9ed053ace67a/data/SRN/data/cars.pickle',transform=None):
        self.path = path
        super().__init__()
        self.picklefile = pickle.load(open(picklefile, 'rb'))
        
        allthevid = sorted(list(self.picklefile.keys()))

        print(len(allthevid))
        
        random.seed(0)
        random.shuffle(allthevid)
        if split == 'train':
            self.ids = allthevid[:int(len(allthevid)*0.9)]
        else:
            self.ids = allthevid[int(len(allthevid)*0.9):]
            
        self.transform = transform
        
        self.transform = nn.Identity() if transform is None else transform

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self,idx):

        item = self.ids[idx]

        i = random.choice(self.picklefile[item])

        img_filename = os.path.join(self.path, item, 'rgb', i)
        img = Image.open(img_filename).convert('RGB')

        img = self.transform(img)

        return img,

class eval_dataset(Dataset):
    
    def __init__(self, split, path='./data/SRN/cars_train', picklefile='./data/cars.pickle', imgsize=128, n_img=8, shuffle_input=False, shuffle_views=True):
        self.imgsize = imgsize
        self.path = path
        super().__init__()
        self.picklefile = pickle.load(open(picklefile, 'rb'))

        self.n_img = n_img
        self.shuffle_views = shuffle_views
        self.shuffle_input = shuffle_input
        
        allthevid = sorted(list(self.picklefile.keys()))

        print(len(allthevid))
        
        random.seed(0)
        random.shuffle(allthevid)
        if split == 'train':
            self.ids = allthevid[:int(len(allthevid)*0.9)]
        else:
            self.ids = allthevid[int(len(allthevid)*0.9):]
            
                
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self,idx):

#        idx = 0
#        first = 1
        item = self.ids[idx]
        
        intrinsics_filename = os.path.join(self.path, item, 'intrinsics', self.picklefile[item][0][:-4] + ".txt")
        K = np.array(open(intrinsics_filename).read().strip().split()).astype(float).reshape((3,3))
        
        #indices = [self.picklefile[item][first]] + random.sample(self.picklefile[item], k=1) #random.sample(self.picklefile[item], k=2)

        if self.n_img:
            if self.shuffle_views:
                if self.shuffle_input:
                    indices = random.sample(self.picklefile[item], k=self.n_img)
                else:
                    indices = [self.picklefile[item][0]]+ random.sample(self.picklefile[item], k=self.n_img-1)
            else:
                indices = self.picklefile[item][:self.n_img]
        else:
            indices = self.picklefile[item]

        imgs = []
        poses = []
        for i in indices:
            img_filename = os.path.join(self.path, item, 'rgb', i)
            img = Image.open(img_filename)
            if self.imgsize != 128:
                img = img.resize((self.imgsize, self.imgsize))
            img = np.array(img) / 255 * 2 - 1
            
            img = img.transpose(2,0,1)[:3].astype(np.float32)
            imgs.append(img)
            
            
            pose_filename = os.path.join(self.path, item, 'pose', i[:-4]+".txt")
            pose = np.array(open(pose_filename).read().strip().split()).astype(float).reshape((4,4))
            R = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
            pose[:3,:3] =  pose[:3,:3] @ R
            poses.append(pose)

       # print('poses', poses)
       # print(la.norm(poses[0][:3,3]))
       # print(la.norm(poses[1][:3,3]))

        new_poses = []
        R0 = poses[0][:3,:3]
        T0 = poses[0][:3,3]
        d0 = la.norm(T0)

        camera_d = d0

        ref_pose = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,d0],[0,0,0,1]])
        #new_poses.append(ref_pose)
        
        ref_transform = np.eye(4)
        ref_transform[:3,:3] = R0.T
        ref_transform[:3,3] = -R0.T @ T0 + np.array([0,0,d0])
        ref_transform = ref_transform

        for p in poses:
            new_poses.append(ref_transform @ p)

        
        imgs = np.stack(imgs, 0)
        poses = np.stack(new_poses, 0).astype(np.float32)
        R = poses[:, :3, :3]
        T = poses[:, :3, 3]
        
        intrinsics = np.array([K[0,0]/8, K[1,1]/8, K[0,2]/8, K[1,2]/8]).astype(np.float32)

        camera_k = K[0,0]/K[0,2]

        return {'imgs':imgs, 'poses':poses, 'intrinsics':intrinsics, 'K':K, 'camera_k': camera_k, 'camera_d': camera_d }
        

    
if __name__ == "__main__":
    
    from torch.utils.data import DataLoader

    d = dataset('train')
    dd = d[0]
    
    for ddd in dd.items():
        print(ddd[1].shape)
    print(dd['K'])
    print(dd['poses'])
