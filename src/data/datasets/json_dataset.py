#!/usr/bin/env python3

"""JSON dataset: support CUB, NABrids, Flower, Dogs and Cars"""

import os
import torch
import torch.utils.data
import torchvision as tv
import numpy as np
from collections import Counter
from ..transforms import get_transforms
from ...utils import logging
from ...utils.io_utils import read_json
logger = logging.get_logger("visual_prompt")

import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import cv2 

class rotation(nn.Module):
    def __init__(self, theta=90.0):
        super(rotation, self).__init__()
        self.theta = theta

    def forward(self, image):
        angle = np.random.uniform(low = -self.theta, high = self.theta, size = (1))
        img = torchvision.transforms.functional.rotate(image, angle[0])
        return img

class JSONDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        assert split in {
            "train",
            "val",
            "test",
        }, "Split '{}' not supported for {} dataset".format(
            split, cfg.DATA.NAME)
        logger.info("Constructing {} dataset {}...".format(
            cfg.DATA.NAME, split))

        self.cfg = cfg
        self._split = split
        self.name = cfg.DATA.NAME
        self.data_dir = cfg.DATA.DATAPATH
        self.data_percentage = cfg.DATA.PERCENTAGE
        self._construct_imdb(cfg)
        self.transform = get_transforms(split, cfg.DATA.CROPSIZE)
        self.base_transform = get_transforms('val', cfg.DATA.CROPSIZE)

        # Create a list of transforms
        self.transform_dict = {
            'h_flip': transforms.RandomHorizontalFlip(p=1.0),
            'rotation': rotation(),
            'grayscale': transforms.Grayscale(num_output_channels=3),
            # Strong color jitter
            'color_jitter': transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
            'blur': transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        }
        self.augmented = cfg.DATA.AUGMENTED

        # Task is to predict rotation angle
        self.predict_rotation = cfg.DATA.PREDICT_ROTATION
        if self.predict_rotation:
            self.angles = [30.0, 60.0, 90.0, 120.0, 180.0, 210.0, 240.0, 270.0]

    def get_anno(self):
        anno_path = os.path.join(self.data_dir, "{}.json".format(self._split))
        if "train" in self._split:
            if self.data_percentage < 1.0:
                anno_path = os.path.join(
                    self.data_dir,
                    "{}_{}.json".format(self._split, self.data_percentage)
                )
        assert os.path.exists(anno_path), "{} dir not found".format(anno_path)

        return read_json(anno_path)

    def get_imagedir(self):
        raise NotImplementedError()

    def _construct_imdb(self, cfg):
        """Constructs the imdb."""

        img_dir = self.get_imagedir()
        assert os.path.exists(img_dir), "{} dir not found".format(img_dir)

        anno = self.get_anno()
        # Map class ids to contiguous ids
        self._class_ids = sorted(list(set(anno.values())))
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}

        # Construct the image db
        self._imdb = []
        for img_name, cls_id in anno.items():
            cont_id = self._class_id_cont_id[cls_id]
            im_path = os.path.join(img_dir, img_name)
            self._imdb.append({"im_path": im_path, "class": cont_id})

        logger.info("Number of images: {}".format(len(self._imdb)))
        logger.info("Number of classes: {}".format(len(self._class_ids)))

    def get_info(self):
        num_imgs = len(self._imdb)
        return num_imgs, self.get_class_num()

    def get_class_num(self):
        return self.cfg.DATA.NUMBER_CLASSES
        # return len(self._class_ids)

    def get_class_weights(self, weight_type):
        """get a list of class weight, return a list float"""
        if "train" not in self._split:
            raise ValueError(
                "only getting training class distribution, " + \
                "got split {} instead".format(self._split)
            )

        cls_num = self.get_class_num()
        if weight_type == "none":
            return [1.0] * cls_num

        id2counts = Counter(self._class_ids)
        assert len(id2counts) == cls_num
        num_per_cls = np.array([id2counts[i] for i in self._class_ids])

        if weight_type == 'inv':
            mu = -1.0
        elif weight_type == 'inv_sqrt':
            mu = -0.5
        weight_list = num_per_cls ** mu
        weight_list = np.divide(
            weight_list, np.linalg.norm(weight_list, 1)) * cls_num
        return weight_list.tolist()

    def __getitem__(self, index):
        # Load the image
        im = tv.datasets.folder.default_loader(self._imdb[index]["im_path"])
        if self.predict_rotation:
            # choose angle randomly
            index = np.random.choice(len(self.angles), size=(1))[0]
            im = torchvision.transforms.functional.rotate(im, self.angles[index])
            im = self.base_transform(im)
            label = index
        else:  
            label = self._imdb[index]["class"]
            im = self.transform(im)
        if self._split == "train":
            index = index
        else:
            index = f"{self._split}{index}"
        
        if self.augmented:
            aug_images = []
            for k in self.transform_dict.keys():
                aug_images.append(self.transform_dict[k](im))
            aug_images = torch.stack(aug_images)
            sample = {
                "image": im,
                "label": label,
                "augmented": aug_images 
            }
        else:
            sample = {
                "image": im,
                "label": label,
                # "id": index
            }
        return sample

    def __len__(self):
        return len(self._imdb)


class CUB200Dataset(JSONDataset):
    """CUB_200 dataset."""

    def __init__(self, cfg, split):
        super(CUB200Dataset, self).__init__(cfg, split)

    def get_imagedir(self):
        if self._split == "val":
            return os.path.join(self.data_dir, "train")
        return os.path.join(self.data_dir, self._split)
    


class CarsDataset(JSONDataset):
    """stanford-cars dataset."""

    def __init__(self, cfg, split):
        super(CarsDataset, self).__init__(cfg, split)

    def get_imagedir(self):
        if self._split == "val":
            return os.path.join(self.data_dir, "train")
        return os.path.join(self.data_dir, self._split)
    
    def _construct_imdb(self, cfg):
        """Constructs the imdb."""

        img_dir = self.get_imagedir()
        assert os.path.exists(img_dir), "{} dir not found".format(img_dir)

        anno = self.get_anno()
        # Map class ids to contiguous ids
        self._class_ids = sorted(list(set(anno.values())))
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}

        # Construct the image db
        self._imdb = []
        for img_name, cls_id in anno.items():
            cont_id = self._class_id_cont_id[cls_id]
            im_path = os.path.join(img_dir, str(cls_id), img_name)
            self._imdb.append({"im_path": im_path, "class": cont_id})

        logger.info("Number of images: {}".format(len(self._imdb)))
        logger.info("Number of classes: {}".format(len(self._class_ids)))


class DogsDataset(JSONDataset):
    """stanford-dogs dataset."""

    def __init__(self, cfg, split):
        super(DogsDataset, self).__init__(cfg, split)

    def get_imagedir(self):
        return os.path.join(self.data_dir, "Images")


class FlowersDataset(JSONDataset):
    """flowers dataset."""

    def __init__(self, cfg, split):
        super(FlowersDataset, self).__init__(cfg, split)

    def get_imagedir(self):
        return self.data_dir


class NabirdsDataset(JSONDataset):
    """Nabirds dataset."""

    def __init__(self, cfg, split):
        super(NabirdsDataset, self).__init__(cfg, split)

    def get_imagedir(self):
        return os.path.join(self.data_dir, "images")

class AnimalPoseDataset(JSONDataset):
    "Animal Pose estimation dataset"
    def __init__(self, cfg, split):
        super(AnimalPoseDataset, self).__init__(cfg, split)
    
    def get_imagedir(self):
        return os.path.join(self.data_dir, "images")
    
    def _construct_imdb(self, cfg):
        """Constructs the imdb."""

        img_dir = self.get_imagedir()
        assert os.path.exists(img_dir), "{} dir not found".format(img_dir)

        anno = self.get_anno()
        # Construct the image db
        self._imdb = []
        for img_name, keypt in anno.items():
            #  Convert keypoints from 0 to 1 by diving by image size
            keypt = np.array(keypt)[:,:2].astype(np.float32)
            image_read = cv2.imread(os.path.join(img_dir, img_name))
            keypt[:,0] = keypt[:, 0]/image_read.shape[1]
            keypt[:,1] = keypt[:, 1]/image_read.shape[0]
            # Flatten the keypoint array
            keypt = keypt.flatten()
            im_path = os.path.join(img_dir, img_name)
            self._imdb.append({"im_path": im_path, "class": keypt})

        logger.info("Number of images: {}".format(len(self._imdb)))
    

