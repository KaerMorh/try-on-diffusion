import random
import torch.utils.data as data
import torchvision.transforms
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as transforms
import torch
import copy, os, collections
import json
from .human_parse_labels import get_label_map, DF_LABEL, YF_LABEL
import pandas as pd
from .pose_utils import load_pose_cords_from_strings, cords_to_map
from .try_on import mask2bbox

TEST_PATCHES = [
    'chequered/chequered_0052.jpg', 'dotted/dotted_0072.jpg',
    'paisley/paisley_0015.jpg', 'striped/striped_0011.jpg'
]

PID = [0, 4, 7, 6]  # bg, face, arm, leg (the first has to be bg and the second has to be face.)
GID = [2, 5, 1, 3]  # hair, top, bottom, jacket


class DFPairDataset(data.Dataset):

    def get_paths(self, root, phase, viton=False):
        pairLst = os.path.join(root, 'fashion-new-pairs-%s.csv' % phase)
        if viton:
            pairLst = os.path.join(root, 'fashion-new-pairs-%s.csv' % 'viton')
        name_pairs = self.init_categories(pairLst)

        image_dir = os.path.join(root, '%s' % phase)
        bonesLst = os.path.join(root, 'fasion-annotation-%s.csv' % phase)

        return image_dir, bonesLst, name_pairs

    def init_categories(self, pairLst):
        with open(pairLst) as f:
            anns = f.readlines()
        anns = [line[:-1].split(",")[1:] for line in anns[1:]]
        ret = []
        for i in range(0, len(anns)):
            ret.append([anns[i][0], anns[(i + 1) % len(anns)][1]])
        return ret

    def __init__(self, dataset_dir, image_size=512, isTrain=True, n_human_part=8, viton=False, **args):
        super(DFPairDataset, self).__init__()
        self.arbitrary_mask_percent = args['arbitrary_mask_percent']
        dim = (image_size, int(image_size / 256 * 192))
        self.root = dataset_dir
        self.isTrain = (args['state'] == 'train')
        self.split = 'train' if self.isTrain else 'test'
        self.n_human_part = n_human_part
        self.dim = dim
        self._init(viton)
        self.mask_dir = self.root + "/%sM_lip" % ('train' if self.isTrain else 'test')
        self.feat_dir = self.root + "/%s_warp" % ('train' if self.isTrain else 'test')

    def _init(self, viton):
        self.image_dir, self.bone_file, self.name_pairs = self.get_paths(self.root, self.split, viton)
        self.annotation_file = pd.read_csv(self.bone_file, sep=':')
        self.annotation_file = self.annotation_file.set_index('name')

        self.aiyu2atr, self.atr2aiyu = get_label_map(self.n_human_part)

        self.load_size = self.dim
        self.crop_size = self.load_size

        # transforms
        self.resize = transforms.Resize(self.crop_size)
        self.toTensor = transforms.ToTensor()
        self.toPIL = transforms.ToPILImage()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # self.padding = transforms.Pad((80, 0, 80, 0), 255)
        # self.mask_padding = transforms.Pad((80, 0, 80, 0), 0)
        self.clip_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                   (0.26862954, 0.26130258, 0.27577711))

        self.length = len(self.name_pairs)

    def __len__(self):
        return len(self.name_pairs)

    def _load_img(self, fn):
        img = Image.open(fn).convert("RGB")
        img = self.resize(img)
        # img = self.padding(img)
        img = self.toTensor(img)
        img = self.normalize(img)

        return img

    def _load_mask(self, fn):
        mask = Image.open(fn + ".png")
        mask = self.resize(mask)
        # mask = self.mask_padding(mask)
        mask = torch.from_numpy(np.array(mask))

        texture_mask = copy.deepcopy(mask)
        for atr in self.atr2aiyu:
            aiyu = self.atr2aiyu[atr]
            texture_mask[mask == atr] = aiyu
        return texture_mask

    def _load_kpt(self, name):
        string = self.annotation_file.loc[name]
        array = load_pose_cords_from_strings(string['keypoints_y'], string['keypoints_x'])
        pose = cords_to_map(array, (256, 192), (256, 176))
        pose = np.transpose(pose, (2, 0, 1))
        # pose = np.pad(pose, ((0, 0), (0, 0), (80, 80)), constant_values=0.0)
        pose = torch.Tensor(pose)
        return pose

    def get_to_item(self, key):
        img = self._load_img(os.path.join(self.image_dir, key))
        kpt = self._load_kpt(key)
        parse = self._load_mask(os.path.join(self.mask_dir, key[:-4]))
        return img, kpt, parse

    def _load_feat(self, name):
        # feat = np.load(os.path.join(self.feat_dir, name + '.npy'))
        # feat = torch.from_numpy(feat)
        img = Image.open(os.path.join(self.feat_dir, name + '.jpg')).convert("RGB")
        img = self.toTensor(img)
        img = self.normalize(img)
        return img

    def __getitem__(self, index):
        from_key, to_key = self.name_pairs[index]

        from_img, from_kpt, from_parse = self.get_to_item(from_key)
        to_img, to_kpt, to_parse = self.get_to_item(to_key)
        _, H, W = to_img.size()
        mask_id = torch.Tensor([5, 3, 7])
        from_mask = torch.isin(from_parse, mask_id)
        inpaint_mask = torch.isin(to_parse, mask_id)

        mask = inpaint_mask.numpy()
        mask = cv2.dilate(mask.astype(np.uint8), kernel=np.ones((10, 10)), iterations=3)
        mask = cv2.erode(mask.astype(np.uint8), kernel=np.ones((10, 10)), iterations=1)
        mask = mask.astype(np.float32)
        mask = 1 - self.toTensor(mask)
        # self.toPIL(mask).save('mask.png')
        inpaint = to_img * mask

        clothes_id = torch.Tensor([5, 3])
        from_mask = torch.isin(from_parse, clothes_id)
        from_clothes = from_mask * from_img

        down, up, left, right = mask2bbox(from_mask.numpy())
        ref_image = from_clothes[:, down:up, left:right]
        ref_image = (ref_image + 1.0) / 2.0
        ref_image = torchvision.transforms.Resize((224, 224))(ref_image)
        ref_image = self.clip_normalize(ref_image)

        to_mask = torch.isin(to_parse, clothes_id)
        to_clothes = to_mask * to_img
        to_label = torch.zeros(8, H, W)
        to_label = torch.scatter(to_label, 0, to_parse.unsqueeze(0).data.long(), 1.0)
        to_label[mask_id.long(), :, :] = 0.0

        feat_name = from_key[:-4] + '-' + to_key
        return from_clothes, from_mask.unsqueeze(0).float(), from_kpt, to_label, \
            to_mask.unsqueeze(0).float(), to_kpt, to_clothes, to_img * to_mask.logical_not(), \
            mask, inpaint, ref_image, to_img, feat_name
