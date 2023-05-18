from cProfile import label
import torch.utils.data as data
import numpy as np
import albumentations as A
import os
from PIL import Image, ImageDraw
import torch, torchvision
import random
import cv2

def mask2bbox(mask):
    up = np.max(np.where(mask)[0])
    down = np.min(np.where(mask)[0])
    left = np.min(np.where(mask)[1])
    right = np.max(np.where(mask)[1])
    center = ((up + down) // 2, (left + right) // 2)
    
    factor = random.random() * 0.1 + 0.1
    
    up = int(min(up * (1 + factor) - center[0] * factor + 1, mask.shape[0]))
    down = int(max(down * (1 + factor) - center[0] * factor, 0))
    left = int(max(left * (1 + factor) - center[1] * factor, 0))
    right = int(min(right * (1 + factor) - center[1] * factor + 1, mask.shape[1]))
    return (down, up, left, right)

def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

class TryOnDataset(data.Dataset):
    def __init__(self, state, arbitrary_mask_percent=0.05, **args):
        self.state=state
        self.args=args
        self.arbitrary_mask_percent = arbitrary_mask_percent
        self.kernel = np.ones((1, 1), np.uint8)
        self.random_trans=A.Compose([
            A.Resize(height=224,width=224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10),
            A.Blur(p=0.2),
            A.ElasticTransform(p=0.2)
            ])
        self.path_list=[]
        
        label_dir = os.path.join(args['dataset_dir'], 'label')
        image_dir = os.path.join(args['dataset_dir'], 'img')
        
        file_names = os.listdir(label_dir)
        
        for label_name in file_names:
            image_name = label_name.replace('.png', '.jpg')
            
            label_path = os.path.join(label_dir, label_name)
            image_path = os.path.join(image_dir, image_name)
            if os.path.exists(label_path) and os.path.exists(image_path):
                self.path_list.append((image_path, label_path))
            
        self.path_list.sort()
        self.length = len(self.path_list)

    def __getitem__(self, index):
        
        image_path, label_path = self.path_list[index]
        
        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path)
        
        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        label = np.array(label).astype(np.int8)

        '''
        useful number 
            0: background
            1: Pants
            2: Hair
            3: Skirt
            4: Face
            5: Upper-clothes
            6: Arm
            7: Leg
        '''
        useful_list = [(1, 7), (3, 7), (5, 6)]
        
        exist_list = []
        for i in useful_list:
            num_i = np.sum(label == i[0])
            if label.size * self.arbitrary_mask_percent < num_i:
                exist_list.append(i)

        if len(exist_list) == 0:
            return self.__getitem__((index + 1) % self.length)        

        mask_number = random.choice(exist_list)
        mask = np.isin(label, mask_number)
        down, up, left, right = mask2bbox(mask)
        ref_image = image_np[down:up, left:right]
        ref_image_tensor= self.random_trans(image=ref_image)
        ref_image_tensor=Image.fromarray(ref_image_tensor["image"])
        ref_image_tensor=get_tensor_clip()(ref_image_tensor)
        
        ### Generate mask
        H, W, C = image_np.shape
        img_size = max(H, W)
        
        image_resize = np.ones((img_size, img_size, C)) * 255
        mask_resize = np.zeros((img_size, img_size))
        free_H = img_size - H
        free_W = img_size - W
        
        mask = cv2.dilate(mask.astype(np.uint8), kernel=np.ones((5, 5)), iterations=3)
        mask = cv2.erode(mask.astype(np.uint8), kernel=np.ones((5, 5)), iterations=1)
        
        mask_resize[free_H // 2: free_H // 2 + H, free_W // 2: free_W // 2 + W] = mask.astype(np.float32)
        image_resize[free_H // 2: free_H // 2 + H, free_W // 2: free_W // 2 + W] = image_np
        
        image_resize = Image.fromarray(image_resize.astype(np.uint8), "RGB")
        
        mask_tensor_resize = 1 - get_tensor(normalize=False, toTensor=True)(mask_resize)
        image_tensor_resize = get_tensor()(image_resize)
        inpaint_tensor_resize = image_tensor_resize * mask_tensor_resize
        return {"GT":image_tensor_resize,"inpaint_image":inpaint_tensor_resize,"inpaint_mask":mask_tensor_resize,"ref_imgs":ref_image_tensor}

    def __len__(self):
        return self.length

if __name__ == '__main__':
    dataset = TryOnDataset(state='train', dataset_dir='dataset/TryOn/train')
    for i in range(len(dataset)):
        w = dataset[i]
        if i % 100 == 0:
            print(i)
