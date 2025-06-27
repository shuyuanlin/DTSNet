from torchvision import transforms
from PIL import Image
import os
import torch
import glob
import numpy as np
import cv2
from einops import rearrange
import imgaug.augmenters as iaa
from .perlin import rand_perlin_2d_np
import matplotlib.pyplot as plt
import random

class ToTensor(object):
    def __call__(self, image):
        try:
            image = torch.from_numpy(image.transpose(2, 0,1))
        except:
            print('Invalid_transpose, please make sure images have shape (H, W, C) before transposing')
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        return image
    
    
class Normalize(object):
    """
    Only normalize images
    """
    def __init__(self, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        self.mean = np.array(mean)
        self.std = np.array(std)
    def __call__(self, image):
        image = (image - self.mean) / self.std
        return image

texture_list = [
    'carpet', 'leather', 'tile', 'wood', 'grid', 'chewinggum', 'candle', 'capsules', 'cashew',
    'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'
]
object_list = [
    'cable', 'capsule', 'pill', 'screw', 'transistor', 'bottle', 'hazelnut', 'metal_nut',
    'toothbrush', '01', '02', '03'
]
class MVTecDataset_train(torch.utils.data.Dataset):
    def __init__(
            self,
            target,
            dataset_path,
            anomaly_source_path,
            resize,
            perlin_scale,
            min_perlin_scale,
            perlin_noise_threshold,
            transparency_range
    ):
        self.target = target
        self.img_path = dataset_path
        self.img_paths = glob.glob(os.path.join(self.img_path, 'good') + "/*.png")
        self.anomaly_source_path = anomaly_source_path
        self.anomaly_source_file_list = glob.glob(os.path.join(self.anomaly_source_path, '*/*'))
        self.resize = resize
        self.transform = transforms.Compose([Normalize(), ToTensor()])
        self.transparency_range = transparency_range
        self.perlin_scale = perlin_scale
        self.min_perlin_scale = min_perlin_scale
        self.perlin_noise_threshold = perlin_noise_threshold
        self.anomaly_switch = False


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img= cv2.resize(img/255., self.resize)

        if self.anomaly_switch:
            img_anomaly, mask = self.generate_anomaly1(img, self.anomaly_source_file_list ,img_path)
            self.anomaly_switch = False
        else:
            img_anomaly = img
            self.anomaly_switch = True
        img_normal = self.transform(img)
        img_anomaly = self.transform(img_anomaly)

        return img_normal, img_anomaly, img_path.split('/')[-1]

    def rand_augment(self):
        augmenters = [
            iaa.GammaContrast((0.5, 2.0), per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
            iaa.Solarize(0.5, threshold=(32, 128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45))
        ]

        aug_idx = np.random.choice(np.arange(len(augmenters)), 3, replace=False)
        aug = iaa.Sequential([
            augmenters[aug_idx[0]],
            augmenters[aug_idx[1]],
            augmenters[aug_idx[2]]
        ])

        return aug

    # def generate_anomaly(self, img: np.ndarray, texture_img_list: list, img_path: str):
    #     img_size = img.shape[:-1]  # H x W
    #     target_foreground_mask = self.generate_target_foreground_mask(img_path=img_path)
    #     perlin_noise_mask = self.generate_perlin_noise_mask(img_size=img_size)
    #
    #     mask = perlin_noise_mask * target_foreground_mask
    #     mask_expanded = np.expand_dims(mask, axis=2)
    #
    #     anomaly_source_img = self.anomaly_source(img=img, texture_img_list=texture_img_list)
    #     factor = np.random.uniform(*self.transparency_range, size=1)[0]
    #     anomaly_mask = factor * (mask_expanded * anomaly_source_img) + (1 - factor) * (mask_expanded * img)
    #
    #     anomaly_img = (((- mask_expanded + 1) * img) + anomaly_mask)
    #
    #     return anomaly_img, mask

    def generate_anomaly1(self, img: np.ndarray, texture_img_list: list, img_path: str):
        img_size = img.shape[:-1]  # H x W

        target_foreground_mask = self.generate_target_foreground_mask(img_path=img_path)
        perlin_noise_mask1 = self.generate_perlin_noise_mask(img_size=img_size)
        perlin_noise_mask2 = self.generate_perlin_noise_mask(img_size=img_size)
        bool_mask1 = perlin_noise_mask1 > 0
        bool_mask2 = perlin_noise_mask2 > 0
        intersection = (bool_mask1 & bool_mask2).astype(np.float64)
        union = (bool_mask1 | bool_mask2).astype(np.float64)
        perlin_noise_masks = [perlin_noise_mask1, perlin_noise_mask2, intersection, union]

        mask = random.choice(perlin_noise_masks) * target_foreground_mask
        mask_expanded = np.expand_dims(mask, axis=2)

        anomaly_source_img = self.anomaly_source(img=img, texture_img_list=texture_img_list)
        factor = np.random.uniform(*self.transparency_range, size=1)[0]
        anomaly_mask = factor * (mask_expanded * anomaly_source_img) + (1 - factor) * (mask_expanded * img)

        anomaly_img = (((- mask_expanded + 1) * img) + anomaly_mask)

        return anomaly_img, mask

    def generate_target_foreground_mask(self, img_path: str) -> np.ndarray:
        target_foreground_mask = np.zeros(self.resize, dtype=np.float64)
        if self.target in object_list:
            foreground_path = img_path.replace('train', 'DISthresh')
            target_foreground_mask = cv2.imread(foreground_path, 0)
            target_foreground_mask = cv2.resize(target_foreground_mask/255.0, dsize=self.resize)
        if self.target in texture_list:
            target_foreground_mask = np.ones(self.resize, dtype=np.float64)
        return target_foreground_mask

    def generate_perlin_noise_mask(self, img_size: tuple) -> np.ndarray:
        perlin_scalex = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np(img_size, (perlin_scalex, perlin_scaley))

        rot = iaa.Affine(rotate=(-90, 90))
        perlin_noise = rot(image=perlin_noise)

        mask_noise = np.where(
            perlin_noise > self.perlin_noise_threshold,
            np.ones_like(perlin_noise),
            np.zeros_like(perlin_noise)
        )

        return mask_noise

    def anomaly_source(self, img: np.ndarray, texture_img_list: list = None) -> np.ndarray:
        p = 0
        if p < 0.5:
            idx = np.random.choice(len(texture_img_list))
            img_size = img.shape[:-1]  # H x W
            anomaly_source_img = self._texture_source(img_size=img_size, texture_img_path=texture_img_list[idx])
        else:
            anomaly_source_img = self._structure_source(img=img)

        return anomaly_source_img

    def _texture_source(self, img_size: tuple, texture_img_path: str) -> np.ndarray:
        texture_source_img = cv2.imread(texture_img_path)
        texture_source_img = cv2.cvtColor(texture_source_img, cv2.COLOR_BGR2RGB)
        texture_source_img = cv2.resize(texture_source_img/255.0, dsize=img_size)

        return texture_source_img

    def _structure_source(self, img: np.ndarray) -> np.ndarray:
        # img = (img * 255).astype(np.uint8)
        structure_source_img = self.rand_augment()(image=img)

        img_size = img.shape[:-1]  # H x W

        assert img_size[0] % 4 == 0, 'structure should be devided by grid size accurately'
        grid_w = img_size[1] // 8
        grid_h = img_size[0] // 4

        structure_source_img = rearrange(
            tensor=structure_source_img,
            pattern='(h gh) (w gw) c -> (h w) gw gh c',
            gw=grid_w,
            gh=grid_h
        )
        disordered_idx = np.arange(structure_source_img.shape[0])
        np.random.shuffle(disordered_idx)

        structure_source_img = rearrange(
            tensor=structure_source_img[disordered_idx],
            pattern='(h w) gw gh c -> (h gh) (w gw) c',
            h=4,
            w=8
        ).astype(np.float32)

        return structure_source_img

class MVTecDataset_test(torch.utils.data.Dataset):
    def __init__(self, dataset_path, resize):
        self.img_path = os.path.join(dataset_path, 'test')
        self.gt_path = os.path.join(dataset_path, 'ground_truth')
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # labels => good : 0, anomaly : 1
        self.resize = resize
        self.transform = transforms.Compose([ Normalize(), ToTensor()])
        self.gt_transform = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor()
        ])

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img= cv2.resize(img/255., self.resize)

        img = self.transform(img)

        if gt == 0:
            gt = torch.zeros([1, img.shape[-1], img.shape[-1]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.shape[1:] == gt.shape[1:], "image.size != gt.size !!!"

        return (img, gt, label, img_type, img_path.split('/')[-1])
class VisADataset_train(torch.utils.data.Dataset):
    def __init__(
            self,
            target,
            dataset_path,
            anomaly_source_path,
            resize,
            perlin_scale,
            min_perlin_scale,
            perlin_noise_threshold,
            transparency_range
    ):
        self.target = target
        self.img_path = dataset_path
        self.img_paths = glob.glob(os.path.join(self.img_path, 'good') + "/*.JPG")
        self.anomaly_source_path = anomaly_source_path
        self.anomaly_source_file_list = glob.glob(os.path.join(self.anomaly_source_path, '*/*'))
        self.resize = resize
        self.transform = transforms.Compose([ Normalize(), ToTensor() ])
        self.transparency_range = transparency_range
        self.perlin_scale = perlin_scale
        self.min_perlin_scale = min_perlin_scale
        self.perlin_noise_threshold = perlin_noise_threshold
        self.anomaly_switch = False

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img / 255., self.resize)
        if self.anomaly_switch:
            img_anomaly, mask = self.generate_anomaly1(img, self.anomaly_source_file_list, img_path)
            self.anomaly_switch = False
        else:
            img_anomaly = img
            self.anomaly_switch = True
        img_normal = self.transform(img)
        img_anomaly = self.transform(img_anomaly)

        return img_normal, img_anomaly, img_path.split('/')[-1]

    def rand_augment(self):
        augmenters = [
            iaa.GammaContrast((0.5, 2.0), per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
            iaa.Solarize(0.5, threshold=(32, 128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45))
        ]

        aug_idx = np.random.choice(np.arange(len(augmenters)), 3, replace=False)
        aug = iaa.Sequential([
            augmenters[aug_idx[0]],
            augmenters[aug_idx[1]],
            augmenters[aug_idx[2]]
        ])

        return aug

    def generate_anomaly1(self, img: np.ndarray, texture_img_list: list, img_path: str):
        img_size = img.shape[:-1]  # H x W

        target_foreground_mask = self.generate_target_foreground_mask(img_path=img_path)
        perlin_noise_mask1 = self.generate_perlin_noise_mask(img_size=img_size)
        perlin_noise_mask2 = self.generate_perlin_noise_mask(img_size=img_size)
        bool_mask1 = perlin_noise_mask1 > 0
        bool_mask2 = perlin_noise_mask2 > 0
        intersection = bool_mask1 & bool_mask2
        union = bool_mask1 | bool_mask2
        perlin_noise_masks = [perlin_noise_mask1, perlin_noise_mask2, intersection, union]

        mask = random.choice(perlin_noise_masks) * target_foreground_mask
        mask_expanded = np.expand_dims(mask, axis=2)

        anomaly_source_img = self.anomaly_source(img=img, texture_img_list=texture_img_list)
        factor = np.random.uniform(*self.transparency_range, size=1)[0]
        anomaly_mask = factor * (mask_expanded * anomaly_source_img) + (1 - factor) * (mask_expanded * img)

        anomaly_img = (((- mask_expanded + 1) * img) + anomaly_mask)

        return anomaly_img, mask

    def generate_target_foreground_mask(self, img_path: str) -> np.ndarray:
        target_foreground_mask = np.zeros(self.resize, dtype=np.float64)
        if self.target in object_list:
            foreground_path = img_path.replace('train', 'DISthresh')
            target_foreground_mask = cv2.imread(foreground_path, 0)
            target_foreground_mask = cv2.resize(target_foreground_mask/255.0, dsize=self.resize)
        if self.target in texture_list:
            target_foreground_mask = np.ones(self.resize, dtype=np.float64)
        return target_foreground_mask

    def generate_perlin_noise_mask(self, img_size: tuple) -> np.ndarray:
        perlin_scalex = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np(img_size, (perlin_scalex, perlin_scaley))

        rot = iaa.Affine(rotate=(-90, 90))
        perlin_noise = rot(image=perlin_noise)

        mask_noise = np.where(
            perlin_noise > self.perlin_noise_threshold,
            np.ones_like(perlin_noise),
            np.zeros_like(perlin_noise)
        )

        return mask_noise

    def anomaly_source(self, img: np.ndarray, texture_img_list: list = None) -> np.ndarray:
        p = 0
        if p < 0.5:
            idx = np.random.choice(len(texture_img_list))
            img_size = img.shape[:-1]  # H x W
            anomaly_source_img = self._texture_source(img_size=img_size, texture_img_path=texture_img_list[idx])
        else:
            anomaly_source_img = self._structure_source(img=img)

        return anomaly_source_img

    def _texture_source(self, img_size: tuple, texture_img_path: str) -> np.ndarray:
        texture_source_img = cv2.imread(texture_img_path)
        texture_source_img = cv2.cvtColor(texture_source_img, cv2.COLOR_BGR2RGB)
        texture_source_img = cv2.resize(texture_source_img/255.0, dsize=img_size)
        return texture_source_img

    def _structure_source(self, img: np.ndarray) -> np.ndarray:
        # img = (img * 255).astype(np.uint8)
        structure_source_img = self.rand_augment()(image=img)

        img_size = img.shape[:-1]  # H x W

        assert img_size[0] % 4 == 0, 'structure should be devided by grid size accurately'
        grid_w = img_size[1] // 8
        grid_h = img_size[0] // 4

        structure_source_img = rearrange(
            tensor=structure_source_img,
            pattern='(h gh) (w gw) c -> (h w) gw gh c',
            gw=grid_w,
            gh=grid_h
        )
        disordered_idx = np.arange(structure_source_img.shape[0])
        np.random.shuffle(disordered_idx)

        structure_source_img = rearrange(
            tensor=structure_source_img[disordered_idx],
            pattern='(h w) gw gh c -> (h gh) (w gw) c',
            h=4,
            w=8
        ).astype(np.float32)

        return structure_source_img

class VisADataset_test(torch.utils.data.Dataset):
    def __init__(self, dataset_path, resize):
        self.img_path = os.path.join(dataset_path, 'test')
        self.gt_path = os.path.join(dataset_path, 'ground_truth')
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # labels => good : 0, anomaly : 1
        self.resize = resize
        self.transform = transforms.Compose([ Normalize(), ToTensor() ])
        self.gt_transform = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor()
        ])

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img / 255., (256, 256))

        img = self.transform(img)

        if gt == 0:
            gt = torch.zeros([1, img.shape[-1], img.shape[-1]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.shape[1:] == gt.shape[1:], "image.size != gt.size !!!"

        return (img, gt, label, img_type, img_path.split('/')[-1])
class BTADDataset_train(torch.utils.data.Dataset):
    def __init__(
            self,
            target,
            dataset_path,
            anomaly_source_path,
            resize,
            perlin_scale,
            min_perlin_scale,
            perlin_noise_threshold,
            transparency_range
    ):
        self.target = target
        self.img_path = dataset_path
        if "02" in self.img_path:
            self.img_paths = glob.glob(os.path.join(self.img_path, 'good') + "/*.png")
        else:
            self.img_paths = glob.glob(os.path.join(self.img_path, 'good') + "/*.bmp")
        self.anomaly_source_path = anomaly_source_path
        self.anomaly_source_file_list = glob.glob(os.path.join(self.anomaly_source_path, '*/*'))
        self.resize = resize
        self.transform = transforms.Compose([ Normalize(), ToTensor()
        ])
        self.transparency_range = transparency_range
        self.perlin_scale = perlin_scale
        self.min_perlin_scale = min_perlin_scale
        self.perlin_noise_threshold = perlin_noise_threshold
        self.anomaly_switch = False

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img= cv2.resize(img/255., self.resize)
        if self.anomaly_switch:
            img_anomaly, mask = self.generate_anomaly1(img, self.anomaly_source_file_list ,img_path)
            self.anomaly_switch = False
        else:
            img_anomaly = img
            self.anomaly_switch = True
        img_normal = self.transform(img)
        img_anomaly = self.transform(img_anomaly)

        return img_normal, img_anomaly, img_path.split('/')[-1]

    def rand_augment(self):
        augmenters = [
            iaa.GammaContrast((0.5, 2.0), per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
            iaa.Solarize(0.5, threshold=(32, 128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45))
        ]

        aug_idx = np.random.choice(np.arange(len(augmenters)), 3, replace=False)
        aug = iaa.Sequential([
            augmenters[aug_idx[0]],
            augmenters[aug_idx[1]],
            augmenters[aug_idx[2]]
        ])

        return aug

    def generate_anomaly1(self, img: np.ndarray, texture_img_list: list, img_path: str):
        img_size = img.shape[:-1]  # H x W

        target_foreground_mask = self.generate_target_foreground_mask(img_path=img_path)
        perlin_noise_mask1 = self.generate_perlin_noise_mask(img_size=img_size)
        perlin_noise_mask2 = self.generate_perlin_noise_mask(img_size=img_size)
        bool_mask1 = perlin_noise_mask1 > 0
        bool_mask2 = perlin_noise_mask2 > 0
        intersection = bool_mask1 & bool_mask2
        union = bool_mask1 | bool_mask2
        perlin_noise_masks = [perlin_noise_mask1, perlin_noise_mask2, intersection, union]

        mask = random.choice(perlin_noise_masks) * target_foreground_mask
        mask_expanded = np.expand_dims(mask, axis=2)

        anomaly_source_img = self.anomaly_source(img=img, texture_img_list=texture_img_list)
        factor = np.random.uniform(*self.transparency_range, size=1)[0]
        anomaly_mask = factor * (mask_expanded * anomaly_source_img) + (1 - factor) * (mask_expanded * img)

        anomaly_img = (((- mask_expanded + 1) * img) + anomaly_mask)

        return anomaly_img, mask

    def generate_target_foreground_mask(self, img_path: str) -> np.ndarray:
        target_foreground_mask = np.zeros(self.resize, dtype=np.float64)
        if self.target in object_list:
            foreground_path = img_path.replace('train', 'DISthresh')
            target_foreground_mask = cv2.imread(foreground_path, 0)
            target_foreground_mask = cv2.resize(target_foreground_mask/255.0, dsize=self.resize)
        if self.target in texture_list:
            target_foreground_mask = np.ones(self.resize, dtype=np.float64)
        return target_foreground_mask

    def generate_perlin_noise_mask(self, img_size: tuple) -> np.ndarray:
        perlin_scalex = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np(img_size, (perlin_scalex, perlin_scaley))

        rot = iaa.Affine(rotate=(-90, 90))
        perlin_noise = rot(image=perlin_noise)

        mask_noise = np.where(
            perlin_noise > self.perlin_noise_threshold,
            np.ones_like(perlin_noise),
            np.zeros_like(perlin_noise)
        )

        return mask_noise

    def anomaly_source(self, img: np.ndarray, texture_img_list: list = None) -> np.ndarray:
        p = 0
        if p < 0.5:
            idx = np.random.choice(len(texture_img_list))
            img_size = img.shape[:-1]  # H x W
            anomaly_source_img = self._texture_source(img_size=img_size, texture_img_path=texture_img_list[idx])
        else:
            anomaly_source_img = self._structure_source(img=img)

        return anomaly_source_img

    def _texture_source(self, img_size: tuple, texture_img_path: str) -> np.ndarray:
        texture_source_img = cv2.imread(texture_img_path)
        texture_source_img = cv2.cvtColor(texture_source_img, cv2.COLOR_BGR2RGB)
        texture_source_img = cv2.resize(texture_source_img/255.0, dsize=img_size)
        return texture_source_img

    def _structure_source(self, img: np.ndarray) -> np.ndarray:
        # img = (img * 255).astype(np.uint8)
        structure_source_img = self.rand_augment()(image=img)

        img_size = img.shape[:-1]  # H x W

        assert img_size[0] % 4 == 0, 'structure should be devided by grid size accurately'
        grid_w = img_size[1] // 8
        grid_h = img_size[0] // 4

        structure_source_img = rearrange(
            tensor=structure_source_img,
            pattern='(h gh) (w gw) c -> (h w) gw gh c',
            gw=grid_w,
            gh=grid_h
        )
        disordered_idx = np.arange(structure_source_img.shape[0])
        np.random.shuffle(disordered_idx)

        structure_source_img = rearrange(
            tensor=structure_source_img[disordered_idx],
            pattern='(h w) gw gh c -> (h gh) (w gw) c',
            h=4,
            w=8
        ).astype(np.float32)

        return structure_source_img

class BTADDataset_test(torch.utils.data.Dataset):
    def __init__(self, dataset_path, resize):
        self.img_path = os.path.join(dataset_path, 'test')
        self.gt_path = os.path.join(dataset_path, 'ground_truth')
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # labels => good : 0, anomaly : 1
        self.resize = resize
        self.transform = transforms.Compose([ Normalize(), ToTensor()
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor()
        ])

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                if "02" in self.img_path:
                    img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                else:
                    img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                if "02" in self.img_path:
                    img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                else:
                    img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                if "03" in self.gt_path:
                    gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.bmp")
                else:
                    gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))
        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img / 255., (256, 256))

        img = self.transform(img)

        if gt == 0:
            gt = torch.zeros([1, img.shape[-1], img.shape[-1]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.shape[1:] == gt.shape[1:], "image.size != gt.size !!!"

        return (img, gt, label, img_type, img_path.split('/')[-1])