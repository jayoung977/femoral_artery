# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from PIL import Image, ImageEnhance
torch.set_num_threads(1)

def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))

# def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
#     assert -0.45 <= v <= 0.45
#     if random.random() > 0.5:
#         v = -v
#     v = v * img.size[0]
#     return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


# def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
#     assert -0.45 <= v <= 0.45
#     if random.random() > 0.5:
#         v = -v
#     v = v * img.size[1]
#     return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)

def Darkening(image, v):
    assert 0.1 <= v <= 0.9

    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(v)

    return image

def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def SolarizeAdd(img, addition=0, threshold=128):
    img_np = np.array(img).astype(np.int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)

def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)

def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


# def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
#     assert 0.0 <= v <= 0.2
#     if v <= 0.:
#         return img

#     v = v * img.size[0]
#     return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


from transformations import noise as noise
from transformations import shadow_ellipse as ellipse
from transformations import shadow_polygon as polygon
from transformations import shadow_single as single
# MIN_SHADOW              = 0.3
# MAX_SHADOW              = 0.7
def add_n_random_shadows(image, v):
    assert 0.1 <= v <= 1.0

    n_shadow  = random.randint(1,3)
    blur_scale = 1.0    

    # intensity = np.random.uniform(const.MIN_SHADOW, const.MAX_SHADOW)
    intensity = np.random.uniform(0.1, v)

    return add_n_shadows(image, n_shadow, intensity, blur_scale)

def add_n_shadows(image, n_shadow = 4, intensity = 0.5, blur_scale = 1.0):
	for i in range(n_shadow ):
		blur_width = noise.get_blur_given_intensity(intensity, blur_scale)
        
		choice = np.random.uniform(0, 10)
		if choice < 1:
			image = polygon.add_n_triangles_shadow(image, intensity, blur_width)
		elif choice >= 1 and choice < 2:
			image = polygon.add_n_triangles_light(image, intensity, blur_width)
		elif choice >= 2 and choice < 3:
			image = polygon.add_polygon_light(image, intensity, blur_width)
		elif choice >= 3 and choice < 4:
			image = polygon.add_polygon_shadow(image, intensity, blur_width)

		elif choice >= 4 and choice < 5:
			image = single.add_single_light(image, intensity, blur_width)
		elif choice >= 5 and choice < 6:
			image = single.add_single_shadow(image, intensity, blur_width)

		elif choice >= 6 and choice < 7:
			image = ellipse.add_n_ellipses_light(image, intensity, blur_width)
		elif choice >= 7 and choice < 8:
			image = ellipse.add_n_ellipses_shadow(image, intensity, blur_width)
		elif choice >= 8 and choice < 9:
			image = ellipse.add_ellipse_light(image, intensity, blur_width)
		else:
			image = ellipse.add_ellipse_shadow(image, intensity, blur_width)

	return Image.fromarray(np.uint8(image))

import cv2 as cv
def HDR(image, v):
    assert 1/256 <= v <= 30
    v2 = random.uniform(v, 30)

    images, times = [np.array(image)], [np.array(v2)]
    times = np.asarray(times, dtype=np.float32)

    calibrate = cv.createCalibrateDebevec()
    response = calibrate.process(images, times)

    merge_debevec = cv.createMergeDebevec()
    hdr = merge_debevec.process(images, times, response)

    tonemap = cv.createTonemap(2.2)
    ldr = tonemap.process(hdr)
    # ldr = cv.normalize(ldr, None, 0, 255, cv.NORM_MINMAX)

    merge_mertens = cv.createMergeMertens()
    fusion = merge_mertens.process(images)
    fusion = cv.normalize(fusion, None, 0, 255, cv.NORM_MINMAX)

    return Image.fromarray(np.uint8(fusion))

import PIL.ImageFilter as Filter
def Edge_Enhance(img, v):
    assert 0 <= v <= 1
    v = int(round(v))
    if v%2 == 0:
        v = v + 1
    imgae = img.filter(Filter.EDGE_ENHANCE)

    return imgae

import PIL.ImageFilter as Filter
def Gaussian_Blur(img, v):
    assert 1 <= v <= 3
    v = int(round(v))
    if v%2 == 0:
        v = v + 1
    image = img.filter(Filter.GaussianBlur(radius = v))

    return image

def Box_Blur(img, v):
    assert 1 <= v <= 3
    v = int(round(v))
    if v%2 == 0:
        v = v + 1
    image = img.filter(Filter.BoxBlur(radius = v))

    return image

def Unsharp_Blur(img, v):
    assert 1 <= v <= 3
    v = int(round(v))
    if v%2 == 0:
        v = v + 1
    image = img.filter(Filter.UnsharpMask(radius = v))

    return image

def Rank_Blur(img, v):
    assert 1 <= v <= 3
    v = int(round(v))
    if v%2 == 0:
        v = v + 1    
    Filter_Type = random.randint(0,2)
    if(Filter_Type == 0):
        Filter_Rank = 0
    elif(Filter_Type == 1):
        Filter_Rank = int(v * v / 2)
    else:
        Filter_Rank = v * v - 1

    image = img.filter(Filter.RankFilter(size = v, rank = Filter_Rank))

    return image

def Median_Blur(img, v):
    assert 1 <= v <= 3
    v = int(round(v))
    if v%2 == 0:
        v = v + 1
    image = img.filter(Filter.MedianFilter(size = v))

    return image

def Min_Blur(img, v):
    assert 1 <= v <= 3
    v = int(round(v))
    if v%2 == 0:
        v = v + 1
    image = img.filter(Filter.MinFilter(size = v))

    return image

def Max_Blur(img, v):
    assert 1 <= v <= 3
    v = int(round(v))
    if v%2 == 0:
        v = v + 1
    image = img.filter(Filter.MaxFilter(size = v))

    return image

def Model_Blur(img, v):
    assert 1 <= v <= 3
    v = int(round(v))
    if v%2 == 0:
        v = v + 1
    image = img.filter(Filter.ModeFilter(size = v))

    return image

from skimage.filters import gaussian
from skimage.util import random_noise
def Gaussian_Noise(image, v):
    assert 0 < v <= 0.005
    image = random_noise(np.array(image), mode='gaussian', clip = True, mean = v) * 255

    return Image.fromarray(np.uint8(image))

def Salt_Noise(image, v):
    # If the specified `prob` is negative or zero, we don't need to do anything.
    assert 0 < v <= 0.005
    image = random_noise(np.array(image), mode='salt', clip = True, amount = v) * 255

    return Image.fromarray(np.uint8(image))

def Pepper_Noise(image, v):
    # If the specified `prob` is negative or zero, we don't need to do anything.
    assert 0 < v <= 0.005
    image = random_noise(np.array(image), mode='pepper', clip = True, amount = v) * 255

    return Image.fromarray(np.uint8(image))

def Salt_Pepper_Noise(image, v):
    # If the specified `prob` is negative or zero, we don't need to do anything.
    assert 0 < v <= 0.005
    image = random_noise(np.array(image), mode='s&p', clip = True, amount = v) * 255

    return Image.fromarray(np.uint8(image))

# def LocalVar_Noise(image, v):
#     # If the specified `prob` is negative or zero, we don't need to do anything.
#     assert 0 < v <= 1

#     Gaussian_Filter = gaussian(np.zeros((np.shape(image)[0], np.shape(image)[1], 3)), v) + 0.001
#     image = random_noise(np.array(image), mode='localvar', clip = True, local_vars = Gaussian_Filter)

#     return Image.fromarray(np.uint8(image))

def Poisson_Noise(image, v):
    # If the specified `prob` is negative or zero, we don't need to do anything.
    assert 0 < v <= 10
    image = random_noise(np.array(image), mode='poisson',  clip = True) * 255

    return Image.fromarray(np.uint8(image))

def Speckle_Noise(image, v):
    assert 0 < v <= 0.005
    image = random_noise(np.array(image), mode='speckle', clip = True, mean = v) * 255

    return Image.fromarray(np.uint8(image))

def augment_list():  # 16 oeprations and their ranges
    l = [
        ### Use
        (AutoContrast, 0, 1), # O
        (Color, 0.1, 1.9), # O
        (Contrast, 0.1, 1.9), # O
        (Brightness, 0.1, 1.9), # O
        (Sharpness, 0.1, 1.9), # O
        (add_n_random_shadows, 0.1, 1), # O
        (Darkening, 0.1, 0.9),
        (HDR, 1/256, 30),
        (Edge_Enhance, 0, 1),

        # (Gaussian_Blur, 1, 3),
        # (Unsharp_Blur, 1, 3),
        # (Box_Blur, 1, 3),
        # (Rank_Blur, 1, 3),
        # (Median_Blur, 1, 3),
        # (Min_Blur, 1, 3),
        # (Max_Blur, 1, 3),
        # (Model_Blur, 1, 3),

        # (Gaussian_Noise, 0.01, 1),
        # (Salt_Noise, 0.001, 0.01),
        # (Pepper_Noise, 0.001, 0.01),
        # (Salt_Pepper_Noise, 0.001, 0.01),
        # (Poisson_Noise, 0.001, 10),
        # (Speckle_Noise, 0.001, 0.02)        
    ]

    return l

def Blur_augment_list():
    l = [
        (Gaussian_Blur, 1, 3),
        (Unsharp_Blur, 1, 3),
        (Box_Blur, 1, 3),
        (Rank_Blur, 1, 3),
        (Median_Blur, 1, 3),
        (Min_Blur, 1, 3),
        (Max_Blur, 1, 3),
        (Model_Blur, 1, 3),
    ]

    return l

def Noise_augment_list():
    l = [
        (Gaussian_Noise, 0.001, 0.005),
        (Salt_Noise, 0.001, 0.005),
        (Pepper_Noise, 0.001, 0.005),
        (Salt_Pepper_Noise, 0.001, 0.005),
        (Poisson_Noise, 0.001, 10),
        (Speckle_Noise, 0.001, 0.005)  
    ]

    return l

class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30]
        self.augment_list = augment_list()
        self.Blur_augment_list = Blur_augment_list()
        self.Noise_augment_list = Noise_augment_list()

    def __call__(self, img):
        ops = []

        # A_ops = random.choices(self.augment_list, k=self.n)
        # ops += A_ops

        Is_Blur = random.randint(0,3)
        Is_Noise = random.randint(0,3)

        if(Is_Blur == 0 and Is_Noise == 0):
            A_ops = random.choices(self.augment_list, k=self.n - 2)
            B_ops = random.choices(self.Blur_augment_list, k=1)
            N_ops = random.choices(self.Noise_augment_list, k=1)
        elif(Is_Blur == 0 and Is_Noise != 0):
            A_ops = random.choices(self.augment_list, k=self.n - 1)
            B_ops = random.choices(self.Blur_augment_list, k=1)
            N_ops = random.choices(self.Noise_augment_list, k=0)
        elif(Is_Blur != 0 and Is_Noise == 0):
            A_ops = random.choices(self.augment_list, k=self.n - 1)
            B_ops = random.choices(self.Blur_augment_list, k=0)
            N_ops = random.choices(self.Noise_augment_list, k=1)  
        else:
            A_ops = random.choices(self.augment_list, k=self.n)
            B_ops = random.choices(self.Blur_augment_list, k=0)
            N_ops = random.choices(self.Noise_augment_list, k=0)             

        ops += B_ops
        ops += N_ops
        ops += A_ops

        # augment_list Blur_augment_list Noise_augment_list
        # ops = random.choices(self.augment_list, k=self.n)

        op_func_name = []
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)
            op_func_name.append(op.__name__)
        # return img, op_func_name
        return img
