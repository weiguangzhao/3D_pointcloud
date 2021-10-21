# _*_coding : UTF-8_*_
# Code writer: Weiguang.Zhao
# Writing time: 2021/10/21  下午2:52
# File Name: data_aug.py
# IDE: PyCharm

import math
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import scipy.ndimage
import scipy.interpolate


# # =====================================matplotlib == 2.2.0==================================
def get_ptcloud_img(xyz, rgb, title):
    fig = plt.figure(figsize=(5, 5))
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    ax = Axes3D(fig)
    ax.view_init(90, -90)  # view angle
    # ax.axis('off')

    # plot point
    # max, min = np.max(xyz), np.min(xyz)
    max_x, min_x = np.max(x), np.min(x)
    max_y, min_y = np.max(y), np.min(y)
    max_z, min_z = np.max(z), np.min(z)
    ax.set_xbound(min_x, max_x)
    ax.set_ybound(min_y, max_y)
    ax.set_zbound(min_z, max_z)
    ax.scatter(x, y, z, zdir='z', c=rgb, marker='.', s=20)
    plt.title(title)
    # plt.savefig('test.png')
    plt.show()


# # ========================aug: jitter,  flip,   rot============================================
def dataAugment(xyz, jitter=False, flip=False, rot=False):
    m = np.eye(3)
    if jitter:
        m += np.random.randn(3, 3) * 0.1
    if flip:
        m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
    if rot:
        theta = np.random.rand() * 2 * math.pi
        m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
    return np.matmul(xyz, m)


# #=================================Elastic distortion==============================================
def elastic(xyz, gran, mag):
    blur0 = np.ones((3, 1, 1)).astype('float32') / 3
    blur1 = np.ones((1, 3, 1)).astype('float32') / 3
    blur2 = np.ones((1, 1, 3)).astype('float32') / 3

    bb = np.abs(xyz).max(0).astype(np.int32) // gran + 3
    noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
    noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
    ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
    interp = [scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]

    def g(x_):
        return np.hstack([i(x_)[:, None] for i in interp])

    return xyz + g(xyz) * mag


# # ================================crop scenes=======================================================
def crop(xyz, full_scale, max_crop_p):
    '''
    :param xyz: (n, 3) >= 0
    '''
    xyz_offset = xyz.copy()
    valid_idxs = (xyz_offset.min(1) >= 0)
    assert valid_idxs.sum() == xyz.shape[0]

    full_scale = np.array([full_scale] * 3)
    room_range = xyz.max(0) - xyz.min(0)
    while (valid_idxs.sum() > max_crop_p):
        offset = np.clip(full_scale - room_range + 0.001, None, 0) * np.random.rand(3)
        xyz_offset = xyz + offset
        valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < full_scale).sum(1) == 3)
        full_scale[:2] -= 32/50.0

    return xyz_offset, valid_idxs


# # ================================main=======================================================
if __name__ == '__main__':
    pt_origin = np.load('input/scene0332_01_vert.npy')   # read pointcloud data [N, 6] : xyz rgb
    xyz = pt_origin[..., :3]
    rgb = pt_origin[..., 3:]
    xyz = xyz - xyz.min(0)                              # offset to First quadrant
    rgb = rgb/255.0                                     # normalize

    get_ptcloud_img(xyz, rgb, title='original')

    xyz_jitter = dataAugment(xyz, jitter=True, flip=False, rot=False)
    get_ptcloud_img(xyz_jitter, rgb, title='jitter')

    xyz_flip = dataAugment(xyz, jitter=False, flip=True, rot=False)
    get_ptcloud_img(xyz_flip, rgb, title='Flip')  # random flip

    xyz_rot = dataAugment(xyz, jitter=False, flip=False, rot=True)
    get_ptcloud_img(xyz_rot, rgb, title='rotation')

    xyz_elastic = elastic(xyz, 12, 80)
    get_ptcloud_img(xyz_elastic, rgb, title='elastic')

    print("original point number is  {}".format(xyz.shape[0]))
    _, crop_index = crop(xyz, 10.0, 50261)   # # xyz scale max_point
    xyz_crop = xyz[crop_index, ...]
    rgb_crop = rgb[crop_index, ...]
    get_ptcloud_img(xyz_crop, rgb_crop, title='crop')
    print("retain point number is  {}".format(xyz_crop.shape[0]))









