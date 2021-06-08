import numpy as np # linear algebra
from PIL import Image
import random
import torch
from numpy.random import choice


def rotate_data(d, rotation):
    img = Image.fromarray(d, mode='L')
    result = torch.tensor(img.rotate(rotation))
    return result


def process_data(data, target, aug_fold=5, min_rot=-45, max_rot=45):
    """
    rotate image with given angle range. Each image rotate for aug_fold times.
    :param data_path: path to load original processed .pt data.
    :param aug_fold: Path to
    :param min_rot:
    :param max_rot:
    :return:
    """
    rot_data = []
    rot_target = []
    for idx in range(data.shape[0]):
        rot_list = choice(np.arange(min_rot, max_rot), aug_fold, replace=False)
        for rot_angle in rot_list:
            rot_data.append(rotate_data(data[idx], rot_angle))
            rot_target.append(torch.tensor([rot_angle, target[idx]]))
    rot_data = torch.cat(rot_data, dim=0)
    rot_target = torch.cat(rot_target, dim=0)

    return rot_data, rot_target


data_path = '/home/yinghua/dataset/mnist/'
f = np.load(data_path+'mnist.npz')
x_tr = f['x_train']
y_tr = f['y_train']
x_te = f['x_test']
y_te = f['y_test']
f.close()
print(x_tr.shape, np.max(x_tr[0]), np.min(x_tr[0]), x_te.shape)

rotate_tr = []
rotate_te = []
min_rot = -45
max_rot = 45
rot_per_img = 5

for i in range(x_tr.shape[0]):
    rot_list = choice(np.arange(min_rot, max_rot), rot_per_img, replace=False)
    for r_idx, rot in enumerate(rot_list):
        rotate_tr.append([rotate_data(x_tr[i], rot), rot,  y_tr[i]])
        np.save(data_path + "train/{i:d}_{rot:d}_{y:d}.npy".format(i=i, rot=rot, y=y_tr[i]), )
for i in range(x_te.shape[0]):
    rot_list = choice(np.arange(min_rot, max_rot), rot_per_img, replace=False)
    for rot in rot_list:
        rotate_te.append([rotate_data(x_te[i], rot), rot,  y_te[i]])

np.save(data_path+'rotate_tr.npy', rotate_tr)
np.save(data_path+'rotate_te.npy', rotate_te)
