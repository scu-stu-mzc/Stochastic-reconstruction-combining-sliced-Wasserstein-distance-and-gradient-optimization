import sys
import os
import cv2
import torch
import time
import torch.nn.functional as F
from torchvision import transforms
import torchvision.utils as vutils
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def expanding_operation(mp1, mp2):
    if mp1.shape[1] == mp2.shape[1]:
        return mp1, mp2
    elif mp1.shape[1] < mp2.shape[1]:
        t_mp = mp1
        mp1 = mp2
        mp2 = t_mp
    t_times = mp1.shape[1] // mp2.shape[1]
    mp2 = torch.cat([mp2] * t_times, dim=1)
    if mp1.shape[1] > mp2.shape[1]:
        t_mod = torch.randperm(mp2.shape[1])[:mp1.shape[1] - mp2.shape[1]]
        mp2 = torch.cat([mp2, mp2[:, t_mod]], dim=1)
    return mp1, mp2


def expanding_operation_v2(mp1, mp2):
    if mp1.shape[2] == mp2.shape[2]:
        return mp1, mp2
    elif mp1.shape[2] < mp2.shape[2]:
        t_mp = mp1
        mp1 = mp2
        mp2 = t_mp
    t_times = mp1.shape[2] // mp2.shape[2]
    mp2 = torch.cat([mp2] * t_times, dim=2)
    if mp1.shape[2] > mp2.shape[2]:
        t_mod = torch.randperm(mp2.shape[2])[:mp1.shape[2] - mp2.shape[2]]
        mp2 = torch.cat([mp2, mp2[:, t_mod]], dim=2)
    return mp1, mp2



class SingleMPSWDLoss3D(torch.nn.Module):
    """single 3D template implementation: require 5D input"""
    def __init__(self, temp_size=7, stride=1, num_proj=256, channels=1):
        super(SingleMPSWDLoss3D, self).__init__()
        self.temp_size = temp_size
        self.stride = stride
        self.num_proj = num_proj
        self.channels = channels

    def forward(self, x, y):
        b, c, l, h, w = x.shape
        rand = torch.randn(self.num_proj, self.channels, self.temp_size, self.temp_size, self.temp_size).to(x.device)
        if self.num_proj > 1:
            rand = rand / torch.std(rand, dim=0, keepdim=True)

        projx = F.conv3d(x, rand).reshape(self.num_proj, -1)
        projy = F.conv3d(y, rand).reshape(self.num_proj, -1)
        projx, projy = expanding_operation(projx, projy)
        projx, _ = torch.sort(projx, dim=1)
        projy, _ = torch.sort(projy, dim=1)
        loss = torch.abs(projx - projy).mean()
        return loss


class MultiMPSWDLoss3D(torch.nn.Module):
    """multiple 3D template implementation: require 5D input"""
    def __init__(self, temp_size=3, stride=1, num_proj=9, channels=1):
        super(MultiMPSWDLoss3D, self).__init__()
        self.temp_size = temp_size
        self.stride = stride
        self.num_proj = num_proj
        self.channels = channels

    def forward(self, x, y):
        b, c, l, h, w = x.shape
        rand_1 = torch.randn(self.num_proj, self.channels, self.temp_size, self.temp_size, self.temp_size).to(
            x.device)
        rand_2 = torch.randn(self.num_proj, self.num_proj, self.temp_size, self.temp_size, self.temp_size).to(
            x.device)
        rand_3 = torch.randn(self.num_proj, self.num_proj, self.temp_size, self.temp_size, self.temp_size).to(
            x.device)

        if self.num_proj > 1:
            rand_1 = rand_1 / torch.std(rand_1, dim=0, keepdim=True)
            rand_2 = rand_2 / torch.std(rand_2, dim=0, keepdim=True)
            rand_3 = rand_3 / torch.std(rand_3, dim=0, keepdim=True)

        projx_1 = F.conv3d(x, rand_1, stride=self.stride)
        projx_2 = F.conv3d(projx_1, rand_2, stride=self.stride)
        projx_3 = F.conv3d(projx_2, rand_3, stride=self.stride)
        projy_1 = F.conv3d(y, rand_1, stride=self.stride)
        projy_2 = F.conv3d(projy_1, rand_2, stride=self.stride)
        projy_3 = F.conv3d(projy_2, rand_3, stride=self.stride)

        outx1 = projx_1.reshape(self.num_proj, -1)
        outx2 = projx_2.reshape(self.num_proj, -1)
        outx3 = projx_3.reshape(self.num_proj, -1)
        outy1 = projy_1.reshape(self.num_proj, -1)
        outy2 = projy_2.reshape(self.num_proj, -1)
        outy3 = projy_3.reshape(self.num_proj, -1)

        px_1, _ = torch.sort(outx1, dim=1)
        px_2, _ = torch.sort(outx2, dim=1)
        px_3, _ = torch.sort(outx3, dim=1)
        py_1, _ = torch.sort(outy1, dim=1)
        py_2, _ = torch.sort(outy2, dim=1)
        py_3, _ = torch.sort(outy3, dim=1)

        loss1 = torch.abs(px_1 - py_1).mean()
        loss2 = torch.abs(px_2 - py_2).mean()
        loss3 = torch.abs(px_3 - py_3).mean()

        # loss1 = torch.nn.functional.mse_loss(px_1, py_1).mean()
        # loss2 = torch.nn.functional.mse_loss(px_2, py_2).mean()
        # loss3 = torch.nn.functional.mse_loss(px_3, py_3).mean()

        loss = 1 * loss1 + 0.1 * loss2 + 0.01 * loss3

        return loss


class SingleMPSWDLoss2Dx3(torch.nn.Module):
    """single 2Dx3 template implementation: require 5D input"""
    def __init__(self, temp_size=3, stride=1, num_proj=9, channels=1):
        super(SingleMPSWDLoss2Dx3, self).__init__()
        self.temp_size = temp_size
        self.stride = stride
        self.num_proj = num_proj
        self.channels = channels

    def forward(self, x, y):
        b, c, l, h, w = x.shape
        rand_dz_1 = torch.randn(self.num_proj, self.channels, 1, self.temp_size, self.temp_size).to(
            x.device)
        rand_dy_1 = torch.randn(self.num_proj, self.channels, self.temp_size, 1, self.temp_size).to(
            x.device)
        rand_dx_1 = torch.randn(self.num_proj, self.channels, self.temp_size, self.temp_size, 1).to(
            x.device)

        if self.num_proj > 1:
            rand_dx_1 = rand_dx_1 / torch.std(rand_dx_1, dim=0, keepdim=True)
            rand_dy_1 = rand_dy_1 / torch.std(rand_dy_1, dim=0, keepdim=True)
            rand_dz_1 = rand_dz_1 / torch.std(rand_dz_1, dim=0, keepdim=True)

        outx_dx_1 = F.conv3d(x, rand_dx_1)
        outx_dy_1 = F.conv3d(x, rand_dy_1)
        outx_dz_1 = F.conv3d(x, rand_dz_1)
        outy_dx_1 = F.conv3d(y, rand_dx_1)
        outy_dy_1 = F.conv3d(y, rand_dy_1)
        outy_dz_1 = F.conv3d(y, rand_dz_1)

        proj_outx_dx_1 = outx_dx_1.reshape(self.num_proj, -1)
        proj_outx_dy_1 = outx_dy_1.reshape(self.num_proj, -1)
        proj_outx_dz_1 = outx_dz_1.reshape(self.num_proj, -1)

        proj_outy_dx_1 = outy_dx_1.reshape(self.num_proj, -1)
        proj_outy_dy_1 = outy_dy_1.reshape(self.num_proj, -1)
        proj_outy_dz_1 = outy_dz_1.reshape(self.num_proj, -1)

        proj_outx_dx_1, proj_outy_dx_1 = expanding_operation(proj_outx_dx_1, proj_outy_dx_1)
        proj_outx_dy_1, proj_outy_dy_1 = expanding_operation(proj_outx_dy_1, proj_outy_dy_1)
        proj_outx_dz_1, proj_outy_dz_1 = expanding_operation(proj_outx_dz_1, proj_outy_dz_1)

        proj_outx_dx_1, _ = torch.sort(proj_outx_dx_1, dim=1)
        proj_outx_dy_1, _ = torch.sort(proj_outx_dy_1, dim=1)
        proj_outx_dz_1, _ = torch.sort(proj_outx_dz_1, dim=1)
        proj_outy_dx_1, _ = torch.sort(proj_outy_dx_1, dim=1)
        proj_outy_dy_1, _ = torch.sort(proj_outy_dy_1, dim=1)
        proj_outy_dz_1, _ = torch.sort(proj_outy_dz_1, dim=1)

        loss_dx_1 = torch.abs(proj_outx_dx_1 - proj_outy_dx_1).mean()
        loss_dy_1 = torch.abs(proj_outx_dy_1 - proj_outy_dy_1).mean()
        loss_dz_1 = torch.abs(proj_outx_dz_1 - proj_outy_dz_1).mean()

        loss = 1 * (loss_dx_1 + loss_dy_1 + 1 * loss_dz_1)
        return loss


class MultiMPSWDLoss2Dx3(torch.nn.Module):
    """multiple 2Dx3 template implementation: require 5D input"""
    def __init__(self, temp_size=3, stride=1, num_proj=9, channels=1):
        super(MultiMPSWDLoss2Dx3, self).__init__()
        self.temp_size = temp_size
        self.stride = stride
        self.num_proj = num_proj
        self.channels = channels

    def forward(self, x, y):
        b, c, l, h, w = x.shape
        rand_dz_1 = torch.randn(self.num_proj, self.channels, 1, self.temp_size, self.temp_size).to(
            x.device)
        rand_dz_2 = torch.randn(self.num_proj, self.num_proj, 1, self.temp_size, self.temp_size).to(
            x.device)
        rand_dz_3 = torch.randn(self.num_proj, self.num_proj, 1, self.temp_size, self.temp_size).to(
            x.device)
        rand_dy_1 = torch.randn(self.num_proj, self.channels, self.temp_size, 1, self.temp_size).to(
            x.device)
        rand_dy_2 = torch.randn(self.num_proj, self.num_proj, self.temp_size, 1, self.temp_size).to(
            x.device)
        rand_dy_3 = torch.randn(self.num_proj, self.num_proj, self.temp_size, 1, self.temp_size).to(
            x.device)
        rand_dx_1 = torch.randn(self.num_proj, self.channels, self.temp_size, self.temp_size, 1).to(
            x.device)
        rand_dx_2 = torch.randn(self.num_proj, self.num_proj, self.temp_size, self.temp_size, 1).to(
            x.device)
        rand_dx_3 = torch.randn(self.num_proj, self.num_proj, self.temp_size, self.temp_size, 1).to(
            x.device)

        if self.num_proj > 1:
            rand_dx_1 = rand_dx_1 / torch.std(rand_dx_1, dim=0, keepdim=True)
            rand_dx_2 = rand_dx_2 / torch.std(rand_dx_2, dim=0, keepdim=True)
            rand_dx_3 = rand_dx_3 / torch.std(rand_dx_3, dim=0, keepdim=True)
            rand_dy_1 = rand_dy_1 / torch.std(rand_dy_1, dim=0, keepdim=True)
            rand_dy_2 = rand_dy_2 / torch.std(rand_dy_2, dim=0, keepdim=True)
            rand_dy_3 = rand_dy_3 / torch.std(rand_dy_3, dim=0, keepdim=True)
            rand_dz_1 = rand_dz_1 / torch.std(rand_dz_1, dim=0, keepdim=True)
            rand_dz_2 = rand_dz_2 / torch.std(rand_dz_2, dim=0, keepdim=True)
            rand_dz_3 = rand_dz_3 / torch.std(rand_dz_3, dim=0, keepdim=True)


        outx_dx_1 = F.conv3d(x, rand_dx_1)
        outx_dx_2 = F.conv3d(outx_dx_1, rand_dx_2)
        outx_dx_3 = F.conv3d(outx_dx_2, rand_dx_3)

        outx_dy_1 = F.conv3d(x, rand_dy_1)
        outx_dy_2 = F.conv3d(outx_dy_1, rand_dy_2)
        outx_dy_3 = F.conv3d(outx_dy_2, rand_dy_3)

        outx_dz_1 = F.conv3d(x, rand_dz_1)
        outx_dz_2 = F.conv3d(outx_dz_1, rand_dz_2)
        outx_dz_3 = F.conv3d(outx_dz_2, rand_dz_3)

        outy_dx_1 = F.conv3d(y, rand_dx_1)
        outy_dx_2 = F.conv3d(outy_dx_1, rand_dx_2)
        outy_dx_3 = F.conv3d(outy_dx_2, rand_dx_3)

        outy_dy_1 = F.conv3d(y, rand_dy_1)
        outy_dy_2 = F.conv3d(outy_dy_1, rand_dy_2)
        outy_dy_3 = F.conv3d(outy_dy_2, rand_dy_3)

        outy_dz_1 = F.conv3d(y, rand_dz_1)
        outy_dz_2 = F.conv3d(outy_dz_1, rand_dz_2)
        outy_dz_3 = F.conv3d(outy_dz_2, rand_dz_3)

        proj_outx_dx_1 = outx_dx_1.reshape(self.num_proj, -1)
        proj_outx_dx_2 = outx_dx_2.reshape(self.num_proj, -1)
        proj_outx_dx_3 = outx_dx_3.reshape(self.num_proj, -1)

        proj_outx_dy_1 = outx_dy_1.reshape(self.num_proj, -1)
        proj_outx_dy_2 = outx_dy_2.reshape(self.num_proj, -1)
        proj_outx_dy_3 = outx_dy_3.reshape(self.num_proj, -1)

        proj_outx_dz_1 = outx_dz_1.reshape(self.num_proj, -1)
        proj_outx_dz_2 = outx_dz_2.reshape(self.num_proj, -1)
        proj_outx_dz_3 = outx_dz_3.reshape(self.num_proj, -1)

        proj_outy_dx_1 = outy_dx_1.reshape(self.num_proj, -1)
        proj_outy_dx_2 = outy_dx_2.reshape(self.num_proj, -1)
        proj_outy_dx_3 = outy_dx_3.reshape(self.num_proj, -1)

        proj_outy_dy_1 = outy_dy_1.reshape(self.num_proj, -1)
        proj_outy_dy_2 = outy_dy_2.reshape(self.num_proj, -1)
        proj_outy_dy_3 = outy_dy_3.reshape(self.num_proj, -1)

        proj_outy_dz_1 = outy_dz_1.reshape(self.num_proj, -1)
        proj_outy_dz_2 = outy_dz_2.reshape(self.num_proj, -1)
        proj_outy_dz_3 = outy_dz_3.reshape(self.num_proj, -1)

        proj_outx_dx_1, proj_outy_dx_1 = expanding_operation(proj_outx_dx_1, proj_outy_dx_1)
        proj_outx_dx_2, proj_outy_dx_2 = expanding_operation(proj_outx_dx_2, proj_outy_dx_2)
        proj_outx_dx_3, proj_outy_dx_3 = expanding_operation(proj_outx_dx_3, proj_outy_dx_3)

        proj_outx_dy_1, proj_outy_dy_1 = expanding_operation(proj_outx_dy_1, proj_outy_dy_1)
        proj_outx_dy_2, proj_outy_dy_2 = expanding_operation(proj_outx_dy_2, proj_outy_dy_2)
        proj_outx_dy_3, proj_outy_dy_3 = expanding_operation(proj_outx_dy_3, proj_outy_dy_3)

        proj_outx_dz_1, proj_outy_dz_1 = expanding_operation(proj_outx_dz_1, proj_outy_dz_1)
        proj_outx_dz_2, proj_outy_dz_2 = expanding_operation(proj_outx_dz_2, proj_outy_dz_2)
        proj_outx_dz_3, proj_outy_dz_3 = expanding_operation(proj_outx_dz_3, proj_outy_dz_3)

        proj_outx_dx_1, _ = torch.sort(proj_outx_dx_1, dim=1)
        proj_outx_dx_2, _ = torch.sort(proj_outx_dx_2, dim=1)
        proj_outx_dx_3, _ = torch.sort(proj_outx_dx_3, dim=1)

        proj_outx_dy_1, _ = torch.sort(proj_outx_dy_1, dim=1)
        proj_outx_dy_2, _ = torch.sort(proj_outx_dy_2, dim=1)
        proj_outx_dy_3, _ = torch.sort(proj_outx_dy_3, dim=1)

        proj_outx_dz_1, _ = torch.sort(proj_outx_dz_1, dim=1)
        proj_outx_dz_2, _ = torch.sort(proj_outx_dz_2, dim=1)
        proj_outx_dz_3, _ = torch.sort(proj_outx_dz_3, dim=1)

        proj_outy_dx_1, _ = torch.sort(proj_outy_dx_1, dim=1)
        proj_outy_dx_2, _ = torch.sort(proj_outy_dx_2, dim=1)
        proj_outy_dx_3, _ = torch.sort(proj_outy_dx_3, dim=1)

        proj_outy_dy_1, _ = torch.sort(proj_outy_dy_1, dim=1)
        proj_outy_dy_2, _ = torch.sort(proj_outy_dy_2, dim=1)
        proj_outy_dy_3, _ = torch.sort(proj_outy_dy_3, dim=1)

        proj_outy_dz_1, _ = torch.sort(proj_outy_dz_1, dim=1)
        proj_outy_dz_2, _ = torch.sort(proj_outy_dz_2, dim=1)
        proj_outy_dz_3, _ = torch.sort(proj_outy_dz_3, dim=1)

        loss_dx_1 = torch.abs(proj_outx_dx_1 - proj_outy_dx_1).mean()
        loss_dx_2 = torch.abs(proj_outx_dx_2 - proj_outy_dx_2).mean()
        loss_dx_3 = torch.abs(proj_outx_dx_3 - proj_outy_dx_3).mean()

        loss_dy_1 = torch.abs(proj_outx_dy_1 - proj_outy_dy_1).mean()
        loss_dy_2 = torch.abs(proj_outx_dy_2 - proj_outy_dy_2).mean()
        loss_dy_3 = torch.abs(proj_outx_dy_3 - proj_outy_dy_3).mean()

        loss_dz_1 = torch.abs(proj_outx_dz_1 - proj_outy_dz_1).mean()
        loss_dz_2 = torch.abs(proj_outx_dz_2 - proj_outy_dz_2).mean()
        loss_dz_3 = torch.abs(proj_outx_dz_3 - proj_outy_dz_3).mean()

        loss = 1 * (loss_dx_1 + loss_dy_1 + loss_dz_1) + 0.1 * (loss_dx_2 + loss_dy_2 + loss_dz_2) + 0.01 * (
                loss_dx_3 + loss_dy_3 + loss_dz_3)
        return loss


# best effect
class MultiMPSWDLoss2D_v3(torch.nn.Module):
    """multiple 2Dx3 template implementation: require 4D input"""
    def __init__(self, temp_size=3, stride=1, num_proj=9, channels=1):
        super(MultiMPSWDLoss2D_v3, self).__init__()
        self.temp_size = temp_size
        self.stride = stride
        self.num_proj = num_proj
        self.channels = channels

    def forward(self, x, y):
        b, c, h, w = x.shape
        rand3_1 = torch.randn(self.num_proj, self.channels, self.temp_size, self.temp_size).to(
            x.device)
        rand3_2 = torch.randn(self.num_proj, self.num_proj, self.temp_size, self.temp_size).to(
            x.device)
        rand3_3 = torch.randn(self.num_proj, self.num_proj, self.temp_size, self.temp_size).to(
            x.device)

        if self.num_proj > 1:
            rand3_1 = rand3_1 / torch.std(rand3_1, dim=0, keepdim=True)  # noramlize
            rand3_2 = rand3_2 / torch.std(rand3_2, dim=0, keepdim=True)
            rand3_3 = rand3_3 / torch.std(rand3_3, dim=0, keepdim=True)

        outx1 = F.conv2d(x, rand3_1)
        outx2 = F.conv2d(outx1, rand3_2)
        outx3 = F.conv2d(outx2, rand3_3)
        outy1 = F.conv2d(y, rand3_1)
        outy2 = F.conv2d(outy1, rand3_2)
        outy3 = F.conv2d(outy2, rand3_3)

        projx1 = outx1.reshape(b, self.num_proj, -1)
        projx2 = outx2.reshape(b, self.num_proj, -1)
        projx3 = outx3.reshape(b, self.num_proj, -1)
        projy1 = outy1.reshape(b, self.num_proj, -1)
        projy2 = outy2.reshape(b, self.num_proj, -1)
        projy3 = outy3.reshape(b, self.num_proj, -1)

        projx1, projy1 = expanding_operation_v2(projx1, projy1)
        projx2, projy2 = expanding_operation_v2(projx2, projy2)
        projx3, projy3 = expanding_operation_v2(projx3, projy3)

        projx1, _ = torch.sort(projx1, dim=2)
        projy1, _ = torch.sort(projy1, dim=2)
        projx2, _ = torch.sort(projx2, dim=2)
        projy2, _ = torch.sort(projy2, dim=2)
        projx3, _ = torch.sort(projx3, dim=2)
        projy3, _ = torch.sort(projy3, dim=2)

        loss1 = torch.abs(projx1 - projy1).mean()
        loss2 = torch.abs(projx2 - projy2).mean()
        loss3 = torch.abs(projx3 - projy3).mean()
        # print(f'loss1:{loss1};loss2:{loss2};loss3:{loss3}')

        loss = loss1 + 0.1 * loss2 + 0.01 * loss3
        return loss


def save_image(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vutils.save_image(torch.clip(img, -1, 1), path, normalize=True)


def cv2pt(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float64) / 255.
    img = img * 2 - 1
    img = torch.from_numpy(img.transpose(2, 0, 1)).float()
    return img


def cv2ptgray(img):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float64) / 255.
    img = img * 2 - 1
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)
    return img


def downscale3d(img, pyr_factor):
    assert 0 < pyr_factor < 1
    b, c, z, y, x = img.shape
    new_z = int(z * pyr_factor)
    new_x = int(x * pyr_factor)
    new_y = int(y * pyr_factor)
    return torch.nn.functional.interpolate(img, size=[new_z, new_y, new_x], mode='trilinear')


def downscale3dby2d(img, pyr_factor):
    assert 0 < pyr_factor < 1
    z, c, y, x = img.shape
    new_z = int(z * pyr_factor)
    new_x = int(x * pyr_factor)
    new_y = int(y * pyr_factor)
    img = transforms.Resize((new_y, new_x))(img)  # z 1 2y 2x
    print('size', img.shape)
    img = img.transpose(0, 2)  # 2y 1 z 2x
    img = transforms.Resize((new_z, new_x))(img)  # 2y 1 2z 2x
    img = img.transpose(0, 2)  # 2z 1 2y 2x
    return img


def get_pyramid3d(img, min_size, pyr_factor):
    res = [img]
    while True:
        img = downscale3dby2d(img, pyr_factor)
        if img.shape[-2] < min_size:
            break
        res = [img] + res
    return res


def Img2Voxel(data_dir):
    files = os.listdir(data_dir)
    files.sort()
    imgs = []
    for file in files:
        file_path = os.path.join(data_dir, file)
        img = cv2.imread(file_path, 0)  # h w
        img = img.astype(np.float64) / 255.
        img = img * 2 - 1
        img = np.expand_dims(img, 0)  # 1 h w
        imgs.append(img)
    img_cat = imgs[0]
    imgs_cat = imgs[1:imgs[0].shape[1]]
    for img in imgs_cat:
        img_cat_j = img
        img_cat = np.concatenate((img_cat, img_cat_j), axis=0)
    voxel = img_cat
    print("the voxel shape:", img_cat.shape)
    return voxel


if __name__ == '__main__':

    random_seed = 9
    # torch.manual_seed(random_seed)
    debug_dir = "3dtest"

    criteria = MultiMPSWDLoss2D_v3(temp_size=3, stride=1, num_proj=9)
    criteria = criteria.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    top_lvl_size = 35
    pyr_factor = 0.8
    lr = 0.05
    num_steps = 400
    decay_steps = 2000

    target_path = 'training_images/3dto3d/TI13'
    img3d = Img2Voxel(target_path)  # l h w
    img3d = torch.from_numpy(img3d).float()  # l h w
    img3d = img3d.unsqueeze(1)  # l 1 h w
    # img3d = img3d.unsqueeze(0).unsqueeze(0)  # b c l h w

    target_pyramid = get_pyramid3d(img3d, top_lvl_size, pyr_factor)
    target_pyramid = [x.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu")) for x in
                      target_pyramid]

    target_img = target_pyramid[-1]
    l, c, h, w = target_pyramid[0].shape[-4:]
    synthesized_image = torch.normal(0, 0.75, size=(l, 1, h, w))
    synthesized_image = synthesized_image.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    synthesized_image.requires_grad_(True)

    loss_history = np.zeros(len(target_pyramid) * num_steps)
    time_begin = time.time()
    for lvl, lvl_target_img in enumerate(target_pyramid):
        if lvl > 0:
            with torch.no_grad():
                l, c, h, w = target_pyramid[lvl].shape[-4:]
                synthesized_image = transforms.Resize((h, w))(synthesized_image)  # z 1 2y 2x
                print('size', synthesized_image.shape)
                synthesized_image = synthesized_image.transpose(0, 2)  # 2y 1 z 2x
                synthesized_image = transforms.Resize((l, w))(synthesized_image)  # 2y 1 2z 2x
                synthesized_image = synthesized_image.transpose(0, 2)  # 2z 1 2y 2x
                # synthesized_image = torch.nn.functional.interpolate(synthesized_image, size=[l, h, w], mode='trilinear')
                synthesized_image = synthesized_image.to(
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                synthesized_image.requires_grad_(True)

        optim = torch.optim.Adam([synthesized_image], lr=lr)
        for i in range(num_steps):
            # if debug_dir and (i % 10 == 0 or i == num_steps - 1):
            # save_image(synthesized_image[0, :, :, :],
            #           os.path.join(debug_dir, 'optimization', f'lvl-{lvl}-iter-{i}.png'))

            optim.zero_grad()

            synthesized_image_Y1_ZX = synthesized_image.transpose(0, 2)
            target_pyramid_Y1_ZX = target_pyramid[lvl].transpose(0, 2)
            synthesized_image_X1_YZ = synthesized_image.transpose(0, 3)
            target_pyramid_X1_YZ = target_pyramid[lvl].transpose(0, 3)

            lossYX = criteria(synthesized_image, target_pyramid[lvl])
            lossZX = criteria(synthesized_image_Y1_ZX, target_pyramid_Y1_ZX)
            lossYZ = criteria(synthesized_image_X1_YZ, target_pyramid_X1_YZ)
            loss = lossYX + lossYZ + lossZX
            # loss = criteria(synthesized_image, target_pyramid[lvl])
            print(f'iteration:{i},loss:{loss}')
            loss.backward()
            optim.step()

            loss_history[lvl * num_steps + i] = loss_history[lvl * num_steps + i] + loss.item()

            if i != 0 and i % decay_steps == 0:
                for g in optim.param_groups:
                    g['lr'] *= 0.9
                    lr *= 0.9

        synthesized_image = torch.clip(synthesized_image, -1, 1)

        if debug_dir:
            save_image(target_pyramid[lvl][0, :, :, :], os.path.join(debug_dir, f'target-lvl-{lvl}.png'))
            save_image(synthesized_image[0, :, :, :], os.path.join(debug_dir, f'output-lvl-{lvl}.png'))
            for j in range(synthesized_image.shape[-4]):
                result = synthesized_image[j, :, :, :]
                save_image(result, f'3dtest/3DRec{lvl}/rec{j}.bmp')
            for j in range(target_pyramid[lvl].shape[-4]):
                result = target_pyramid[lvl][j, :, :, :]
                save_image(result, f'3dtest/Target{lvl}/Image{j}.bmp')

    time_end = time.time()
    time = time_end - time_begin
    print(time)

    nphase = torch.unique(target_pyramid[-1])
    if len(nphase) == 2:
        print('2phase')
        for j in range(synthesized_image.shape[-4]):
            result = synthesized_image[j, :, :, :]
            save_image(result, f'3dtest/3DRec_gray_results/rec{j}.bmp')
        synthesized_image[synthesized_image >= 0] = 1
        synthesized_image[synthesized_image < 0] = -1
        for j in range(synthesized_image.shape[-4]):
            result = synthesized_image[j, :, :, :]
            save_image(result, f'3dtest/3DRec_results/rec{j}.bmp')
    elif len(nphase) == 3:
        print('3phase')
        for j in range(synthesized_image.shape[-4]):
            result = synthesized_image[j, :, :, :]
            save_image(result, f'3dtest/3DRec_gray_results/rec{j}.bmp')
        synthesized_image[synthesized_image >= 0.5] = 1
        synthesized_image = torch.where(torch.gt(synthesized_image, -0.5) & torch.lt(synthesized_image, 0.5), 0,
                                        synthesized_image)
        synthesized_image[synthesized_image <= -0.5] = -1
        for j in range(synthesized_image.shape[-4]):
            result = synthesized_image[j, :, :, :]
            save_image(result, f'3dtest/3DRec_results/rec{j}.bmp')
    else:
        print('gray or color')
        for j in range(synthesized_image.shape[-4]):
            result = synthesized_image[j, :, :, :]
            save_image(result, f'3dtest/3DRec_results/rec{j}.bmp')

    txt_path = debug_dir + '/loss.txt'
    file_handle = open(txt_path, mode='w')
    file_handle.write("time:" + str(time) + '\n')
    file_handle.write("seed:" + str(random_seed) + '\n')
    for n_iter in range(len(target_pyramid) * num_steps):
        file_handle.write((str(n_iter) + ":" + str(loss_history[n_iter])) + '\n')
