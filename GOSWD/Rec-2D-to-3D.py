import sys
import os
import cv2
import torch
import time
from torchvision import transforms
import torch.nn.functional as F
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


class MultiMPSWDLoss2D_v1(torch.nn.Module):
    def __init__(self, temp_size=3, stride=1, num_proj=9, channels=1):
        super(MultiMPSWDLoss2D_v1, self).__init__()
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

        projx1 = outx1.reshape(self.num_proj, -1)
        projx2 = outx2.reshape(self.num_proj, -1)
        projx3 = outx3.reshape(self.num_proj, -1)
        projy1 = outy1.reshape(self.num_proj, -1)
        projy2 = outy2.reshape(self.num_proj, -1)
        projy3 = outy3.reshape(self.num_proj, -1)

        projx1, projy1 = expanding_operation(projx1, projy1)
        projx2, projy2 = expanding_operation(projx2, projy2)
        projx3, projy3 = expanding_operation(projx3, projy3)

        projx1, _ = torch.sort(projx1, dim=1)
        projy1, _ = torch.sort(projy1, dim=1)
        projx2, _ = torch.sort(projx2, dim=1)
        projy2, _ = torch.sort(projy2, dim=1)
        projx3, _ = torch.sort(projx3, dim=1)
        projy3, _ = torch.sort(projy3, dim=1)

        loss1 = torch.abs(projx1 - projy1).mean()
        loss2 = torch.abs(projx2 - projy2).mean()
        loss3 = torch.abs(projx3 - projy3).mean()
        # print(f'loss1:{loss1};loss2:{loss2};loss3:{loss3}')
        loss = loss1 + 0.1 * loss2 + 0.01 * loss3

        return loss


class MultiMPSWDLoss2D_v2(torch.nn.Module):
    def __init__(self, temp_size=3, stride=1, num_proj=9, channels=1):
        super(MultiMPSWDLoss2D_v2, self).__init__()
        self.temp_size = temp_size
        self.stride = stride
        self.num_proj = num_proj
        self.channels = channels

    def forward(self, x, y):
        b, c, h, w = x.shape
        rand = torch.randn(self.num_proj, self.channels, self.temp_size, self.temp_size).to(x.device)
        if self.num_proj > 1:
            rand = rand / torch.std(rand, dim=0, keepdim=True)  # noramlize
        projx1 = F.conv2d(x, rand)
        projy1 = F.conv2d(y, rand)

        mpy = torch.cat([projy1] * b, dim=0)

        projx = projx1.reshape(b, self.num_proj, -1)
        projy = mpy.reshape(b, self.num_proj, -1)

        projx, projy = expanding_operation_v2(projx, projy)

        projx, _ = torch.sort(projx, dim=2)
        projy, _ = torch.sort(projy, dim=2)

        loss = torch.abs(projx - projy).mean()

        return loss


class MultiMPSWDLoss2D_v3(torch.nn.Module):
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

        mpy1 = torch.cat([outy1] * b, dim=0)
        mpy2 = torch.cat([outy2] * b, dim=0)
        mpy3 = torch.cat([outy3] * b, dim=0)

        projx1 = outx1.reshape(b, self.num_proj, -1)
        projx2 = outx2.reshape(b, self.num_proj, -1)
        projx3 = outx3.reshape(b, self.num_proj, -1)
        projy1 = mpy1.reshape(b, self.num_proj, -1)
        projy2 = mpy2.reshape(b, self.num_proj, -1)
        projy3 = mpy3.reshape(b, self.num_proj, -1)

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


def cv2ptgray(img):
    img = img.astype(np.float64) / 255.
    img = img * 2 - 1
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)
    return img


def downscale(img, pyr_factor):
    assert 0 < pyr_factor < 1
    c, y, x = img.shape
    new_x = int(x * pyr_factor)
    new_y = int(y * pyr_factor)
    return transforms.Resize((new_y, new_x))(img)


def get_pyramid(img, min_size, pyr_factor):
    pyd = [img]
    while True:
        img = downscale(img, pyr_factor)
        if img.shape[-2] < min_size:
            break
        pyd = [img] + pyd
    return pyd


def get_pyramid_noise_size(target_size, lvl, pyr_factor):
    pyd = [target_size]
    for i in range(lvl - 1):
        target_size = int(target_size * pyr_factor)
        pyd = [target_size] + pyd
    return pyd


def get_cut_pyramid_size(target_size, lvl, pyr_factor):
    target_size = target_size + 8
    pyd = [target_size]
    for i in range(lvl - 1):
        target_size = int(target_size * pyr_factor + 8)
        pyd = [target_size] + pyd
    return pyd


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

    random_seed = 0
    # torch.manual_seed(random_seed)
    debug_dir = "2dto3dtest"
    target_img_YX_path = "training_images/2dto3d/TI1.bmp"

    criteria = MultiMPSWDLoss2D_v3(temp_size=3, stride=1, num_proj=9)
    criteria = criteria.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    top_lvl_size = 35
    pyr_factor = 0.8
    lr = 0.05
    num_steps = 200
    decay_steps = 2000

    target_img_YX = cv2.imread(target_img_YX_path, 0)

    target_img_YX = cv2ptgray(target_img_YX)  # c,h,w

    target_pyramid_YX = get_pyramid(target_img_YX, top_lvl_size, pyr_factor)

    target_pyramid_YX = [x.unsqueeze(0).to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu")) for x in
                         target_pyramid_YX]  # b c h w

    h, w = target_pyramid_YX[0].shape[-2:]
    l = h
    synthesized_image = torch.normal(0, 0.75, size=(l, 1, h, w))
    synthesized_image = synthesized_image.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    synthesized_image.requires_grad_(True)

    loss_history = np.zeros(len(target_pyramid_YX) * num_steps)
    time_begin = time.time()
    for lvl, lvl_target_img in enumerate(target_pyramid_YX):
        if lvl > 0:
            with torch.no_grad():
                h, w = target_pyramid_YX[lvl].shape[-2:]
                l = h
                synthesized_image = transforms.Resize((h, w))(synthesized_image)  # z 1 2y 2x
                print('size',synthesized_image.shape)
                synthesized_image = synthesized_image.transpose(0, 2)   # 2y 1 z 2x
                synthesized_image = transforms.Resize((h, w))(synthesized_image)  # 2y 1 2z 2x
                synthesized_image = synthesized_image.transpose(0, 2)  # 2z 1 2y 2x
                # synthesized_image = torch.nn.functional.interpolate(synthesized_image, size=[l, h, w],
                #                                                     mode='trilinear')
                synthesized_image = synthesized_image.to(
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                synthesized_image.requires_grad_(True)

        optim = torch.optim.Adam([synthesized_image], lr=lr)
        for i in range(num_steps):
            optim.zero_grad()

            synthesized_image_Y1_ZX = synthesized_image.transpose(0, 2)
            synthesized_image_X1_YZ = synthesized_image.transpose(0, 3)

            lossYX = criteria(synthesized_image, target_pyramid_YX[lvl])
            lossZX = criteria(synthesized_image_Y1_ZX, target_pyramid_YX[lvl])
            lossYZ = criteria(synthesized_image_X1_YZ, target_pyramid_YX[lvl])
            loss = lossYX + lossYZ + lossZX
            # loss = lossYX
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
            save_image(target_pyramid_YX[lvl][:, :, :, :], os.path.join(debug_dir, f'target_YX-lvl-{lvl}.png'))
            save_image(synthesized_image[0, :, :, :], os.path.join(debug_dir, f'output-lvl-{lvl}.png'))
            for j in range(synthesized_image.shape[0]):
                result = synthesized_image[j, :, :, :]
                save_image(result, f'2dto3dtest/3DRec{lvl}/rec{j}.bmp')

    time_end = time.time()
    time = time_end - time_begin
    print(time)
    nphase = torch.unique(target_pyramid_YX[-1])
    th, tw = synthesized_image.shape[-2:]
    if len(nphase) == 2:
        print('2phase')
        for j in range(synthesized_image.shape[0]):
            result = synthesized_image[j, :, :, :]
            save_image(result, f'2dto3dtest/3DRec_gray_results/rec{j}.bmp')
        synthesized_image[synthesized_image >= 0] = 1
        synthesized_image[synthesized_image < 0] = -1
        for j in range(synthesized_image.shape[0]):
            result = synthesized_image[j, :, :, :]
            save_image(result, f'2dto3dtest/3DRec_results/rec{j}.bmp')
    elif len(nphase) == 3:
        print('3phase')
        for j in range(synthesized_image.shape[0]):
            result = synthesized_image[j, :, :, :]
            save_image(result, f'2dto3dtest/3DRec_gray_results/rec{j}.bmp')
        synthesized_image[synthesized_image >= 0.5] = 1
        synthesized_image = torch.where(torch.gt(synthesized_image, -0.5) & torch.lt(synthesized_image, 0.5), 0,
                                        synthesized_image)
        synthesized_image[synthesized_image <= -0.5] = -1
        for j in range(synthesized_image.shape[0]):
            result = synthesized_image[j, :, :, :]
            save_image(result, f'2dto3dtest/3DRec_results/rec{j}.bmp')
    else:
        print('gray or color')
        for j in range(synthesized_image.shape[0]):
            result = synthesized_image[j, :, :, :]
            save_image(result, f'2dto3dtest/3DRec_results/rec{j}.bmp')
    txt_path = debug_dir + '/loss.txt'
    file_handle = open(txt_path, mode='w')
    file_handle.write("time:" + str(time) + '\n')
    file_handle.write("seed:" + str(random_seed) + '\n')
    for n_iter in range(len(target_pyramid_YX) * num_steps):
        file_handle.write((str(n_iter) + ":" + str(loss_history[n_iter])) + '\n')
