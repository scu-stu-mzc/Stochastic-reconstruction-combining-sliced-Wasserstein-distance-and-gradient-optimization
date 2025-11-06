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


def extract_patterns(x, temp_size, stride):
    b, c, h, w = x.shape
    unfold = torch.nn.Unfold(kernel_size=temp_size, stride=stride)
    x_patches = unfold(x).transpose(1, 2).reshape(b, -1, 1, temp_size, temp_size)
    # print(x_patches.shape)
    return x_patches.view(b, -1, 1, temp_size * temp_size)


class SingleMPSWDLoss2D(torch.nn.Module):
    """single 2D template implementation"""
    def __init__(self, temp_size=7, stride=1, num_proj=256, channels=1):
        super(SingleMPSWDLoss2D, self).__init__()
        self.temp_size = temp_size
        self.stride = stride
        self.num_proj = num_proj
        self.channels = channels

    def forward(self, x, y):
        b, c, h, w = x.shape
        rand = torch.randn(self.num_proj, self.channels, self.temp_size, self.temp_size).to(x.device)
        if self.num_proj > 1:
            rand = rand / torch.std(rand, dim=0, keepdim=True)  # noramlize
        projx = F.conv2d(x, rand).reshape(self.num_proj, -1)
        projy = F.conv2d(y, rand).reshape(self.num_proj, -1)

        projx, projy = expanding_operation(projx, projy)

        projx, _ = torch.sort(projx, dim=1)
        projy, _ = torch.sort(projy, dim=1)

        loss = torch.abs(projx - projy).mean()

        return loss


class SingleMPSWDLoss2D_with_bord(torch.nn.Module):
    """single 2D template implementation with periodic boundary condition"""
    def __init__(self, temp_size=7, stride=1, num_proj=256, channels=1):
        super(SingleMPSWDLoss2D_with_bord, self).__init__()
        self.temp_size = temp_size
        self.stride = stride
        self.num_proj = num_proj
        self.channels = channels

    def forward(self, x, y):
        b, c, h, w = x.shape
        rand = torch.randn(self.num_proj, self.channels, self.temp_size, self.temp_size).to(x.device)
        if self.num_proj > 1:
            rand = rand / torch.std(rand, dim=0, keepdim=True)  # noramlize
        x_pad = F.pad(x, (1, 3, 2, 3), mode='circular')
        projx = F.conv2d(x_pad, rand).reshape(self.num_proj, -1)
        projy = F.conv2d(y, rand).reshape(self.num_proj, -1)

        projx, projy = expanding_operation(projx, projy)

        projx, _ = torch.sort(projx, dim=1)
        projy, _ = torch.sort(projy, dim=1)

        loss = torch.abs(projx - projy).mean()

        return loss


class MultiMPSWDLoss2D_v1(torch.nn.Module):
    """multiple 2D template implementation"""
    def __init__(self, temp_size=3, stride=1, num_proj=9, channels=1):
        super(MultiMPSWDLoss2D_v1, self).__init__()
        self.temp_size = temp_size
        self.stride = stride
        self.num_proj = num_proj
        self.channels = channels

    def forward(self, x, y):
        b, c, h, w = x.shape
        assert b == 1, "Batches not implemented"
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


class MSELoss2D(torch.nn.Module):
    """MSE implementation"""
    def __init__(self, temp_size=3, stride=1, channels=1):
        super(MSELoss2D, self).__init__()
        self.temp_size = temp_size
        self.stride = stride
        self.channels = channels

    def forward(self, x, y):
        b, c, h, w = x.shape
        # b len 1 t*t
        y_patches = extract_patterns(y, self.temp_size, self.stride)
        x_patches = extract_patterns(x, self.temp_size, self.stride)

        # b len 1 t*t
        loss = torch.nn.functional.mse_loss(y_patches, x_patches)
        print(f'loss:{loss};')
        return loss


class MSELoss2D_rand(torch.nn.Module):
    """MSE implementation out of pattern order"""
    def __init__(self, temp_size=3, stride=1, channels=1):
        super(MSELoss2D_rand, self).__init__()
        self.temp_size = temp_size
        self.stride = stride
        self.channels = channels

    def forward(self, x, y):
        b, c, h, w = x.shape
        # b len 1 t*t
        y_patches = extract_patterns(y, self.temp_size, self.stride)
        torch.manual_seed(2)
        y_patches_rand = y_patches[:, torch.randperm(y_patches.shape[1]), :, :]
        x_patches = extract_patterns(x, self.temp_size, self.stride)

        # b len 1 t*t
        loss = torch.nn.functional.mse_loss(y_patches_rand, x_patches)
        print(f'loss:{loss};')
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


def get_pyramid_rec_size(rec_size, rec_lvl, pyr_factor):
    pyd_size = [rec_size]
    tmp_size = rec_size
    tmp_lvl = rec_lvl
    while True:
        if tmp_lvl == 1:
            break
        tmp_size = int(tmp_size * pyr_factor)
        pyd_size = [tmp_size] + pyd_size
        tmp_lvl = tmp_lvl - 1
    return pyd_size


if __name__ == '__main__':

    random_seed = 1
    # torch.manual_seed(random_seed)

    debug_dir = "2dtest"
    target_img_path = "training_images/2dto2d/TI1.bmp"

    criteria = SingleMPSWDLoss2D(temp_size=5, stride=1, num_proj=25)
    # criteria = SingleMPSWDLoss2D_with_bord(temp_size=5, stride=1, num_proj=256)
    # criteria = MultiMPSWDLoss2D_v1(temp_size=3, stride=1, num_proj=49)
    # criteria = MSELoss2D_rand(temp_size=5, stride=1)
    criteria = criteria.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    top_lvl_size = 35
    pyr_factor = 0.8
    lr = 0.05
    num_steps = 200
    decay_steps = 2000
    rec_size = 128

    target_img = cv2.imread(target_img_path, 0)
    target_img = cv2ptgray(target_img)  # c,h,w
    target_pyramid = get_pyramid(target_img, top_lvl_size, pyr_factor)
    target_pyramid = [x.unsqueeze(0).to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu")) for x in
                      target_pyramid]

    target_img = target_pyramid[-1]
    rec_lvl = len(target_pyramid)
    rec_size_pyd = get_pyramid_rec_size(rec_size, rec_lvl, pyr_factor)
    h = rec_size_pyd[0]
    w = h
    # h, w = target_pyramid[0].shape[-2:]
    synthesized_image = torch.normal(0, 0.75, size=(1, 1, h, w))
    synthesized_image = synthesized_image.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    synthesized_image.requires_grad_(True)

    loss_history = np.zeros(len(target_pyramid) * num_steps)
    time_begin = time.time()
    for lvl, lvl_target_img in enumerate(target_pyramid):
        if lvl > 0:
            with torch.no_grad():
                h = rec_size_pyd[lvl]
                w = h
                # h, w = target_pyramid[lvl].shape[-2:]
                synthesized_image = transforms.Resize((h, w))(synthesized_image)
                synthesized_image = synthesized_image.to(
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                synthesized_image.requires_grad_(True)

        optim = torch.optim.Adam([synthesized_image], lr=lr)
        for i in range(num_steps):
            """
            if debug_dir and (i % 20 == 0 or i == num_steps - 1):
                save_image(synthesized_image, os.path.join(debug_dir, 'optimization', f'lvl-{lvl}-{i}.png'))
            """
            optim.zero_grad()
            loss = criteria(synthesized_image, target_pyramid[lvl])
            print(f'iteration:{i},loss:{loss}')
            loss.backward()
            optim.step()

            loss_history[lvl * num_steps + i] = loss_history[lvl * num_steps + i] + loss.item()

            if i != 0 and i % decay_steps == 0:
                for g in optim.param_groups:
                    g['lr'] *= 0.9
                    lr *= 0.9

        synthesized_image = torch.clip(synthesized_image, -1, 1)

        """
        if debug_dir:
            save_image(lvl_target_img, os.path.join(debug_dir, f'target-lvl-{lvl}.png'))
            save_image(synthesized_image, os.path.join(debug_dir, f'output-lvl-{lvl}.png'))
        """

    time_end = time.time()
    time = time_end - time_begin
    print(time)

    nphase = torch.unique(target_pyramid[-1])
    if len(nphase) == 2:
        print('2phase')
        save_image(synthesized_image, os.path.join(debug_dir, f'gray_result.png'))
        synthesized_image[synthesized_image >= 0] = 1
        synthesized_image[synthesized_image < 0] = -1
        save_image(synthesized_image, os.path.join(debug_dir, f'output_result.png'))
    elif len(nphase) == 3:
        print('3phase')
        save_image(synthesized_image, os.path.join(debug_dir, f'gray_result.png'))
        synthesized_image[synthesized_image >= 0.5] = 1
        synthesized_image = torch.where(torch.gt(synthesized_image, -0.5) & torch.lt(synthesized_image, 0.5), 0,
                                        synthesized_image)
        synthesized_image[synthesized_image <= -0.5] = -1
        save_image(synthesized_image, os.path.join(debug_dir, f'output_result.png'))
    else:
        print('gray or color')
        save_image(synthesized_image, os.path.join(debug_dir, f'output_result.png'))

    txt_path = debug_dir + '/loss.txt'
    file_handle = open(txt_path, mode='w')
    file_handle.write("time:" + str(time) + '\n')
    file_handle.write("seed:" + str(random_seed) + '\n')
    for n_iter in range(len(target_pyramid) * num_steps):
        file_handle.write((str(n_iter) + ":" + str(loss_history[n_iter])) + '\n')
