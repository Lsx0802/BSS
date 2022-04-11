
# coding: utf-8
"""
通过实现Grad-CAM学习module中的forward_hook和backward_hook函数
"""
import cv2
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from SE_ResNet import  resnet34

IMAGE_SIZE=448

def img_transform(img_in, transform):
    """
    将img进行预处理，并转换成模型输入所需的形式—— B*C*H*W
    :param img_roi: np.array
    :return:
    """
    img = img_in.copy()
    img = Image.fromarray(np.uint8(img))
    img = transform(img)
    img = img.unsqueeze(0)  # C*H*W --> B*C*H*W
    return img


def img_preprocess(img_in):
    """
    读取图片，转为模型可读的形式
    :param img_in: ndarray, [H, W, C]
    :return: PIL.image
    """
    img = img_in.copy()
    img = cv2.resize(img, (IMAGE_SIZE,IMAGE_SIZE))
    img = img[:, :, ::-1]  # BGR --> RGB
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img_input = img_transform(img, transform)
    return img_input


def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())


def farward_hook(module, input, output):
    fmap_block.append(output)


def show_cam_on_image(img, mask, out_dir,patient):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)

    path_cam_img = os.path.join(out_dir, patient+"cam.jpg")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite(path_cam_img, np.uint8(255 * cam))



def comp_class_vec(ouput_vec, index=None):
    """
    计算类向量
    :param ouput_vec: tensor
    :param index: int，指定类别
    :return: tensor
    """
    if not index:
        index = np.argmax(ouput_vec.cpu().data.numpy())
    else:
        index = np.array(index)
    index = index[np.newaxis, np.newaxis]
    index = torch.from_numpy(index)
    one_hot = torch.zeros(1, 2).scatter_(1, index, 1)
    one_hot.requires_grad = True
    class_vec = torch.sum(one_hot.to(device) * output)  # one_hot = 11.8605

    return class_vec


def gen_cam(feature_map, grads):
    """
    依据梯度和特征图，生成cam
    :param feature_map: np.array， in [C, H, W]
    :param grads: np.array， in [C, H, W]
    :return: np.array, [H, W]
    """
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam shape (H, W)

    weights = np.mean(grads, axis=(1, 2))  #

    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (IMAGE_SIZE,IMAGE_SIZE))
    cam -= np.min(cam)
    cam /= np.max(cam)

    return cam


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fold='4'
    path=r'C:\Users\hello\PycharmProjects\tongue\data\dataYX_fusion\calssification\NYX\5'
    patient=os.listdir(path)
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    for i in patient:

        path_img = os.path.join(path,i)

        # path_net = os.path.join(BASE_DIR, "..", "..", "Data", "net_params_72p.pkl")
        path_net='weights/SE_face_448_fold'+fold+'1.pkl'

        # output_dir = os.path.join(BASE_DIR, "..", "..", "Result", "backward_hook_cam")
        output_dir='./save_img/'+'face_NYX_fold'+fold+'/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fmap_block = list()
        grad_block = list()

        # 图片读取；网络加载
        img = cv2.imread(path_img, 1)  # H*W*C
        img_input = img_preprocess(img)
        net = resnet34(num_classes=2).to(device)
        net.load_state_dict(torch.load(path_net))

        # 注册hook
        net.layer4.register_forward_hook(farward_hook)
        net.layer4.register_backward_hook(backward_hook)

        # forward
        output = net(img_input.to(device))

        # backward
        net.zero_grad()
        class_loss = comp_class_vec(output)
        class_loss.backward()

        # 生成cam
        grads_val = grad_block[0].cpu().data.numpy().squeeze()
        fmap = fmap_block[0].cpu().data.numpy().squeeze()
        cam = gen_cam(fmap, grads_val)

        # 保存cam图片
        img_show = np.float32(cv2.resize(img, (IMAGE_SIZE,IMAGE_SIZE))) / 255
        show_cam_on_image(img_show, cam, output_dir,i[0:-4])








