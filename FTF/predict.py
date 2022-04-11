import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2

from SE_ResNet import resnet34


def main(img, cam, label):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(448),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image

    img_path = img
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)

    img = Image.open(img_path)
    img_2=img.resize((448,448))
    cam_img = Image.open(cam)

    plt.subplot(121)
    plt.imshow(img_2)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(122)
    plt.imshow(cam_img)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    img=Image.open(img_path)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = resnet34(num_classes=2).to(device)

    # load model weights
    weights_path = "weights/SE_face_448_fold41.pkl"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "original:{} predict: {}  prob: {:.4}".format(label, class_indict[str(predict_cla)],
                                                              predict[predict_cla].numpy())
    plt.title(print_res,size=14,loc = 'right')
    print(print_res)
    save_path = r'C:\Users\hello\PycharmProjects\tongue\FTF\save_img\face_'+label
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    final_path = os.path.join(save_path, i)
    plt.savefig(final_path)


if __name__ == '__main__':
    fold = '4'
    path = r'C:\Users\hello\PycharmProjects\tongue\data\dataYX_fusion\calssification'
    path0 = os.path.join(path ,'NYX/5')
    path1 =os.path.join(path ,'YX/5')
    path00 = os.listdir(path0)
    path11 = os.listdir(path1)
    cam_path0 = r'C:\Users\hello\PycharmProjects\tongue\FTF\save_img\face_NYX_fold' + fold
    cam_path1=r'C:\Users\hello\PycharmProjects\tongue\FTF\save_img\face_YX_fold' + fold


    for i in path00:
        cam = os.path.join(cam_path0 , i[0:-4] + 'cam.jpg')
        path000 = os.path.join(path0 , i)
        label = 'non_BSS'
        main(path000, cam, label)

    for i in path11:
        cam = os.path.join(cam_path1 , i[0:-4] + 'cam.jpg')
        path111 = os.path.join(path1 , i)
        label = 'BSS'
        main(path111, cam, label)
