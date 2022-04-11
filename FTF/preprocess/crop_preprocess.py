
import os
from PIL import Image
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image
from skimage import transform

bace_path = r'C:\Users\Lsx\Desktop\YX_temp'

save_path_1 = r'C:\Users\Lsx\Desktop\YX\1'
if not os.path.exists(save_path_1):
    os.makedirs(save_path_1)

save_path_2 = r'C:\Users\Lsx\Desktop\YX\2'
if not os.path.exists(save_path_2):
    os.makedirs(save_path_2)

save_path_3 = r'C:\Users\Lsx\Desktop\YX\3'
if not os.path.exists(save_path_3):
    os.makedirs(save_path_3)

save_path_4 = r'C:\Users\Lsx\Desktop\YX\4'
if not os.path.exists(save_path_4):
    os.makedirs(save_path_4)

save_path_5 = r'C:\Users\Lsx\Desktop\YX\5'
if not os.path.exists(save_path_5):
    os.makedirs(save_path_5)

save_path_6 = r'C:\Users\Lsx\Desktop\YX\6'
if not os.path.exists(save_path_6):
    os.makedirs(save_path_6)

patient=os.listdir(bace_path)


for im in tqdm(patient):
    p=os.path.join(bace_path,im,'label.png')
    image = Image.open(os.path.join(bace_path, im, 'img.png'))

    # cv2.imshow('imshow', image)
    
    label=Image.open(p)
    label_2 = label.convert('L')
    # label_3=label_2.resize((h,w))
    label_4 = np.asarray(label_2)
    h, w = label_4.shape[0],label_4.shape[1]

    label_face = np.zeros((h,w))
    label_tongue = np.zeros((h, w))

    left1, top1, right1, bottom1 = 100000, 100000, 0, 0
    left2,top2,right2,bottom2=100000,100000,0,0

    #脸38，舌75
    for row in range(h):
        for con in range(w):
            if label_4[row,con]==38:
                label_face[row, con]=255
                if top1>row:
                    top1=row
                if bottom1<row:
                    bottom1=row
                if left1>con:
                    left1=con
                if right1<con:
                    right1=con
            elif label_4[row,con]==75:
                label_tongue[row, con] = 255
                if top2>row:
                    top2=row
                if bottom2<row:
                    bottom2=row
                if left2>con:
                    left2=con
                if right2<con:
                    right2=con

    label_face_2=Image.fromarray(np.uint8(label_face))
    label_tongue_2 = Image.fromarray(np.uint8(label_tongue))

    left1=max(0,left1-8)
    right1=min(w,right1+8)
    top1=max(0,top1-8)
    bottom1=min(h,bottom1+8)

    left2=max(0,left2-8)
    right2=min(w,right2+8)
    top2=max(0,top2-8)
    bottom2=min(h,bottom2+8)

    cropped_mask_face = label_face_2.crop((left1, top1, right1, bottom1))
    cropped_image_face=image.crop((left1, top1, right1, bottom1))

    cropped_mask_tongue = label_tongue_2.crop((left2, top2, right2, bottom2))
    cropped_image_tongue=image.crop((left2, top2, right2, bottom2))


    cropped_mask_face.save(os.path.join(save_path_1,im+'.jpg'))
    cropped_image_face.save(os.path.join(save_path_2,im+'.jpg'))

    cropped_mask_tongue.save(os.path.join(save_path_3,im+'.jpg'))
    cropped_image_tongue.save(os.path.join(save_path_4,im+'.jpg'))

    mask_face=cv2.imread(os.path.join(save_path_1, im+'.jpg'))
    image_face=cv2.imread(os.path.join(save_path_2, im+'.jpg'))

    mask_tongue = cv2.imread(os.path.join(save_path_3, im+'.jpg'))
    image_tongue=cv2.imread(os.path.join(save_path_4, im+'.jpg'))

    save_face = cv2.bitwise_and(image_face, mask_face)
    save_tongue=cv2.bitwise_and(image_tongue,mask_tongue)

    cv2.imwrite(os.path.join(save_path_5, im + '.jpg'),save_face)
    cv2.imwrite(os.path.join(save_path_6, im + '.jpg'),save_tongue)