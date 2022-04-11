# coding=utf-8

from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np

class MyDataset(Dataset):
    def __init__(self, data_dir_face,data_dir_tongue, txt,transform=None):
        super(MyDataset, self).__init__()

        data = []
        fh = open(txt, 'r')
        # 按照传入的路径和txt文本参数，以只读的方式打开这个文本
        for line in fh:  # 迭代该列表#按行循环txt文本中的内
            line = line.strip('\n')
            line = line.rstrip('\n')
            # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()
            # 用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
            data.append((words[0], int(words[1])))

        self.data_dir_face = data_dir_face
        self.data_dir_tongue = data_dir_tongue
        self.transform = transform
        self.data=data

    def __len__(self):
        return len(self.data)

    # 得到数据内容和标签
    def __getitem__(self, index):
        fn, label= self.data[index]
        face_image_path=os.path.join(self.data_dir_face,fn)
        tongue_image_path = os.path.join(self.data_dir_tongue, fn)
        image_face = Image.open(face_image_path).convert("RGB")
        image_tongue = Image.open(tongue_image_path).convert("RGB")

        if self.transform is not None:
            image_face =self.transform(image_face)
            image_tongue = self.transform(image_tongue)

        return image_face,image_tongue, label

def test():
    train_dataset = MyDataset(data_dir='data/image',mask_dir='data/mask',txt='train.txt', transform=transforms.ToTensor())
    print(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        num_workers=0)
    print(train_loader)

if __name__ == '__main__':
    test()

