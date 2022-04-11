# coding=utf-8
import os
import time

from DSN_ResNet import resnet34
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F
from itertools import cycle
import warnings
from MyDataset_fusion import MyDataset
from utils import create_lr_scheduler

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_confusion_matrix(trues, preds):
    labels = [0, 1]
    conf_matrix = confusion_matrix(trues, preds, labels)
    return conf_matrix


def roc_auc(trues, preds):
    fpr, tpr, thresholds = roc_curve(trues, preds)
    auc = roc_auc_score(trues, preds)
    return fpr, tpr, auc


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_confusion_matrix(conf_matrix):
    plt.imshow(conf_matrix, cmap=plt.cm.Greens)
    indices = range(conf_matrix.shape[0])
    labels = [0, 1]
    plt.xticks(indices, labels)
    plt.yticks(indices, labels)
    plt.colorbar()
    plt.xlabel('y_pred')
    plt.ylabel('y_true')
    # 显示数据
    for first_index in range(conf_matrix.shape[0]):
        for second_index in range(conf_matrix.shape[1]):
            plt.text(first_index, second_index, conf_matrix[first_index, second_index])
    plt.savefig('heatmap_confusion_matrix.jpg')
    plt.show()


def main():
    EPOCH = 50
    best_accuracy = 0.0
    trigger = 0
    early_stop_step = 10
    BATCH_SIZE = 32
    LR = 1e-4
    WEIGHT_DECAY = 5e-4
    IMAGE_HEIGHT, IMAGE_WIDTH = 448,448
    model_use_pretain_weight = True
    lowest_loss = 100000

    model = resnet34(num_classes=2)

    if model_use_pretain_weight:
        model_weight_path = "pre_weight/resnet34-333f7ec4.pth"
        assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)

        pre_weights = torch.load(model_weight_path, map_location=device)
        del_key = []
        for key, _ in pre_weights.items():
            if "fc" in key:
                del_key.append(key)

        for key in del_key:
            del pre_weights[key]

        missing_keys, unexpected_keys = model.load_state_dict(pre_weights, strict=False)
        print("[missing_keys]:", *missing_keys, sep="\n")
        print("[unexpected_keys]:", *unexpected_keys, sep="\n")

    print(model)
    print('params:' + str(count_params(model)))
    model.to(device)

    tongue_image_path = r'C:\Users\hello\PycharmProjects\tongue\data\dataYX_fusion\non_classification\6'  # tongue data set path
    face_image_path = r'C:\Users\hello\PycharmProjects\tongue\data\dataYX_fusion\non_classification\5'
    train_txt_path = 'txt/train4.txt'
    val_txt_path = 'txt/val4.txt'
    name = 'DSN_448_fold41'

    data_transform = {
        "train": transforms.Compose([transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.RandomRotation(degrees=(0,15)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ]),

        "val": transforms.Compose([transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ])}

    train_dataset = MyDataset(data_dir_face=face_image_path, data_dir_tongue=tongue_image_path,txt=train_txt_path,
                                     transform=data_transform['train'])
    val_dataset = MyDataset(data_dir_face=face_image_path,data_dir_tongue=tongue_image_path, txt=val_txt_path,
                                   transform=data_transform['val'])

    nw = min([os.cpu_count(), BATCH_SIZE if BATCH_SIZE > 1 else 0, 8])  # number of workers

    data_loader_train = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        num_workers=nw)

    data_loader_val = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=nw)

    train_step = len(data_loader_train)
    val_step = len(data_loader_val)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = create_lr_scheduler(optimizer, len(data_loader_train), EPOCH, warmup=True, warmup_epochs=1)

    print("Start training")

    val_loss = []
    val_precision = []
    val_recall = []
    val_f1 = []
    val_accuracy = []
    val_fpr = []
    val_tpr = []
    val_AUC = []

    before = time.time()
    for epoch in range(EPOCH):
        train_tot_loss = 0.0
        val_tot_loss = 0.0

        train_preds = []
        train_trues = []

        model.train()
        for i, (train_data_batch_face, train_data_batch_tongue,train_label_batch) in tqdm(enumerate(data_loader_train),
                                                             total=len(data_loader_train)):
            train_data_batch_face = train_data_batch_face.float().to(device)  # 将double数据转换为float
            train_data_batch_tongue = train_data_batch_tongue.float().to(device)  # 将double数据转换为float
            train_label_batch= train_label_batch.to(device)

            outputs1,outputs2,outputs3 = model(train_data_batch_face,train_data_batch_tongue)
            # _, preds = torch.max(outputs.data, 1)
            loss1 = criterion(outputs1, train_label_batch)
            loss2 = criterion(outputs2, train_label_batch)
            loss3 = criterion(outputs3, train_label_batch)
            loss=loss1+loss2+loss3
            # print(loss)
            # 反向传播优化网络参数
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            # 累加每个step的损失
            train_tot_loss += loss1.item()
            train_outputs = outputs1.argmax(dim=1)

            train_preds.extend(train_outputs.detach().cpu().numpy())
            train_trues.extend(train_label_batch.detach().cpu().numpy())


        train_tot_loss = train_tot_loss / train_step
        accuracy = accuracy_score(train_trues, train_preds)
        precision = precision_score(train_trues, train_preds)
        recall = recall_score(train_trues, train_preds)
        f1 = f1_score(train_trues, train_preds)

        print("[train] Epoch:{} accuracy:{:.2f} precision:{:.2f} recall:{:.2f} f1:{:.2f} loss:{:.4f}".format(
            epoch, accuracy * 100, precision * 100, recall * 100, f1 * 100, train_tot_loss))

        val_preds = []
        val_trues = []
        val_softmax = []
        model.eval()
        with torch.no_grad():
            for i, (val_data_batch_face, val_data_batch_tongue,val_label_batch) in tqdm(enumerate(data_loader_val), total=len(data_loader_val)):
                val_data_batch_face = val_data_batch_face.float().to(device)  # 将double数据转换为float
                val_data_batch_tongue = val_data_batch_tongue.float().to(device)  # 将double数据转换为float
                val_label_batch = val_label_batch.to(device)
                val_outputs1,val_outputs2,val_outputs3 = model(val_data_batch_face,val_data_batch_tongue)
                val_loss_1 = criterion(val_outputs1, val_label_batch)
                val_loss_2 = criterion(val_outputs2, val_label_batch)
                val_loss_3 = criterion(val_outputs3, val_label_batch)
                val_loss_ = val_loss_1+val_loss_2+val_loss_3

                softmax_output = torch.softmax(val_outputs1, dim=1)
                val_outputs = val_outputs1.argmax(dim=1)

                val_preds.extend(val_outputs.detach().cpu().numpy())
                val_trues.extend(val_label_batch.detach().cpu().numpy())
                val_softmax.extend(softmax_output[:, 1].detach().cpu().numpy())

                val_tot_loss += val_loss_1.item()

            val_tot_loss = val_tot_loss / val_step
            accuracy = accuracy_score(val_trues, val_preds)
            precision = precision_score(val_trues, val_preds)
            recall = recall_score(val_trues, val_preds)
            f1 = f1_score(val_trues, val_preds)
            fpr, tpr, AUC = roc_auc(val_trues, val_softmax)

            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), "weights/" + name + ".pkl")
                print("save best weighted ")
                trigger = 0

            # if lowest_loss>=val_tot_loss:
            #     lowest_loss=val_tot_loss
            #     trigger = 0

            trigger += 1
            if trigger >= early_stop_step:
                print("=> early stopping")
                break

            if epoch == EPOCH - 1:
                print(classification_report(val_trues, val_preds))

            print("[valadation] Epoch:{} accuracy:{:.2f} precision:{:.2f} recall:{:.2f} f1:{:.2f} AUC:{:.2f} loss:{:.4f} ".format(
                epoch, accuracy * 100, precision * 100, recall * 100, f1 * 100,AUC * 100, val_tot_loss))

            val_accuracy.append(accuracy), val_precision.append(precision), val_recall.append(recall), val_f1.append(
                f1), val_AUC.append(AUC), val_fpr.append(fpr), val_tpr.append(tpr), val_loss.append(val_tot_loss)

    result_path = 'runs/result_' + name
    np.savez(result_path, val_accuracy=val_accuracy, val_precision=val_precision, val_recall=val_recall, val_f1=val_f1,
             val_AUC=val_AUC, val_fpr=val_fpr, val_tpr=val_tpr, val_loss=val_loss)
    # print('val_accuracy = '+str(val_accuracy))
    # print('\n')
    # print('val_precision = '+str(val_precision))
    # print('\n')
    # print('val_recall = '+str(val_recall))
    # print('\n')
    # print('val_f1 = '+str(val_f1))
    # print('\n')
    # print('val_loss = '+str(val_loss))
    # print('\n')

    after = time.time()
    total_time = after - before
    print('total_time: ' + str(total_time))
    print('best_accuracy: ' + str(best_accuracy))
    # print('trigger: ' + str(trigger))


if __name__ == '__main__':
    main()
