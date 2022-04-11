import numpy as np
import matplotlib.pyplot as plt

def smooth(scalar, weight=0.5):
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

fig = plt.figure(dpi=300,figsize=(18,12))
ax1 = fig.add_subplot(2, 3, 1)
ax2 = fig.add_subplot(2, 3, 2)
ax3 = fig.add_subplot(2, 3, 3)
ax4 = fig.add_subplot(2, 3, 4)
ax5 = fig.add_subplot(2, 3, 5)
ax6 = fig.add_subplot(2, 3, 6)

######################################################################
data1=np.load('runs/result_tongue_fold.npz',allow_pickle=True)

val_accuracy1=data1['val_accuracy']
val_precision1=data1['val_precision']
val_recall1=data1['val_recall']
val_f11=data1['val_f1']
val_AUC1=data1['val_AUC']
val_fpr1=data1['val_fpr']
val_tpr1=data1['val_tpr']
val_loss1=data1['val_loss']

val_accuracy1=smooth(val_accuracy1)
val_precision1=smooth(val_precision1)
val_recall1=smooth(val_recall1)
val_f11=smooth(val_f11)
val_loss1=smooth(val_loss1)
val1_x = np.arange(0,len(val_accuracy1))

val_fpr1_=val_fpr1[np.argmax(val_accuracy1)]
val_tpr1_=val_tpr1[np.argmax(val_accuracy1)]
val_AUC1_=a1 = round(val_AUC1[np.argmax(val_accuracy1)], 2)

# ########################################################
labels=['224',]

ax1.plot(val1_x, val_accuracy1, 'r*-', label=labels[0], linewidth=2.2)
ax2.plot(val1_x, val_loss1, 'r*-', label=labels[0], linewidth=2.2)
ax3.plot(val1_x, val_precision1, 'r*-', label=labels[0], linewidth=2.2)
ax4.plot(val1_x, val_recall1, 'r*-', label=labels[0], linewidth=2.2)
ax5.plot(val1_x, val_f11, 'r*-', label=labels[0],linewidth=2.2)
ax6.plot(val_fpr1_, val_tpr1_, 'r*-', label=labels[0]+ ', AUC = ' + str(val_AUC1_), linewidth=2.2)
ax6.plot([0, 1], [0, 1], color='navy', linewidth=1.2, linestyle='--')
#

ax1.legend(loc='lower right',fontsize=18)
ax1.set_title('Accuracy',fontsize=18)
ax1.set_xlabel('Epoch',fontsize=18)

ax2.legend(loc='upper right',fontsize=18)
ax2.set_title('Loss',fontsize=18)
ax2.set_xlabel('Epoch',fontsize=18)

ax3.legend(loc='lower right',fontsize=18)
ax3.set_title('Precision',fontsize=18)
ax3.set_xlabel('Epoch',fontsize=18)

ax4.legend(loc='lower right',fontsize=18)
ax4.set_title('Recall',fontsize=18)
ax4.set_xlabel('Epoch',fontsize=18)

ax5.legend(loc='lower right',fontsize=18)
ax5.set_title('F1_score',fontsize=18)
ax5.set_xlabel('Epoch',fontsize=18)

ax6.legend(loc='lower right',fontsize=18)
ax6.set_title('ROC',fontsize=18)
ax6.set_xlabel('False Positive Rate',fontsize=18)
ax6.set_ylabel('True Positive Rate',fontsize=18)

plt.tick_params(labelsize=18)
plt.show()