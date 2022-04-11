# coding=utf-8
# coding=utf-8
import os
path0=r'C:\Users\hello\PycharmProjects\tongue\data\dataYX_fusion\calssification\NYX\6' #150 26
patient0=os.listdir(path0)
path1=r'C:\Users\hello\PycharmProjects\tongue\data\dataYX_fusion\calssification\YX\6' #150 27
patient1=os.listdir(path1)

NYX_=[0,37,37+38,37+38+37,150]
YX_=[0,38,37+38,37+38+38,150]

with open('trainval.txt','w') as f:
    for i in patient0:
        f.write(i+' 0\n')
    for i in patient1:
        f.write(i+' 1\n')

with open("val1.txt","w") as fval1:
    for i in range(NYX_[0],NYX_[1]):
        fval1.write(patient0[i]+' 0\n')
    for i in range(YX_[0],YX_[1]):
        fval1.write(patient1[i]+' 1\n')

with open("trainval.txt", "r") as f:
    with open("val1.txt", "r") as fval1:
        with open("train1.txt", "w") as ftrain1:
            lines = f.readlines()
            lines1 = fval1.readlines()
            for line in lines:
                if line not in lines1:
                    ftrain1.write(line)

with open("val2.txt","w") as fval1:
    for i in range(NYX_[1],NYX_[2]):
        fval1.write(patient0[i]+' 0\n')
    for i in range(YX_[1],YX_[2]):
        fval1.write(patient1[i]+' 1\n')

with open("trainval.txt", "r") as f:
    with open("val2.txt", "r") as fval1:
        with open("train2.txt", "w") as ftrain1:
            lines = f.readlines()
            lines1 = fval1.readlines()
            for line in lines:
                if line not in lines1:
                    ftrain1.write(line)


with open("val3.txt","w") as fval1:
    for i in range(NYX_[2],NYX_[3]):
        fval1.write(patient0[i]+' 0\n')
    for i in range(YX_[2],YX_[3]):
        fval1.write(patient1[i]+' 1\n')

with open("trainval.txt", "r") as f:
    with open("val3.txt", "r") as fval1:
        with open("train3.txt", "w") as ftrain1:
            lines = f.readlines()
            lines1 = fval1.readlines()
            for line in lines:
                if line not in lines1:
                    ftrain1.write(line)

with open("val4.txt","w") as fval1:
    for i in range(NYX_[3],NYX_[4]):
        fval1.write(patient0[i]+' 0\n')
    for i in range(YX_[3],YX_[4]):
        fval1.write(patient1[i]+' 1\n')

with open("trainval.txt", "r") as f:
    with open("val4.txt", "r") as fval1:
        with open("train4.txt", "w") as ftrain1:
            lines = f.readlines()
            lines1 = fval1.readlines()
            for line in lines:
                if line not in lines1:
                    ftrain1.write(line)
# with open("trainval.txt", "w") as ftraintest:
#     with open("test.txt", "w") as ftest:
#         for i in range(20):
#             ftest.write(patient1[i][0:12]+' 1\n')
#         for i in range(88):
#             ftest.write(patient0[i][0:12]+' 0\n')
#         for i in range(20,len(patient1)):
#             ftraintest.write(patient1[i][0:12] + ' 1\n')
#         for i in range(88,len(patient0)):
#             ftraintest.write(patient0[i][0:12] + ' 0\n')

#
# count0 = 0
# count1 = 0
# # with open("trainval.txt", "r") as f:
#     with open("val1.txt", "w") as fval1:
#         with open("val2.txt", "w") as fval2:
#             with open("val3.txt", "w") as fval3:
#                 with open("val4.txt", "w") as fval4:
#                     lines=f.readlines()
#                     lines2 = fval2.readlines()
#                     lines3 = fval3.readlines()
#                     lines4 = fval4.readlines()
#                     for line in lines:
#                         if line in lines4 or line in lines3 or line in lines2:
#                             continue
#                         if line[-2] == '1' and count1<44:
#                             fval1.write(line)
#                             count1=count1+1
#                         if line[-2] == '0'and count0 < 206:
#                             fval1.write(line)
#                             count0=count0+1

#
# with open("trainval.txt", "r") as f:
#     with open("train1.txt", "w") as ftrain1:
#         with open("train2.txt", "w") as ftrain2:
#             with open("train3.txt", "w") as ftrain3:
#                 with open("train4.txt", "w") as ftrain4:
#                     with open("val1.txt", "r") as fval1:
#                         with open("val2.txt", "r") as fval2:
#                             with open("val3.txt", "r") as fval3:
#                                 with open("val4.txt", "r") as fval4:
#                                     lines = f.readlines()
#                                     lines1 = fval1.readlines()
#                                     lines2 = fval2.readlines()
#                                     lines3 = fval3.readlines()
#                                     lines4 = fval4.readlines()
#                                     for line in lines:
#                                         if line in lines1 :
#                                             ftrain2.write(line)
#                                             ftrain3.write(line)
#                                             ftrain4.write(line)
#                                         elif line in lines2 :
#                                             ftrain1.write(line)
#                                             ftrain3.write(line)
#                                             ftrain4.write(line)
#                                         elif line in lines3 :
#                                             ftrain1.write(line)
#                                             ftrain2.write(line)
#                                             ftrain4.write(line)
#                                         else:
#                                             ftrain1.write(line)
#                                             ftrain2.write(line)
#                                             ftrain3.write(line)
