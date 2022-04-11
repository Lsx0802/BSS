import os
from shutil import copytree

path=r'C:\Users\Lsx\Desktop\NYX'
save=r'C:\Users\Lsx\Desktop\NYX_temp'
if not os.path.exists(save):
    os.makedirs(save)

patient=os.listdir(path)

for i in patient:
    json_path=os.path.join(path,i)
    copytree(json_path,os.path.join(save,i[0:12]))

path=r'C:\Users\Lsx\Desktop\YX'
save=r'C:\Users\Lsx\Desktop\YX_temp'
if not os.path.exists(save):
    os.makedirs(save)

patient=os.listdir(path)

for i in patient:
    json_path=os.path.join(path,i)
    copytree(json_path,os.path.join(save,i[0:12]))

