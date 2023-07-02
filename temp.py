import os
import pathlib
import shutil 

path = pathlib.Path(__file__).parent.resolve()
print(path)
path1 = os.path.join(path, 'www')
for subfolder in os.listdir(path1):
    cur_dir = os.path.join(path1, subfolder)
    if os.path.isfile(cur_dir):
        os.remove(cur_dir)
    for file in os.listdir(cur_dir):
        if os.path.isdir(os.path.join(cur_dir, file)):
            shutil.rmtree(os.path.join(cur_dir, file))