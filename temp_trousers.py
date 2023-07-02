import os
import pathlib
import shutil 

path = pathlib.Path(__file__).parent.resolve()
for file in os.listdir(os.path.join(path, 'data/image')):
    if '_1' in file:
        shutil.move(os.path.join(path, 'data/image', file), os.path.join(path, 'data/cloth', file.replace('_1', '_0')))