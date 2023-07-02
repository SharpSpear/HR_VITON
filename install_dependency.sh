conda create -n env python=3.8
conda activate env
pip install -r requirements.txt
pip uninstall nvidia-cublas-cu11
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

