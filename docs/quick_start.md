# Quick Start

### Set up a new virtual environment
```bash
virtualenv mm_sparse4d --python=python3.8
source mm_sparse4d/bin/activate
```

### Install packpages using pip3 我安装的是torch1.9.1+cu11.1
```bash
sparse4d_path="path/to/sparse4d"
cd ${sparse4d_path}
pip3 install --upgrade pip
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install -r requirement.txt
```
### mmcv—full请手动下载安装1.4.0，接着手动安装mmdet3d
```bash
# mmcv—full下载地址: https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# 手动下载mmdet3d-0.17.1, 到mmdetection3d的官网https://github.com/open-mmlab/mmdetection3d/tree/v0.17.1?tab=readme-ov-file
pip install -v -e .
# 手动下载安装mmdet3d后, 请将拉取的mmdet3d的文件夹替换这个手动安装的mmdet3d文件夹, 更改文件名即可(因为为了适配robo比赛的一些文件，对sparse4d的文件做了改动,所以这里略显麻烦)
```

### 安装detectron
```bash
git clone https://github.com/facebookresearch/detectron2.git

cd detectron2
python setup.py install

pip install setuptools==59.5.0
```


### Compile the deformable_aggregation CUDA op
```bash
cd projects/mmdet3d_plugin/ops
python3 setup.py develop
cd ../../../
```

### Prepare the data
Download the [NuScenes dataset](https://www.nuscenes.org/nuscenes#download) and create symbolic links.
```bash
cd ${sparse4d_path}
mkdir data
ln -s path/to/nuscenes ./data/nuscenes
```

Pack the meta-information and labels of the dataset, and generate the required .pkl files.
```bash
pkl_path="data/nuscenes_anno_pkls"
mkdir -p ${pkl_path}
python3 tools/nuscenes_converter.py --version v1.0-mini --info_prefix ${pkl_path}/nuscenes-mini
python3 tools/nuscenes_converter.py --version v1.0-trainval,v1.0-test --info_prefix ${pkl_path}/nuscenes
```

### Generate anchors by K-means
```bash
python3 tools/anchor_generator.py --ann_file ${pkl_path}/nuscenes_infos_train.pkl
```

### Download pre-trained weights
Download the required backbone [pre-trained weights](https://download.pytorch.org/models/resnet50-19c8e357.pth).
```bash
mkdir ckpt
wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O ckpt/resnet50-19c8e357.pth
```

### Commence training and testing
```bash
# train
bash local_train.sh sparse4dv3_temporal_r50_1x8_bs6_256x704

# test
bash local_test.sh sparse4dv3_temporal_r50_1x8_bs6_256x704  path/to/checkpoint
```

For inference-related guidelines, please refer to the [tutorial/tutorial.ipynb](../tutorial/tutorial.ipynb).
