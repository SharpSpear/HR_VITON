#!/bin/bash
ROOT_PATH=$(pwd)
cd ~
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose
cd openpose/
OPENPOSE_PATH=$(pwd)
git submodule update --init --recursive --remote
apt-get -qq install -y libatlas-base-dev libprotobuf-dev libleveldb-dev \
libsnappy-dev libhdf5-serial-dev protobuf-compiler libgflags-dev libgoogle-glog-dev \
liblmdb-dev opencl-headers ocl-icd-opencl-dev libviennacl-dev \
libboost-all-dev libopencv-dev
rm -rf build || true && mkdir build && cd build && cmake .. -DUSE_CUDNN=OFF && make -j8
cd "${ROOT_PATH}"
mkdir openpose
cp "${OPENPOSE_PATH}/build/examples/openpose/openpose.bin" "${ROOT_PATH}/openpose"
rsync -a "${OPENPOSE_PATH}/models" "${ROOT_PATH}/openpose"
# rm -rf ~/openpose
