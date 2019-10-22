set -e
set -x

# replace placeholder in data/*.scp to current working directory
sed -i 's?PLEASEREPLACE?'`pwd`'?g' data/train.scp
sed -i 's?PLEASEREPLACE?'`pwd`'?g' data/dev.scp
sed -i 's?PLEASEREPLACE?'`pwd`'?g' data/eval.scp

# install miniconda
CONDA_URL="https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh"
wget --tries=3 ${CONDA_URL} -O miniconda.sh
bash miniconda.sh -b -p `pwd`/venv

# install dependencies under virtualenv
. venv/bin/activate
# install pytorch, you may want to change the cuda version for pytorch
conda install -y pytorch=1.0.1 cuda80 -c pytorch
# install dependencies required by ESPnet
pip install matplotlib scipy h5py chainer scikit-learn librosa soundfile editdistance protobuf tensorboardX pillow kaldiio configargparse PyYAML
# install tqdm for data preparation
pip install tqdm
conda deactivate
# clean up
rm miniconda.sh
