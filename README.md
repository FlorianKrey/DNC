# Discriminative Neural Clustering (DNC) for Speaker Diarisation

This repository is the code used in our paper:

>**[Discriminative Neural Clustering for Speaker Diarisation](https://arxiv.org/abs/1910.09703)**
>
>*Qiujia Li\*, Florian Kreyssig\*, Chao Zhang, Phil Woodland* (\* indicates equal contribution)

## Overview
We propose to use encoder-decoder models for supervised clustering. This repository contains:
* a submodule for spectral clustering, a modified version of [this repository by Google](https://github.com/wq2012/SpectralCluster)
* a submodule for DNC using Transformers, implemented in [ESPnet](https://github.com/espnet/espnet)
* data processing procedures for data augmentation & curriculum learning in our paper

## Dependencies
First, as this repository contains two submodules, after cloning this repository, please run
```bash
git submodule update --init --recursive
```
Then execute the following command to install MiniConda for virtualenv with related packages:
```bash
cd DNC
./install.sh
```
Note that you may want to change the CUDA version for PyTorch in `install.sh` according to your own driver.

## Data generation

First activate the virtual environment:
```bash
source venv/bin/activate
```

To generate training and validation data with sub-meeting length 50 and 1000 random shifts:
```bash
python3 datapreperation/gen_augment_data.py --input-scps data/train.scp --input-mlfs data/train.mlf --filtEncomp --maxlen 50 --augment 1000 --varnormalise /path/to/datadir/m50.real.augment

python3 datapreperation/gen_augment_data.py --input-scps data/dev.scp --input-mlfs data/dev.mlf --filtEncomp --maxlen 50 --augment 1000 --varnormalise /path/to/datadir/m50.real.augment
```

To generate training data with sub-meeting length 50 and 1000 random shifts using the meeting randomisation:
```bash
python3 datapreperation/gen_dvecdict.py --input-scps data/train.scp --input-mlfs data/train.mlf --filtEncomp --segLenConstraint 100 --meetingLevelDict /path/to/datadir/dvecdict.meeting.split100

python3 datapreperation/gen_augment_data.py --input-scps data/train.scp --input-mlfs data/train.mlf --filtEncomp --maxlen 50 --augment 100 --varnormalise --randomspeaker  --dvectordict /path/to/datadir/dvecdict.meeting.split100/train.npz /path/to/datadir/m50.meeting.augment/
```

To generate evaluation data:
```bash
python3 datapreperation/gen_augment_data.py --input-scps data/eval.scp --input-mlfs data/eval.mlf --filtEncomp --maxlen 50 --varnormalise /path/to/datadir/m50.real
```

## Training and decoding of DNC models
### Train a DNC Transformer
The example setup for AMI is in
```bash
cd espnet/egs/ami/dnc1
```
There are multiple configuration files you may want to change:
* model training config: `config/tuning/train_transformer.yaml`
* model decoding config: `config/decode.yaml`
* submission config: `cmd_backend` variable should be set in `cmd.sh` to use your preferred setup. You may also want to modify the corresponding submission settings for the queuing system, *e.g.* `config/queue.conf` for SGE or `conf/slurm.conf` for SLURM.

To start training, run
```bash
./run.sh --stage 4 --stop_stage 4 --train_json path/to/train.json --dev_json path/to/dev.json --tag tag.for.model --init_model path/to/model/for/initialisation
```
If the model trains from scratch, the `--init_model` option should be omitted. For more options, please look into `run.sh` and `config/tuning/train_transformer.yaml`.

To track the progress of the training, run
```bash
tail -f exp/mdm_train_pytorch_tag.for.model/train.log
```

### Decode a DNC Tranformer
Similar to the command used for training, run
```bash
./run.sh --stage 5 --decode_json path/to/eval.json --tag tag.for.model
```
For more options, please look into `run.sh` and `config/decode.yaml`.

The decoding results are, by default, stored in multiple json files in `exp/mdm_train_pytorch_tag.for.model/decode_dev_xxxxx/data.JOB.json`

## Running spectral clustering

To run spectral clustering on previously generated evalutation data, for example for sub-meeting lengths 50:
```bash
python3 scoring/run_spectralclustering.py --p-percentile 0.95 --custom-dist cosine --json-out /path/to/scoringdir/eval95k24.1.json  /path/to/datadir/m50.real/eval.json
```
## Evaluation of clustering results

First the DNC or SC output has to be converted into the RTTM format:
For SC:
```bash
python3 scoring/gen_rttm.py --input-scp data/eval.scp --js-dir /path/to/scoringdir --js-num 1 --js-name eval95k24 --rttm-name eval95k24
```

For DNC:
```bash
python3 scoring/gen_rttm.py --input-scp data/eval.scp --js-dir espnet/egs/ami/dnc1/exp/mdm_train_pytorch_tag.for.model/decode_dev_xxxxx/ --js-num 16 --js-name data --rttm-name evaldnc
```

To score the result the reference rttm has to first be split into the appropriate sub-meeting lengths:
```bash
python3 scoring/split_rttm.py --submeeting-rttm /path/to/scoringdir/eval95k24.rttm --input-rttm scoring/refoutputeval.rttm --output-rttm /path/to/scoringdir/reference.rttm
```

Finally, the speaker error rate has to be calculated using:
```bash
python3 scoring/score_rttm.py --score-rttm /path/to/scoringdir/eval95k24.rttm --ref-rttm /path/to/scoringdir/reference.rttm --output-scoredir /path/to/scoringdir/eval95k24
```

## Reference
```plaintext
@misc{LiKreyssig2019DNC,
  title={Discriminative Neural Clustering for Speaker Diarisation},
  author={Li, Qiujia and Kreyssig, Florian L. and Zhang, Chao and Woodland, Philip C.},
  journal={ArXiv.org},
  eprint={1910.09703}
  year={2019},
  url={https://arxiv.org/abs/1910.09703}
}
```
