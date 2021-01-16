Automated-BERT-Regularization
===

This package provides implementations of Automated-BERT-Regularization.

## Overview
#### Code structure
``` Unicode
Automated-BERT-Regularization
  │ 
  ├── src
  │    │     
  │    ├── finetuned_models: BERT finetuned on downstream tasks
  │    │      
  │    ├── lib
  │    │     ├── agent.py: code for agent
  │    │     ├── memory.py: code for memory
  │    │     └── reward.py: code for reward
  │    │  
  │    ├── script: script for training and evaluating a pruned BERT on downstream tasks
  │    │ 
  │    ├─── utils
  │    │     ├── default_param.py: default cfgs
  │    │     └── utils.py: utility functions
  │    │ 
  │    ├─── transformers: refer to https://github.com/huggingface/transformers/
  │    │     
  │    ├─── main.py: main file to run Automated-BERT-Regularization
  │    │    
  │    └─── train.py: code for training the agent
  │
  ├─── data: GLUE data
  │
  └─── script: shell scripts for demo
```

#### Data description
* GLUE datasets with `train.tsv` and `dev.tsv` can be downloaded from https://github.com/nyu-mll/GLUE-baselines
* In each dataset directory, there should be two folders, `train` and `dev`.
    * `dev/dev.tsv`: original `dev.tsv` for evaluating the performance of AUBER
    * `dev/train.tsv`: mini-training set
    * `train/dev.tsv`: mini-dev set
    * `train/train.tsv`: original `train.tsv` to get the finetuned model

#### Output
* Trained models will be saved in `src/trained_models/[MODEL_NAME]_[TASK_NAME]_[LAYER_NUMBER]` after training.
* You can test the model only if:
    * There is a finetuned model saved in `src/finetuned_models/`.
    * There are train/evaluate scripts for the pruned model saved in `src/script/`.
        * train/evaluate scripts should take three arguments: model path, gpu to use, and the dataset on which the model will be evaluated.

## Install
#### Environment 
* Ubuntu
* CUDA 10.0
* Python 3.6.12
* torch 1.5.1
* torchvision 0.6.1
* sklearn
* transformers 3.5.0

## How to use 
#### Clone the repository
```shell 
git clone https://monet.snu.ac.kr/gitlab/snudatalab/vet/VTT-project.git
cd VTT-project/Automated-BERT-Regularization/src/
git clone https://github.com/huggingface/transformers
cd transformers
git checkout tags/v3.5.0
pip install .
pip install -r ./examples/requirements.txt
cd ..
mv ./run_glue.py ./transformers/examples/text-classification
```

#### Finetune BERT on downstream tasks
Depending on the target downstream tasks, `TASK_NAME` can be changed.
```shell
export TASK_NAME=MRPC
python transformers/examples/text-classification/run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./finetuned_model/$TASK_NAME
cd ..
```

#### DEMO
* To train the model on the MRPC dataset, run script:
    ```shell    
    cd script
    ./demo.sh
    ```
* Intermediate models after pruning each layer will be saved in `src/trained_models/`.

## Contact us
- Hyun Dong Lee (hl2787@columbia.edu)
- Seongmin Lee (ligi214@snu.ac.kr)
- U Kang (ukang@snu.ac.kr)
- Data Mining Lab at Seoul National University.
