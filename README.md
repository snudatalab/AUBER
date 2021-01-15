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
  │    ├── finetuned_models
  │    │     └── mrpc_original: BERT finetuned on MRPC (for demo)
  │    │      
  │    ├── lib
  │    │     ├── agent.py: code for agent
  │    │     ├── memory.py: code for memory
  │    │     └── reward.py: code for reward
  │    │  
  │    ├── script
  │    │     ├── MRPC_train_split.sh: script for training a pruned BERT on MRPC (for demo)
  │    │     └── MRPC_eval_split.sh: script for evaluating a pruned BERT on MRPC (for demo)
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
* MRPC: Microsoft Research Paraphrase Corpus
* Note:
    * Other GLUE datasets can be downloaded from https://github.com/nyu-mll/GLUE-baselines
    * In each dataset directory, there should be two folders, `train` and `dev`.
    In `dev`, there should be a copy of train and dev datasets.
    In `train`, there should be two copies of the train dataset, one named as `train.tsv` and the other named as `dev.tsv`.

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
    git clone https://monet.snu.ac.kr/gitlab/snudatalab/vet/VTT-project.git
    cd VTT-project/Automated-BERT-Regularization/src/
    git clone https://github.com/huggingface/transformers
    cd transformers
    git checkout tags/v3.5.0
    pip install .
    pip install -r ./examples/requirements.txt
    mv ../run_glue.py ./examples/text-classification
    cd ..
#### DEMO
* To train the model on the MRPC dataset, run script:
    ```    
    cd script
    ./demo.sh
    ```
    Intermediate models after pruning each layer will be saved in `src/trained_models/`.

## Contact us
- Hyun Dong Lee (hl2787@columbia.edu)
- Seongmin Lee (ligi214@snu.ac.kr)
- U Kang (ukang@snu.ac.kr)
- Data Mining Lab at Seoul National University.
