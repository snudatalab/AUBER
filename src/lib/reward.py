"""
AUBER: Automated BERT-Regularization

Authors:
- Hyun Dong Lee (hl2787@columbia.edu)
- Seongmin Lee (ligi214@snu.ac.kr)
- U Kang (ukang@snu.ac.kr)
- Data Mining Lab at Seoul National University.

File: src/lib/reward.py
 - Contains source code for reward schemes.
"""

import subprocess
import torch

def get_reward(model,directory,dataset,do_train,eval_script,train_script,gpu_num,lr):
    '''
    evaluate the model on the specified dataset and return the reward
    :param model: model to evaluate
    :param directory: where to save the model
    :param dataset: dataset on which the model will be evaluated
    :param do_train: whether to train the model before evaluating
    :param eval_script: evaluation script
    :param train_script: train script
    :param gpu_num: gpu for running the script
    :param lr: learning rate for finetunine BERT
    '''
    model.save_pretrained(directory)
    if do_train == False:
        process = subprocess.Popen(' '.join([eval_script,directory,gpu_num,dataset,lr]), 
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    elif do_train == True:
        process = subprocess.Popen(' '.join([train_script,directory,gpu_num,dataset,lr]), 
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, err = process.communicate()
    #print(err)
    result = output.strip()
    result = result.decode("utf-8")
    print(result)

    try:
        reward = torch.Tensor([float(result)])
    except ValueError:
        print(err)
        print(result)
        exit(0)

    return reward
