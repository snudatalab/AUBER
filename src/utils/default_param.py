"""
AUBER: Automated BERT-Regularization

Authors:
- Hyun Dong Lee (hl2787@columbia.edu)
- Seongmin Lee (ligi214@snu.ac.kr)
- U Kang (ukang@snu.ac.kr)
- Data Mining Lab at Seoul National University.

File: src/utils/default_param.py
 - Contain source code for receiving arguments.
"""

import argparse


def get_default_param():
    """
    Receive arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-model", "--model_class",
                        help="model class",
                        choices=["bert"],
                        type=str,
                        default='bert')

    parser.add_argument("-original_dir",
                        help="original directory of pretrained model",
                        type=str,
                        default='./finetuned_models/mrpc_original')

    parser.add_argument("-do_train",
                        help="do train",
                        type=bool,
                        default=True)

    parser.add_argument("-num_episodes",
                        help="number of episodes for each layer",
                        type=int,
                        default=1)

    parser.add_argument("-task",
                        help="finetuning task",
                        type=str,
                        default='mrpc')

    parser.add_argument("-opt",
                        help="optimizer for training RL",
                        type=str,
                        default='SGD')

    parser.add_argument("-eval_script",
                        help="eval script",
                        type=str,
                        default='./script/MRPC_eval.sh')

    parser.add_argument("-train_script",
                        help="train script",
                        type=str,
                        default='./script/MRPC_train.sh')

    parser.add_argument("-gpu_num",
                        help="gpu to use",
                        type=str,
                        default='0')

    parser.add_argument("-resume",
                        help="resume",
                        type=bool,
                        default=False)

    parser.add_argument("-split",
                        help="split training data",
                        type=bool,
                        default=False)

    parser.add_argument("-state",
                        help="state vector (key/query/value)",
                        type=str,
                        default='value')

    parser.add_argument("-lr",
                        help="learning rate",
                        type=str,
                        default='2e-5')

    return parser
