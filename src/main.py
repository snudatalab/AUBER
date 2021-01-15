"""
AUBER: Automated BERT-Regularization

Authors:
 - Hyun Dong Lee (hl2787@columbia.edu)
 - Seongmin Lee (ligi214@snu.ac.kr)
 - U Kang (ukang@snu.ac.kr)
 - Data Mining Lab at Seoul National University.

File: src/main.py
 - Contains source code for running AUBER.
"""

from train import train
from utils.default_param import get_default_param

def main(args):

    model_type = args.model_class
    task = args.task
    original_dir = args.original_dir
    opt = args.opt
    num_episodes = args.num_episodes
    eval_script = args.eval_script
    train_script = args.train_script
    gpu_num = args.gpu_num
    resume = args.resume
    lr = args.lr
    split = args.split
    state = args.state

    # train AUBER
    train(model_type=model_type,
            task=task,
            original_dir=original_dir,
            opt=opt,
            num_episodes=num_episodes,
            eval_script=eval_script,
            train_script=train_script,
            gpu_num=gpu_num,
            resume=resume,
            lr=lr,
            split = split,
            state_m = state)

if __name__ == "__main__":

    # get default parameters
    # and parse arguments
    parser = get_default_param()
    args = parser.parse_args()

    print(args)

    main(args)
