import argparse

import torch
import random
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
            description="Few Shot Text Classification with Stable-PROTO.")

    parser.add_argument("--data_path", type=str,
                        default="../data/amazon.json",
                        help="path to dataset")
    parser.add_argument("--dataset", type=str, default="amazon",
                        help="name of the dataset. "
                        "Options: [20newsgroup, amazon, huffpost, "
                        "reuters, rcv1, fewrel]")
    parser.add_argument("--n_train_class", type=int, default=10,
                        help="number of meta-train classes")
    parser.add_argument("--n_val_class", type=int, default=5,
                        help="number of meta-val classes")
    parser.add_argument("--n_test_class", type=int, default=9,
                        help="number of meta-test classes")

    parser.add_argument("--n_workers", type=int, default=10,
                        help="Num. of cores used for loading data. Set this "
                        "to zero if you want to use all the cpus.")

    parser.add_argument("--way", type=int, default=5,
                        help="#classes for each task")
    parser.add_argument("--shot", type=int, default=5,
                        help="#support examples for each class for each task")
    parser.add_argument("--query", type=int, default=25,
                        help="#query examples for each class for each task")

    parser.add_argument("--train_epochs", type=int, default=10000,
                        help="max num of training epochs")
    parser.add_argument("--train_episodes", type=int, default=2,
                        help="#tasks sampled during each training epoch")
    parser.add_argument("--val_epochs", type=int, default=100,
                        help="#asks sampled during each validation epoch")
    parser.add_argument("--test_epochs", type=int, default=1000,
                        help="#tasks sampled during each testing epoch")

    parser.add_argument("--wv_path", type=str,
                        default='../pretrain_wordvec',
                        help="path to word vector cache")
    parser.add_argument("--word_vector", type=str, default='../pretrain_wordvec/wiki.en.vec',
                        help=("Name of pretrained word embeddings."))
    parser.add_argument("--finetune_ebd", action="store_true", default=False,
                        help=("Finetune embedding during meta-training"))

    parser.add_argument("--embedding", type=str, default="mlada",
                        help=("document embedding method."))
    parser.add_argument("--classifier", type=str, default="r2d2",
                        help=("classifier."))
    parser.add_argument("--auxiliary", type=str, nargs="*", default=[],
                        help=("auxiliary embeddings (used for fewrel)."))
    parser.add_argument("--seed", type=int, default=330, help="seed")
    parser.add_argument("--dropout", type=float, default=0.1, help="drop rate")
    parser.add_argument("--patience", type=int, default=20, help="patience")
    parser.add_argument("--clip_grad", type=float, default=None,
                        help="gradient clipping")
    parser.add_argument("--cuda", type=int, default=-1,
                        help="cuda device, -1 for cpu")
    parser.add_argument("--mode", type=str, default="test",
                        help=("Running mode."
                              "Options: [train, test]"
                              "[Default: test]"))
    parser.add_argument("--save", action="store_true", default=False,
                        help="train the model")
    parser.add_argument("--notqdm", action="store_true", default=False,
                        help="disable tqdm")
    parser.add_argument("--result_path", type=str, default="")
    parser.add_argument("--snapshot", type=str, default="",
                        help="path to the pretraiend weights")

    parser.add_argument("--pretrain", type=str, default=None, help="path to the pretraiend weights for MLADA")
    parser.add_argument("--train_iter", type=int, default=5, help="Number of iterations of training(in)")
    parser.add_argument("--test_iter", type=int, default=10, help="Number of iterations of testing(in)")
    parser.add_argument("--meta_lr", type=float, default=1e-3, help="learning rate of meta(out)")
    parser.add_argument("--task_lr", type=float, default=1, help="learning rate of task(in)")
    parser.add_argument("--lr_scheduler", type=str, default=None, help="lr_scheduler")
    parser.add_argument("--ExponentialLR_gamma", type=float, default=0.98, help="ExponentialLR_gamma")
    parser.add_argument("--train_mode", type=str, default=None, help="you can choose t_add_v or None")
    parser.add_argument("--ablation", type=str, default="", help="ablation study:[-DAN, -IL]")
    parser.add_argument("--path_drawn_data", type=str, default="reuters_False_data.json", help="path_drawn_data")
    parser.add_argument("--Comments", type=str, default="", help="Comments")
    parser.add_argument("--id2word", default=None, help="id2word")

    return parser.parse_args()


def print_args(args):
    """
        Print arguments (only show the relevant arguments)
    """
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))
    print("""                          
          _________ __        ___.   .__                  ____________________ ___________________________   
         /   _____//  |______ \\_ |_ |  |   ____          \\______   \\______   \\_____  \\__    ___/\\_____   \\  
         \\_____  \\   __\\__  \\ | __ \\|  | _/ __ \\   ______ |     ___/|       _/ /   |   \\|    |    /   |   \\ 
         /        \\|  |  / __\\| \\_\\ \\  |_\\  ___/  /_____/ |    |    |    |   \\/    |    \\    |   /    |    \\
        /_______  /|__|  (___ /___  /____/\\___  >         |____|    |____|_  /\\_______  /____|   \\_______  /
                \\/          \\/    \\/          \\/                           \\/         \\/                 \\/                                                            
        
    """)


def set_seed(seed):
    """
        Setting random seeds
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_model_state_dict(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    keys = []
    for k, v in pretrained_dict.items():
           keys.append(k)

    i = 0

    print("_____________pretrain_parameters______________________________")
    for k, v in model_dict.items():
        if v.size() == pretrained_dict[keys[i]].size():
            model_dict[k] = pretrained_dict[keys[i]]
            print(model_dict[k])
            i = i + 1
        # print(model_dict[k])
    print("___________________________________________________________")
    model.load_state_dict(model_dict)
    return model


def neg_dist(instances, class_proto):  # ins:[N*K, 256], cla:[N, 256]
    return -torch.pow(torch.pow(class_proto.unsqueeze(0) - instances.unsqueeze(1), 2).sum(-1), 0.5)


def reidx_y(args, YS, YQ):
    '''
        Map the labels into 0,..., way
        @param YS: batch_size
        @param YQ: batch_size

        @return YS_new: batch_size
        @return YQ_new: batch_size
    '''
    unique1, inv_S = torch.unique(YS, sorted=True, return_inverse=True)
    unique2, inv_Q = torch.unique(YQ, sorted=True, return_inverse=True)

    if len(unique1) != len(unique2):
        raise ValueError(
            'Support set classes are different from the query set')

    if len(unique1) != args.way:
        raise ValueError(
            'Support set classes are different from the number of ways')

    if int(torch.sum(unique1 - unique2).item()) != 0:
        raise ValueError(
            'Support set classes are different from the query set classes')

    Y_new = torch.arange(start=0, end=args.way, dtype=unique1.dtype,
            device=unique1.device)

    return Y_new[inv_S], Y_new[inv_Q]