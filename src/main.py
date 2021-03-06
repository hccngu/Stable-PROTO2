import os
import pickle

from classifier.classifier_getter import get_classifier
from train.train import train
from train.test import test

from tools.tool import parse_args, print_args, set_seed
# from tools.visualization import Print_Attention

import dataset.loader as loader
from embedding.embedding import get_embedding


def main():

    args = parse_args()

    print_args(args)

    set_seed(args.seed)

    # load data
    train_data, val_data, test_data, class_names, vocab = loader.load_dataset(args)

    args.id2word = vocab.itos

    # initialize model
    model = {}
    model["G"] = get_embedding(vocab, args)  #main里面这里是最原始的pytorch的Embedding层
    model["clf"] = get_classifier(model["G"].hidden_size * 2, args)

    if args.mode == "train":
        # train model on train_data, early stopping based on val_data
        optCLF = train(train_data, val_data, model, class_names, args)

    # val_acc, val_std, _ = test(val_data, model, args,
    #                                         args.val_episodes)

    test_acc, test_std = test(test_data, class_names, optCLF, model, args, args.test_epochs, False)

    # path_drawn = args.path_drawn_data
    # with open(path_drawn, 'w') as f_w:
    #     json.dump(drawn_data, f_w)
    #     print("store drawn data finished.")

    # file_path = r'../data/attention_data.json'
    # Print_Attention(file_path, vocab, model, args)

    if args.result_path:
        directory = args.result_path[:args.result_path.rfind("/")]
        if not os.path.exists(directory):
            os.mkdirs(directory)

        result = {
            "test_acc": test_acc,
            "test_std": test_std,
            # "val_acc": val_acc,
            # "val_std": val_std
        }

        for attr, value in sorted(args.__dict__.items()):
            result[attr] = value

        with open(args.result_path, "wb") as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()