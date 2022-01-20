import os
import random
from torchtext.data.utils import get_tokenizer
import numpy as np
import torch
from text_federated import Client, Server
from dataset.amazon_utils import read_dir
from models.amazon import *
from utils import parse_args
from tqdm import tqdm


def initialize_clients(
    train_data, test_data, device="cuda", augment=True, threshold=0.8
):
    clients = []
    for client_id in train_data.keys():
        clients.append(
            Client(
                client_id=client_id,
                train_data=train_data[client_id],
                test_data=test_data[client_id],
                device=device,
                augment=augment,
                threshold=threshold,
            )
        )
    return clients


def main():
    args = parse_args()

    random.seed(1 + args.seed)
    np.random.seed(12 + args.seed)
    torch.manual_seed(123 + args.seed)
    torch.cuda.manual_seed(123 + args.seed)

    # load the model
    Model = eval(args.model)
    model = Model(output_dim=args.num_classes)
    model.init_weight()
    try:
        model.load_state_dict(torch.load(args.save_dir + '/weights_50000.pt'))
    except:
        print("No weight file to load!")
    model = model.to(args.device)

    # load data
    train_data, test_data = read_dir(root_dir="dataset/amazon_reviews/data_by_user_books.npy")
    # train_data, test_data = read_dir("dataset/yelp/data_by_user.npy")

    clients = initialize_clients(
        train_data,
        test_data,
        device=args.device,
        augment=args.augment,
        threshold=args.threshold,
    )

    # initialize pipelines
    vec = GloVe(cache="dataset/amazon_reviews")
    tokenizer = get_tokenizer("spacy", language="en_core_web_lg")
    text_pipeline = lambda x: vec.get_vecs_by_tokens(tokenizer(x.lower()))
    if args.num_classes <= 2:
        label_pipeline = lambda x: 1 if x > 3 else 0
    else:
        label_pipeline = lambda x: int(x) - 1

    # set up the server
    server = Server(
        model,
        clients=clients,
        text_pipeline=text_pipeline,
        label_pipeline=label_pipeline,
        lr=args.lr,
        tune_lr=args.tune_lr,
        loss_func=nn.CrossEntropyLoss() if args.num_classes != 1 else nn.BCEWithLogitsLoss(),
        augment=args.augment
    )

    # training
    print("Start Training!")
    os.makedirs(args.save_dir, exist_ok=True)
    # f = open(args.save_dir + "/log.txt", "w")
    # f.close()
    # f = open(args.save_dir + "/all_clients.txt", "w")
    # f.close()

    for round in tqdm(range(args.num_rounds), desc="Round"):
        # if round > 500:
        #     server.change_lr(args.lr/10, "train")
        selected_clients = server.select_clients(round, args.clients_per_round)
        loss = server.train(selected_clients, args.num_epochs, batch_size=args.batch_size)
        # all_loss.append(loss)
        if (round+1) % args.report_interval == 0:
            # avg_loss = np.mean(np.array(all_loss))
            print(f"Round {round + 1}/{args.num_rounds}, Average Loss: {loss:.4f}")
            with open(args.save_dir + "/loss_record.txt", "a") as f:
                f.write(f"{loss:.4f}\n")
            # all_loss = []

        if (round + 1) % args.eval_every == 0 or (round + 1) == args.num_rounds:
            train_acc, test_acc, full_test_acc = server.test(
                batch_size=args.batch_size, use_tune=args.use_tune
            )
            # test_acc = server.test(batch_size=args.batch_size, usage="test")
            print(
                f"Round {round + 1}, train accuracy: {train_acc:.4f}, test accuracy: {test_acc:.4f}"
            )
            with open(args.save_dir + "/log.txt", "a") as f:
                f.write(f"{train_acc:.4f}\t{test_acc:.4f}\n")
            with open(args.save_dir + "/all_clients.txt", "a") as f:
                for acc, weight, id in full_test_acc:
                    f.write(f"{acc:.4f} {weight} {id}\t")
                f.write("\n")
    f.close()
    torch.save(server.model_state, args.save_dir+f'/weights_{50000+args.num_rounds}.pt')


if __name__ == "__main__":
    main()
