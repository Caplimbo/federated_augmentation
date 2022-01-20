import os
import random

import numpy as np
from federated import Client, Server
from dataset.data_utils import read_data
from models.femnist import *
from utils import parse_args
from tqdm import tqdm

def initialize_clients(
    client_ids, train_data, test_data, only_digits=True, device="cuda", augment=True, threshold=0.8
):
    clients = []
    for client_id in client_ids:
        clients.append(
            Client(
                client_id=client_id,
                train_data=train_data[client_id],
                test_data=test_data[client_id],
                only_digits=only_digits,
                device=device,
                augment=augment,
                threshold=threshold
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
    model = Model(args.num_classes)
    model.init_weight()
    model = model.to(args.device)

    # load data
    train_clients, train_data, test_data = read_data(
        "dataset/femnist/data/train", "dataset/femnist/data/test", only_digits=args.only_digits
    )
    print(f"Number of Clients: {len(train_clients)}")
    # set up all clients
    clients = initialize_clients(train_clients, train_data, test_data, only_digits=args.only_digits, device=args.device, augment=args.augment, threshold=args.threshold)

    # set up the server
    server = Server(model, clients=clients, lr=args.lr, tune_lr=args.tune_lr)

    # training
    print("Start Training!")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.save_dir+"_NoTune", exist_ok=True)
    os.makedirs(args.save_dir+"_WithTune", exist_ok=True)
    for round in tqdm(range(args.num_rounds), desc="Round"):
        selected_clients = server.select_clients(round, args.clients_per_round)
        server.train(selected_clients, args.num_epochs, batch_size=args.batch_size)

        if (round + 1) % args.eval_every == 0 or (round + 1) == args.num_rounds:
            train_acc, test_acc, full_test_acc = server.test(batch_size=args.batch_size, use_tune=False)
            print(f"Round {round + 1}, Without Tune, train accuracy: {train_acc:.4f}, test accuracy: {test_acc}")
            with open(args.save_dir+"_NoTune" + '/log.txt', 'a') as f:
                f.write(f"{train_acc:.4f}\t{test_acc:.4f}\n")
            with open(args.save_dir+"_NoTune" + "/all_clients.txt", 'a') as f:
                for acc, weight, id in full_test_acc:
                    f.write(f'{acc:.4f} {weight} {id}\t')
                f.write('\n')
            train_acc, test_acc, full_test_acc = server.test(batch_size=args.batch_size, use_tune=args.use_tune)
            print(f"Round {round+1}, With Tune, train accuracy: {train_acc:.4f}, test accuracy: {test_acc}")
            with open(args.save_dir+"_WithTune"+'/log.txt', 'a') as f:
                f.write(f"{train_acc:.4f}\t{test_acc:.4f}\n")
            with open(args.save_dir+"_WithTune" +"/all_clients.txt", 'a') as f:
                for acc, weight, id in full_test_acc:
                    f.write(f'{acc:.4f} {weight} {id}\t')
                f.write('\n')
    f.close()
    torch.save(server.model_state, args.save_dir+'/weights.pt')

if __name__ == "__main__":
    main()
