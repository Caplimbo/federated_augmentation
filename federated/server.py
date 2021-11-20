import numpy as np
import torch.nn as nn
import torch
from federated.client import Client
from collections import defaultdict
from tqdm import tqdm


class Server:
    def __init__(
            self, model, clients: [Client], loss_func=nn.CrossEntropyLoss(), lr=1e-4, tune_lr=1e-2
    ):
        self.model = model
        # self.gen_model = gen_model
        # self.dis_model = dis_model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.tune_optimizer = torch.optim.SGD(self.model.parameters(), lr=tune_lr)
        self.loss_func = loss_func
        self.clients = clients
        self.model_state = model.state_dict()

    def select_clients(self, my_round, num_clients=3):
        np.random.seed(my_round)
        selected_clients = np.random.choice(self.clients, num_clients, replace=False)
        #print(f"round {my_round}, selected_clients:{[client.id for client in selected_clients]}")
        return selected_clients

    def train(self, selected_clients, num_epochs=3, batch_size=16):
        #  def train_gan():

        # print(f"selected clients: {[selected_client.id for selected_client in selected_clients]}")
        total_weight = 0
        self.model.train()
        sum_state_dict = None
        for client in selected_clients:
            new_state_dict, weight = client.train(
                self.model,
                self.model_state,
                self.optimizer,
                self.loss_func,
                num_epochs,
                batch_size,
                local=False,
            )
            total_weight += weight
            # total_weight += 1
            if sum_state_dict is None:
                # print("initializing!")
                sum_state_dict = new_state_dict
                for var in new_state_dict:
                    try:
                        sum_state_dict[var] *= weight
                    except:
                        print(f"Problem var is {var}")
                        exit()
                    # sum_state_dict[var] *= torch.tensor(weight).type_as(sum_state_dict[var])
            else:
                # print("updating!")
                for var in sum_state_dict:
                    sum_state_dict[var] = sum_state_dict[var] + new_state_dict[var] * weight
        for var in sum_state_dict:
            sum_state_dict[var] /= total_weight
        # print(self.model_state.keys())
        # print("-----")
        # print(sum_state_dict.keys())
        # assert self.model_state.keys == sum_state_dict.keys()
        self.model_state = sum_state_dict



    def test(self, batch_size=64, use_tune=True):
        sum_train_acc = 0
        sum_test_acc = 0
        sum_train_weight = 0
        sum_test_weight = 0
        all_acc = []
        for client in tqdm(self.clients, desc="Testing..."):
            train_acc, test_acc, train_weight, test_weight = client.test(
                self.model,
                self.model_state,
                batch_size=batch_size,
                optimizer=self.tune_optimizer,
                loss_func=self.loss_func,
                use_tune=use_tune
            )
            sum_train_acc += train_acc * train_weight
            sum_test_acc += test_acc * test_weight
            sum_train_weight += train_weight
            sum_test_weight += test_weight
            all_acc.append((test_acc, test_weight, client.id))
        return sum_train_acc / sum_train_weight, sum_test_acc / sum_test_weight, all_acc

    def change_lr(self, new_lr, usage="train"):
        if usage == "train":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=new_lr)
        elif usage == "tune":
            self.tune_optimizer = torch.optim.SGD(self.model.parameters(), lr=new_lr)

    def save_model(self, path):
        pass
