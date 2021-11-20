import numpy as np
import torch.nn as nn
import torch
from federated.client import Client
from collections import defaultdict
from tqdm import tqdm
from util.gpu_mem_track import MemTracker
# from sentiment_transfer.augment import transfer_sentiment
from text_generation import sample_by_ratings
import random
gpu_tracker = MemTracker()


class Server:
    def __init__(
        self,
        model,
        clients: [Client],
        text_pipeline,
        label_pipeline,
        loss_func=nn.CrossEntropyLoss(),
        lr=1e-4,
        tune_lr=1e-2,
        augment=False,
        task="amazon"
    ):
        self.model = model
        # self.gen_model = gen_model
        # self.dis_model = dis_model
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        # self.model, self.optimizer = amp.initialize(model, optimizer, opt_level="O0")
        # print("AMP: ", amp.state_dict())
        self.tune_optimizer = torch.optim.SGD(model.parameters(), lr=tune_lr)
        #  _, self.tune_optimizer = amp.initialize(model, tune_optimizer, opt_level="O2")
        self.loss_func = loss_func
        self.clients = clients
        self.model_state = model.state_dict()
        self.text_pipeline = text_pipeline
        self.label_pipeline = label_pipeline
        self.augment = augment

        self.task = task

    def select_clients(self, my_round, num_clients=3):
        np.random.seed(my_round)
        torch.manual_seed(my_round)
        torch.cuda.manual_seed(my_round)
        random.seed(my_round)
        selected_clients = np.random.choice(self.clients, num_clients, replace=False)
        # print(f"round {my_round}, selected_clients:{selected_clients}")
        return selected_clients

    def train(self, selected_clients, num_epochs=3, batch_size=16):
        augment_samples = self.get_augment_samples(selected_clients) if self.augment else None
        total_weight = 0
        self.model.train()
        sum_state_dict = None
        all_loss = []
        for (i, client) in enumerate(selected_clients):
            # gpu_tracker.track()
            new_state_dict, weight, loss = client.train(
                self.model,
                self.model_state,
                self.text_pipeline,
                self.label_pipeline,
                self.optimizer,
                self.loss_func,
                num_epochs,
                batch_size,
                local=False,
                augment_data=augment_samples[i] if augment_samples else None
            )
            # print("weight is: ", weight)
            all_loss.append(loss)
            total_weight += weight
            # total_weight += 1
            if sum_state_dict is None:
                sum_state_dict = {}
                # sum_state_dict = new_state_dict
                for var in new_state_dict:
                    if var == "embedding.weight":
                        continue
                    # sum_state_dict[var] *= 1
                    # sum_state_dict[var] = new_state_dict[var].detach().cpu() * weight

                    sum_state_dict[var] = new_state_dict[var] * weight
                    # sum_state_dict[var] *= torch.HalfTensor([1.0]).to("cuda")#.type_as(sum_state_dict[var])
                #     '''
                #     try:
                #         sum_state_dict[var] *= torch.HalfTensor(weight).type_as(sum_state_dict[var])
                #         #  sum_state_dict[var] *= weight.half()
                #     except:
                #         print(f"Problem var is {var}")
                #         exit()
                #     '''

            else:
                # print("updating!")
                for var in sum_state_dict:
                    if var == "embedding.weight":
                        continue
                    # print(new_state_dict[var].dtype)
                    sum_state_dict[var] = (
                        sum_state_dict[var] + new_state_dict[var] * weight
                    )
                    # sum_state_dict[var] = sum_state_dict[var] + new_state_dict[var].detach().cpu() * weight

                    # sum_state_dict[var] = sum_state_dict[var] + new_state_dict[var] * torch.HalfTensor([1.0]).to("cuda")

            torch.cuda.empty_cache()
        for var in sum_state_dict:
            sum_state_dict[var] /= total_weight
            # sum_state_dict[var] /= torch.HalfTensor([total_weight*1.0]).to("cuda")
            # sum_state_dict[var] = sum_state_dict[var].cuda()
        # print(self.model_state.keys())
        # print("-----")
        # print(sum_state_dict.keys())
        # assert self.model_state.keys == sum_state_dict.keys()
        sum_state_dict["embedding.weight"] = new_state_dict["embedding.weight"]
        self.model_state = sum_state_dict
        avg_loss = np.mean(np.array(all_loss))
        return avg_loss

    def get_augment_samples(self, clients):
        texts_to_transfer = []
        split_index = [0]
        for client in clients:
            text_to_transfer = client.get_samples_to_transfer(self.label_pipeline)
            texts_to_transfer.extend(text_to_transfer)
            split_index.append(len(texts_to_transfer))

        transferred = []
        for index in range(0, len(texts_to_transfer), 64):
            to_transfer = texts_to_transfer[index: index + 64]
            # assert self.task in ['yelp', 'amazon']
            if self.task == "yelp":
                batch_result = transfer_sentiment(to_transfer)
            else:
                batch_result = sample_by_ratings(to_transfer)
            transferred.extend(batch_result)
        if self.task == "yelp":
            return [transferred[split_index[i]: split_index[i+1]] for i in range(len(clients))]

        else:  # amazon
            res = []
            for i in range(len(clients)):
                texts = transferred[split_index[i]: split_index[i+1]]
                labels = texts_to_transfer[split_index[i]: split_index[i+1]]
                res.append([texts, labels])
            return res

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
                self.text_pipeline,
                self.label_pipeline,
                batch_size=batch_size,
                optimizer=self.tune_optimizer,
                loss_func=self.loss_func,
                use_tune=use_tune,
            )
            torch.cuda.empty_cache()
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
