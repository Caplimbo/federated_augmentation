import torch
import torchvision
import numpy as np
import time
import matplotlib.pyplot as plt
import argparse
import torch.optim as optim
import os
from tqdm import tqdm
from models.femnist import CNN
import torch.nn as nn
from dataset.data_utils import load_dataset_from_saved_file
from torch.utils.data import DataLoader
import random


def seed(value: int) -> None:
    """Seed random number generators to get reproducible results"""
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)  # type: ignore


class Trainer:
    def __init__(
        self,
        data_directory,
        class_size,
        embedding_dim,
        batch_size,
        latent_size=100,
        device="cpu",
        lr=0.004,
        num_workers=1,
        augment=False,
    ):

        # get dataset from directory. If not present, download to directory
        self.num_classes = class_size

        seed(2021)

        train_dataset, test_dataset = load_dataset_from_saved_file(
            data_directory + "/train", shape=(-1, 28 * 28)
        ), load_dataset_from_saved_file(data_directory + "/test", shape=(-1, 28 * 28))

        self.train_data_loader, self.test_data_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        ), DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.batch_size = batch_size
        self.device = device
        self.class_size = class_size

        self.model = CNN(num_classes=self.num_classes).to(device)
        self.model.init_weight()
        self.loss_func = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        # self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-6))

    def train(self, epochs, path):
        def eval(i):
            self.model.eval()
            correct, sum = 0, 0
            with torch.no_grad():
                for (
                    images,
                    labels,
                ) in self.test_data_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    preds = self.model(images)
                    preds = torch.argmax(preds, dim=1)
                    correct += (preds == labels).sum()
                    sum += labels.size(0)

            test_acc = correct / sum
            with open(path + "/round_log.txt", 'a') as f:
                f.write(
                    f"{i}-th Test Acc: {test_acc:.4f}"
                )
                f.write("\n")

        os.makedirs(path, exist_ok=True)
        f = open(path + "/log.txt", "w")
        open(path + "/round_log.txt", 'w')
        round = 0
        eval(0)
        for epoch in range(epochs):
            self.model.train()
            train_loss = []
            for images, labels in tqdm(
                self.train_data_loader, desc=f"Epoch {epoch}, Training..."
            ):
                # print(texts, labels)
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(images)
                # print(images[0])
                loss = self.loss_func(preds, labels)
                # print(f"loss: {loss}")
                loss.backward()
                train_loss.append(loss.item())
                self.optimizer.step()

                if (round + 1) % (len(self.train_data_loader) // (340 // 15) * (self.batch_size / 16)) == 0:
                    eval(round+1)
                round += 1
            train_avg_loss = torch.mean(torch.FloatTensor(train_loss)).item()

            self.model.eval()
            test_loss = []
            correct, sum = 0, 0
            with torch.no_grad():
                for (
                    images,
                    labels,
                ) in tqdm(self.test_data_loader, desc=f"Epoch {epoch}, Testing..."):
                    images, labels = images.to(self.device), labels.to(self.device)
                    preds = self.model(images)
                    loss = self.loss_func(preds, labels)
                    test_loss.append(loss.item())
                    preds = torch.argmax(preds, dim=1)
                    correct += (preds == labels).sum()
                    sum += labels.size(0)
                test_avg_loss = torch.mean(torch.FloatTensor(test_loss)).item()
                test_acc = correct / sum

            print(
                f"Epoch {epoch}: Train Loss: {train_avg_loss:.4f}, Test Loss: {test_avg_loss:.4f}, Test Acc: {test_acc:.4f}"
            )
            # if (epoch + 1) % 5 == 0:
            f.write(
                f"Epoch {epoch+1}: Train Loss: {train_avg_loss:.4f}, Test Loss: {test_avg_loss:.4f}, Test Acc: {test_acc:.4f}"
            )
            f.write("\n")
        f.close()


def main():
    parser = argparse.ArgumentParser(description="Hyperparameters for training CNN")
    # hyperparameter loading
    parser.add_argument(
        "--data_directory",
        type=str,
        default="dataset/femnist/full",
        help="directory to FEMNIST dataset files",
    )
    parser.add_argument(
        "--class_size", type=int, default=62, help="number of unique classes in dataset"
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=50, help="size of embedding vector"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="size of batches passed through networks at each step",
    )
    parser.add_argument(
        "--latent_size", type=int, default=100, help="size of gaussian noise vector"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cpu or gpu depending on availability and compatability",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-3, help="learning rate of models"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="workers simultaneously putting data into RAM",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of iterations of dataset through network for training",
    )

    parser.add_argument(
        "-s",
        "--save-dir",
        type=str,
        help="path to save the model and logs",
        default="checkpoints/run",
        required=False,
    )
    parser.add_argument(
        "-a", "--augment", help="whether we apply augmentation", action="store_true"
    )

    args = parser.parse_args()

    data_dir = args.data_directory
    class_size = args.class_size
    embedding_dim = args.embedding_dim
    batch_size = args.batch_size
    latent_size = args.latent_size
    device = args.device
    lr = args.lr
    num_workers = args.num_workers
    epochs = args.epochs
    path = args.save_dir
    gan = Trainer(
        data_dir,
        class_size,
        embedding_dim,
        batch_size,
        latent_size,
        device,
        lr,
        num_workers,
        args.augment,
    )
    gan.train(epochs, path)


if __name__ == "__main__":
    main()
