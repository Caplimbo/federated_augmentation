import torch
import torchvision
import torchvision.transforms as transforms
from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer
import numpy as np
import time
import matplotlib.pyplot as plt
import argparse
import torch.optim as optim
import os
from tqdm import tqdm
from models.amazon import RNN
from dataset.amazon_utils import load_full_dataset
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
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
        num_classes=1,
        sample_ratio=0.5
    ):

        # get dataset from directory. If not present, download to directory

        vec = GloVe(cache="dataset/amazon_reviews")
        tokenizer = get_tokenizer("spacy", language="en_core_web_lg")
        self.num_classes = num_classes
        text_pipeline = lambda x: vec.get_vecs_by_tokens(tokenizer(x.lower()))
        if self.num_classes == 1:
            label_pipeline = lambda x: 1 if x > 3 else 0
        else:
            label_pipeline = lambda x: int(x) - 1

        seed(2021)

        train_dataset, test_dataset = load_full_dataset(
            "dataset/amazon_reviews/data_by_user_books.npy", augment=augment, ratio=sample_ratio
        )

        def collate_fn(batch):
            texts = pad_sequence([text_pipeline(str(item[0])) for item in batch], batch_first=True)
            labels = torch.tensor([label_pipeline(item[1]) for item in batch])
            return texts, labels

        self.train_data_loader, self.test_data_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        ), DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )
        self.batch_size = batch_size
        self.device = device
        self.class_size = class_size

        # define models
        self.latent_size = 100

        self.model = RNN(output_dim=self.num_classes).to(device)
        # self.model.init_weight()
        if self.num_classes == 5:
            self.loss_func = nn.CrossEntropyLoss()
        else:
            self.loss_func = nn.BCEWithLogitsLoss()

        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

    def train(self, epochs, path):
        os.makedirs(path, exist_ok=True)
        f = open(path+"/log.txt", 'w')
        for epoch in range(epochs):

            self.model.train()
            train_loss = []
            for texts, labels in tqdm(
                self.train_data_loader, desc=f"Epoch {epoch}, Training..."
            ):
                texts, labels = texts.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(texts)
                if self.num_classes == 5:
                    loss = self.loss_func(preds, labels)
                else:
                    loss = self.loss_func(torch.flatten(preds), labels.float())
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())
            train_avg_loss = torch.mean(torch.FloatTensor(train_loss)).item()

            self.model.eval()
            test_loss = []
            correct, sum = 0, 0
            with torch.no_grad():
                for (
                    texts,
                    labels,
                ) in tqdm(self.test_data_loader, desc=f"Epoch {epoch}, Testing..."):
                    texts, labels = texts.to(self.device), labels.to(self.device)

                    preds = self.model(texts)

                    if self.num_classes == 5:
                        loss = self.loss_func(preds, labels)
                    else:
                        loss = self.loss_func(torch.flatten(preds), labels.float())
                    test_loss.append(loss.item())
                    # preds = preds.detach().cpu().flatten()

                    if self.num_classes == 1:
                        preds = preds.detach().flatten()
                        preds = torch.where(preds > 0.5, 1, 0)
                    else:
                        preds = preds.detach()
                        preds = torch.argmax(preds, dim=1)
                    correct += (preds == labels).sum().item()
                    sum += labels.size(0)
                test_avg_loss = torch.mean(torch.FloatTensor(test_loss)).item()
                test_acc = correct / sum
            print(
                f"Epoch {epoch}: Train Loss: {train_avg_loss:.4f}, Test Loss: {test_avg_loss:.4f}, Test Acc: {test_acc:.4f}"
            )

            f.write(
                f"Epoch {epoch+1}: Train Loss: {train_avg_loss:.4f}, Test Loss: {test_avg_loss:.4f}, Test Acc: {test_acc:.4f}"
            )
            f.write("\n")
        f.close()


def main():
    parser = argparse.ArgumentParser(description="Hyperparameters for training GAN")
    # hyperparameter loading
    parser.add_argument(
        "--data_directory",
        type=str,
        default="dataset/pytorch_emnist",
        help="directory to EMNIST dataset files",
    )
    parser.add_argument(
        "--class_size", type=int, default=5, help="number of unique classes in dataset"
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=50, help="size of embedding vector"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
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
        "--lr", type=float, default=5e-2, help="learning rate of models"
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
        default=300,
        help="number of iterations of dataset through network for training",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1,
        help="binary or 5-class classification",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.5,
        help="sample ratio of positive labels",
    )
    parser.add_argument('-s', '--save-dir', type=str, help='path to save the model and logs', default='checkpoints/run', required=False)
    parser.add_argument('-a', '--augment',
                        help='whether we apply augmentation',
                        action='store_true')

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
        args.num_classes,
        args.ratio
    )
    gan.train(epochs, path)


if __name__ == "__main__":
    main()
