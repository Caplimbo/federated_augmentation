import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import matplotlib.pyplot as plt
import argparse
import torch.optim as optim
import os
from tqdm import tqdm
from mymodel import *

import sys

sys.path.append("..")
from dataset.data_utils import combine_femnist_data

CRITIC_ITERATIONS = 5


class Trainer():

    def __init__(self, data_directory, class_size, embedding_dim, batch_size, latent_size=100, device='cpu', lr=0.0002,
                 num_workers=1):
        # load dataset
        transformation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomRotation(degrees=(90, 90)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
            # transforms.Lambda(lambda x: 1 - x),
        ])

        # check if data directory exists, if not, create it.
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
            print('Directory created.')
        else:
            print('Directory exists.')

        # get dataset from directory. If not present, download to directory
        # self.dataset = torchvision.datasets.EMNIST(data_directory, split='digits', train=True, transform=transformation,
        #                                            download=True)
        self.dataset = combine_femnist_data('../dataset/femnist/data/train', (class_size == 10), path='../dataset/gan_train/all')
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True,
                                                       num_workers=num_workers)

        self.batch_size = batch_size
        self.device = device
        self.class_size = class_size

        # define models
        self.latent_size = 100

        self.dis = Discriminator(class_size, embedding_dim).to(device)
        self.dis.init_weight()

        self.gen = Generator(latent_size, class_size, 2).to(device)
        self.gen.init_weight()
        # self.dis.load_state_dict(torch.load('output/letter2/saved_models/discriminator_20.pt'))
        # self.gen.load_state_dict(torch.load('output/letter2/saved_models/generator_20.pt'))

        self.loss_func = nn.BCELoss().to(device)

        # self.optimizer_d = optim.RMSprop(self.dis.parameters(), lr=lr)
        # self.optimizer_g = optim.RMSprop(self.gen.parameters(), lr=lr)

        self.optimizer_d = optim.Adam(self.dis.parameters(), lr=lr, betas=(0.5, 0.9))
        self.optimizer_g = optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.9))

    def gradient_penalty(self, real_samples, fake_samples, labels):
        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.dis(interpolates, labels)
        fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).to(self.device)
        fake.requires_grad = False
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        gradients = gradients[0].view(gradients[0].size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train(self, epochs, saved_image_directory, saved_model_directory):
        start_time = time.time()

        gen_loss_list = []
        dis_loss_list = []
        was_loss_list = []

        lmbda_gp = 10

        for epoch in tqdm(range(epochs)):
            if epoch == 50:
                lmbda_gp = 1
            gen_loss = 0
            dis_loss = 0
            w_loss = 0
            cur_time = time.time()
            for images, labels in tqdm(self.data_loader, desc="BATCH..."):
                # print(labels)
                b_size = len(images)
                # train Discriminator with Wasserstein Loss
                fake = torch.zeros((b_size, 1)).to(self.device)
                valid = torch.ones((b_size, 1)).to(self.device)

                for _ in range(CRITIC_ITERATIONS):
                    self.optimizer_d.zero_grad()

                    # fake loss
                    z = torch.randn(b_size, self.latent_size).to(self.device)
                    fake_images = self.gen(z, labels.to(self.device))
                    fake_pred = self.dis(fake_images, labels.to(self.device))
                    ddd_loss_fake = self.loss_func(fake_pred, fake)
                    d_loss_fake = torch.mean(fake_pred)

                    # real loss
                    real_pred = self.dis(images.to(self.device), labels.to(self.device))
                    d_loss_real = -torch.mean(real_pred)
                    ddd_loss_real = self.loss_func(real_pred, valid)

                    gp = self.gradient_penalty(images.to(self.device), fake_images, labels.to(self.device))

                    ddd_loss = (ddd_loss_fake + ddd_loss_real) / 2
                    was_loss = (d_loss_fake + d_loss_real) + lmbda_gp * gp
                    was_loss.backward()
                    self.optimizer_d.step()
                    w_loss += was_loss.item()
                    dis_loss += ddd_loss.item()

                # train Generator
                self.optimizer_g.zero_grad()

                z = torch.randn(b_size, self.latent_size).to(self.device)
                fake_images = self.gen(z, labels.to(self.device))
                fake_pred = self.dis(fake_images, labels.to(self.device))
                g_loss = -torch.mean(fake_pred)
                g_loss.backward()
                self.optimizer_g.step()

                gen_loss += g_loss.item()

            cur_time = time.time() - cur_time
            dis_loss /= 5 * len(self.data_loader)
            gen_loss /= len(self.data_loader)
            was_loss = w_loss / len(self.data_loader)
            print('Epoch {},    Gen Loss: {:.4f},   Dis Loss: {:.4f},   Was Loss: {:.4f}'.format(epoch, gen_loss,
                                                                                                 dis_loss, was_loss))
            print('Time Taken: {:.4f} seconds. Estimated {:.4f} hours remaining'.format(cur_time, (epochs - epoch) * (
                cur_time) / 3600))
            gen_loss_list.append(gen_loss)
            dis_loss_list.append(dis_loss)
            was_loss_list.append(was_loss)

            # show samples
            labels = torch.LongTensor(np.repeat(np.arange(self.class_size), 5)).to(self.device)
            z = torch.randn(self.class_size * 5, self.latent_size).to(self.device)
            sample_images = self.gen(z, labels)

            # save models to model_directory
            torch.save(self.gen.state_dict(), saved_model_directory + '/generator_{}.pt'.format(epoch + 1))
            torch.save(self.dis.state_dict(), saved_model_directory + '/discriminator_{}.pt'.format(epoch + 1))

            image_grid = torchvision.utils.make_grid(sample_images.cpu().detach(), nrow=10, normalize=True)
            _, plot = plt.subplots(figsize=(12, 12))
            plt.axis('off')
            plot.imshow(image_grid.permute(1, 2, 0))
            plt.savefig(saved_image_directory + '/epoch_{}_checkpoint.jpg'.format(epoch + 1), bbox_inches='tight')
            plt.close()

        finish_time = time.time() - start_time
        print('Training Finished. Took {:.4f} seconds or {:.4f} hours to complete.'.format(finish_time,
                                                                                           finish_time / 3600))
        return gen_loss_list, dis_loss_list


def main():
    parser = argparse.ArgumentParser(description='Hyperparameters for training GAN')
    # hyperparameter loading
    parser.add_argument('--data_directory', type=str, default='../dataset/pytorch_emnist',
                        help='directory to EMNIST dataset files')
    parser.add_argument('--saved_image_directory', type=str, default='output/saved_images',
                        help='directory to where image samples will be saved')
    parser.add_argument('--saved_model_directory', type=str, default='output/saved_models',
                        help='directory to where model weights will be saved')
    parser.add_argument('--class_size', type=int, default=62, help='number of unique classes in dataset')
    parser.add_argument('--embedding_dim', type=int, default=3, help='size of embedding vector')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='size of batches passed through networks at each step')
    parser.add_argument('--latent_size', type=int, default=100, help='size of gaussian noise vector')
    parser.add_argument('--device', type=str, default='cuda',
                        help='cpu or gpu depending on availability and compatability')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate of models')
    parser.add_argument('--num_workers', type=int, default=0, help='workers simultaneously putting data into RAM')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of iterations of dataset through network for training')
    args = parser.parse_args()

    data_dir = args.data_directory
    saved_image_dir = args.saved_image_directory
    saved_model_dir = args.saved_model_directory
    class_size = args.class_size
    embedding_dim = args.embedding_dim
    batch_size = args.batch_size
    latent_size = args.latent_size
    device = args.device
    lr = args.lr
    num_workers = args.num_workers
    epochs = args.epochs
    os.makedirs(args.saved_image_directory, exist_ok=True)
    os.makedirs(args.saved_model_directory, exist_ok=True)
    gan = Trainer(data_dir, class_size, embedding_dim, batch_size, latent_size, device, lr, num_workers)
    gen_loss_lost, dis_loss_list = gan.train(epochs, saved_image_dir, saved_model_dir)


if __name__ == "__main__":
    main()
