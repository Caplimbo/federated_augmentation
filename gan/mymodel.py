import torch
import torch.nn as nn


class Generator(nn.Module):
    # initializers
    def __init__(self, latent_size=100, class_size=10, embedding_dim=1):
        super(Generator, self).__init__()
        self.embed = nn.Embedding(class_size, latent_size * embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(latent_size + latent_size * embedding_dim, 3 * 3 * 256),
            nn.ReLU(inplace=True),
        )

        self.deconv = nn.Sequential(
            # nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 4, 2, padding=1, output_padding=1),
            # nn.BatchNorm2d(128, 0.8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, padding=1),
            # nn.BatchNorm2d(64, 0.8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh()
        )

    # weight_init
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.02)
                nn.init.zeros_(m.bias.data)

    # forward method
    def forward(self, inputs, labels):
        embedding = self.embed(labels)
        x = torch.cat([inputs, embedding], 1)
        x = self.fc(x).view(-1, 256, 3, 3)
        x = self.deconv(x)
        return x


class Discriminator(nn.Module):
    # initializers
    def __init__(self, class_size=10, embedding_dim=1):
        super(Discriminator, self).__init__()
        self.class_size = class_size
        self.embed = nn.Embedding(class_size, 28 * 28 * embedding_dim)
        self.embedding_dim = embedding_dim

        self.seq = nn.Sequential(
            nn.Conv2d(1 + embedding_dim, 64, 4, 2, 1),
            nn.InstanceNorm2d(64, affine=True),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128, affine=True),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(0.25),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256, affine=True),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # nn.Dropout2d(0.25),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 3 * 256, 1),
            nn.Sigmoid()
        )
        # self.init_weight()

    # weight_init
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.02)
                nn.init.zeros_(m.bias.data)

    # forward method
    def forward(self, inputs, labels):
        inputs = inputs.view(-1, 1, 28, 28)
        embedding = self.embed(labels).view(-1, self.embedding_dim, 28, 28)
        x = torch.cat([inputs, embedding], 1)
        x = self.seq(x)
        x = self.fc(x)
        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
