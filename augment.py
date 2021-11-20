import torch
import torch.nn as nn
from gan.mymodel import Generator, Discriminator


class CGAN_DataGenerator:
    def __init__(
        self,
        gen: Generator,
        dis: Discriminator,
        generator_weight_path,
        discriminator_weight_path,
        latent_size=100,
        num_classes=10,
        device="cuda",
    ):

        self.gen = gen.to(device)
        self.gen.load_state_dict(torch.load(generator_weight_path))
        self.gen.eval()

        self.dis = dis.to(device)
        self.dis.load_state_dict(torch.load(discriminator_weight_path))
        self.dis.eval()

        self.latent_size = latent_size
        self.num_classes = num_classes
        self.device = device

    def generate_by_labels(self, labels, threshold=0.5):
        PATIENCE = 5
        labels = labels.to(self.device)
        generated_images, generated_labels, labels = self.generate_once(labels, threshold)
        p = PATIENCE
        while len(labels) > 0:
            if p == 0:
                break
            generated_once_images, generated_once_labels, labels = self.generate_once(labels, threshold)
            generated_images = torch.cat([generated_images, generated_once_images])
            generated_labels = torch.cat([generated_labels, generated_once_labels])
            if len(generated_once_labels) == 0:
                p -= 1
            else:
                p = PATIENCE
        # 二值化 ?
        generated_images = generated_images.view(-1, 28*28)
        # generated_images = torch.where(generated_images > 0.2, 1.0, -1.0)# .view(-1, 28*28)
        return generated_images, generated_labels

    def generate_once(self, labels, threshold):
        num_samples = len(labels)
        noise = torch.randn(num_samples, self.latent_size).to(self.device)
        with torch.no_grad():
            generated_images = (
                self.gen(noise, labels)
            )
            score = self.dis(generated_images, labels).view(-1, )
        selected = score >= threshold
        left = score < threshold
        taken_images, taken_labels = generated_images.cpu().detach()[selected], labels.cpu().detach()[selected]
        left_labels = labels[left]
        torch.cuda.empty_cache()
        # print(f"scores this time: {score}")
        return taken_images, taken_labels, left_labels

    def load_weight(self, generator_weight_path, discriminator_weight_path):
        self.gen.load_state_dict(torch.load(generator_weight_path))
        self.dis.load_state_dict(torch.load(discriminator_weight_path))


# embedding_dim=3, num=82 for 62 classes (all)
# embedding_Dim=1, num=62 for 10 classes (digits)
cGAN_DataGenerator = CGAN_DataGenerator(
    Generator(class_size=10, embedding_dim=1),
    Discriminator(class_size=10, embedding_dim=1),
    "gan/output/femnist/digits/saved_models/generator_62.pt",
    "gan/output/femnist/digits/saved_models/discriminator_62.pt",
)

if __name__ == "__main__":
    images = cGAN_DataGenerator.generate_by_labels(labels=torch.tensor([1, 2, 3]))
    print(images)
