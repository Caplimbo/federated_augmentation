import torch
from torch.utils.data import DataLoader
from dataset.data_utils import make_torch_dataset


class Client:

    def __init__(self, client_id, only_digits=True, device="cuda", train_data={'x': [], 'y': []},
                 test_data={'x': [], 'y': []}, augment=True, threshold=0.8):
        self.id = client_id
        self.train_data = train_data
        self.test_data = test_data
        self.only_digits = only_digits
        self.device = device
        self.augment = augment
        self.threshold = threshold

    def prepare_client_data(self, usage="test", batch_size=16):
        if usage == "test":
            dataset = make_torch_dataset(self.test_data, only_digits=self.only_digits, augment=False)
            return DataLoader(dataset, batch_size=batch_size, shuffle=False), len(dataset)
        elif usage == "eval":
            dataset = make_torch_dataset(self.train_data, only_digits=self.only_digits, usage="eval", augment=False)
            return DataLoader(dataset, batch_size=batch_size, shuffle=False), len(dataset)
        else:
            dataset = make_torch_dataset(self.train_data, only_digits=self.only_digits, usage=usage, augment=self.augment, threshold=self.threshold)
            return DataLoader(dataset, batch_size=batch_size, shuffle=True), len(dataset)

    def train(self, model, base_state, optimizer, loss_func, num_epochs=1, batch_size=16, local=True):
        model.load_state_dict(base_state, strict=True)
        model.train()
        loss_func = loss_func.to(self.device)
        if not local:
            train_data_loader, length = self.prepare_client_data(usage='global', batch_size=batch_size)
        else:
            train_data_loader, length = self.prepare_client_data(usage='tune', batch_size=batch_size)
        for epoch in range(num_epochs):
            for images, labels in train_data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                preds = model(images)
                loss = loss_func(preds, labels)
                loss.backward()
                optimizer.step()

        return model.state_dict(), float(length)/100

    def train_gan(self, gen_model, dis_model, gen_state, dis_state, gen_optimizer, dis_optimizer, ):
        pass

    def test(self, model, state, optimizer, loss_func, batch_size=16, use_tune=True):
        model.load_state_dict(state)
        # first do local fine-tune
        if use_tune:
            tune_data_loader, _ = self.prepare_client_data(usage="tune", batch_size=batch_size)
            model.train()
            for images, labels in tune_data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                preds = model(images)
                loss = loss_func(preds, labels)

                loss.backward()
                optimizer.step()
        eval_data_loader, eval_length = self.prepare_client_data(usage="eval", batch_size=128)
        test_data_loader, test_length = self.prepare_client_data(usage="test", batch_size=128)
        eval_acc = self.compute_accuracy(model, eval_data_loader)
        test_acc = self.compute_accuracy(model, test_data_loader)
        return eval_acc, test_acc, eval_length, test_length

    def compute_accuracy(self, model, data_loader):
        model.eval()
        correct, sum = 0, 0
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                preds = model(images)
                preds = torch.argmax(preds, dim=1)
                correct += (preds == labels).sum()
                sum += labels.size(0)
        return correct / sum
