import torch
from torch.utils.data import DataLoader
from dataset.amazon_utils import make_client_dataset, find_texts_to_transfer, incorporate_augment_and_make_client_dataset
from torch.nn.utils.rnn import pad_sequence

class Client:

    def __init__(self, client_id, train_data, test_data, device="cuda", augment=True, threshold=0.8):
        self.id = client_id
        self.train_data = train_data
        self.test_data = test_data
        self.device = device
        self.augment = augment
        self.threshold = threshold

    def get_samples_to_transfer(self, label_pipeline):
        return find_texts_to_transfer(self.train_data, label_pipeline)

    def prepare_client_data(self, text_pipeline, label_pipeline, usage="test", batch_size=16, augment_data=None):

        def collate_fn(batch):
            texts = pad_sequence([text_pipeline(item[0]) for item in batch], batch_first=True)
            labels = torch.tensor([label_pipeline(item[1]) for item in batch])
            return texts, labels

        if usage == "test":
            dataset = make_client_dataset(self.test_data, text_pipeline, label_pipeline, usage=usage, augment=False)
            return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn), len(dataset)
        elif usage == "eval":
            dataset = make_client_dataset(self.train_data, text_pipeline, label_pipeline, usage=usage, augment=False)
            return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn), len(dataset)
        else:
            if augment_data:
                dataset = incorporate_augment_and_make_client_dataset(self.train_data, text_pipeline, label_pipeline, augment_data)
            else:
                dataset = make_client_dataset(self.train_data, text_pipeline, label_pipeline, usage=usage, augment=self.augment)
            return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn), len(dataset)

    def train(self, model, base_state, text_pipeline, label_pipeline, optimizer, loss_func, num_epochs=1, batch_size=16, local=True, augment_data=None):
        model.load_state_dict(base_state, strict=True)
        model.train()
        loss_func = loss_func.to(self.device)

        if not local:
            train_data_loader, length = self.prepare_client_data(text_pipeline, label_pipeline, usage='global', batch_size=batch_size, augment_data=augment_data)
        else:
            train_data_loader, length = self.prepare_client_data(text_pipeline, label_pipeline, usage='tune', batch_size=batch_size)
        for epoch in range(num_epochs):
            all_loss = []
            for texts, labels in train_data_loader:
                texts, labels = texts.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                preds = model(texts)
                loss = loss_func(preds, labels)
                # loss = loss_func(torch.flatten(preds), labels.float())
                loss.backward()
                optimizer.step()
                all_loss.append(loss.item())
        full_loss = torch.mean(torch.FloatTensor(all_loss)).item()
        torch.cuda.empty_cache()
        return model.state_dict(), length, full_loss

    def test(self, model, state, text_pipeline, label_pipeline, optimizer, loss_func, batch_size=16, use_tune=True):
        model.load_state_dict(state)
        # first do local finetune
        if use_tune:
            tune_data_loader, _ = self.prepare_client_data(text_pipeline, label_pipeline, usage="tune", batch_size=batch_size)
            model.train()
            for texts, labels in tune_data_loader:
                texts, labels = texts.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                preds = model(texts)
                loss = loss_func(preds, labels)
                # loss = loss_func(torch.flatten(preds), labels.float())
                loss.backward()
                optimizer.step()
        eval_data_loader, eval_length = self.prepare_client_data(text_pipeline, label_pipeline, usage="eval", batch_size=128)
        test_data_loader, test_length = self.prepare_client_data(text_pipeline, label_pipeline, usage="test", batch_size=128)
        eval_acc = self.compute_accuracy(model, eval_data_loader)
        test_acc = self.compute_accuracy(model, test_data_loader)
        return eval_acc, test_acc, eval_length, test_length

    def compute_accuracy(self, model, data_loader):
        model.eval()
        correct, sum = 0, 0
        with torch.no_grad():
            for texts, labels in data_loader:
                texts, labels = texts.to(self.device), labels
                preds = model(texts)
                preds = preds.detach().cpu()
                # preds = torch.where(preds.flatten() > 0.5, 1, 0)
                preds = torch.argmax(preds, dim=1)
                # preds = torch.argmax(preds, dim=1)
                correct += (preds == labels).sum().item()
                sum += labels.size(0)

        return correct / sum
