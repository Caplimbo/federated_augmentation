import torch.nn as nn
import torchtext.legacy.data
from torchtext.vocab import GloVe
import torch
from torchsummary import summary


class RNN(nn.Module):
    def __init__(self, embedding_dim=300, hidden_dim=256, output_dim=5, n_layers=2,
                 bidirectional=True, dropout=0.5):
        super().__init__()

        vec = GloVe(cache='dataset/amazon_reviews')
        self.embedding = nn.Embedding.from_pretrained(vec.vectors)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           batch_first=True)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)


        self.dropout = nn.Dropout(dropout)
        self.output_dim = output_dim

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.01)
                nn.init.zeros_(m.bias.data)

    def forward(self, text):
        # text = [sent len, batch size]

        # embedded = self.dropout(self.embedding(text))
        # print(embedded.shape)
        # embedded = text
        # embedded = [sent len, batch size, emb dim]

        embedded = self.dropout(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded,
                                                            lengths=[int(embedded.size()[1])] * len(embedded), batch_first=True)
        # print("packed_embedding shape: ", packed_embedded)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # packed_output, (hidden, cell) = self.rnn(text)
        # unpack sequence
        # output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors

        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        # hidden = [batch size, hid dim * num directions]
        x = self.fc(hidden)
        # if self.output_dim == 1:
        #     x = nn.Sigmoid()(x)
        return x

if __name__ == "__main__":
    summary(RNN().cuda(), input_size=(100), batch_size=4)
