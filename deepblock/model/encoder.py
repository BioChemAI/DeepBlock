from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncoderRNN(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, hid_layers, dropout, 
                bidirectional, emb_weight=None):
        super().__init__()
        
        if emb_weight:
            self.embedding = nn.Embedding.from_pretrained(emb_weight)
        else:
            self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, hid_layers, batch_first=True, dropout=dropout, 
            bidirectional=bidirectional)
        self.flatten = nn.Flatten(1, -1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_len):
        #src = [batch size, src len]
        #src_len = [batch size]
        
        embedded = self.dropout(self.embedding(src))
        #embedded = [batch size, src len, emb dim]

        packed_embedded = pack_padded_sequence(embedded, src_len, 
            batch_first=True, enforce_sorted=False)
        packed_outputs, hidden = self.rnn(packed_embedded)
        #hidden = [bid num * hid layers, batch size, hid dim]

        output, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        state = self.flatten(hidden.permute(1,0,2))
        #output = [batch size, src len, hid dim]
        #state = [batch size, bid num * hid dim * hid layers]
        return output, state

class DecoderRNN(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, hid_layers, dropout, emb_weight=None):
        super().__init__()

        if emb_weight:
            self.embedding = nn.Embedding.from_pretrained(emb_weight)
        else:
            self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, hid_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.flatten = nn.Flatten(1, -1)
        self.unflatten = nn.Unflatten(1, (hid_layers, hid_dim))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, trg_len, state):
        #trg = [batch size, trg_len] -> Tensor([[sos, 1, 2, 3, eos, pad, pad]])
        #trg_len = [batch size] -> Tensor([4]) Use trg_len - 1 to remove eos
        #state = [batch size, hid dim * hid layers]
        
        hidden = self.unflatten(state).permute(1,0,2)
        #hidden = [hid layers, batch size, hid dim]

        embedded = self.dropout(self.embedding(trg))
        #embedded = [batch size, trg len, emb dim]

        packed_embedded = pack_padded_sequence(embedded, trg_len, 
            batch_first=True, enforce_sorted=False)
        packed_outputs, hidden = self.rnn(packed_embedded, hidden)
        #hidden = [batch size, hid layers, hid dim]
        
        output, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        #output = [batch size, trg len, hid dim]

        output = self.fc_out(output)
        state = self.flatten(hidden.permute(1,0,2))
        #output = [batch size, trg len, output dim] means Tensor([[1, 2, 3, eos]])
        #state = [batch size, hid dim * hid layers]
        return output, state
