import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.init as I


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.2):
        super().__init__()
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fcl = nn.Linear(hidden_size, vocab_size)
        self.initialize_weights()
    
    def forward(self, features, captions):
        captions = captions[:, :-1]
        caption_embeds = self.embed(captions)
        features_and_captions = torch.cat((features.unsqueeze(1), caption_embeds), 1)
        
        output, hidden = self.lstm(features_and_captions)
        output = self.dropout(output)
        output = self.fcl(output)
        
        return output
    
    def initialize_weights(self):
        I.xavier_normal_(self.fcl.weight)
        self.fcl.bias.data.fill_(0.01)
        
        # from my research, setting the bias for all forget gates to 1 in some cases performs better
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        tokens = []
        for i in range(max_len):
            output, states = self.lstm(inputs, states)
            output = self.fcl(output.squeeze(1))
            _, prediction = output.max(1) 
            tokens.append(prediction.item())
            inputs = self.embed(prediction) 
            inputs = inputs.unsqueeze(1)
        return tokens