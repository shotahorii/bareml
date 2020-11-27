import bareml.deeplearning.functions as F
import bareml.deeplearning.layers as L
from .utils import UnigramSampler
from .core import get_array_module, Tensor


class MLP(L.Module):
    def __init__(self, fc_output_sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, 'l'+str(i), layer)
            self.layers.append(layer)
    
    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)


class VGG16(L.Module):
    """ VGG16 net """
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()
        features = L.Sequential(
            L.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            L.ReLU(),
            L.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            L.ReLU(),
            L.MaxPool2d(kernel_size=2, stride=2, padding=0),
            L.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            L.ReLU(),
            L.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            L.ReLU(),
            L.MaxPool2d(kernel_size=2, stride=2, padding=0),
            L.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            L.ReLU(),
            L.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            L.ReLU(),
            L.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            L.ReLU(),
            L.MaxPool2d(kernel_size=2, stride=2, padding=0),
            L.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            L.ReLU(),
            L.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            L.ReLU(),
            L.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            L.ReLU(),
            L.MaxPool2d(kernel_size=2, stride=2, padding=0),
            L.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            L.ReLU(),
            L.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            L.ReLU(),
            L.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            L.ReLU(),
            L.MaxPool2d(kernel_size=2, stride=2, padding=0)
            )

        classifier = L.Sequential(
            L.Linear(in_features=25088, out_features=4096, bias=True),
            L.ReLU(),
            L.Dropout(p=0.5),
            L.Linear(in_features=4096, out_features=4096, bias=True),
            L.ReLU(),
            L.Dropout(p=0.5),
            L.Linear(in_features=4096, out_features=num_classes, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.shape[0],-1)
        x = self.classifier(x)
        return x
        

class CBOW(L.Module):
    """
    Parameters
    ----------
    corpus: a list of word_id or a list of list of word_id ( = list of docs)
    """

    def __init__(self, corpus, embedding_dim=100, neg_sample_size=5):
        super().__init__()

        if isinstance(corpus[0],list):
            corpus = [item for sublist in corpus for item in sublist] # flatten

        self.embedding_dim = embedding_dim
        self.neg_sample_size = neg_sample_size
        self.num_embeddings = len(set(corpus)) # vocab size
        
        self.emb_in = L.Embedding(self.num_embeddings, self.embedding_dim)
        self.emb_out = L.Embedding(self.num_embeddings, self.embedding_dim)
        self.sampler = UnigramSampler(corpus, sample_size=neg_sample_size, include_positive=True,prioritise_speed=True)
    
    def forward(self, x, t):
        """
        Parameters
        ----------
        x: bareml.Tensor (n, c) int 
            n: batch size
            c: number of contexts to be embedded
        
        t: np.ndarray (n,) int
            n: batch size
        """
        num_context = x.shape[1]

        x = self.emb_in(x) # (n, c, embedding_dim)
        x = x.sum(axis=1)/num_context # (n, embedding_dim)
        
        t = self.sampler.get_negative_samples(t) # (n, 1 + neg_sample_size)
        t = self.emb_out(t) # (n, 1 + neg_sample_size, embedding_dim)
        
        y = x.repeat_interleave(dim=0,repeats=self.neg_sample_size+1).reshape(*t.shape) * t # (n, 1 + neg_sample_size, embedding_dim)
        y = y.sum(axis=2) # (n, 1 + neg_sample_size)
        y = F.sigmoid(y)

        xp = get_array_module(y)
        correct_labels = xp.zeros_like(y.data, dtype=xp.float32)
        correct_labels[:,0] = 1

        y = y.reshape(-1).astype(xp.float32)
        correct_labels = correct_labels.reshape(-1)

        return y, correct_labels


class RNNLM(L.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, stateful=True):
        super().__init__()
        self.embedding = L.Embedding(vocab_size, embedding_dim)
        self.rnn = L.RNN(input_size=embedding_dim, hidden_size=hidden_size)
        self.fc = L.Linear(in_features=hidden_size, out_features=vocab_size)
        self.stateful = stateful
        self.h_0 = Tensor(None, name='h_0')

    def forward(self, x):
        """
        Parameters
        ----------
        x: bareml.Tensor (n, l_seq)
            n: batch size
            l_seq: length of the sequence
        """
        if self.h_0.data is not None:
            # make sure not to go further before h_0 when backprop
            self.h_0.unchain_backward()

        n, l_seq = x.shape
        embedded = self.embedding(x) # embedded.shape is (n, l_seq, embedding_dim)
        embedded = embedded.transpose(1,0,2) # embedded (l_seq, n, embedding_dim)
        hs, h_n = self.rnn(embedded, self.h_0) # hs (l_seq, n, hidden_size)  h_n (n, hidden_size)

        if self.stateful:
            self.h_0 = h_n

        #out = self.fc(h_n) # out (n, vocab_size)
        #out = F.softmax(out) # out (n, vocab_size)
        #return out

        hs = hs.reshape(l_seq * n, -1) # hs (l_seq * n, hidden_size)
        outs = self.fc(hs) # outs (l_seq * n, vocab_size)
        outs = F.softmax(outs) # outs (l_seq * n, vocab_size)
        #outs = outs.reshape(l_seq, n, -1) # outs (l_seq, n, vocab_size)
        #outs = outs.transpose(1,0,2) # outs (n, l_seq, vocab_size)
        return outs


