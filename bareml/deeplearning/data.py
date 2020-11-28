from abc import ABCMeta, abstractmethod
import os
import gzip
import tarfile
import urllib.request
import math
import numpy as np
from .transforms import Compose, Flatten, ToFloat, Normalise
from .utils import replace_symbols, replace_email, replace_numbers, replace_tab, single_spacing, fix_multidots

# matplotlib is not in dependency list (optional)
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# -------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------


# path to the directory to put any cache files.
CACHE_DIR = os.path.join(os.path.expanduser('~'), '.bareml')


def show_progress(block_num, block_size, total_size):
    bar_template = "\r[{}] {:.2f}%"

    downloaded = block_num * block_size
    p = downloaded / total_size * 100
    i = int(downloaded / total_size * 30)
    if p >= 100.0: p = 100.0
    if i >= 30: i = 30
    bar = "#" * i + "." * (30 - i)
    print(bar_template.format(bar, p), end='')


def get_file(url, file_name=None):
    """
    Download a file from the given url if it is not in the cache.
    The file at the given url is downloaded to the '~/.bareml'.

    Parameters
    ----------
    url: str
        url of the file.
    
    file_name: str
        name of the file. If none, the original file name is used.

    Returns
    -------
    file_path: str
        Absolute path to the saved file.
    """
    if file_name is None:
        file_name = url[url.rfind('/') + 1:]
    file_path = os.path.join(CACHE_DIR, file_name)

    if not os.path.exists(CACHE_DIR):
        os.mkdir(CACHE_DIR)

    if os.path.exists(file_path):
        return file_path

    print("Downloading: " + file_name)
    try:
        urllib.request.urlretrieve(url, file_path, show_progress)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    print(" Done")

    return file_path


# -------------------------------------------------------------
# Dataset core classes: RawDataset / Dataset / RefRawDataset
# -------------------------------------------------------------


class RawDataset(metaclass=ABCMeta):
    """
    Dataset which can be either ready / not ready to be used in the models directly.
    dtype of the data can be non np.number. 
    """

    def __init__(self, train=True, transform=None, target_transform=None):
        self.train = train

        self.transform = transform 
        self.target_transform = target_transform
        if self.transform is None:
            self.transform = lambda x:x
        if self.target_transform is None:
            self.target_transform = lambda x:x

        self.data = None
        self.target = None
        self.prepare()

        if not self.valid_type():
            raise ValueError

    def __getitem__(self, index):
        if self.target is None:
            return self.transform(self.data[index]), None
        else:
            return self.transform(self.data[index]), self.target_transform(self.target[index])

    def __len__(self):
        return len(self.data)

    def valid_type(self):
        return True

    @abstractmethod
    def prepare(self):
        """ implement data creation for self.data and self.target """
        pass


class Dataset(RawDataset):
    """
    Dataset ready to be an input of models.
    dtype of the data must be one of np.number. 
    """
    def valid_type(self):
        return np.issubdtype(self.data.dtype, np.number) \
               and np.issubdtype(self.target.dtype, np.number)
    

class RefRawDataset(RawDataset):
    """
    self.data only stores filepath to the data, instead of putting all data in the init. 
    actual data is read only when it's called. 
    """

    def __init__(self, train=True, transform=None, target_transform=None,
                 mode='rb', encoding=None):
        self.mode = mode
        self.encoding = encoding

        if transform is None:
            new_transform = self._load_data
        else:
            new_transform = Compose([self._load_data, transform])

        if target_transform is None:
            new_target_transform = self._load_target
        else:
            new_target_transform = Compose([self._load_target, target_transform])
        
        super().__init__(train, new_transform, new_target_transform)

    def _load_data(self, x):
        """
        By default, just read given filepath(s) and return as a list.
        Override if any other process needed. 

        Parameters
        ----------
        x: list of filepaths or a filepath

        Returns 
        -------
        y: list of data
           note - dtype is always list, even if only 1 data is read. 
        """
        if not isinstance(x, list):
            x = [x]
        return [open(f, self.mode, encoding=self.encoding).read() for f in x]
        
    def _load_target(self, t):
        """
        By default, assume self.target contains actual target data, unlike self.data. 
        Override if any other process needed. 

        Parameters
        ----------
        t: list of target values or a target value

        Returns 
        -------
        y: list of target values or a target value
           note - if you override this, make sure this returns a value when len(t)==1, 
                  and returns a list when len(t) > 1
        """
        return t


# -------------------------------------------------------------
# Data Loader core classes: DataLoader / SequentialDataLoader
# -------------------------------------------------------------


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / batch_size)

        self.reset()

    def reset(self):
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(len(self.dataset))
        else:
            self.index = np.arange(len(self.dataset))

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i*batch_size : (i+1)*batch_size]
        batch_x, batch_t = self.dataset[batch_index]

        self.iteration += 1
        return batch_x, batch_t

    def next(self):
        return self.__next__()


class SequentialDataLoader(DataLoader):
    def __init__(self, dataset, batch_size):
        super().__init__(dataset, batch_size, shuffle=False)
    
    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        jump = self.data_size // self.batch_size
        batch_index = [(i*jump + self.iteration) % self.data_size for i in range(self.batch_size)]
        batch_x, batch_t = self.dataset[batch_index]

        self.iteration += 1
        return batch_x, batch_t


# -------------------------------------------------------------
# For NLP
# -------------------------------------------------------------


class Corpus:
    """
    Parameters
    ----------
    docs: a list of strings, a string or bareml.RawDataset

    flatten: bool
        if True, consider all docs in the given docs as one long document
        hence self.corpus will be a long 1d list of word
        if False, consider each doc in the given docs as separated
        hence self.corpus will be a list of list (each inner list is a list of word)
    
    max_doc: int
        if not None, limit the number of doc to read.
        set if you want to make smaller corpus not to use entire given dataset.
    """
    def __init__(self, docs, flatten=False, max_doc=None):
        if isinstance(docs, str):
            docs = [docs]
        self.docs = docs
        self.flatten = flatten
        self.max_doc = max_doc

        self.word2id = {}
        self.id2word = {}
        self.corpus = []
        self.make_corpus()

    def make_corpus(self):
        print("-- Creating the corpus --")

        for i, doc in enumerate(self.docs):
            
            if self.max_doc is not None and i >= self.max_doc:
                break

            if isinstance(self.docs, RawDataset):
                doc = doc[0][0] # doc[0] is Dataset.data, where doc[1] here is Dataset.target

            encoded_doc = []
            for word in doc.split(' '):
                if word not in self.word2id:
                    new_id = len(self.word2id)
                    self.word2id[word] = new_id
                    self.id2word[new_id] = word
                encoded_doc.append(self.word2id[word])
            if self.flatten:
                self.corpus += encoded_doc
            else:
                self.corpus.append(encoded_doc)

            if i % 1000 == 0 and i != 0:
                print(str(i)+"/"+str(len(self.docs)) + " docs")
                
    def get_word(self, word_id):
        return self.id2word[word_id]
    
    def get_id(self, word):
        return self.word2id[word]


# -------------------------------------------------------------
# Datasets: MNIST, 20Newsgroups
# -------------------------------------------------------------


class MNIST(Dataset):
    """ MNIST dataset 28x28 """

    def __init__(self, train=True,
                 transform=Compose([Flatten(), ToFloat(), Normalise(0., 255.)]),
                 target_transform=None, digits=None):
        super().__init__(train, transform, target_transform)

        if digits is not None:
            self._specify_digits(digits)

    def prepare(self):
        url = 'http://yann.lecun.com/exdb/mnist/'
        train_files = {'target': 'train-images-idx3-ubyte.gz',
                       'label': 'train-labels-idx1-ubyte.gz'}
        test_files = {'target': 't10k-images-idx3-ubyte.gz',
                      'label': 't10k-labels-idx1-ubyte.gz'}

        files = train_files if self.train else test_files
        data_path = get_file(url + files['target'])
        label_path = get_file(url + files['label'])

        self.data = self._load_data(data_path)
        self.target = self._load_label(label_path)

    def _load_label(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels

    def _load_data(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data

    def _specify_digits(self, digits):
        """
        Parameters
        ----------
        digits: int [0,9] or list of int [0,9]
        """
        if isinstance(digits, list) or isinstance(digits, np.ndarray):
            digits = list(set(digits)) # remove duplicates if any
            idx = np.array([])
            for d in digits:
                idx = np.concatenate([idx, np.where(self.target == d)[0]])
        else: # digits is an int
            idx = np.where(self.target == digits)[0]

        idx = idx.astype(int)

        self.data = self.data[idx]
        self.target = self.target[idx]

    def show(self, row=10, col=10):
        if plt is None:
            print('Visualisation not available. Install matplotlib.')
        else:
            H, W = 28, 28
            img = np.zeros((H * row, W * col))
            for r in range(row):
                for c in range(col):
                    img[r * H:(r + 1) * H, c * W:(c + 1) * W] = self.data[
                        np.random.randint(0, len(self.data) - 1)].reshape(H, W)
            plt.imshow(img, cmap='gray', interpolation='nearest')
            plt.axis('off')
            plt.show()

    @staticmethod
    def labels():
        return {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}


class NewsGroups(RefRawDataset):
    """ 20 News groups data set"""   

    def __init__(self, train=True, transform=None, target_transform=None, preprocess=True):
        self.preprocess = preprocess
        super().__init__(train, transform, target_transform, mode='r', encoding='latin1')
        
    def prepare(self):
        url = 'http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz'
        filepath = get_file(url)

        dataname = '20news-bydate-train' if self.train else '20news-bydate-test'
        datapath = os.path.join(CACHE_DIR, dataname)

        if not os.path.exists(datapath):
            tar = tarfile.open(filepath, "r:gz")
            tar.extractall(CACHE_DIR)
            tar.close()

        folders = [f for f in os.listdir(datapath)]

        self.data = []
        self.target = []
        self.labels = {}
        for i, folder_name in enumerate(folders):
            self.labels[i] = folder_name
            folderpath = os.path.join(datapath, folder_name)
            self.data += [os.path.join(folderpath, file_name) for file_name in os.listdir(folderpath)]
            self.target += [i] * len(os.listdir(folderpath))

    def _load_data(self, x):
        if not isinstance(x, list):
            x = [x]

        if self.preprocess:
            return [NewsGroups.preprocess(open(f, self.mode, encoding=self.encoding).readlines()) for f in x]
        else:
            return [open(f, self.mode, encoding=self.encoding).read() for f in x]

    def labels(self):
        return self.labels

    @staticmethod
    def preprocess(lines):
        
        def is_header(line):
            headers = ('from:', 'subject:', 'organization:', 'lines:', 'nntp-posting-host:', 
                       'reply-to:', 'in-reply-to:', 'keywords:', 'nf-id:', 'nf-from:', 'originator:',
                       'in article ', 'distribution:', '> in article', '>>in article', '>> in article',
                       'article-i.d.:')
            return line.lower().startswith(headers)
        
        def clean(line):
            line = replace_tab(line)
            line = replace_email(line)
            line = line.replace("'", '') # don't -> dont
            line = replace_symbols(line)
            line = line.replace(',', ' ') # remove commas
            line = line.replace('.', ' .') # consider . as a single word
            line = replace_numbers(line, '<N>')
            line = fix_multidots(line)
            line = single_spacing(line)
            return line.strip()

        lines = [clean(line) for line in lines if not is_header(line)]
        return ' '.join([line for line in lines if line != '']).lower() 


class NewsGroupsWordInference(Dataset):
    def __init__(self, train=True, window_size=5):
        self.window_size = window_size
        super().__init__(train)

    def prepare(self):

        print('Creating NewsGroupsWordInference Dataset: Step 1/2')
        self.corpus = Corpus(NewsGroups(self.train))
        
        print('Creating NewsGroupsWordInference Dataset: Step 2/2')
        target_col = self.window_size//2
        for i, doc in enumerate(self.corpus.corpus):
            y = np.array([doc[i:i+self.window_size] for i in range(len(doc) - self.window_size + 1)])
            if len(y) != 0:
                if self.data is None:
                    self.data = np.delete(y, target_col, 1)
                    self.target = y[:,target_col]
                else:
                    self.data = np.concatenate((self.data, np.delete(y, target_col, 1)))
                    self.target = np.concatenate((self.target, y[:,target_col]))

            if i % 1000 == 0 and i != 0:
                print(str(i)+"/"+str(len(self.corpus.corpus)) + " docs")


class NewsGroupsLanguageModel(Dataset):
    def __init__(self, train=True, len_seq=10, max_doc=None):
        self.len_seq = len_seq
        self.max_doc = max_doc
        super().__init__(train)

    def prepare(self):

        print('Creating NewsGroupsLanguageModel Dataset: Step 1/2')
        self.corpus = Corpus(NewsGroups(self.train), flatten=True, max_doc=self.max_doc)
        
        print('Creating NewsGroupsLanguageModel Dataset: Step 2/2')
        if len(self.corpus.corpus) <= self.len_seq + 1:
            raise ValueError('corpus is too short.')

        num_data = len(self.corpus.corpus) - self.len_seq

        self.data = np.zeros((num_data, self.len_seq), dtype=np.int32)
        self.target = np.zeros((num_data, self.len_seq), dtype=np.int32)

        # display progress
        num_display = 10
        display_points = [(len(self.corpus.corpus)//num_display)*j for j in range(num_display)]

        for i in range(num_data):
            x = self.corpus.corpus[i:i+self.len_seq]
            t = self.corpus.corpus[i+1:i+1+self.len_seq]
            self.data[i] = x
            self.target[i] = t

            if i in display_points:
                perc = round(display_points.index(i) * (100.0/num_display), 2)
                print(str(perc), "%")

