import numpy as np
import torch


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        # if self.verbose:
        # print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __delattr__ = dict.__delitem__
    __setattr__ = dict.__setitem__


class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean


def get_default_args():
    args = dotdict()
    args.model = 'informer'  # model of experiment, options: [informer, informerstack, informerlight(TBD)]
    args.data = 'custom'  # data
    args.root_path = './'  # root path of data file
    args.data_path = 'SP500 2019.csv'  # data file
    args.target = 'close'  # target feature in S or MS task
    args.checkpoints = './informer_checkpoints'  # location of model checkpoints
    args.base_decoder = 'default'  # Options are: LSTM, default
    args.enc_in = 1  # encoder input size
    args.dec_in = 1  # decoder input size
    args.c_out = 1  # output size
    args.factor = 1  # probsparse attn factor
    args.dropout = 0.05  # dropout
    args.attn = 'prob'  # attention used in encoder, options:[prob, full]
    args.embed = 'timeF'  # time features encoding, options:[timeF, fixed, learned]
    args.activation = 'gelu'  # activation
    args.output_attention = False  # whether to output attention in ecoder
    args.mix = True
    args.padding = 0
    args.loss = 'mse'
    args.lradj = 'type1'
    args.use_amp = False  # whether to use automatic mixed precision training
    args.num_workers = 0
    args.itr = 1
    args.des = 'exp'
    args.use_gpu = True if torch.cuda.is_available() else False
    args.gpu = 0
    args.use_multi_gpu = False
    args.devices = '0,1,2,3'

    return args
