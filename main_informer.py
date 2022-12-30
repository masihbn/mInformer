import argparse
import os
import torch

from exp.exp_informer import Exp_Informer
from utils.tools import dotdict

file_name = 'Gold Indicator 2019 12.csv'
data_base_dir = 'data_classified'
enc_in = 15
days_in_a_year = 252


def get_default_args(f_name=file_name):
    args = dotdict()
    args.model = 'informer'  # model of experiment, options: [informer, informerstack, informerlight(TBD)]
    args.data = 'custom'  # data
    args.root_path = './'  # root path of data file
    args.data_path = f_name  # data file
    args.target = 'color'  # target feature in S or MS task
    args.checkpoints = './informer_checkpoints'  # location of model checkpoints
    args.enc_in = enc_in  # encoder input size
    args.c_out = 1  # output size
    args.factor = 1  # probsparse attn factor
    args.dropout = 0.05  # dropout
    args.attn = 'prob'  # attention used in encoder, options:[prob, full]
    args.embed = 'timeF'  # time features encoding, options:[timeF, fixed, learned]
    args.activation = 'gelu'  # activation
    args.mix = True
    args.lradj = 'type1'
    args.use_amp = False  # whether to use automatic mixed precision training
    args.num_workers = 0
    args.use_gpu = True if torch.cuda.is_available() else False
    args.gpu = 0
    args.use_multi_gpu = False
    args.devices = '0,1,2,3'

    return args


def get_args(args):
    args.features = 'M'  # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
    args.freq = 'b'  # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h

    args.seq_len = 20  # input sequence length of Informer encoder

    args.d_model = 1024  # dimension of model
    args.n_heads = 4  # num of heads
    args.e_layers = 1  # num of encoder layers
    args.d_layers = 1  # num of decoder layers
    args.d_ff = 2048  # dimension of fcn in model
    args.distil = False  # whether to use distilling in encoder

    # args.base_decoder = 'IE-NN'
    args.base_decoder = 'IE-NNSoftmax1'
    args.decoder_layers = [args.d_model, 512, 128, 16]

    args.train_epochs = 5
    args.patience = 100
    args.learning_rate = 0.0001

    ################################################################################################################################################################################
    args.print_log = True  #########################################################################################################################################################
    args.inverse = False  ###########################################################################################################################################################
    ################################################################################################################################################################################

    args.pred_data_path = file_name  #########################################################################################################################
    args.detail_freq = 'b'  #########################################################################################################################################################
    args.test_set_length = 24  # For monthly Data
    #   args.test_set_length = 7 # For weekly Data
    args.batch_size = 32
    #   args.batch_size = args.test_set_length - 1
    #   args.test_set_length = None

    return args


args = get_args(get_default_args(file_name))

Exp = Exp_Informer

for ii in range(0, 1):
    # setting record of experiments
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model,
                                                                                                         args.data,
                                                                                                         args.features,
                                                                                                         args.seq_len,
                                                                                                         args.label_len,
                                                                                                         args.pred_len,
                                                                                                         args.d_model,
                                                                                                         args.n_heads,
                                                                                                         args.e_layers,
                                                                                                         args.d_layers,
                                                                                                         args.d_ff,
                                                                                                         args.attn,
                                                                                                         args.factor,
                                                                                                         args.embed,
                                                                                                         args.distil,
                                                                                                         args.mix,
                                                                                                         ii)

    exp = Exp(args)  # set experiments

    # train_data, train_loader = exp._get_data(flag='test')

    if args.print_log:
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)

    if args.print_log:
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    acc = exp.test(setting)
    print(f'test accuracy: {acc}')

    preds, trues = exp.predict(setting, flag='test')
    print(exp.binary_acc(torch.Tensor(preds[0][0]), torch.Tensor(trues[0][0])))

    torch.cuda.empty_cache()
