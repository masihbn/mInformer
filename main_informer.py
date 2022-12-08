import argparse
import os
import torch

from exp.exp_informer import Exp_Informer
from utils.tools import dotdict

args = dotdict()
file_name = 'Gold Indicator 2019 12.csv'


args.model = 'informer'  # model of experiment, options: [informer, informerstack, informerlight(TBD)]
args.print_log = True

args = dotdict()
args.model = 'informer'  # model of experiment, options: [informer, informerstack, informerlight(TBD)]
args.base_decoder = 'default'  # Options are: LSTM, default, IE-SBiGRU
args.data = 'custom'  # data
args.root_path = './'  # root path of data file
args.data_path = file_name  # data file
args.target = 'color'  # target feature in S or MS task
args.checkpoints = './informer_checkpoints'  # location of model checkpoints
args.enc_in = 15  # encoder input size
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

args.d_model = 1024  # dimension of model
args.n_heads = 4  # num of heads
args.e_layers = 1  # num of encoder layers
args.d_layers = 1  # num of decoder layers
args.d_ff = 2048  # dimension of fcn in model
args.distil = False  # whether to use distilling in encoder

# args.data = 'custom' # data
# args.root_path = './' # root path of data file
# args.data_path = 'SP500.csv' # data file
# args.features = 'S' # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
# args.target = 'close' # target feature in S or MS task
# args.freq = 'b' # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
# args.checkpoints = './informer_checkpoints' # location of model checkpoints

args.seq_len = 20  # input sequence length of Informer encoder
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

args.test_set_length = 21   ############### Set to None if you want the default informer setting for this ###########

# args.base_decoder = 'default'

args.base_decoder = 'IE-NN'
args.decoder_layers = [args.d_model, 512, 128]

# args.base_decoder = 'LSTM'
# args.LSTM_hidden_units = 25
# args.LSTM_num_layers = 32
# args.LSTM_input_size = 25

# args.base_decoder = 'IE-SBiGRU'
# args.BiGRU_pred_len = 5
# args.BiGRU_input_size = 25
# args.BiGRU_hidden_size = 10
# args.BiGRU_num_layers = 5
# args.BiGRU_seq_length = 40

args.features = 'M'  # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
args.freq = 'b'  # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
args.detail_freq = 'b'

args.batch_size = 30  ############## Set according to args.test_set_length #############################
args.learning_rate = 0.0001

args.train_epochs = 100
args.patience = 100

args.print_log = True
args.inverse = False

args.use_gpu = True if torch.cuda.is_available() else False
args.gpu = 0

args.use_multi_gpu = False
args.devices = '0,1,2,3'

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

# data_parser = {
#     'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
#     'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
#     'ETTm1': {'data': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
#     'ETTm2': {'data': 'ETTm2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
#     'WTH': {'data': 'WTH.csv', 'T': 'WetBulbCelsius', 'M': [12, 12, 12], 'S': [1, 1, 1], 'MS': [12, 12, 1]},
#     'ECL': {'data': 'ECL.csv', 'T': 'MT_320', 'M': [321, 321, 321], 'S': [1, 1, 1], 'MS': [321, 321, 1]},
#     'Solar': {'data': 'solar_AL.csv', 'T': 'POWER_136', 'M': [137, 137, 137], 'S': [1, 1, 1], 'MS': [137, 137, 1]},
# }
data_parser = {
    'Gold': {'data': file_name, 'T': 'close', 'M': [14, 14, 14], 'S': [1, 1, 1], 'MS': [14, 14, 1]},
}

if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

Exp = Exp_Informer

for ii in range(args.itr):
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
                                                                                                         args.des, ii)

    exp = Exp(args)  # set experiments

    # train_data, train_loader = exp._get_data(flag='test')

    if args.print_log:
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)

    if args.print_log:
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    acc = exp.test(setting)
    print(f'test accuracy: {acc}')
    if args.do_predict:
        if args.print_log:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting, True)

    torch.cuda.empty_cache()
