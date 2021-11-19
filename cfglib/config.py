from easydict import EasyDict
import torch
import os
import yaml
import collections

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def init_config():
    config = EasyDict()
    config.gpu = "1"
    config.cuda = True
    # dataloader jobs number
    config.num_workers = 24
    # batch_size
    config.batch_size = 12
    # training epoch number
    config.max_epoch = 200
    config.start_epoch = 0
    # learning rate
    config.lr = 1e-4
    # using GPU
    config.cuda = True
    config.output_dir = 'output'
    config.input_size = 640
    # max polygon per image
    # synText, total-text:600; CTW1500: 1200; icdar: ; MLT: ; TD500: .
    config.max_annotation = 64
    # control points number
    config.num_points = 20
    # adj num for graph
    config.adj_num = 4
    # max point per polygon for annotation
    config.max_points = 20
    # use hard examples (annotated as '#')
    config.use_hard = True
    # prediction on 1/scale feature map
    config.scale = 1
    # # clip gradient of loss
    config.grad_clip = 0
    # demo tcl threshold
    config.dis_threshold = 0.3
    config.cls_threshold = 0.8
    # Contour approximation factor
    config.approx_factor = 0.007
    conf = EasyDict(yaml.load(open('cfglib/Config.yml', 'rb'), Loader=yaml.Loader)['Global'])
    if conf.use_gpu==False:
        config.cuda = False
        config.gpu = -1
    else:
        config.cuda = True
        config.gpu = conf.gpu
    update_config(config, conf)
    conf = EasyDict(flatten(yaml.load(open('cfglib/Config.yml', 'rb'), Loader=yaml.Loader)))
    for i in config:
        for j in conf:
            if i in j:
                config[i] = conf[j]
    update_config(config, conf)
    config =  {k.lower(): v for k, v in config.items()}
    config = fix_types(config)
    return EasyDict(config)

def fix_types(config):
    '''
    Convert strings back to numerics (float or int)
    '''
    for k, v in config.items():
        if isinstance(v, str):
            try:
                config[k] = eval(v)
            except:
                pass
    return config

def update_config(config, extra_config):
    for k, v in vars(extra_config).items():
        config[k] = v
    config.device = torch.device('cuda') if config.cuda else torch.device('cpu')
    config = fix_types(config)

def print_config(config):
    print('==========Options============')
    for k, v in sorted(config.items()):
        print('{}: {}'.format(k, v))
    print('=============End=============')

config = init_config()
