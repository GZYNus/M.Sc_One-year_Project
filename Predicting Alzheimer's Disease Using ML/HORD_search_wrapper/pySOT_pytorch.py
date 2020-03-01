"""
Hord Algorithm Implementation
Author:Zongyi & Lijun
"""
import os
import warnings
import csv
import threading
from itertools import chain
from subprocess import Popen
from subprocess import PIPE
import numpy as np
warnings.filterwarnings('ignore')


def dict2paramlist(mapping):
    return list(chain(*[('--%s' % k, v) for k, v in mapping.items()]))


class optim_base:
    def __init__(self, path_args, fix_args, namestem):
        self.args = dict(path_args)
        self.args.update(fix_args)
        self.fix_args = fix_args
        # self.logfile = 'log/%s_%s.csv' % (namestem, self.__class__.__name__)
        self.logfile = 'log/%s_%s_%s.csv' % (fix_args['test_fold'], namestem,
                                             self.__class__.__name__)
        assert not os.path.isfile(self.logfile), '%s exists. Remove file or host anther name' % self.logfile
        self.bestResult = 100000
        self.lock = threading.Lock()
        self.f_eval_count = 0

        self.init_param_list()

        dup_keys = set(self.fix_args.keys()) & set(self.hyper_map.keys())
        assert len(dup_keys) == 0, 'Must not specific %s for this experiment' %dup_keys

        with open(self.logfile, 'w') as f:
            args = list(self.fix_args.keys()) + list(self.hyper_map.keys())
            csv.writer(f).writerow(['bestResult', 'current mAUC',
                                    'current BCA', 'current ADAS13', 'Current VenICV', 'f_eval_count'] + args)

    def init_param_list(self):
        raise NotImplementedError()

    # support parameter types are 'int', 'float', and 'enum'
    def get_gp_param_space(self):
        ret = {}
        for name, idx in self.hyper_map.items():
            argtype = 'float' if idx in self.continuous else 'int'
            ret[name] = {'type': argtype, 'min': self.xlow[idx], 'max': self.xup[idx]}
        return ret

    def param_vec2dict(self, x):
        return {name: self.func[i](x[i]) for name, i in self.hyper_map.items()}

    def param_dict2vec(self, x):
        ret = np.full(len(self.hyper_map), np.nan)
        for k, v in x.items():
            ret[self.hyper_map[k]] = v
        return ret

    def objfunction(self, x, gpu_id):
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')

        var_arg = self.param_vec2dict(x)
        model_name = self.args['out_path'].split('/')[1]
        # print('model_name:',model_name)
        args = ['python', 'train_' + model_name + '.py', '--gpu', gpu_id] + \
               dict2paramlist(self.args) + dict2paramlist(var_arg)
        proc = Popen(args, stdout=PIPE)
        out, err = proc.communicate()
        # print("out1:",out)
        out = out.decode('utf-8').strip('\n')
        # print("out2:", out)
        if proc.returncode != 0:
            print(err)
            raise ValueError('Function evaluation error')
        if len(out.split(' ')) > 6:
            print(out)
            raise ValueError('Function evaluation error')
        # when searching hyperparameter, we only care about BCA
        tmp = np.array(out.split(' ')).astype(np.float)
        # print('tmp:',tmp)
        mAUC = 0 if np.isnan(tmp[0]) else tmp[0]
        BCA = 0 if np.isnan(tmp[0]) else tmp[1]
        znorm_adas13 = 0 if np.isnan(tmp[0]) else tmp[2]
        adas13 = 0 if np.isnan(tmp[0]) else tmp[3]
        znorm_VenICV = 0 if np.isnan(tmp[0]) else tmp[4]
        VenICV = 0 if np.isnan(tmp[0]) else tmp[5]
        result = 2 - mAUC - BCA + znorm_adas13 + znorm_VenICV     # the lower the better
        with self.lock:
            if self.bestResult > result:
                self.bestResult = result

            self.f_eval_count += 1
            f_eval_count = self.f_eval_count

            # print('*args', *args)
            print('--------')
            print('*****Now at', self.f_eval_count)
            print('Best:%.5f| Current:%.5f| Current mAUC: %.5f| Current BCA: %.5f | Current ADAS13: %.5f |Current VenICV: %.5f'
                % (self.bestResult, result, mAUC, BCA, adas13, VenICV))
        row = [self.bestResult, mAUC, BCA, adas13, VenICV, f_eval_count] + list(self.fix_args.values())
        row += [var_arg[k] for k in self.hyper_map.keys()]
        with open(self.logfile, 'a') as f:
            csv.writer(f).writerow(row)
        return result


class lstm_mod(optim_base):
    def init_param_list(self):
        self.dim = 4
        self.continuous = np.arange(0, 2)
        self.integer = np.arange(2, self.dim)

        # Hyperparameters to optimize
        self.hyper_map = {
            'lr': 0,
            'l2': 1,
            'hidden_size': 2,
            'nb_layers': 3
        }
        m = self.hyper_map

        self.xlow = np.zeros(self.dim)
        self.xup = np.zeros(self.dim)
        self.func = [None for _ in range(self.dim)]


        self.xlow[m['hidden_size']] = 64
        self.xup[m['hidden_size']] = 1024
        self.func[m['hidden_size']] = lambda x: '%d' % x

        self.xlow[m['l2']] = -7
        self.xup[m['l2']] = -5
        self.func[m['l2']] = lambda x: '%e' % (10 ** x)

        self.xlow[m['lr']] = -5.
        self.xup[m['lr']] = -2.
        self.func[m['lr']] = lambda x: '%e' % (10 ** x)

        self.xlow[m['nb_layers']] = 1
        self.xup[m['nb_layers']] = 4
        self.func[m['nb_layers']] = lambda x: '%d' % x




