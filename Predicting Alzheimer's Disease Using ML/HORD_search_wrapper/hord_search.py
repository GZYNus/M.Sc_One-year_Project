"""
HORD algorithm implementation
Author: Zongyi@CBIG
"""
import os
import argparse
import warnings
import datetime
from poap.controller import ThreadController
from poap.controller import BasicWorkerThread
from pySOT import *
import pySOT_pytorch

warnings.filterwarnings('ignore')


def hord_search(args):
    start_time = datetime.datetime.now()
    np.random.seed(17)
    # print('Experiment:', args.experiment)
    print('Running on:', args.test_fold, 'th Fold')
    print('Number of threads:', args.nthreads)
    print('Maximum number of evaluations:', args.maxeval)
    # fixed parameters used in lstm
    fixed_params = {'data_path': str(args.data_path),
                    'seed': str(args.seed),
                    'test_fold': str(args.test_fold),
                    'batch_size': str(args.batch_size),
                    'epochs': str(args.epochs),
                    'checkpoint': str(args.checkpoint),
                    'input_size': str(args.input_size),
                    'label': str(args.label),
                    'window': str(args.window),
                    'log_path': str(args.log_path)}
    path_dict = {'out_path': os.path.join('out', args.experiment)}
    # what is used for ? what is data?
    stem = '%s_%s_%deval' % (args.logstem, args.experiment, args.maxeval)
    # print(stem)    # log_adminimalrnn_mod_60eval
    exp_class = getattr(pySOT_pytorch, args.experiment)
    # print(exp_class)    # <class 'pySOT_pytorch.adminimalrnn_mod'>
    data = exp_class(path_dict, fixed_params, stem)
    # print(data)  # <pySOT_pytorch.adminimalrnn_mod object at 0x7f75274db450>

    # create a strategy and a controller
    controller = ThreadController()
    controller.strategy = \
        SyncStrategyNoConstraints(worker_id=0,
                                  data=data,
                                  maxeval=args.maxeval,
                                  nsamples=args.nthreads,
                                  exp_design=LatinHypercube(dim=data.dim, npts=2*(data.dim + 1)),
                                  response_surface=RBFInterpolant(kernel=CubicKernel, maxp=args.maxeval),
                                  sampling_method=CandidateDYCORS(data=data, numcand=100*data.dim))
    # launch threads and give them access to objective functions
    for i in range(args.nthreads):
        function = lambda x, n=str(i): data.objfunction(x, gpu_id=n)
        controller.launch_worker(BasicWorkerThread(controller, function))

    result = controller.run()

    print('Best value found:', result.value)
    print('Best solution found:', data.param_vec2dict(result.params[0]))

    print('Started: %s' % start_time)
    print('Ended: %s' % datetime.datetime.now())
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")



