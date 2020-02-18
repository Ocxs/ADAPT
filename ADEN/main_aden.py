# -*- coding: utf-8 -*-
# author: Xusong Chen
# email: cxs2016@mail.ustc.edu.cn

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from InputFN.input_fn import MixedSeqMultiModalData
from Model.ADEN import MixedSeqMultiModalModel
from Solver import estimator_solver
from Utils.params_config import MixedSeqMultiModalParams
from Utils import utils

def main(_):
  project_name = 'ADEN'
  seq_params = MixedSeqMultiModalParams(project_name)
  seq_params.post_process()
  mode = seq_params.params['mode']

  model_fn = MixedSeqMultiModalModel(seq_params.params)
  input_fn = MixedSeqMultiModalData(seq_params.params)

  if mode == 'train':
    print('train&eval')
    estimator_solver.train(
      input_fn=input_fn,
      model_fn=model_fn,
      params=seq_params.params
    )
  elif mode == 'distributed_train':
    seq_params.params['mode'] = 'train'
    estimator_solver.distributed_train(
      input_fn=input_fn,
      model_fn=model_fn,
      params=seq_params.params
    )
  elif mode == 'eval':
    print('eval')
    estimator_solver.evaluate(
      input_fn=input_fn,
      model_fn=model_fn,
      params=seq_params.params
    )
  elif mode == 'infer':
    print('predict')
    estimator_solver.predict(
      input_fn=input_fn,
      model_fn=model_fn,
      params=seq_params.params
    )
  else:
    raise ValueError('invalid mode: {}'.format(mode))

  print('------------ evaluate ------------')
  tf.logging.info('------------ evaluate ------------')
  estimator_solver.evaluate(
    input_fn=input_fn,
    model_fn=model_fn,
    params=seq_params.params
  )
  model_dir = seq_params.params['model_dir']
  utils.stat_eval_results(model_dir, 'eval_result.txt')

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)
