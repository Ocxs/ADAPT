# -*- coding: utf-8 -*-
# author: Xusong Chen
# email: cxs2016@mail.ustc.edu.cn

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from InputFN.input_fn import PiNetData
from Model.baseline_model import PiNetModel
from Solver import estimator_solver
from Utils.params_config import PiNetParams
from Utils import utils

def main(_):
  project_name = 'PiNet'
  seq_params = PiNetParams(project_name)
  seq_params.post_process()
  mode = seq_params.params['mode']

  model_fn = PiNetModel(seq_params.params)
  input_fn = PiNetData(seq_params.params)

  if mode == 'train':
    print('train&eval')
    estimator_solver.train(
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
