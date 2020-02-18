# -*- coding: utf-8 -*-
# author: Xusong Chen
# email: cxs2016@mail.ustc.edu.cn


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from InputFN.input_fn import DUALBaseData
from Model.DUAL import DUALModel
from Solver import estimator_solver as estimator_solver
from Utils.params_config import DualParams
from Utils import utils




def main(_):
  project_name = 'DUAL'
  ft_params = DualParams(project_name)
  ft_params.post_process()
  mode = ft_params.params['mode']

  model_fn = DUALModel(ft_params.params)
  input_fn = DUALBaseData(ft_params.params)

  if mode == 'train':
    print('train&eval')
    estimator_solver.train(
      input_fn=input_fn,
      model_fn=model_fn,
      params=ft_params.params
    )
  elif mode == 'eval':
    print('eval')
    estimator_solver.evaluate(
      input_fn=input_fn,
      model_fn=model_fn,
      params=ft_params.params
    )
  elif mode == 'infer':
    print('predict')
    estimator_solver.predict(
      input_fn=input_fn,
      model_fn=model_fn,
      params=ft_params.params
    )
  else:
    raise ValueError('invalid mode: {}'.format(mode))

  model_dir = ft_params.params['model_dir']
  utils.stat_eval_results(model_dir, 'eval_result.txt')

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)
