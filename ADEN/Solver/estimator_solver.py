# -*- coding: utf-8 -*-
# author: Xusong Chen

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import os


def train(input_fn=None, model_fn=None, params=None):
  save_summary_steps = params['save_summary_steps']
  save_checkpoints_steps = params['save_checkpoints_steps']
  keep_checkpoint_max = params['keep_checkpoint_max']
  model_dir = params['model_dir']
  train_table = params['train_table']
  eval_table = params['eval_table']

  session_config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False,
    # inter_op_parallelism_threads=16,
    # intra_op_parallelism_threads=16
  )
  session_config.gpu_options.allow_growth = False

  config = tf.estimator.RunConfig(
    model_dir=model_dir,
    session_config=session_config,
    save_summary_steps=save_summary_steps,
    save_checkpoints_steps=save_checkpoints_steps,
    keep_checkpoint_max=keep_checkpoint_max+3,
    log_step_count_steps=save_summary_steps
  )

  estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=model_dir,
    params=params,
    config=config)

  early_stop_hook = tf.contrib.estimator.stop_if_no_increase_hook(
    estimator=estimator,
    metric_name='metric/test/content_roc_auc',
    max_steps_without_increase=save_checkpoints_steps*keep_checkpoint_max,
    eval_dir=os.path.join(model_dir, 'eval/'),  # the path must end with '/'
    run_every_secs=None,
    run_every_steps=save_checkpoints_steps
  )
  # profile_hook = tf.train.ProfilerHook(
  #   save_steps=save_summary_steps,
  #   output_dir=os.path.join(model_dir, 'tmp/'),
  #   show_memory=True
  # )

  train_spec = tf.estimator.TrainSpec(
    input_fn=lambda: input_fn(train_table, params=params, mode='train'),
    max_steps=None,
    hooks=[early_stop_hook] #, profile_hook]
  )

  evaluate_spec = tf.estimator.EvalSpec(
    input_fn=lambda: input_fn(eval_table, params=params, mode='eval'),
    steps=None,
    start_delay_secs=600,
    throttle_secs=1
  )

  # step 3. call tf.estimator.train_and_evaluate()
  tf.estimator.train_and_evaluate(
    estimator=estimator, train_spec=train_spec, eval_spec=evaluate_spec)


def evaluate(input_fn=None, model_fn=None, params=None):
  # step 1. get hyper parameters
  eval_table = params['eval_table']
  model_dir = params['model_dir']
  print('eval_table: ', eval_table)

  session_config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False
  )
  config = tf.estimator.RunConfig(
    model_dir=model_dir,
    session_config=session_config
  )

  # step 2. get model from estimator
  estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=model_dir,
    config=config,
    params=params
  )

  # step 3. make evaluation
  estimator.evaluate(
    input_fn=lambda: input_fn(eval_table, params=params, mode='eval')
  )


def predict(input_fn=None, model_fn=None, params=None):
  # step 1. get hyper parameters
  eval_table = params['eval_table']
  output_table = params['output_table']
  display = params['every_n_iter']
  local = params['local']
  model_dir = params['model_dir']

  session_config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False
  )
  config = tf.estimator.RunConfig(
    model_dir=model_dir,
    session_config=session_config
  )

  # step 2. get model from estimator
  model = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=model_dir,
    config=config,
    params=params
  )

  # step 3. make prediction
  predictions = model.predict(
    input_fn=lambda: input_fn(eval_table, params=params, mode='eval')
  )

  # step 4. build a new graph to write results into odps
  new_graph = tf.Graph()
  with new_graph.as_default():
    user_id_ph = tf.placeholder(dtype=tf.string)
    content_id_ph = tf.placeholder(dtype=tf.string)
    label_ph = tf.placeholder(dtype=tf.int32)
    ctr_prob_ph = tf.placeholder(dtype=tf.float32)

    if not local:
      writer = tf.TableRecordWriter(output_table)
      write_to_table = writer.write(
        range(4),
        [user_id_ph, content_id_ph, label_ph, ctr_prob_ph]
      )
      close_table = writer.close()

  # check checkpoint
  _print_graph_weight(model_dir)

  sess = tf.Session(graph=new_graph)
  # with  as sess:
  try:
    for i, preds in enumerate(predictions):
      if (i + 1) % display == 0:
        tf.logging.info('step {}'.format(i + 1))

      if not local:
        sess.run(write_to_table, feed_dict={
          user_id_ph: preds['user_id'],
          content_id_ph: preds['content_id'],
          label_ph: preds['click_cnt'],
          ctr_prob_ph: preds['ctr_prob']
        })
      else:
        print(preds['user_id'], preds['content_id'],
              preds['click_cnt'], preds['ctr_prob'])
  except tf.errors.OutOfRangeError:
    print('final step: {}'.format(i + 1))
  finally:
    if not local:
      print(sess.run(close_table))


def _print_graph_weight(checkpoint_dir):
  checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
  checkpoint_path = checkpoint.model_checkpoint_path

  reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)

  param_dict = reader.get_variable_to_shape_map()

  for key, val in param_dict.items():
    try:
      # if 'mlp' in key:
      print(key, val)
      # reader.get_tensor(key)
    except Exception, ex:
      print(ex)


class BestExporter(tf.estimator.BestExporter):
  def export(self, estimator, export_path, checkpoint_path, eval_result,
             is_the_final_export):
    if self._best_eval_result is None or \
        self._compare_fn(self._best_eval_result, eval_result):
      tf.logging.info(
        'Exporting a better model ({} instead of {})...'.format(
          eval_result, self._best_eval_result))
      result = self._saved_model_exporter.export(
        estimator, export_path, checkpoint_path, eval_result,
        is_the_final_export)
      self._best_eval_result = eval_result
      self._garbage_collect_exports(export_path)
      return result
    else:
      tf.logging.info(
        'Keeping the current best model ({} instead of {}).'.format(
          self._best_eval_result, eval_result))




if __name__ == '__main__':
  pass

