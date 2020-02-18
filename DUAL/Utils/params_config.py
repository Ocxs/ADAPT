# -*- coding: utf-8 -*-
# author: Xusong Chen

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
from tensorflow.python.platform import gfile


flags = tf.app.flags
flags.DEFINE_boolean(
  "local",
  True,
  "run in local or pai")
# hyper parameters for distributed training
flags.DEFINE_integer("task_index", 0, "任务编号")
flags.DEFINE_string("ps_hosts", "", "分布式集群中parameter server列表")
flags.DEFINE_string("worker_hosts", "", "分布式集群中worker列表")
flags.DEFINE_string("job_name", "", "任务名称")

flags.DEFINE_string(
    "mode",
    "train",
    "job mode: train/eval/infer/")
flags.DEFINE_string(
  "model_date",
  time.strftime('%Y%m%d-%H%M%S'),
  "what time to save model"
)
flags.DEFINE_string(
  "tables",
  "SRD-1_pretrain_data.txt,SRD-1_pretrain_data.txt",
  "table for training and evaluating, splited by ','"
)
flags.DEFINE_string(
  "outputs",
  "/Users/chenxs/Documents/Research/WeiTao/data/output/",
  "table for output"
)
flags.DEFINE_string(
  "checkpointDir",
  "./Files/output/",
  "directory to save model"
)
flags.DEFINE_integer(
  "save_summary_steps",
  10,
  "save summary every save_summary_steps"
)
flags.DEFINE_integer(
  "keep_checkpoint_max",
  5,
  "maximum checkpoints to keep")
flags.DEFINE_integer(
  'num_threads',
  24,
  'the number of threads loading data.'
)
flags.DEFINE_integer(
  "num_epochs",
  50,
  "number of epochs to run")
flags.DEFINE_integer(
  "batch_size",
  128,
  "bath size")
flags.DEFINE_string(
  "optimizer",
  "adamw",
  "optimizer")
flags.DEFINE_float(
  "init_lr",
  0.0001,
  "initial learning rate"
)
flags.DEFINE_float(
  "l2_reg",
  0.01,
  "l2 norm weight"
)
flags.DEFINE_integer(
  "emb_dim",
  64,
  "embedding dimension"
)
flags.DEFINE_string(
  "agg_layer",
  "256,128",
  "the number of mlp layers in aggregation layer")
flags.DEFINE_string(
  "agg_act",
  "relu",
  "which activation function used in agg_layer"
)
flags.DEFINE_string(
  "selected_cols",
  'user_id,content_id,click_cnt',
  "which columns to use"
)
flags.DEFINE_float(
  "dropout_prob",
  0.1,
  "dropout prob"
)
flags.DEFINE_float(
  "att_dropout_prob",
  0.1,
  "dropout prob in attention."
)
flags.DEFINE_integer(
  "every_n_iter",
  100,
  "show log every n iteration"
)
flags.DEFINE_integer(
  "vocab_size",
  21128,
  "the number of yuyi_chinese_vocab.txt"
)
flags.DEFINE_integer(
  "max_text_len",
  32,
  "max length of content/item/query title"
)
flags.DEFINE_integer(
  "num_warmup_steps",
  20000,
  "the number of warmup steps"
)
flags.DEFINE_integer(
  'train_data_len',
  128,
  "the length of train data"
)

flags.DEFINE_string(
  'content_text_encoder_type',
  'multi_att',
  'avg/multi_att/single_att/dot_att/cnn'
)
flags.DEFINE_string(
  'product_text_encoder_type',
  'avg',
  'avg/multi_att/single_att/dot_att/cnn'
)
flags.DEFINE_integer(
  'shuffle_size',
  10000,
  'shuffle size in tf.Data'
)
flags.DEFINE_string(
  'feature_transfer_type',
  'sub_dann',
  'inner/sub/mmd/sub_mmd/sub_dann'
)
flags.DEFINE_float(
  'margin',
  1.0,
  'margin in pairwise learning'
)
flags.DEFINE_string(
  'restore_model_dir',
  'ftp_dual_pairwise/20200106-160953/product_sub_True_hinge_1.0_dot_att_adamw_0.001_1e-05',
  'restore_model_dir'
)
flags.DEFINE_string(
  'target',
  'content',
  'content/product'
)
flags.DEFINE_boolean(
  'use_shuffle',
  True,
  'True/False'
)

FLAGS = flags.FLAGS


def write_params(params):
  model_dir = params['model_dir']
  path = os.path.join(model_dir, 'params.txt')
  with gfile.GFile(path, mode='w') as writer:
    for key in params:
      writer.write('{}: {}\n'.format(key, params[key]))
  print('write parameters into {}'.format(path))


class DUALBaseParams(object):
  def __init__(self, project_name):
    train_table, eval_table = None, None
    if FLAGS.tables:
      temp_tables = FLAGS.tables.split(',')
      if len(temp_tables) > 1:
        train_table, eval_table = temp_tables[0], temp_tables[1]
      elif len(temp_tables) > 0:
        train_table = temp_tables[0]
      else:
        raise ValueError('invalid tables: {}'.format(FLAGS.tables))
    print('train_table: {}'.format(train_table))
    print('eval_table: {}'.format(eval_table))

    if FLAGS.local:
      data_dir = './Files'
      train_table = os.path.join(data_dir, train_table)
      eval_table = os.path.join(data_dir, eval_table)

    num_epochs = FLAGS.num_epochs if 'train' in FLAGS.mode else 1
    # save ckpt at the end of every epoch
    save_checkpoints_steps = FLAGS.train_data_len // FLAGS.batch_size
    num_train_steps = FLAGS.train_data_len//FLAGS.batch_size * FLAGS.num_epochs
    output_dir = FLAGS.checkpointDir

    params = {
      'local': FLAGS.local,
      'task_index': FLAGS.task_index,
      'ps_hosts': FLAGS.ps_hosts,
      'worker_hosts': FLAGS.worker_hosts,
      'job_name': FLAGS.job_name,
      'mode': FLAGS.mode,
      'model_date': FLAGS.model_date,
      'train_table': train_table,
      'eval_table': eval_table,
      'output_table': FLAGS.outputs,
      'output_dir': output_dir,
      'save_summary_steps': FLAGS.save_summary_steps,
      'save_checkpoints_steps': save_checkpoints_steps,
      'keep_checkpoint_max': FLAGS.keep_checkpoint_max,
      'num_threads': FLAGS.num_threads,
      'num_epochs': num_epochs,
      'batch_size': FLAGS.batch_size,
      'optimizer': FLAGS.optimizer,
      'init_lr': FLAGS.init_lr,
      'l2_reg': FLAGS.l2_reg,
      'emb_dim': FLAGS.emb_dim,
      'agg_layer': FLAGS.agg_layer,
      'agg_act': FLAGS.agg_act,
      'selected_cols': FLAGS.selected_cols,
      'dropout_prob': FLAGS.dropout_prob,
      'att_dropout_prob': FLAGS.att_dropout_prob,
      'every_n_iter': FLAGS.every_n_iter,
      'max_text_len': FLAGS.max_text_len,
      'vocab_size': FLAGS.vocab_size,
      'num_warmup_steps': FLAGS.num_warmup_steps,
      'num_train_steps': num_train_steps,
      'content_text_encoder_type': FLAGS.content_text_encoder_type,
      'product_text_encoder_type': FLAGS.product_text_encoder_type,
      'train_data_len': FLAGS.train_data_len,
      'shuffle_size': FLAGS.shuffle_size,
      'use_shuffle': FLAGS.use_shuffle,
      'feature_transfer_type': FLAGS.feature_transfer_type,
      'margin': FLAGS.margin,
    }
    self.params = params
    self.params['project_name'] = project_name
    self.params['selected_cols'] = \
      'content_id,product_id,' \
      'content_title_input_ids,content_title_input_len,' \
      'product_title_input_ids,product_title_input_len,' \
      'content_image_feat,product_image_feat,' \
      'neg_content_id,neg_product_id,' \
      'neg_content_title_input_ids,neg_content_title_input_len,' \
      'neg_product_title_input_ids,neg_product_title_input_len,' \
      'neg_content_image_feat,neg_product_image_feat'

    if self.params['job_name'] != "":
      self.params['model_date'] = 'distributed_train'
    self.params['model_dir_list'] = []
    self.params['model_dir_list'].extend([
      self.params['use_shuffle']
      ]
    )

  def post_process(self):
    model_dir = self.get_model_dir()
    if self.params['mode'] == 'train':
      if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    self.params['model_dir'] = model_dir
    write_params(self.params)
    for key in self.params:
      print(key, self.params[key])

  def get_model_dir(self):
    sub_dir = '{}/{}/{}'.format(
      self.params['project_name'],
      self.params['model_date'],
      '_'.join(map(str, self.params['model_dir_list'][::-1]))
    )
    model_dir = os.path.join(self.params['output_dir'], sub_dir)
    return model_dir


class DUALTitleParams(DUALBaseParams):
  def __init__(self, project_name):
    super(DUALTitleParams, self).__init__(project_name)
    self.params['model_dir_list'].extend(
      [self.params['product_text_encoder_type'],
       self.params['content_text_encoder_type'],
       self.params['feature_transfer_type']])


class DUALImageParams(DUALTitleParams):
  def __init__(self, project_name):
    super(DUALImageParams, self).__init__(project_name)


class DualParams(DUALImageParams, DUALTitleParams):
  def __init__(self, project_name):
    super(DualParams, self).__init__(project_name)


class DUALInferParams(DualParams):
  def __init__(self, project_name):
    super(DUALInferParams, self).__init__(project_name)
    self.params['model_dir'] = os.path.join(FLAGS.checkpointDir,
                                            FLAGS.restore_model_dir)
    self.params['target'] = FLAGS.target
    self.params['selected_cols'] = \
      '{}_id,' \
      '{}_title_input_ids,{}_title_input_len,' \
      '{}_image_feat'.format(
        self.params['target'], self.params['target'],
        self.params['target'], self.params['target']
      )
