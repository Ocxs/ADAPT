# -*- coding: utf-8 -*-
# author: Xusong Chen

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
from tensorflow.python.platform import gfile
import collections

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
  "SRD-1_train_data.txt,SRD-1_test_data.txt",
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
  "save summary at each save_summary_steps"
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
flags.DEFINE_string(
  "user_filename",
  "./Files/SRD-1_user_ids.txt",
  "user ids"
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
  "max_seq_len",
  20,
  "max sequence length"
)
flags.DEFINE_string(
  "seq_encoder_type",
  "trm_att",
  "avg/single_att/multi_att/dot_att/trm/trm_att")
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
  10,
  "the number of warmup steps"
)
flags.DEFINE_integer(
  'train_data_len',
  128,
  "the length of train data"
)
flags.DEFINE_string(
  'loss_type',
  'bce',
  'bce/wbce/bwbce/'
)
flags.DEFINE_string(
  'product_text_encoder_type',
  'avg',
  'avg/max/multi_att/single_att/dot_att/cnn'
)
flags.DEFINE_string(
  'content_text_encoder_type',
  'multi_att',
  'avg/max/multi_att/single_att/dot_att/cnn'
)
flags.DEFINE_integer(
  'shuffle_size',
  2048,
  'shuffle size in tf.Data'
)
flags.DEFINE_string(
  'domain_transfer_type',
  'concat',
  'early/cross/concat/gate/att/add'
)
flags.DEFINE_boolean(
  'seq_is_share_params',
  False,
  'True or False'
)
flags.DEFINE_string(
  'restore_model_path',
  'DUAL/SRD-1/sub_dann_multi_att_avg_True',
  'the pretrained model dir'
)
flags.DEFINE_boolean(
  'emb_trainable',
  True,
  'True/False'
)
flags.DEFINE_boolean(
  'use_target',
  True,
  'whether to use target as query of attention model in sequenece encoder'
)
flags.DEFINE_string(
  'target',
  'content',
  'content/product'
)
FLAGS = flags.FLAGS


def write_params(params):
  model_dir = params['model_dir']
  path = os.path.join(model_dir, 'params.txt')
  with gfile.GFile(path, mode='w') as writer:
    for key in params:
      writer.write('{}: {}\n'.format(key, params[key]))
  print('write parameters into {}'.format(path))


class SeqBaseParams(object):
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
    #
    # if FLAGS.outputs:
    #   output_table = os.path.join(FLAGS.outputs, '/{}'.format(FLAGS.model_date))
    #   print(output_table)
    # else:
    #   output_table = FLAGS.outputs

    if FLAGS.local:
      data_dir = '/Users/chenxs/Documents/Research/WeiTao/src/ADAPT/ADEN/Files'
      train_table = os.path.join(data_dir, train_table)
      eval_table = os.path.join(data_dir, eval_table)

    num_epochs = FLAGS.num_epochs if 'train' in FLAGS.mode else 1
    # save ckpt at the end of every epoch
    save_checkpoints_steps = FLAGS.train_data_len // FLAGS.batch_size
    num_train_steps = FLAGS.train_data_len//FLAGS.batch_size * FLAGS.num_epochs
    output_dir = FLAGS.checkpointDir

    params = collections.OrderedDict({
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
      'user_filename': FLAGS.user_filename,
      'emb_dim': FLAGS.emb_dim,
      'agg_layer': FLAGS.agg_layer,
      'agg_act': FLAGS.agg_act,
      'selected_cols': FLAGS.selected_cols,
      'dropout_prob': FLAGS.dropout_prob,
      'att_dropout_prob': FLAGS.att_dropout_prob,
      'every_n_iter': FLAGS.every_n_iter,
      'max_seq_len': FLAGS.max_seq_len,
      'max_text_len': FLAGS.max_text_len,
      'seq_encoder_type': FLAGS.seq_encoder_type,
      'vocab_size': FLAGS.vocab_size,
      'num_warmup_steps': FLAGS.num_warmup_steps,
      'num_train_steps': num_train_steps,
      'loss_type': FLAGS.loss_type,
      'product_text_encoder_type': FLAGS.product_text_encoder_type,
      'content_text_encoder_type': FLAGS.content_text_encoder_type,
      'train_data_len': FLAGS.train_data_len,
      'shuffle_size': FLAGS.shuffle_size,
      'use_target': FLAGS.use_target,
      'target': FLAGS.target
    })
    self.params = params
    self.params['project_name'] = project_name
    self.params['selected_cols'] = ['user_id']

    self.params['selected_cols'].extend([
      'content_click', 'content_id',
      'content_title_input_ids', 'content_title_input_len',
      'content_image_feat'
    ])


    if self.params['job_name'] != "":
      self.params['model_date'] = 'distributed_train'
    self.params['model_dir_list'] = []
    self.params['model_dir_list'].extend([
      self.params['emb_dim'],
      self.params['loss_type'],
      self.params['product_text_encoder_type'],
      self.params['content_text_encoder_type'],
      self.params['seq_encoder_type']]
    )

  def post_process(self):
    self.params['selected_cols'] = ','.join(self.params['selected_cols'])
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


class CidSeqBaseParams(SeqBaseParams):
  def __init__(self, project_name):
    super(CidSeqBaseParams, self).__init__(project_name)
    self.params['selected_cols'].extend(['content_seq_len',
                                         'content_seq_time_ids'])


class PidSeqBaseParams(SeqBaseParams):
  def __init__(self, project_name):
    super(PidSeqBaseParams, self).__init__(project_name)
    self.params['selected_cols'].extend(['product_seq_len',
                                         'product_seq_time_ids'])


class CidSeqTitleBaseParams(CidSeqBaseParams):
  def __init__(self, project_name):
    super(CidSeqTitleBaseParams, self).__init__(project_name)
    self.params['selected_cols'].extend([
      'content_title_seq_input_ids', 'content_title_seq_input_len',
    ])


class PidSeqTitleBaseParams(PidSeqBaseParams):
  def __init__(self, project_name):
    super(PidSeqTitleBaseParams, self).__init__(project_name)
    self.params['selected_cols'].extend([
      'product_title_seq_input_ids', 'product_title_seq_input_len'
    ])


class CidSeqImageBaseParams(CidSeqBaseParams):
  def __init__(self, project_name):
    super(CidSeqImageBaseParams, self).__init__(project_name)
    self.params['selected_cols'].extend([
      'content_image_seq_feat',
    ])


class PidSeqImageBaseParams(PidSeqBaseParams):
  def __init__(self, project_name):
    super(PidSeqImageBaseParams, self).__init__(project_name)
    self.params['selected_cols'].extend([
      'product_image_seq_feat'
    ])


class MixedSeqBaseParams(CidSeqBaseParams, PidSeqBaseParams):
  def __init__(self, project_name):
    super(MixedSeqBaseParams, self).__init__(project_name)
    self.params['seq_is_share_params'] = FLAGS.seq_is_share_params
    self.params['domain_transfer_type'] = FLAGS.domain_transfer_type

    self.params['model_dir_list'].extend([
      self.params['seq_is_share_params'],
      self.params['domain_transfer_type']
    ])


class MixedSeqTitleBaseParams(CidSeqTitleBaseParams,
                              PidSeqTitleBaseParams,
                              MixedSeqBaseParams):
  def __init__(self, project_name):
    super(MixedSeqTitleBaseParams, self).__init__(project_name)


class MixedSeqImageBaseParams(CidSeqImageBaseParams,
                            PidSeqImageBaseParams,
                            MixedSeqBaseParams):
  def __init__(self, project_name):
    super(MixedSeqImageBaseParams, self).__init__(project_name)


class MixedSeqMultiModalParams(MixedSeqImageBaseParams,
                               MixedSeqTitleBaseParams):
  def __init__(self, project_name):
    super(MixedSeqMultiModalParams, self).__init__(project_name)


class SeqFinetuneParams(SeqBaseParams):
  def __init__(self, project_name):
    super(SeqFinetuneParams, self).__init__(project_name)
    self.params['emb_trainable'] = FLAGS.emb_trainable
    restore_model_path = os.path.join(
      self.params['output_dir'], FLAGS.restore_model_path
    )
    if os.path.isfile('{}.index'.format(restore_model_path)):
      pass
    else:
      has_ckpt = tf.train.get_checkpoint_state(restore_model_path)
      if has_ckpt:
        restore_model_path = has_ckpt.model_checkpoint_path
      else:
        raise ValueError("No ckpt found in {}".format(restore_model_path))

    self.params['restore_model_path'] = restore_model_path
    if self.params['mode'] != 'train':
      self.params['restore_model_path'] = None


class CidSeqTitleBaseFtParams(CidSeqTitleBaseParams, SeqFinetuneParams):
  def __init__(self, project_name):
    super(CidSeqTitleBaseFtParams, self).__init__(project_name)


class PidSeqTitleBaseFtParams(PidSeqTitleBaseParams, SeqFinetuneParams):
  def __init__(self, project_name):
    super(PidSeqTitleBaseFtParams, self).__init__(project_name)


class CidSeqImageBaseFtParams(CidSeqImageBaseParams, SeqFinetuneParams):
  def __init__(self, project_name):
    super(CidSeqImageBaseFtParams, self).__init__(project_name)


class PidSeqImageBaseFtParams(PidSeqImageBaseParams, SeqFinetuneParams):
  def __init__(self, project_name):
    super(PidSeqImageBaseFtParams, self).__init__(project_name)


class MixedSeqTitleBaseFtParams(MixedSeqTitleBaseParams, SeqFinetuneParams):
  def __init__(self, project_name):
    super(MixedSeqTitleBaseFtParams, self).__init__(project_name)


class MixedSeqImageBaseFtParams(MixedSeqImageBaseParams, SeqFinetuneParams):
  def __init__(self, project_name):
    super(MixedSeqImageBaseFtParams, self).__init__(project_name)


class MixedSeqMultiModalFtParams(MixedSeqMultiModalParams,
                                 SeqFinetuneParams):
  def __init__(self, project_name):
    super(MixedSeqMultiModalFtParams, self).__init__(project_name)


# ----------- params of baseline model -----------
class CMFParams(SeqBaseParams):
  def __init__(self, project_name):
    super(CMFParams, self).__init__(project_name)

    self.params['selected_cols'] = [
      'user_id', 'content_id', 'content_click', 'product_click',
      'content_image_feat',
      'content_title_input_ids', 'content_title_input_len',
      'product_image_feat',
      'product_title_input_ids', 'product_title_input_len'
    ]


class CoNetParams(CMFParams):
  def __init__(self, project_name):
    super(CoNetParams, self).__init__(project_name)


class DINParams(CidSeqImageBaseParams, CidSeqTitleBaseParams):
  def __init__(self, project_name):
    super(DINParams, self).__init__(project_name)
    self.params['model_dir_list'] = []
    self.params['model_dir_list'].extend([
      self.params['emb_dim'],
      self.params['loss_type'],
      self.params['product_text_encoder_type'],
      self.params['content_text_encoder_type'],
      self.params['agg_act']])


class NCFParams(SeqBaseParams):
  def __init__(self, project_name):
    super(NCFParams, self).__init__(project_name)
    self.params['model_dir_list'] = []
    self.params['model_dir_list'].extend([
      self.params['emb_dim'],
      self.params['loss_type'],
      self.params['product_text_encoder_type'],
      self.params['content_text_encoder_type'],
      self.params['agg_act']]
    )


class DIENParams(DINParams):
  def __init__(self, project_name):
    super(DIENParams, self).__init__(project_name)


class MFParams(NCFParams):
  def __init__(self, project_name):
    super(MFParams, self).__init__(project_name)


class PiNetParams(MixedSeqMultiModalParams):
  def __init__(self, project_name):
    super(PiNetParams, self).__init__(project_name)
    self.params['selected_cols'].extend(['posb'])
    self.params['model_dir_list'] = []
    self.params['model_dir_list'].extend([
      self.params['emb_dim'],
      self.params['loss_type'],
      self.params['product_text_encoder_type'],
      self.params['content_text_encoder_type']
      ]
    )


class BSTParams(DINParams):
  def __init__(self, project_name):
    super(BSTParams, self).__init__(project_name)


class YouTubeNetParams(DINParams):
  def __init__(self, project_name):
    super(YouTubeNetParams, self).__init__(project_name)


# Ablation study
class CidSeqMultiModalParams(CidSeqTitleBaseParams, CidSeqImageBaseParams):
  def __init__(self, project_name):
    super(CidSeqMultiModalParams, self).__init__(project_name)


class PidSeqMultiModalParams(PidSeqTitleBaseParams, PidSeqImageBaseParams):
  def __init__(self, project_name):
    super(PidSeqMultiModalParams, self).__init__(project_name)


class CidSeqMultiModalFtParams(CidSeqMultiModalParams, SeqFinetuneParams):
  def __init__(self, project_name):
    super(CidSeqMultiModalFtParams, self).__init__(project_name)


class PidSeqMultiModalFtParams(PidSeqMultiModalParams, SeqFinetuneParams):
  def __init__(self, project_name):
    super(PidSeqMultiModalFtParams, self).__init__(project_name)