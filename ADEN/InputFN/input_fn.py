# -*- coding: utf-8 -*-
# author: Xusong Chen

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from Utils import utils


def _padding_seq(feature_seq, max_seq_len, batch_size,
                 sep=',', default_value='0'):
  temp_feature_seq = tf.reshape(feature_seq, [batch_size,])
  temp_feature_seq = tf.string_split(temp_feature_seq, delimiter=sep)
  feature_seq = tf.sparse_to_dense(
      temp_feature_seq.indices,
      [batch_size, max_seq_len],
      temp_feature_seq.values,
      default_value=default_value)
  return feature_seq


def _load_seq(feature_seq, shape, sep=',', dtype=tf.int32, local=True):

  if local:
    batch_size = shape[0]
    temp_feature_seq = tf.reshape(feature_seq, [batch_size,])
    temp_feature_seq = tf.string_split(temp_feature_seq, delimiter=sep)
    feature_seq = tf.reshape(temp_feature_seq.values, shape=shape)
    feature_seq = tf.string_to_number(feature_seq, dtype)
  else:
    flatten_len = np.asarray(shape[1:]).prod()
    temp_feature_seq = tf.trans_csv_to_dense(feature_seq, flatten_len, sep)
    feature_seq = tf.reshape(temp_feature_seq, shape=shape)
    if dtype == tf.int32:
      feature_seq = tf.cast(feature_seq, dtype=dtype)
  return feature_seq


class SeqBaseData(object):
  def __init__(self, params):
    # step 1. read hyper parameters
    self.batch_size = params['batch_size']
    self.num_epochs = params['num_epochs']
    self.selected_cols = params['selected_cols']
    self.max_seq_len = params['max_seq_len']
    self.num_threads = params['num_threads']
    self.local = params['local']
    self.max_text_len = params['max_text_len']
    self.to_number_cols = ['content_click', 'content_title_input_len']
    self.to_float_cols = []
    if params['train_data_len'] < params['shuffle_size']:
      self.shuffle_size = params['train_data_len']
    else:
      self.shuffle_size = params['shuffle_size'] #500000

    if self.local:
      self.num_threads = 0

  def __call__(self, filename, params, mode, slice_id=None, slice_count=None):
    # step 1. load data
    feature_cols = self.load_data(
      filename=filename,
      slice_id=slice_id,
      slice_count=slice_count,
      capacity=0,
      mode=mode
    )
    if self.local:
      feature_cols = tf.transpose(feature_cols)
    print(feature_cols)

    # split feature columns to get features and labels
    self.split_feature_cols(feature_cols)

    # preprocess feature for training
    self.preprocess()
    # features = tf.staged(self.features, capacity=5, num_threads=10)
    #  features = pai.data.prefetch(self.features, capacity=5, num_threads=5)
    return self.features, self.features['content_click']
    #return self.features, self.features['content_click']

  def split_feature_cols(self, feature_cols):
    features = {}
    for idx, feature_name in enumerate(self.selected_cols.split(',')):
      if feature_name in self.to_number_cols:
        features[feature_name] = tf.string_to_number(feature_cols[idx],
                                                     tf.int32)
      elif feature_name in self.to_float_cols:
        features[feature_name] = tf.string_to_number(feature_cols[idx],
                                                     tf.float32)
      else:
        features[feature_name] = feature_cols[idx]
    self.features = features
    self.labels = features['content_click']

  def load_data(self,
                filename,
                mode='train',
                slice_id=0,
                slice_count=1,
                capacity=0):

    record_defaults = [''] * len(self.selected_cols.split(','))

    if self.local:
      self.data = utils.local_load_data(filename, self.selected_cols, '\t')
      dataset = tf.data.Dataset.from_tensor_slices(self.data)
    else:
      dataset = tf.data.TableRecordDataset(
        filenames=filename,
        record_defaults=record_defaults,
        selected_cols=self.selected_cols,
        slice_id=slice_id,
        slice_count=slice_count,
        num_threads=self.num_threads,
        capacity=capacity)

    # repeat -> shuffle -> batch -> iterator
    if 'train' in mode:
      dataset = dataset.repeat(self.num_epochs)
      # It's better to set buffer_size > len(data_set)
      dataset = dataset.shuffle(self.shuffle_size)
      #dataset = dataset.shuffle(self.batch_size*1000)
    else:
      # we only run over evaluate dataset once.
      dataset = dataset.repeat(1)

    # omit the final small batch because tf.sparse_to_dense in _padding_seq()
    dataset = dataset.batch(self.batch_size, drop_remainder=True).prefetch(4)
    feature_cols = dataset.make_one_shot_iterator().get_next()
    return feature_cols

  def preprocess(self):
    col_name = 'content_title_input_ids'
    self.features[col_name] = _load_seq(
      feature_seq=self.features[col_name],
      shape=[self.batch_size, self.max_text_len],
      sep=',', dtype=tf.int32, local=self.local
    )

    col_name = 'content_image_feat'
    self.features[col_name] = _load_seq(
      feature_seq=self.features[col_name],
      shape=[self.batch_size, 512],
      sep=',', dtype=tf.float32, local=self.local
    )


class CidSeqBaseData(SeqBaseData):
  def __init__(self, params):
    super(CidSeqBaseData, self).__init__(params)
    self.to_number_cols.extend(['content_seq_len'])

  def preprocess(self):
    super(CidSeqBaseData, self).preprocess()
    col_name = 'content_seq_time_ids'
    self.features[col_name] = _load_seq(
      feature_seq=self.features[col_name],
      shape=[self.batch_size, self.max_seq_len],
      sep=',', dtype=tf.int32, local=self.local
    )


class CidSeqTitleBaseData(CidSeqBaseData):
  def __init__(self, params):
    super(CidSeqTitleBaseData, self).__init__(params)

  def preprocess(self):
    super(CidSeqTitleBaseData, self).preprocess()

    col_name = 'content_title_seq_input_len'
    self.features[col_name] = _load_seq(
      feature_seq=self.features[col_name],
      shape=[self.batch_size, self.max_seq_len],
      sep=',', dtype=tf.int32, local=self.local
    )

    col_name = 'content_title_seq_input_ids'
    self.features[col_name] = _load_seq(
      feature_seq=self.features[col_name],
      shape=[self.batch_size, self.max_seq_len, self.max_text_len],
      sep=',', dtype=tf.int32, local=self.local
    )


class CidSeqImageBaseData(CidSeqBaseData):
  def __init__(self, params):
    super(CidSeqImageBaseData, self).__init__(params)

  def preprocess(self):
    super(CidSeqImageBaseData, self).preprocess()
    col_name = 'content_image_seq_feat'
    self.features[col_name] = _load_seq(
      feature_seq=self.features[col_name],
      shape=[self.batch_size, self.max_seq_len, 512],
      sep=',', dtype=tf.float32, local=self.local
    )


class PidSeqBaseData(SeqBaseData):
  def __init__(self, params):
    super(PidSeqBaseData, self).__init__(params)
    self.to_number_cols.extend(['product_seq_len', 'product_seq_len_new'])

  def preprocess(self):
    super(PidSeqBaseData, self).preprocess()
    col_name = 'product_seq_time_ids'
    self.features[col_name] = _load_seq(
      feature_seq=self.features[col_name],
      shape=[self.batch_size, self.max_seq_len],
      sep=',', dtype=tf.int32, local=self.local
    )


class PidSeqTitleBaseData(PidSeqBaseData):
  def __init__(self, params):
    super(PidSeqTitleBaseData, self).__init__(params)

  def preprocess(self):
    super(PidSeqTitleBaseData, self).preprocess()
    col_name = 'product_title_seq_input_len'
    self.features[col_name] = _load_seq(
      feature_seq=self.features[col_name],
      shape=[self.batch_size, self.max_seq_len],
      sep=',', dtype=tf.int32, local=self.local
    )

    col_name = 'product_title_seq_input_ids'
    self.features[col_name] = _load_seq(
      feature_seq=self.features[col_name],
      shape=[self.batch_size, self.max_seq_len, self.max_text_len],
      sep=',', dtype=tf.int32, local=self.local
    )


class PidSeqImageBaseData(PidSeqBaseData):
  def __init__(self, params):
    super(PidSeqImageBaseData, self).__init__(params)

  def preprocess(self):
    super(PidSeqImageBaseData, self).preprocess()
    col_name = 'product_image_seq_feat'
    self.features[col_name] = _load_seq(
      feature_seq=self.features[col_name],
      shape=[self.batch_size, self.max_seq_len, 512],
      sep=',', dtype=tf.float32, local=self.local
    )


class MixedSeqTitleBaseData(CidSeqTitleBaseData, PidSeqTitleBaseData):
  def __init__(self, params):
    super(MixedSeqTitleBaseData, self).__init__(params)


class MixedSeqImageBaseData(CidSeqImageBaseData, PidSeqImageBaseData):
  def __init__(self, params):
    super(MixedSeqImageBaseData, self).__init__(params)


class MixedSeqMultiModalBaseData(MixedSeqImageBaseData, MixedSeqTitleBaseData):
  def __init__(self, params):
    super(MixedSeqMultiModalBaseData, self).__init__(params)


class MixedSeqMultiModalData(MixedSeqMultiModalBaseData):
  def __init__(self, params):
    super(MixedSeqMultiModalData, self).__init__(params)

  def preprocess(self):
    super(MixedSeqMultiModalData, self).preprocess()

    # col_name = 'product_cate_level1_id_seq'
    # self.features[col_name] = _load_seq(
    #   feature_seq=self.features[col_name],
    #   shape=[self.batch_size, self.max_seq_len],
    #   sep=',', dtype=tf.int32, local=self.local
    # )
    # self.features[col_name] = self.features[col_name][:, :self.pid_recent_k]


class CMFData(SeqBaseData):
  def __init__(self, params):
    super(CMFData, self).__init__(params)
    self.to_number_cols.extend(['product_title_input_len',
                                'content_click', 'product_click'])

  def preprocess(self):
    super(CMFData, self).preprocess()
    col_name = 'product_title_input_ids'
    self.features[col_name] = _load_seq(
      feature_seq=self.features[col_name],
      shape=[self.batch_size, self.max_text_len],
      sep=',', dtype=tf.int32, local=self.local
    )

    col_name = 'product_image_feat'
    self.features[col_name] = _load_seq(
      feature_seq=self.features[col_name],
      shape=[self.batch_size, 512],
      sep=',', dtype=tf.float32, local=self.local
    )


class CoNetData(CMFData):
  def __init__(self, params):
    super(CoNetData, self).__init__(params)


class DINData(CidSeqImageBaseData, CidSeqTitleBaseData):
  def __init__(self, params):
    super(DINData, self).__init__(params)


class NCFData(SeqBaseData):
  def __init__(self, params):
    super(NCFData, self).__init__(params)


class MFData(NCFData):
  def __init__(self, params):
    super(MFData, self).__init__(params)


class PiNetData(MixedSeqMultiModalBaseData):
  def __init__(self, params):
    super(PiNetData, self).__init__(params)

  def preprocess(self):
    super(PiNetData, self).preprocess()
    col_name = 'posb'
    self.features[col_name] = _load_seq(
      feature_seq=self.features[col_name],
      shape=[self.batch_size, 20],
      sep=',', dtype=tf.int32, local=self.local
    )
    batch_index = tf.tile(tf.expand_dims(tf.range(0, self.batch_size), 1),
                          [1, self.max_seq_len])  #[B, S]
    self.features[col_name] = tf.stack([batch_index, self.features[col_name]],
                                       axis=2)


class DIENData(DINData):
  def __init__(self, params):
    super(DIENData, self).__init__(params)


class BSTData(DINData):
  def __init__(self, params):
    super(BSTData, self).__init__(params)

  def preprocess(self):
    super(BSTData, self).preprocess()


class YouTubeNetData(BSTData):
  def __init__(self, params):
    super(YouTubeNetData, self).__init__(params)


class CidSeqMultiModalData(CidSeqTitleBaseData, CidSeqImageBaseData):
  def __init__(self, params):
    super(CidSeqMultiModalData, self).__init__(params)


class PidSeqMultiModalData(PidSeqTitleBaseData, PidSeqImageBaseData):
  def __init__(self, params):
    super(PidSeqMultiModalData, self).__init__(params)



if __name__ == "__main__":
  pass
