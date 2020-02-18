# -*- coding: utf-8 -*-
# author: Xusong Chen
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
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


class DUALBaseData(object):
  def __init__(self, params):
    # step 1. read hyper parameters
    self.batch_size = params['batch_size']
    self.num_epochs = params['num_epochs']
    self.selected_cols = params['selected_cols']
    self.num_threads = params['num_threads']
    self.local = params['local']
    self.max_text_len = params['max_text_len']
    self.to_number_cols = ['content_title_input_len',
                           'product_title_input_len',
                           'neg_content_title_input_len',
                           'neg_product_title_input_len',]
    self.to_float_cols = []
    if params['train_data_len'] < params['shuffle_size']:
      self.shuffle_size = params['train_data_len']
    else:
      self.shuffle_size = params['shuffle_size'] #500000

    self.col_indices = range(16)

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
    return self.features, self.labels

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
    self.labels = None

  def load_data(self,
                filename,
                mode='train',
                slice_id=0,
                slice_count=1,
                capacity=0):

    record_defaults = [''] * len(self.selected_cols.split(','))

    if self.local:
      self.data = utils.local_load_data(filename, self.col_indices, '\t')
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
    dataset = dataset.batch(self.batch_size, drop_remainder=True)
    feature_cols = dataset.make_one_shot_iterator().get_next()
    return feature_cols

  def preprocess(self):
    self.features['content_title_input_ids'] = _load_seq(
      feature_seq=self.features['content_title_input_ids'],
      shape=[self.batch_size, self.max_text_len],
      sep=',', dtype=tf.int32, local=self.local
    )
    self.features['product_title_input_ids'] = _load_seq(
      feature_seq=self.features['product_title_input_ids'],
      shape=[self.batch_size, self.max_text_len],
      sep=',', dtype=tf.int32, local=self.local
    )
    self.features['neg_content_title_input_ids'] = _load_seq(
      feature_seq=self.features['neg_content_title_input_ids'],
      shape=[self.batch_size, self.max_text_len],
      sep=',', dtype=tf.int32, local=self.local
    )
    self.features['neg_product_title_input_ids'] = _load_seq(
      feature_seq=self.features['neg_product_title_input_ids'],
      shape=[self.batch_size, self.max_text_len],
      sep=',', dtype=tf.int32, local=self.local
    )
    self.features['content_image_feat'] = _load_seq(
      feature_seq=self.features['content_image_feat'],
      shape=[self.batch_size, 512],
      sep=',', dtype=tf.float32, local=self.local
    )
    self.features['product_image_feat'] = _load_seq(
      feature_seq=self.features['product_image_feat'],
      shape=[self.batch_size, 512],
      sep=',', dtype=tf.float32, local=self.local
    )
    self.features['neg_content_image_feat'] = _load_seq(
      feature_seq=self.features['neg_content_image_feat'],
      shape=[self.batch_size, 512],
      sep=',', dtype=tf.float32, local=self.local
    )
    self.features['neg_product_image_feat'] = _load_seq(
      feature_seq=self.features['neg_product_image_feat'],
      shape=[self.batch_size, 512],
      sep=',', dtype=tf.float32, local=self.local
    )


class DUALInferData(DUALBaseData):
  def __init__(self, params):
    super(DUALInferData, self).__init__(params)
    self.target = params['target']
    self.col_indices = [0, 1, 2, 3, 4]
    self.to_number_cols.extend([
      '{}_title_input_len'.format(self.target)
    ])

  def preprocess(self):
    self.features['{}_title_input_ids'.format(self.target)] = _load_seq(
      feature_seq=self.features['{}_title_input_ids'.format(self.target)],
      shape=[self.batch_size, self.max_text_len],
      sep=',', dtype=tf.int32, local=self.local
    )

    self.features['{}_image_feat'.format(self.target)] = _load_seq(
      feature_seq=self.features['{}_image_feat'.format(self.target)],
      shape=[self.batch_size, 512],
      sep=',', dtype=tf.float32, local=self.local
    )

