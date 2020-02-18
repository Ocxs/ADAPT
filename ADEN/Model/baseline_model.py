# -*- coding: utf-8 -*-
# author: Xusong Chen
# email: cxs2016@mail.ustc.edu.cn

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from Utils import utils
from Utils import bert
from Utils import optimization
from Utils import PiNet
from Utils import dien_rnn
from Utils import dien_utils
from Model.ADEN import *


class CMFModel(object):
  def __init__(self, params):
    # step 1. read hyper parameters
    self.batch_size = params['batch_size']
    self.emb_dim = params['emb_dim']
    self.every_n_iter = params['every_n_iter']
    self.init_lr = params['init_lr']
    self.optimizer = params['optimizer']
    self.l2_reg = params['l2_reg']
    self.agg_layer = params['agg_layer']
    self.agg_act = params['agg_act']
    self.vocab_size = params['vocab_size']
    self.num_train_steps = params['num_train_steps']
    self.num_warmup_steps = params['num_warmup_steps']
    self.model_dir = params['model_dir']

    self.max_seq_len = params['max_seq_len']
    self.max_text_len = params['max_text_len']
    self.max_cate_text_len = params['max_cate_text_len']
    self.seq_encoder_type = params['seq_encoder_type']
    self.loss_type = params['loss_type']
    self.content_text_encoder_type = params['content_text_encoder_type']
    self.product_text_encoder_type = params['product_text_encoder_type']
    self.use_target = params['use_target']
    self.user_filename = params['user_filename']

    self.window_size = params['window_size']
    self.text_att_act = params['text_att_act']
    self.text_num_attention_heads = params['text_num_attention_heads']
    self.text_dot_att_use_fc = params['text_dot_att_use_fc']
    self.text_encoder_params = {
      'window_size': self.window_size,
      'att_act': self.text_att_act,
      'num_attention_heads': self.text_num_attention_heads,
      'dot_att_use_fc': self.text_dot_att_use_fc
    }

    self.seq_att_act = params['seq_att_act']
    self.seq_num_attention_heads = params['seq_num_attention_heads']
    self.seq_dot_att_use_fc = params['seq_dot_att_use_fc']
    self.seq_encoder_params = {
      'att_act': self.seq_att_act,
      'num_attention_heads': self.seq_num_attention_heads,
      'dot_att_use_fc': self.seq_dot_att_use_fc
    }

  def __call__(self, features, labels, mode, params):
    training = True if 'train' in mode else False
    if training:
      dropout_prob = params['dropout_prob']
      att_dropout_prob = params['att_dropout_prob']
      print('train phase: dropout_prob {:.4f}'.format(dropout_prob))
    else:
      dropout_prob = 0.0
      att_dropout_prob = 0.0
      print('eval phase: dropout_prob {:.4f}'.format(dropout_prob))

    self.content_labels = tf.cast(features['content_click'] >= 1, tf.int32)
    self.product_labels = tf.cast(features['product_click'] >= 1, tf.int32)

    # step 1. build embedding layer
    with tf.variable_scope('embedding_layer'):
      self.build_embedding_layer()

    # step 2. build input layer
    with tf.variable_scope('input_layer'):
      self.build_input_layer(features, dropout_prob)

    # step 3. build user interest layer
    with tf.variable_scope('user_interest_layer'):
      self.build_user_interest_layer(
        dropout_prob=dropout_prob,
        att_dropout_prob=att_dropout_prob,
        activation='relu'
      )

    # step 4. build aggregation layer for final prediction
    with tf.variable_scope('agg_layer'):
      self.build_aggregation_layer(dropout_prob=dropout_prob)

    # step 5. build EstimatorSpec
    if mode == tf.estimator.ModeKeys.PREDICT:
      self.build_predict_estimator_spec(mode, features)
      return self.pred_est_spec

    self.build_loss_layer()
    if mode == tf.estimator.ModeKeys.TRAIN:
      self.build_train_operation()
    self.build_metric_fn()

    if mode == tf.estimator.ModeKeys.TRAIN:
      self.build_train_estimator_spec(mode)
      return self.train_est_spec

    elif mode == tf.estimator.ModeKeys.EVAL:
      self.build_eval_estimator_spec(mode)
      return self.eval_est_spec
    else:
      raise ValueError('invalid mode {}'.format(mode))

  def build_embedding_layer(self):
    self.word_embeddings = utils.build_embeddings(
      emb_name='word_embeddings',
      vocab_size=self.vocab_size,
      emb_dim=self.emb_dim
    )
    self.word_position_embeddings = utils.build_embeddings(
      emb_name='word_position_embeddings',
      vocab_size=32,
      emb_dim=self.emb_dim
    )
    self.user_columns = utils.build_columns(
      column_key='user_id',
      vocab_file=self.user_filename,
      emb_dim=self.emb_dim
    )


  def load_title_emb_and_mask(self,
                              emb_input_ids,
                              emb_input_len,
                              query,
                              encoder_type,
                              name='text_encoder',
                              reuse=True,
                              dropout_prob=0.0,
                              use_4d=False):
    emb = tf.nn.embedding_lookup(self.word_embeddings, emb_input_ids)
    mask = tf.sequence_mask(emb_input_len, self.max_text_len, tf.float32)
    if use_4d:
      emb = utils.text_encoder_4d(
        emb, mask, encoder_type, name, reuse,
        position_emb=self.word_position_embeddings,
        dropout_prob=dropout_prob,
        query=query,
        **self.text_encoder_params
      )  # [B, s, D]
    else:
      emb = utils.text_encoder(
        emb, mask, encoder_type, name, reuse,
        position_emb=self.word_position_embeddings,
        dropout_prob=dropout_prob,
        query=query,
        **self.text_encoder_params
      )  # [B, D]
    return emb

  def build_input_layer(self, features, dropout_prob):
    self.text_encoder_name = 'title'
    content_title_emb = self.load_title_emb_and_mask(
      features['content_title_input_ids'],
      features['content_title_input_len'],
      None, self.content_text_encoder_type,
      self.text_encoder_name, False, dropout_prob,
    )
    product_title_emb = self.load_title_emb_and_mask(
      features['product_title_input_ids'],
      features['product_title_input_len'],
      None, self.product_text_encoder_type,
      self.text_encoder_name, True, dropout_prob,
    )

    content_image_emb = tf.layers.dense(
      features['content_image_feat'], self.emb_dim, name='img_fc'
    )
    product_image_emb = tf.layers.dense(
      features['product_image_feat'], self.emb_dim, name='img_fc', reuse=True
    )

    user_id_emb = tf.feature_column.input_layer(features, self.user_columns)

    self.content_emb = content_title_emb + content_image_emb
    self.product_emb = product_title_emb + product_image_emb
    self.user_emb = user_id_emb

  def build_user_interest_layer(self,
                                dropout_prob=0.0,
                                att_dropout_prob=0.0,
                                activation='relu'):
    pass

  def build_aggregation_layer(self, dropout_prob=0.0):
    self.content_logits = tf.reduce_sum(self.user_emb*self.content_emb, axis=1)
    self.product_logits = tf.reduce_sum(self.user_emb*self.product_emb, axis=1)
    self.logits = self.content_logits

  def build_loss_layer(self):
    content_labels = tf.cast(self.content_labels, tf.float32)
    content_loss = utils.compute_loss(
      content_labels, self.content_logits, self.loss_type, 1.0
    )

    product_labels = tf.cast(self.product_labels, tf.float32)
    product_loss = utils.compute_loss(
      product_labels, self.product_logits, self.loss_type, 1.0
    )

    self.content_loss = content_loss
    self.product_loss = product_loss
    self.loss = content_loss + product_loss
    tf.summary.scalar('loss/content_ctr_loss', content_loss)
    tf.summary.scalar('loss/product_ctr_loss', product_loss)
    tf.summary.scalar('loss/ctr_loss', self.loss)

  def build_train_operation(self, train_vars=None):
    for var in tf.trainable_variables():
      print(var.name)

    learning_rate = optimization.warmup(
      init_lr=self.init_lr,
      num_train_steps=self.num_train_steps,
      num_warmup_steps=self.num_warmup_steps
    )
    if self.optimizer.lower() == 'adamw':
      train_op = optimization.adam_weight_decay_optimizer(
        loss=self.loss,
        learning_rate=learning_rate,
        l2_reg=self.l2_reg,
        train_vars=train_vars
      )
    else:
      train_op = optimization.custom_optimizer(
        loss=self.loss,
        optimizer_type=self.optimizer.lower(),
        learning_rate=learning_rate,
        l2_reg=self.l2_reg,
        train_vars=train_vars
      )
    tf.summary.scalar('learning_rate', learning_rate)
    self.train_op = train_op

  def build_metric_fn(self):
    content_ctr_probs = tf.nn.sigmoid(self.content_logits)
    roc_auc = tf.metrics.auc(self.content_labels, content_ctr_probs)

    predictions = tf.cast(tf.round(content_ctr_probs), tf.int32)
    accuracy = tf.metrics.accuracy(self.content_labels, predictions)
    # precision = tf.metrics.precision(self.labels, predictions)
    # recall = tf.metrics.recall(self.labels, predictions)
    # f1_func = lambda p, r: p * r / (p + r) * 2
    # f1 = f1_func(precision[0], recall[0]), f1_func(precision[1], recall[1])

    # self.precision = precision
    # self.recall = recall
    # self.f1 = f1
    self.roc_auc = roc_auc
    self.accuracy = accuracy

  def build_train_estimator_spec(self, mode):
    train_print = {
      'loss': self.loss,
      'product_loss': self.product_loss,
      'content_loss': self.content_loss,
      'roc_auc': self.roc_auc[1]
    }
    tf.summary.scalar('metric/train/roc_auc', self.roc_auc[1])
    tf.summary.scalar('metric/train/accuracy', self.accuracy[1])
    # tf.summary.scalar('metric/train/precision', self.precision[1])
    # tf.summary.scalar('metric/train/recall', self.recall[1])
    # tf.summary.scalar('metric/train/f1', self.f1[1])
    train_hook = tf.train.LoggingTensorHook(
      train_print, every_n_iter=self.every_n_iter)
    train_est_spec = tf.estimator.EstimatorSpec(
      mode, loss=self.loss, train_op=self.train_op, training_hooks=[train_hook]
    )
    self.train_est_spec = train_est_spec

  def build_eval_estimator_spec(self, mode):
    eval_est_spec = tf.estimator.EstimatorSpec(
      mode,
      loss=self.loss,
      eval_metric_ops={
        'metric/test/content_accuracy': self.accuracy,
        'metric/test/content_roc_auc': self.roc_auc,
      }
    )
    utils.stat_eval_results(self.model_dir, 'eval_result_temp.txt')
    self.eval_est_spec = eval_est_spec

  def build_predict_estimator_spec(self, mode, features):
    ctr_probs = tf.nn.sigmoid(self.logits)
    predict_results = {
      'user_id': features['user_id'],
      'content_id': features['content_id'],
      'content_click': features['content_click'],
      'ctr_prob': ctr_probs
    }
    pred_est_spec = tf.estimator.EstimatorSpec(mode, predict_results)
    self.pred_est_spec = pred_est_spec


class CoNetModel(CMFModel):
  def __init__(self, params):
    super(CoNetModel, self).__init__(params)

  def build_aggregation_layer(self, dropout_prob=0.0):
    content_agg_vec, product_agg_vec = utils.collaborative_cross_networks(
      input1=tf.concat([self.user_emb, self.content_emb], axis=1),
      input2=tf.concat([self.user_emb, self.product_emb], axis=1),
      mlp_layers=self.agg_layer,
      activation=self.agg_act,
      dropout_prob=dropout_prob
    )
    content_logits = tf.layers.dense(content_agg_vec, 1)
    product_logits = tf.layers.dense(product_agg_vec, 1)
    self.content_logits = tf.squeeze(content_logits)
    self.product_logits = tf.squeeze(product_logits)


class PiNetModel(MixedSeqMultiModalModel):
  def __init__(self, params):
    super(PiNetModel, self).__init__(params)
    self.num_rnn_layers = 1
    self.num_members = 1

  def build_input_layer(self, features, dropout_prob):
    super(PiNetModel, self).build_input_layer(features, dropout_prob)
    self.content_seq_len = features['content_seq_len']
    self.product_seq_len = features['product_seq_len']
    self.posB = features['posb']

  def build_user_interest_layer(self,
                                content_seq_emb=None,
                                content_seq_mask=None,
                                product_seq_emb=None,
                                product_seq_mask=None,
                                seq_query=None,
                                dropout_prob=0.0,
                                att_dropout_prob=0.0,
                                activation='relu'):

    content_seq_emb = self.content_seq_title_emb + self.content_seq_image_emb
    product_seq_emb = self.product_seq_title_emb + self.product_seq_image_emb

    keep_prob = 1.0 - dropout_prob

    with tf.variable_scope('encoder_A'):
      encoder_output_A, encoder_state_A = self.encoder(
        content_seq_emb, self.content_seq_len, self.emb_dim,
        self.num_rnn_layers, keep_prob
      )
    with tf.variable_scope('encoder_B'):
      encoder_output_B, encoder_state_B = self.encoder(
        product_seq_emb, self.product_seq_len, self.emb_dim,
        self.num_rnn_layers, keep_prob
      )
    with tf.variable_scope('filter_B'):
      filter_output_B, filter_state_B = self.filter_B(
        encoder_output_A, encoder_output_B, self.product_seq_len,
        self.posB, self.num_members, self.emb_dim, self.emb_dim,
        self.num_rnn_layers, keep_prob
      )

    with tf.variable_scope('transfer_B'):
      transfer_output_B, transfer_state_B = self.transfer_B(
        filter_output_B, self.product_seq_len, self.emb_dim,
        self.num_rnn_layers, keep_prob
      )
    with tf.name_scope('prediction_A'):
      user_emb = self.prediction_A(transfer_state_B, encoder_state_A, keep_prob)

    self.user_emb = user_emb

  def get_gru_cell(self, hidden_size, keep_prob):
    gru_cell = tf.contrib.rnn.GRUCell(
      hidden_size,
      kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False)
    )
    gru_cell = tf.contrib.rnn.DropoutWrapper(
      gru_cell, input_keep_prob=keep_prob,
      output_keep_prob=keep_prob, state_keep_prob=keep_prob
    )
    return gru_cell

  def get_filter_cell(self, hidden_size, member_embedding, keep_prob):
    filter_cell = PiNet.FilterCell(hidden_size, member_embedding)
    filter_cell = tf.contrib.rnn.DropoutWrapper(filter_cell,
                                                input_keep_prob=keep_prob,
                                                output_keep_prob=keep_prob,
                                                state_keep_prob=keep_prob)
    return filter_cell

  def encoder(self, embbed_seq_A, len_A, hidden_size,
                num_layers, keep_prob):
    encoder_cell_A = tf.contrib.rnn.MultiRNNCell(
      [self.get_gru_cell(hidden_size, keep_prob)
       for _ in range(num_layers)]
    )
    encoder_output_A, encoder_state_A = tf.nn.dynamic_rnn(
      encoder_cell_A, embbed_seq_A, sequence_length=len_A, dtype=tf.float32
    )
    return encoder_output_A, encoder_state_A

  def filter_B(self, encoder_output_A, encoder_output_B, len_B, pos_B,
               num_members, embedding_size, hidden_size, num_layers, keep_prob):
    zero_state = tf.zeros(
      dtype=tf.float32,
      shape=(tf.shape(encoder_output_A)[0], 1, tf.shape(encoder_output_A)[-1])
    )  # zero_state=[batch_size, 1, hidden_size]
    print(zero_state)

    # encoder_output=[batch_size,timestamp_A+1,hidden_size]
    encoder_output = tf.concat([zero_state, encoder_output_A], axis=1)
    print(encoder_output)

    # select_output_A=[batch_size,timestamp_B,hidden_size]
    select_output_A = tf.gather_nd(encoder_output, pos_B)
    print(select_output_A)
    # filter_input_B=[batch_size,timestamp_B,hidden_size+hidden_size]
    filter_input_B = tf.concat([encoder_output_B, select_output_A], axis=-1)
    print(filter_input_B)
    member_embedding_B = tf.get_variable(
      dtype=tf.float32, name='member_embedding_B',
      shape=[num_members, embedding_size],
      initializer=tf.contrib.layers.xavier_initializer(uniform=False)
    )
    print(member_embedding_B)
    filter_cell_B = tf.contrib.rnn.MultiRNNCell(
      [self.get_filter_cell(hidden_size, member_embedding_B, keep_prob)
       for _ in range(num_layers)]
    )

    # filter_output_B=[batch_size,timestamp_B,hidden_size]，
    # filter_state_B=[batch_size,hidden_size]
    filter_output_B, filter_state_B = tf.nn.dynamic_rnn(
      filter_cell_B, filter_input_B, sequence_length=len_B, dtype=tf.float32
    )
    print(filter_output_B)
    print(filter_state_B)
    return filter_output_B, filter_state_B

  def transfer_B(self, filter_output_B, len_B, hidden_size, num_layers,
                 keep_prob):
    with tf.variable_scope('transfer_B'):
      transfer_cell_B = tf.contrib.rnn.MultiRNNCell(
        [self.get_gru_cell(hidden_size, keep_prob) for _ in range(num_layers)])
      # transfer_output_B=[batch_size,timestamp_B,hidden_size],
      # transfer_state_B=([batch_size,hidden_size]*num_layers)
      transfer_output_B, transfer_state_B = tf.nn.dynamic_rnn(
        transfer_cell_B, filter_output_B,
        sequence_length=len_B, dtype=tf.float32
      )
      print(transfer_output_B)
      print(transfer_state_B)
    return transfer_output_B, transfer_state_B

  def prediction_A(self, transfer_state_B, encoder_state_A, keep_prob):
    with tf.variable_scope('prediction_A'):
      concat_output = tf.concat([transfer_state_B[-1], encoder_state_A[-1]],
                                axis=-1)
      print(concat_output)
      # concat_output=[batch_size,hidden_size+hidden_size]
      user_emb = tf.nn.dropout(concat_output, keep_prob)
    return user_emb


class DINModel(CidSeqImageBaseModel, CidSeqTitleBaseModel):
  def __init__(self, params):
    super(DINModel, self).__init__(params)

  def build_input_layer(self, features, dropout_prob):
    super(DINModel, self).build_input_layer(features, dropout_prob)
    self.content_seq_len = features['content_seq_len']

  def build_user_interest_layer(self,
                                seq_query=None,
                                dropout_prob=0.0,
                                att_dropout_prob=0.0,
                                activation='relu'):

    content_seq_emb = self.content_seq_title_emb + self.content_seq_image_emb

    user_emb = utils.din_attention(
      queries=self.target_emb,
      keys=content_seq_emb,
      keys_length=self.content_seq_len
    )
    self.user_emb = user_emb


class NCFModel(SeqBaseModel):
  def __init__(self, params):
    super(NCFModel, self).__init__(params)
    self.user_filename = params['user_filename']

  def build_embedding_layer(self):
    super(NCFModel, self).build_embedding_layer()
    self.user_columns = utils.build_columns(
      column_key='user_id',
      vocab_file=self.user_filename,
      emb_dim=self.emb_dim
    )

  def build_input_layer(self, features, dropout_prob):
    super(NCFModel, self).build_input_layer(features, dropout_prob)
    user_id_emb = tf.feature_column.input_layer(features, self.user_columns)
    self.user_id_emb = user_id_emb

  def build_user_interest_layer(self,
                                seq_query=None,
                                dropout_prob=0.0,
                                att_dropout_prob=0.0,
                                activation='relu'):
    self.user_emb = self.user_id_emb

  def build_aggregation_layer(self, dropout_prob=0.0):
    logits = utils.neural_collaborative_filtering(
      user_emb=self.user_emb,
      target_emb=self.target_emb,
      emb_size=self.emb_dim,
      mlp_layers=self.agg_layer,
      mlp_activation=self.agg_act,
      dropout_prob=dropout_prob
    )
    self.logits = logits


class MFModel(NCFModel):
  def __init__(self, params):
    super(MFModel, self).__init__(params)

  def build_aggregation_layer(self, dropout_prob=0.0):
    logits = tf.reduce_sum(self.user_emb * self.target_emb, axis=1)
    self.logits = logits


class DIENModel(DINModel, NCFModel):
  def __init__(self, params):
    super(DIENModel, self).__init__(params)

  def build_input_layer(self, features, dropout_prob):
    super(DIENModel, self).build_input_layer(features, dropout_prob)
    self.content_seq_len = features['content_seq_len']

  def build_user_interest_layer(self,
                                seq_query=None,
                                dropout_prob=0.0,
                                att_dropout_prob=0.0,
                                activation='relu'):
    content_seq_emb = self.content_seq_title_emb + self.content_seq_image_emb
    self.content_seq_emb = content_seq_emb

    with tf.variable_scope('rnn'):
      rnn_outputs, _ = dien_rnn.dynamic_rnn(
        tf.nn.rnn_cell.GRUCell(self.emb_dim),
        inputs=content_seq_emb,
        sequence_length=self.content_seq_len,
        dtype=tf.float32,
        scope='gru1'
      )
    with tf.variable_scope('attention_layer_1'):
      att_outputs, alphas = dien_utils.din_fcn_attention(
        query=self.target_emb,
        facts=rnn_outputs,
        mask=self.content_seq_mask,
        softmax_stag=1,
        stag='1_1',
        mode='LIST',
        return_alphas=True
      )
    with tf.variable_scope('rnn_2'):
      rnn_outputs2, final_state2 = dien_rnn.dynamic_rnn(
        dien_utils.VecAttGRUCell(self.emb_dim),
        inputs=rnn_outputs,
        att_scores=tf.expand_dims(alphas, -1),
        sequence_length=self.content_seq_len,
        dtype=tf.float32,
        scope='gru2'
      )
    self.user_emb = final_state2
    self.rnn_outputs = rnn_outputs

  def build_aggregation_layer(self, dropout_prob=0.0):
    seq_emb_sum = tf.reduce_sum(self.content_seq_emb, 1)
    inp = tf.concat(
      [self.user_id_emb,
       self.target_emb,
       seq_emb_sum,
       self.target_emb*seq_emb_sum,
       self.user_emb],
      axis=1
    )
    logits = dien_utils.build_fcn_net(inp, use_dice=True)
    self.logits = logits


class BSTModel(DINModel):
  def __init__(self, params):
    super(BSTModel, self).__init__(params)

  def build_user_interest_layer(self,
                                seq_query=None,
                                dropout_prob=0.0,
                                att_dropout_prob=0.0,
                                activation='relu'):

    content_seq_emb = self.content_seq_title_emb + self.content_seq_time_emb + \
                      self.content_seq_image_emb

    batch_size, to_seq_len, emb_dim = content_seq_emb.get_shape().as_list()
    trm_mask = bert.create_attention_mask_from_input_mask(content_seq_emb,
                                                          self.content_seq_mask)
    user_emb = bert.transformer_model(
      input_tensor=content_seq_emb,
      attention_mask=trm_mask,
      hidden_size=emb_dim,
      num_hidden_layers=1,
      num_attention_heads=8,
      intermediate_size=emb_dim * 4,
      intermediate_act_fn=utils.gelu,
      hidden_dropout_prob=dropout_prob,
      attention_probs_dropout_prob=dropout_prob,
      initializer_range=0.02,
      do_return_all_layers=False
    )
    #self.user_emb = tf.reshape(user_emb, [batch_size, -1])
    self.user_emb = tf.reduce_mean(
      user_emb * tf.expand_dims(self.content_seq_mask, axis=-1),
      axis=1
    )


class YouTubeNetModel(DINModel):
  def __init__(self, params):
    super(YouTubeNetModel, self).__init__(params)

  def build_user_interest_layer(self,
                                seq_query=None,
                                dropout_prob=0.0,
                                att_dropout_prob=0.0,
                                activation='relu'):

    content_seq_emb = self.content_seq_title_emb + self.content_seq_time_emb + \
                      self.content_seq_image_emb
    self.user_emb = tf.reduce_mean(
      content_seq_emb * tf.expand_dims(self.content_seq_mask, axis=-1),
      axis=1
    )







if __name__ == '__main__':
  from Utils.params_config import PidSeqBaseFtParams
  from InputFN.input_fn import PidSeqBaseData
  import numpy as np
  import sys

  np.set_printoptions(threshold=sys.maxsize)
  project_name = 'pid_seq_base_debug'
  params = PidSeqBaseFtParams(project_name)
  params.post_process()
  params.params['batch_size'] = 2
  seq_data = PidSeqBaseData(params.params)

  _features, _labels = seq_data(params.params['train_table'], params.params,
                                params.params['mode'])
  seq_model = PidSeqBaseFtDebugModel(params.params)
  seq_model.build_embedding_layer()
  _emb, _mask = seq_model.load_cate_emb_and_mask(
    _features['product_cate_name_seq_input_ids'],
    _features['product_cate_name_seq_input_len'],
    use_4d=True
  )

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1):
      print('{}'.format(i) + '-' * 40)
      # features, labels = sess.run(data_loader(filenames, params))
      features, labels, emb, mask = sess.run([_features, _labels, _emb, _mask])
      print(features['user_id'])
      print(features['content_id'])
      print(features['product_cate_name_seq_input_ids'])
      print(features['product_cate_name_seq_input_len'])
      print(features['product_cate_name_seq_input_ids'].shape)
      print(features['product_cate_name_seq_input_len'].shape)
      print(emb)

      print(mask)
