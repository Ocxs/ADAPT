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


class SeqBaseModel(object):
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
    self.seq_encoder_type = params['seq_encoder_type']
    self.loss_type = params['loss_type']
    self.content_text_encoder_type = params['content_text_encoder_type']
    self.product_text_encoder_type = params['product_text_encoder_type']
    self.use_target = params['use_target']

    self.text_encoder_params = {
      'window_size': '3,5',
      'att_act': 'tanh',
      'num_attention_heads': 4,
      'dot_att_use_fc': False
    }

    self.seq_encoder_params = {
      'att_act': 'relu',
      'num_attention_heads': 4,
      'dot_att_use_fc': False
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

    self.labels = tf.cast(labels >= 1, tf.int32)

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
    self.time_embeddings = utils.build_embeddings(
      emb_name='time_embeddings',
      vocab_size=24*7+2,
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
    target_title_emb = self.load_title_emb_and_mask(
      features['content_title_input_ids'],
      features['content_title_input_len'],
      None, self.content_text_encoder_type,
      self.text_encoder_name, False, dropout_prob,
    )

    target_image_emb = tf.layers.dense(
      features['content_image_feat'], self.emb_dim, name='img_fc'
    )

    self.target_title_emb = target_title_emb
    self.target_image_emb = target_image_emb
    self.target_emb = target_image_emb + target_title_emb
    self.seq_emb = None
    self.seq_mask = None

  def build_user_interest_layer(self,
                                seq_query=None,
                                dropout_prob=0.0,
                                att_dropout_prob=0.0,
                                activation='relu'):
    user_emb = utils.seq_encoder(
      seq_emb=self.seq_emb,
      seq_mask=self.seq_mask,
      encoder_type=self.seq_encoder_type,
      query=seq_query,
      dropout_prob=dropout_prob,
      **self.seq_encoder_params
    )
    self.user_emb = user_emb

  def build_aggregation_layer(self, dropout_prob=0.0):
    input = tf.concat([self.user_emb, self.target_emb], axis=1)
    agg_vector = utils.multi_layer_perceptron(
      input=input,
      mlp_layers=self.agg_layer,
      activation=self.agg_act,
      dropout_prob=dropout_prob
    )
    logits = tf.layers.dense(agg_vector, 1, name='pred')
    self.logits = tf.squeeze(logits)

  def build_loss_layer(self):
    labels = tf.cast(self.labels, tf.float32)
    loss = utils.compute_loss(
      labels, self.logits, self.loss_type
    )
    tf.summary.scalar('loss/ctr_loss', loss)
    self.loss = loss

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
    ctr_probs = tf.nn.sigmoid(self.logits)
    roc_auc = tf.metrics.auc(labels=self.labels, predictions=ctr_probs)

    predictions = tf.cast(tf.round(ctr_probs), tf.int32)
    accuracy = tf.metrics.accuracy(self.labels, predictions)
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


class CidSeqBaseModel(SeqBaseModel):
  def __init__(self, params):
    super(CidSeqBaseModel, self).__init__(params)

  def build_input_layer(self, features, dropout_prob):
    super(CidSeqBaseModel, self).build_input_layer(features, dropout_prob)
    content_seq_mask = tf.sequence_mask(
      features['content_seq_len'], self.max_seq_len, tf.float32
    )  # [B, S]
    content_seq_time_emb = tf.nn.embedding_lookup(
      self.time_embeddings, features['content_seq_time_ids']
    )

    self.content_seq_time_emb = content_seq_time_emb
    self.content_seq_mask = content_seq_mask


class CidSeqTitleBaseModel(CidSeqBaseModel):
  def __init__(self, params):
    super(CidSeqTitleBaseModel, self).__init__(params)

  def build_input_layer(self, features, dropout_prob):
    super(CidSeqTitleBaseModel, self).build_input_layer(features, dropout_prob)
    content_seq_title_emb = self.load_title_emb_and_mask(
      features['content_title_seq_input_ids'],
      features['content_title_seq_input_len'],
      None, self.content_text_encoder_type,
      self.text_encoder_name, reuse=True,
      dropout_prob=dropout_prob, use_4d=True
    )  # [B, S, D]
    self.content_seq_title_emb = content_seq_title_emb

  def build_user_interest_layer(self,
                                seq_query=None,
                                dropout_prob=0.0,
                                att_dropout_prob=0.0,
                                activation='relu'):
    self.seq_emb = self.content_seq_title_emb + self.content_seq_time_emb
    self.seq_mask = self.content_seq_mask
    if self.use_target:
      seq_query = tf.expand_dims(self.target_title_emb, axis=1)

    super(CidSeqTitleBaseModel, self).build_user_interest_layer(
      seq_query=seq_query,
      dropout_prob=dropout_prob,
      att_dropout_prob=att_dropout_prob,
      activation=activation
    )


class CidSeqImageBaseModel(CidSeqBaseModel):
  def __init__(self, params):
    super(CidSeqImageBaseModel, self).__init__(params)

  def build_input_layer(self, features, dropout_prob):
    super(CidSeqImageBaseModel, self).build_input_layer(features, dropout_prob)
    content_seq_image_emb = tf.layers.dense(
      features['content_image_seq_feat'], self.emb_dim,
      name='img_fc', reuse=True
    )
    self.content_seq_image_emb = content_seq_image_emb

  def build_user_interest_layer(self,
                                seq_query=None,
                                dropout_prob=0.0,
                                att_dropout_prob=0.0,
                                activation='relu'):
    self.seq_emb = self.content_seq_image_emb + self.content_seq_time_emb
    cid_recent_k = min(self.cid_recent_k, 20)
    self.seq_mask = self.content_seq_mask[:, :cid_recent_k]
    if self.use_target:
      seq_query = tf.expand_dims(self.target_image_emb, axis=1)

    super(CidSeqImageBaseModel, self).build_user_interest_layer(
      seq_query=seq_query,
      dropout_prob=dropout_prob,
      att_dropout_prob=att_dropout_prob,
      activation=activation
    )


class PidSeqBaseModel(SeqBaseModel):
  def __init__(self, params):
    super(PidSeqBaseModel, self).__init__(params)

  def build_input_layer(self, features, dropout_prob):
    super(PidSeqBaseModel, self).build_input_layer(features, dropout_prob)
    product_seq_mask = tf.sequence_mask(
      features['product_seq_len'], self.max_seq_len, tf.float32
    )
    product_seq_time_emb = tf.nn.embedding_lookup(
      self.time_embeddings, features['product_seq_time_ids']
    )

    self.product_seq_time_emb = product_seq_time_emb
    self.product_seq_mask = product_seq_mask


class PidSeqTitleBaseModel(PidSeqBaseModel):
  def __init__(self, params):
    super(PidSeqTitleBaseModel, self).__init__(params)

  def build_input_layer(self, features, dropout_prob):
    super(PidSeqTitleBaseModel, self).build_input_layer(features, dropout_prob)

    product_seq_title_emb = self.load_title_emb_and_mask(
      features['product_title_seq_input_ids'],
      features['product_title_seq_input_len'],
      None, self.product_text_encoder_type,
      self.text_encoder_name, reuse=True,
      dropout_prob=dropout_prob, use_4d=True
    )  # [B, S, D]

    self.product_seq_title_emb = product_seq_title_emb

  def build_user_interest_layer(self,
                                seq_query=None,
                                dropout_prob=0.0,
                                att_dropout_prob=0.0,
                                activation='relu'):
    self.seq_emb = self.product_seq_title_emb + self.product_seq_time_emb
    self.seq_mask = self.product_seq_mask
    if self.use_target:
      seq_query = tf.expand_dims(self.target_title_emb, axis=1)
    super(PidSeqTitleBaseModel, self).build_user_interest_layer(
      seq_query=seq_query,
      dropout_prob=dropout_prob,
      att_dropout_prob=att_dropout_prob,
      activation=activation
    )


class PidSeqImageBaseModel(PidSeqBaseModel):
  def __init__(self, params):
    super(PidSeqImageBaseModel, self).__init__(params)

  def build_input_layer(self, features, dropout_prob):
    super(PidSeqImageBaseModel, self).build_input_layer(features, dropout_prob)
    product_seq_image_emb = tf.layers.dense(
      features['product_image_seq_feat'], self.emb_dim,
      name='img_fc', reuse=True
    )
    self.product_seq_image_emb = product_seq_image_emb

  def build_user_interest_layer(self,
                                seq_query=None,
                                dropout_prob=0.0,
                                att_dropout_prob=0.0,
                                activation='relu'):
    self.seq_emb = self.product_seq_image_emb + self.product_seq_time_emb
    self.seq_mask = self.product_seq_mask
    if self.use_target:
      seq_query = tf.expand_dims(self.target_image_emb, axis=1)

    super(PidSeqImageBaseModel, self).build_user_interest_layer(
      seq_query=seq_query,
      dropout_prob=dropout_prob,
      att_dropout_prob=att_dropout_prob,
      activation=activation
    )


class MixedSeqBaseModel(CidSeqBaseModel, PidSeqBaseModel):
  def __init__(self, params):
    super(MixedSeqBaseModel, self).__init__(params)
    self.domain_transfer_type = params['domain_transfer_type']
    self.seq_is_share_params = params['seq_is_share_params']
    self.mixed_seq_encoder_params = {
      'seq_is_share_params': self.seq_is_share_params
    }
    self.mixed_seq_encoder_params.update(self.seq_encoder_params)

  def build_user_interest_layer(self,
                                content_seq_emb=None,
                                content_seq_mask=None,
                                product_seq_emb=None,
                                product_seq_mask=None,
                                seq_query=None,
                                dropout_prob=0.0,
                                att_dropout_prob=0.0,
                                activation='relu'):
    user_emb = utils.mixed_seq_encoder(
      content_seq_emb=content_seq_emb,
      content_seq_mask=content_seq_mask,
      product_seq_emb=product_seq_emb,
      product_seq_mask=product_seq_mask,
      query=seq_query,
      domain_transfer_type=self.domain_transfer_type,
      seq_encoder_type=self.seq_encoder_type,
      dropout_prob=dropout_prob,
      **self.mixed_seq_encoder_params
    )
    self.user_emb = user_emb


class MixedSeqTitleBaseModel(MixedSeqBaseModel,
                             CidSeqTitleBaseModel,
                             PidSeqTitleBaseModel):
  def __init__(self, params):
    super(MixedSeqTitleBaseModel, self).__init__(params)

  def build_user_interest_layer(self,
                                content_seq_emb=None,
                                content_seq_mask=None,
                                product_seq_emb=None,
                                product_seq_mask=None,
                                seq_query=None,
                                dropout_prob=0.0,
                                att_dropout_prob=0.0,
                                activation='relu'):
    if self.use_target:
      seq_query = tf.expand_dims(self.target_title_emb, axis=1)

    super(MixedSeqTitleBaseModel, self).build_user_interest_layer(
      content_seq_emb=self.content_seq_title_emb + self.content_seq_time_emb,
      content_seq_mask=self.content_seq_mask,
      product_seq_emb = self.product_seq_title_emb + self.product_seq_time_emb,
      product_seq_mask=self.product_seq_mask,
      seq_query=seq_query, dropout_prob=dropout_prob,
      att_dropout_prob=att_dropout_prob, activation=activation
    )


class MixedSeqImageBaseModel(MixedSeqBaseModel,
                             CidSeqImageBaseModel,
                             PidSeqImageBaseModel):
  def __init__(self, params):
    super(MixedSeqImageBaseModel, self).__init__(params)

  def build_user_interest_layer(self,
                                content_seq_emb=None,
                                content_seq_mask=None,
                                product_seq_emb=None,
                                product_seq_mask=None,
                                seq_query=None,
                                dropout_prob=0.0,
                                att_dropout_prob=0.0,
                                activation='relu'):
    if self.use_target:
      seq_query = tf.expand_dims(self.target_image_emb, axis=1)

    super(MixedSeqImageBaseModel, self).build_user_interest_layer(
      content_seq_emb=self.content_seq_image_emb + self.content_seq_time_emb,
      content_seq_mask=self.content_seq_mask,
      product_seq_emb=self.product_seq_image_emb + self.product_seq_time_emb,
      product_seq_mask=self.product_seq_mask,
      seq_query=seq_query, dropout_prob=dropout_prob,
      att_dropout_prob=att_dropout_prob, activation=activation
    )


class MixedSeqMultiModalModel(MixedSeqTitleBaseModel,
                              MixedSeqImageBaseModel,
                              MixedSeqBaseModel):
  def __init__(self, params):
    super(MixedSeqMultiModalModel, self).__init__(params)

  def build_user_interest_layer(self,
                                content_seq_emb=None,
                                content_seq_mask=None,
                                product_seq_emb=None,
                                product_seq_mask=None,
                                seq_query=None,
                                dropout_prob=0.0,
                                att_dropout_prob=0.0,
                                activation='relu'):
    if self.use_target:
      seq_query = tf.expand_dims(self.target_emb, axis=1)

    content_seq_emb = self.content_seq_title_emb + self.content_seq_time_emb + \
                      self.content_seq_image_emb
    product_seq_emb = self.product_seq_title_emb + self.product_seq_time_emb + \
                      self.product_seq_image_emb

    super(MixedSeqImageBaseModel, self).build_user_interest_layer(
      content_seq_emb=content_seq_emb,
      content_seq_mask=self.content_seq_mask,
      product_seq_emb=product_seq_emb,
      product_seq_mask=self.product_seq_mask,
      seq_query=seq_query, dropout_prob=dropout_prob,
      att_dropout_prob=att_dropout_prob, activation=activation
    )


class SeqFinetuneModel(SeqBaseModel):
  def __init__(self, params):
    super(SeqFinetuneModel, self).__init__(params)
    self.restore_model_path = params['restore_model_path']
    self.emb_trainable = params['emb_trainable']

  def build_train_operation(self, train_vars=None):
    if self.emb_trainable:
      train_vars = None
    else:
      tvars = tf.trainable_variables()
      (_, initialized_variable_names) = \
        bert.get_assignment_map_from_checkpoint(tvars, self.restore_model_path)
      print('restored variable names: ', initialized_variable_names)
      train_vars = [v for v in tf.trainable_variables()
                    if v.name not in initialized_variable_names]

    super(SeqFinetuneModel, self).build_train_operation(train_vars)

  def build_train_estimator_spec(self, mode):
    self.restore_model()
    super(SeqFinetuneModel, self).build_train_estimator_spec(mode)

  def restore_model(self):
    tvars = tf.trainable_variables()
    (assignment_map, initialized_variable_names) = \
      bert.get_assignment_map_from_checkpoint(tvars, self.restore_model_path)

    tf.train.init_from_checkpoint(self.restore_model_path, assignment_map)
    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = {}, shape = {}{}".format(
        var.name, var.shape, init_string))


class CidSeqTitleBaseFtModel(CidSeqTitleBaseModel, SeqFinetuneModel):
  def __init__(self, params):
    super(CidSeqTitleBaseFtModel, self).__init__(params)


class CidSeqImageBaseFtModel(CidSeqImageBaseModel, SeqFinetuneModel):
  def __init__(self, params):
    super(CidSeqImageBaseFtModel, self).__init__(params)


class PidSeqTitleBaseFtModel(PidSeqTitleBaseModel, SeqFinetuneModel):
  def __init__(self, params):
    super(PidSeqTitleBaseFtModel, self).__init__(params)


class PidSeqImageBaseFtModel(PidSeqImageBaseModel, SeqFinetuneModel):
  def __init__(self, params):
    super(PidSeqImageBaseFtModel, self).__init__(params)


class MixedSeqTitleBaseFtModel(MixedSeqTitleBaseModel, SeqFinetuneModel):
  def __init__(self, params):
    super(MixedSeqTitleBaseFtModel, self).__init__(params)


class MixedSeqImageBaseFtModel(MixedSeqImageBaseModel, SeqFinetuneModel):
  def __init__(self, params):
    super(MixedSeqImageBaseFtModel, self).__init__(params)


class MixedSeqMultiModalFtModel(MixedSeqMultiModalModel, SeqFinetuneModel):
  def __init__(self, params):
    super(MixedSeqMultiModalFtModel, self).__init__(params)


class CidSeqMultiModalModel(CidSeqTitleBaseModel,
                            CidSeqImageBaseModel,
                            CidSeqBaseModel):
  def __init__(self, params):
    super(CidSeqMultiModalModel, self).__init__(params)

  def build_user_interest_layer(self,
                                seq_query=None,
                                dropout_prob=0.0,
                                att_dropout_prob=0.0,
                                activation='relu'):
    self.seq_emb = self.content_seq_image_emb + self.content_seq_time_emb + \
                   self.content_seq_title_emb
    self.seq_mask = self.content_seq_mask
    if self.use_target:
      seq_query = tf.expand_dims(self.target_emb, axis=1)

    super(CidSeqImageBaseModel, self).build_user_interest_layer(
      seq_query=seq_query,
      dropout_prob=dropout_prob,
      att_dropout_prob=att_dropout_prob,
      activation=activation
    )


class PidSeqMultiModalModel(PidSeqTitleBaseModel,
                            PidSeqImageBaseModel,
                            PidSeqBaseModel):
  def __init__(self, params):
    super(PidSeqMultiModalModel, self).__init__(params)

  def build_user_interest_layer(self,
                                seq_query=None,
                                dropout_prob=0.0,
                                att_dropout_prob=0.0,
                                activation='relu'):
    self.seq_emb = self.product_seq_image_emb + self.product_seq_time_emb + \
                   self.product_seq_title_emb
    self.seq_mask = self.product_seq_mask
    if self.use_target:
      seq_query = tf.expand_dims(self.target_emb, axis=1)

    super(PidSeqImageBaseModel, self).build_user_interest_layer(
      seq_query=seq_query,
      dropout_prob=dropout_prob,
      att_dropout_prob=att_dropout_prob,
      activation=activation
    )


class CidSeqMultiModalFtModel(CidSeqMultiModalModel, SeqFinetuneModel):
  def __init__(self, params):
    super(CidSeqMultiModalFtModel, self).__init__(params)


class PidSeqMultiModalFtModel(PidSeqMultiModalModel, SeqFinetuneModel):
  def __init__(self, params):
    super(PidSeqMultiModalFtModel, self).__init__(params)


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
