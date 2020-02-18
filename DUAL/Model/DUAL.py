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
from Utils import gradient_reversal_layer as grl
import random


class DUALBaseModel(object):
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

    self.max_text_len = params['max_text_len']
    self.content_text_encoder_type = params['content_text_encoder_type']
    self.product_text_encoder_type = params['product_text_encoder_type']
    self.feature_transfer_type = params['feature_transfer_type']
    self.margin = params['margin']
    self.use_shuffle = params['use_shuffle']
    self.text_encoder_params = {
      'window_size': '3,5',
      'att_act': 'tanh',
      'num_attention_heads': 4,
      'dot_att_use_fc': False
    }

  def __call__(self, features, labels, mode, params):
    training = True if 'train' in mode else False
    if training:
      dropout_prob = params['dropout_prob']
      print('train phase: dropout_prob {:.4f}'.format(dropout_prob))
    else:
      dropout_prob = 0.0
      print('eval phase: dropout_prob {:.4f}'.format(dropout_prob))

    # step 1. build embedding layer
    with tf.variable_scope('embedding_layer'):
      self.build_embedding_layer()

    # step 2. build input layer
    with tf.variable_scope('input_layer'):
      self.build_input_layer(features, dropout_prob)
      self.build_feature_transfer_layer(dropout_prob)
      self.build_loss_layer(dropout_prob)

    # step 4. build EstimatorSpec
    if mode == tf.estimator.ModeKeys.PREDICT:
      return self.build_predict_estimator_spec(mode, features)

    if mode == tf.estimator.ModeKeys.TRAIN:
      self.build_train_operation()
      return self.build_train_estimator_spec(mode)
    elif mode == tf.estimator.ModeKeys.EVAL:
      return self.build_eval_estimator_spec(mode)
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

  def load_title_emb_and_mask(self,
                              emb_input_ids,
                              emb_input_len,
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
        **self.text_encoder_params
      )  # [B, s, D]
    else:
      emb = utils.text_encoder(
        emb, mask, encoder_type, name, reuse,
        position_emb=self.word_position_embeddings,
        dropout_prob=dropout_prob,
        **self.text_encoder_params
      )  # [B, D]
    return emb

  def build_input_layer(self, features, dropout_prob):
    pass

  def build_feature_transfer_layer(self, dropout_prob=0.0):
    pass

  def build_loss_layer(self, dropout_prob):
    pass

  def build_train_operation(self):
    learning_rate = optimization.warmup(
      init_lr=self.init_lr,
      num_train_steps=self.num_train_steps,
      num_warmup_steps=self.num_warmup_steps
    )
    if self.optimizer.lower() == 'adamw':
      train_op = optimization.adam_weight_decay_optimizer(
        loss=self.loss,
        learning_rate=learning_rate,
        l2_reg=self.l2_reg
      )
    else:
      train_op = optimization.custom_optimizer(
        loss=self.loss,
        optimizer_type=self.optimizer.lower(),
        learning_rate=learning_rate,
        l2_reg=self.l2_reg
      )
    tf.summary.scalar('learning_rate', learning_rate)
    self.train_op = train_op

  def build_train_estimator_spec(self, mode):
    for var in tf.trainable_variables():
      print(var.name)
    train_print = {
      'loss': self.loss,
    }
    train_hook = tf.train.LoggingTensorHook(
      train_print, every_n_iter=self.every_n_iter)
    return tf.estimator.EstimatorSpec(
      mode, loss=self.loss, train_op=self.train_op, training_hooks=[train_hook]
    )

  def build_eval_estimator_spec(self, mode):
    utils.stat_eval_results(self.model_dir, 'eval_result_temp.txt')
    return tf.estimator.EstimatorSpec(
      mode,
      loss=self.loss,
    )

  def hinge_loss(self, low, high):
    return tf.reduce_mean(tf.nn.relu(low + self.margin - high))


class DUALTitleModel(DUALBaseModel):
  def __init__(self, params):
    super(DUALTitleModel, self).__init__(params)

  def build_input_layer(self, features, dropout_prob):
    super(DUALTitleModel, self).build_input_layer(features, dropout_prob)

    content_encoder_name = product_encoder_name = 'title'
    pos_content_title_emb = self.load_title_emb_and_mask(
      features['content_title_input_ids'],
      features['content_title_input_len'],
      self.content_text_encoder_type,
      content_encoder_name, False, dropout_prob
    )
    pos_product_title_emb = self.load_title_emb_and_mask(
      features['product_title_input_ids'],
      features['product_title_input_len'],
      self.product_text_encoder_type,
      product_encoder_name, True, dropout_prob
    )
    neg_content_title_emb = self.load_title_emb_and_mask(
      features['neg_content_title_input_ids'],
      features['neg_content_title_input_len'],
      self.content_text_encoder_type,
      content_encoder_name, True, dropout_prob
    )
    neg_product_title_emb = self.load_title_emb_and_mask(
      features['neg_product_title_input_ids'],
      features['neg_product_title_input_len'],
      self.product_text_encoder_type,
      product_encoder_name, True, dropout_prob
    )

    if self.use_shuffle:
      rnd_index = tf.random_shuffle(range(self.batch_size))
      neg_content_title_emb = tf.gather(neg_content_title_emb, rnd_index)
      neg_product_title_emb = tf.gather(neg_product_title_emb, rnd_index)

    self.pos_content_title_emb = pos_content_title_emb
    self.neg_content_title_emb = neg_content_title_emb
    self.pos_product_title_emb = pos_product_title_emb
    self.neg_product_title_emb = neg_product_title_emb

  def title_hinge_loss(self):
    pos_pos_dist = self.pos_content_title_emb - self.pos_product_title_emb
    pos_neg_dist = self.pos_content_title_emb - self.neg_product_title_emb
    neg_pos_dist = self.neg_content_title_emb - self.pos_product_title_emb
    neg_neg_dist = self.neg_content_title_emb - self.neg_product_title_emb
    pos_pos_dist = tf.reduce_sum(tf.square(pos_pos_dist), axis=1)
    pos_neg_dist = tf.reduce_sum(tf.square(pos_neg_dist), axis=1)
    neg_pos_dist = tf.reduce_sum(tf.square(neg_pos_dist), axis=1)
    neg_neg_dist = tf.reduce_sum(tf.square(neg_neg_dist), axis=1)
    loss1 = self.hinge_loss(pos_pos_dist, pos_neg_dist)
    loss2 = self.hinge_loss(pos_pos_dist, neg_pos_dist)
    loss3 = self.hinge_loss(neg_neg_dist, pos_neg_dist)
    loss4 = self.hinge_loss(neg_neg_dist, neg_pos_dist)
    loss = loss1 + loss2 + loss3 + loss4
    return loss

  def build_feature_transfer_layer(self, dropout_prob=0.0):
    super(DUALTitleModel, self).build_feature_transfer_layer(dropout_prob)
    if self.feature_transfer_type == 'inner':
      pos_pos_sim = tf.reduce_sum(
        self.pos_content_title_emb * self.pos_product_title_emb, axis=1)
      pos_neg_sim = tf.reduce_sum(
        self.pos_content_title_emb * self.neg_product_title_emb, axis=1)
      neg_pos_sim = tf.reduce_sum(
        self.neg_content_title_emb * self.pos_product_title_emb, axis=1)
      neg_neg_sim = tf.reduce_sum(
        self.neg_content_title_emb * self.neg_product_title_emb, axis=1)
      loss1 = self.hinge_loss(pos_neg_sim, pos_pos_sim)
      loss2 = self.hinge_loss(neg_pos_sim, pos_pos_sim)
      loss3 = self.hinge_loss(neg_pos_sim, neg_neg_sim)
      loss4 = self.hinge_loss(pos_neg_sim, neg_neg_sim)
      loss = loss1 + loss2 + loss3 + loss4

    elif self.feature_transfer_type == 'sub':
      loss = self.title_hinge_loss()
    elif self.feature_transfer_type == 'mmd':
      content_title_emb = tf.concat(
        [self.pos_content_title_emb, self.neg_content_title_emb], axis=0)
      product_title_emb = tf.concat(
        [self.pos_product_title_emb, self.neg_product_title_emb], axis=0)
      mmd_loss = utils.mmd_loss(content_title_emb, product_title_emb)
      loss = mmd_loss
    elif self.feature_transfer_type == 'sub_mmd':
      loss = self.title_hinge_loss()
      content_title_emb = tf.concat(
        [self.pos_content_title_emb, self.neg_content_title_emb], axis=0)
      product_title_emb = tf.concat(
        [self.pos_product_title_emb, self.neg_product_title_emb], axis=0)
      mmd_loss = utils.mmd_loss(content_title_emb, product_title_emb)
      loss += mmd_loss
    elif self.feature_transfer_type == 'sub_dann':
      loss = self.title_hinge_loss()

      content_title_emb = tf.concat(
        [self.pos_content_title_emb, self.neg_content_title_emb], axis=0)
      product_title_emb = tf.concat(
        [self.pos_product_title_emb, self.neg_product_title_emb], axis=0)
      dann_loss, dann_acc = utils.dann_loss(
        content_title_emb, product_title_emb, scope='title_dann')
      loss += dann_loss
      tf.summary.scalar('dann_acc', dann_acc[1])
    else:
      raise NotImplementedError

    self.title_loss = loss
    self.loss = loss
    tf.summary.scalar('title/loss', self.loss)


class DUALImageModel(DUALBaseModel):
  def __init__(self, params):
    super(DUALImageModel, self).__init__(params)


  def build_input_layer(self, features, dropout_prob):
    super(DUALImageModel, self).build_input_layer(features, dropout_prob)
    pos_content_image_emb = tf.layers.dense(
      features['content_image_feat'], self.emb_dim, name='img_fc')
    pos_product_image_emb = tf.layers.dense(
      features['product_image_feat'], self.emb_dim, name='img_fc', reuse=True)
    neg_content_image_emb = tf.layers.dense(
      features['neg_content_image_feat'], self.emb_dim, name='img_fc',
      reuse=True)
    neg_product_image_emb = tf.layers.dense(
      features['neg_content_image_feat'], self.emb_dim, name='img_fc',
      reuse=True)

    if self.use_shuffle:
      rnd_index = tf.random_shuffle(range(self.batch_size))
      neg_content_image_emb = tf.gather(neg_content_image_emb, rnd_index)
      neg_product_image_emb = tf.gather(neg_product_image_emb, rnd_index)

    self.pos_content_image_emb = pos_content_image_emb
    self.pos_product_image_emb = pos_product_image_emb
    self.neg_content_image_emb = neg_content_image_emb
    self.neg_product_image_emb = neg_product_image_emb

  def image_hinge_loss(self):
    pos_pos_dist = self.pos_content_image_emb - self.pos_product_image_emb
    pos_neg_dist = self.pos_content_image_emb - self.neg_product_image_emb
    neg_pos_dist = self.neg_content_image_emb - self.pos_product_image_emb
    neg_neg_dist = self.neg_content_image_emb - self.neg_product_image_emb
    pos_pos_dist = tf.reduce_sum(tf.square(pos_pos_dist), axis=1)
    pos_neg_dist = tf.reduce_sum(tf.square(pos_neg_dist), axis=1)
    neg_pos_dist = tf.reduce_sum(tf.square(neg_pos_dist), axis=1)
    neg_neg_dist = tf.reduce_sum(tf.square(neg_neg_dist), axis=1)
    loss1 = self.hinge_loss(pos_pos_dist, pos_neg_dist)
    loss2 = self.hinge_loss(pos_pos_dist, neg_pos_dist)
    loss3 = self.hinge_loss(neg_neg_dist, pos_neg_dist)
    loss4 = self.hinge_loss(neg_neg_dist, neg_pos_dist)
    loss = loss1 + loss2 + loss3 + loss4
    return loss

  def build_feature_transfer_layer(self, dropout_prob=0.0):
    super(DUALImageModel, self).build_feature_transfer_layer(dropout_prob)
    if self.feature_transfer_type == 'inner':
      pos_pos_sim = tf.reduce_sum(
        self.pos_content_image_emb * self.pos_product_image_emb, axis=1)
      pos_neg_sim = tf.reduce_sum(
        self.pos_content_image_emb * self.neg_product_image_emb, axis=1)
      neg_pos_sim = tf.reduce_sum(
        self.neg_content_image_emb * self.pos_product_image_emb, axis=1)
      neg_neg_sim = tf.reduce_sum(
        self.neg_content_image_emb * self.neg_product_image_emb, axis=1)
      loss1 = self.hinge_loss(pos_neg_sim, pos_pos_sim)
      loss2 = self.hinge_loss(neg_pos_sim, pos_pos_sim)
      loss3 = self.hinge_loss(neg_pos_sim, neg_neg_sim)
      loss4 = self.hinge_loss(pos_neg_sim, neg_neg_sim)
      loss = loss1 + loss2 + loss3 + loss4
    elif self.feature_transfer_type == 'sub':
      loss = self.image_hinge_loss()
    elif self.feature_transfer_type == 'mmd':
      content_image_emb = tf.concat(
        [self.pos_content_image_emb, self.neg_content_image_emb], axis=0)
      product_image_emb = tf.concat(
        [self.pos_product_image_emb, self.neg_product_image_emb], axis=0)
      mmd_loss = utils.mmd_loss(content_image_emb, product_image_emb)
      loss = mmd_loss
    elif self.feature_transfer_type == 'sub_mmd':
      loss = self.image_hinge_loss()

      content_image_emb = tf.concat(
        [self.pos_content_image_emb, self.neg_content_image_emb], axis=0)
      product_image_emb = tf.concat(
        [self.pos_product_image_emb, self.neg_product_image_emb], axis=0)
      mmd_loss = utils.mmd_loss(content_image_emb, product_image_emb)
      loss += mmd_loss

    elif self.feature_transfer_type == 'sub_dann':
      loss = self.image_hinge_loss()

      content_image_emb = tf.concat(
        [self.pos_content_image_emb, self.neg_content_image_emb], axis=0)
      product_image_emb = tf.concat(
        [self.pos_product_image_emb, self.neg_product_image_emb], axis=0)
      dann_loss, dann_acc = utils.dann_loss(
        content_image_emb, product_image_emb, scope='image_dann')
      loss += dann_loss
      tf.summary.scalar('dann_acc', dann_acc[1])
    else:
      raise NotImplementedError

    self.image_loss = loss
    self.loss = loss


class DUALModel(DUALImageModel, DUALTitleModel):
  def __init__(self, params):
    super(DUALModel, self).__init__(params)

  def build_input_layer(self, features, dropout_prob):
    super(DUALModel, self).build_input_layer(features, dropout_prob)

  def build_feature_transfer_layer(self, dropout_prob=0.0):
    super(DUALModel, self).build_feature_transfer_layer(dropout_prob)
    self.loss = self.title_loss + self.image_loss



class DUALInferModel(DUALModel):
  def __init__(self, params):
    super(DUALInferModel, self).__init__(params)
    self.target = params['target']

  def build_input_layer(self, features, dropout_prob):
    text_encoder_name = 'title'
    pos_query = None
    if self.target == 'content':
      encoder_type = self.content_text_encoder_type
    else:
      encoder_type = self.product_text_encoder_type

    target_title_emb = self.load_title_emb_and_mask(
      features['{}_title_input_ids'.format(self.target)],
      features['{}_title_input_len'.format(self.target)],
      pos_query, encoder_type,
      text_encoder_name, False, dropout_prob
    )
    target_image_emb = tf.layers.dense(
      features['{}_image_feat'.format(self.target)], self.emb_dim, name='img_fc'
    )
    self.target_emb = tf.concat([target_title_emb, target_image_emb], axis=1)

  def build_feature_transfer_layer(self, dropout_prob):
    pass

  def build_loss_layer(self, dropout_prob):
    pass

  def build_predict_estimator_spec(self, mode, features):
    predict_results = {
      '{}_id'.format(self.target): features['{}_id'.format(self.target)],
      '{}_emb'.format(self.target): self.target_emb
    }
    return tf.estimator.EstimatorSpec(mode, predict_results)









