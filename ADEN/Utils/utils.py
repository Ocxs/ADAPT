# -*- coding: utf-8 -*-
# author: Xusong Chen

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tensorflow.contrib.estimator.python.estimator import early_stopping
from tensorflow.python.platform import gfile

import six
import os
from Utils import bert
import math


def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.truncated_normal_initializer(stddev=initializer_range)


def gelu(x, name=''):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf


def prelu(_x, name=''):
  """parametric ReLU activation"""
  with tf.variable_scope(name_or_scope=name, default_name="prelu"):
    _alpha = tf.get_variable("prelu_" + name, shape=_x.get_shape()[-1],
                             dtype=_x.dtype,
                             initializer=tf.constant_initializer(0.1))
    return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)


def dice(_x, name='dice', axis=-1, epsilon=0.000000001):
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    alphas = tf.get_variable('alpha'+name, _x.get_shape()[-1],
                         initializer=tf.constant_initializer(0.0),
                         dtype=tf.float32)
    input_shape = list(_x.get_shape())

    reduction_axes = list(range(len(input_shape)))
    del reduction_axes[axis]
    broadcast_shape = [1] * len(input_shape)
    broadcast_shape[axis] = input_shape[axis]

  # case: train mode (uses stats of the current batch)
  mean = tf.reduce_mean(_x, axis=reduction_axes)
  brodcast_mean = tf.reshape(mean, broadcast_shape)
  std = tf.reduce_mean(tf.square(_x - brodcast_mean) + epsilon,
                       axis=reduction_axes)
  std = tf.sqrt(std)
  brodcast_std = tf.reshape(std, broadcast_shape)
  x_normed = (_x - brodcast_mean) / (brodcast_std + epsilon)
  # x_normed = tf.layers.batch_normalization(_x, center=False, scale=False)
  x_p = tf.sigmoid(x_normed)
  return alphas * (1.0 - x_p) * _x + x_p * _x


def get_activation(activation_string):
  """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

  Args:
    activation_string: String name of the activation function.

  Returns:
    A Python function corresponding to the activation function. If
    `activation_string` is None, empty, or "linear", this will return None.
    If `activation_string` is not a string, it will return `activation_string`.

  Raises:
    ValueError: The `activation_string` does not correspond to a known
      activation.
  """

  # We assume that anything that"s not a string is already an activation
  # function, so we just return it.
  if not isinstance(activation_string, six.string_types):
    return activation_string

  if not activation_string:
    return None

  act = activation_string.lower()
  if act == "linear":
    return None
  elif act == "relu":
    return tf.nn.relu
  elif act == 'prelu':
    return prelu
  elif act == 'lrelu':
    return tf.nn.leaky_relu
  elif act == 'dice':
    return dice
  elif act == "gelu":
    return gelu
  elif act == "tanh":
    return tf.tanh
  elif act == "sigmoid":
    return tf.nn.sigmoid
  else:
    raise ValueError("Unsupported activation: %s" % act)


def average_pooling(keys, mask, avg_axis=1):
  """
  :param keys: [N, m, d] or [N, m, k, d], tf.float32
  :param key_mask: [N, m] or [N, m, k]
  :return:
  """
  numerator = tf.reduce_sum(
    keys * tf.expand_dims(tf.cast(mask, tf.float32), axis=-1),
    axis=avg_axis
  )
  denominator = tf.cast(
    tf.reduce_sum(mask, avg_axis, keepdims=True),
    tf.float32
  )
  output = numerator / (denominator + 1e-8)
  return output


def additive_attention_layer(keys,
                             query=None,
                             values=None,
                             mask=None,
                             dropout_prob=0.0,
                             activation='tanh',
                             att_weight_type='single',
                             return_attention_weight=False,
                             initializer=create_initializer(0.02)):
  """
  perform additive attention layer. w_2*f(w_1*[k, q]+b)
  :param keys: [B, S, D]
  :param query: if query is None, which can be seen as additive self attention.
                [B, S, D], [B, 1, D]
  :param values:
  :param mask: [B, S]
  :param att_axis: which axis will be performed attention
  :param dropout_prob: apply dropout on attention weights
  :param att_weight_type: single or multi,
                          single indicates vector-wise attention,
                          multi indicates element-wise attention
  :return:
  """
  batch_size, seq_len, dim = keys.get_shape().as_list()

  act_func = get_activation(activation)

  keys_proj = tf.layers.dense(
    keys, dim, name='K', kernel_initializer=initializer, use_bias=False)

  if query is not None:
    query_proj = tf.layers.dense(
      query, dim, name='Q', kernel_initializer=initializer, use_bias=False)
    proj = act_func(query_proj + keys_proj)
  else:
    proj = act_func(keys_proj)

  att_dim = 1 if att_weight_type == 'single' else dim
  # [B, S, att_dim]
  att_score = tf.layers.dense(
    proj, att_dim, name='att', kernel_initializer=initializer)

  if mask is not None:
    # mask = [B, S, 1]
    mask = tf.expand_dims(mask, axis=-1)
    adder = (1.0 - mask) * -1000000.0
    att_score += adder
  alpha = tf.nn.softmax(att_score, axis=1)

  # this operation may improve performance when seq_len is large. (> 80)
  alpha = tf.nn.dropout(alpha, 1.0 - dropout_prob)

  if values is None:
    output = tf.reduce_sum(alpha * keys, axis=1)
  else:
    output = tf.reduce_sum(alpha * values, axis=1)

  if return_attention_weight:
    return output, alpha
  return output


def dot_scale_attention_layer(keys,
                              query=None,
                              values=None,
                              mask=None,
                              dropout_prob=0.0,
                              num_attention_heads=4,
                              use_fc=True,
                              return_attention_weight=False,
                              initializer=create_initializer(0.02)):
  """
  perform dot-scale attention softmax{(Q*K)/sqrt(d)}*V
  :param keys: [B, T, D]
  :param query: [B, F, D]
  :param values: values = keys
  :param mask: [B, from_seq_len, to_seq_len]
  :param dropout_prob:
  :param activation:
  :param att_axis:
  :param use_fc: perform linear transform on keys and query
  :param initializer:
  :return:
  """
  def transpose_for_scores(input_tensor, batch_size, seq_length,
                           num_attention_heads, width):
    output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_attention_heads, width])

    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor

  batch_size, to_seq_len, dim = keys.get_shape().as_list()

  if query is None:
    query = keys
    from_seq_len = to_seq_len
  else:
    _, from_seq_len, dim = query.get_shape().as_list()
  mask = bert.create_attention_mask_from_input_mask(query, mask)

  if values is None:
    values = keys

  if use_fc:
    query = tf.layers.dense(query, dim, name='Q', kernel_initializer=initializer)
    keys = tf.layers.dense(keys, dim, name='K', kernel_initializer=initializer)

  size_per_head = dim // num_attention_heads
  # [B, num_att_heads, F, size_per_head]
  query = transpose_for_scores(query, batch_size, from_seq_len,
                               num_attention_heads, size_per_head)
  # [B, num_att_heads, T, size_per_head]
  keys = transpose_for_scores(keys, batch_size, to_seq_len,
                              num_attention_heads, size_per_head)

  # [B, num_att_heads, F, T]
  att_scores = tf.matmul(query, keys, transpose_b=True)
  att_scores = tf.multiply(att_scores, 1.0 / math.sqrt(float(size_per_head)))

  if mask is not None:
    # mask = [B, 1, F, T]
    mask = tf.expand_dims(mask, axis=1)
    adder = (1.0 - mask) * -100000.0
    att_scores += adder
  # [B, num_att_heads, F, T]
  att_probs = tf.nn.softmax(att_scores)
  att_probs = tf.nn.dropout(att_probs, 1.0 - dropout_prob)

  # [B, num_att_heads, T, size_per_head]
  values = transpose_for_scores(values, batch_size, to_seq_len,
                                num_attention_heads, size_per_head)
  # [B, num_att_heads, F, size_per_head]
  context = tf.matmul(att_probs, values)
  # [B, F, num_att_heads, size_per_head]
  context = tf.transpose(context, [0, 2, 1, 3])
  output = tf.reshape(context, [batch_size, from_seq_len, dim])

  if return_attention_weight:
    return output, att_probs
  return output


def multi_layer_perceptron(input,
                           mlp_layers,
                           activation,
                           dropout_prob):

  act_func = get_activation(activation)
  mlp_layers = [int(unit) for unit in mlp_layers.split(',')]
  mlp_net = input

  for i in range(len(mlp_layers)):
    mlp_net = tf.layers.dense(mlp_net, mlp_layers[i], name='fc_{}'.format(i))
    mlp_net = tf.nn.dropout(mlp_net, 1.0 - dropout_prob)
    mlp_net = act_func(mlp_net, name='{}'.format(i+1))

  return mlp_net


def get_shape(inputs, name=None):
  name = "shape" if name is None else name
  with tf.name_scope(name):
    static_shape = inputs.get_shape().as_list()
    dynamic_shape = tf.shape(inputs)
    shape = []
    for i, dim in enumerate(static_shape):
      dim = dim if dim is not None else dynamic_shape[i]
      shape.append(dim)
    return (shape)


def max_pooling(keys, mask, max_axis=1):
  """
  perform max pooling on keys along max_axis
  :param keys: [N, M, K, d] or [N, M, d]
  :param mask: [N, M, K] or [N, M]
  :param max_axis:
  :return:
  """
  keys_shape = get_shape(keys)
  emb_dim = keys_shape[-1]
  if mask is not None:
    mask_shape = get_shape(mask)
    # assert len(mask_shape) + 1 == len(keys_shape), \
    #   'invalid mask shape: {}'.format(mask_shape)
    if len(mask_shape) == 2:
      mask = tf.tile(tf.expand_dims(mask, axis=-1), [1, 1, emb_dim])
    elif len(mask_shape) == 3:
      mask = tf.tile(tf.expand_dims(mask, axis=-1), [1, 1, 1, emb_dim])
    else:
      raise ValueError('invalid mask shape: {}'.format(mask_shape))
    mask = tf.cast(mask, tf.bool)
    padding_values = tf.ones_like(mask, tf.float32) * (-2**32+1)
    keys = tf.where(mask, keys, padding_values)

  return tf.reduce_max(keys, axis=max_axis)


def stat_eval_results(model_dir, save_name='eval_results.txt'):
  eval_dir = os.path.join(model_dir, 'eval/')
  print('eval_dir: ', eval_dir)
  eval_results = early_stopping.read_eval_metrics(eval_dir)
  #print(eval_results)

  eval_step_list = []
  eval_loss_list = []
  eval_auc_list = []
  eval_acc_list = []
  for step, metrics in eval_results.items():
    if step < 0:
      continue
    eval_step_list.append(step)
    eval_loss_list.append(metrics['loss'])
    eval_auc_list.append(metrics['metric/test/content_roc_auc'])
    eval_acc_list.append(metrics['metric/test/content_accuracy'])
    print(step, metrics)

  if len(eval_auc_list) <= 0:
    return

  best_auc = max(eval_auc_list)
  best_auc_epoch = np.argsort(-np.asarray(eval_auc_list))[0] + 1

  save_path = os.path.join(model_dir, save_name)
  with gfile.GFile(save_path, mode='w') as writer:
    cols_name = 'epoch xx:   auc,  acc,' \
                '   loss,    step\n'
    writer.write(cols_name)
    print(cols_name, end='')
    for i in range(len(eval_step_list)):
      line_to_write = 'epoch {:0>2}: {:.4f},{:.4f},' \
                      '{:.5f},{:0>10}\n'.format(
        i+1,
        eval_auc_list[i],
        eval_acc_list[i],
        eval_loss_list[i],
        eval_step_list[i]
      )
      print(line_to_write, end='')
      writer.write(line_to_write)
    writer.write('best epoch: {}, best auc: {:.4f}'.format(
      best_auc_epoch, best_auc))

  print('best epoch: {}, best auc: {:.4f}'.format(best_auc_epoch, best_auc))


def text_cnn(text_emb, window_size='3,5', dropout_prob=0.0):
  """
  perform text CNN with conv1d
  :param text_emb: [B, S, D]
  :param window_size: [3,5]/[5,7]/[3,5,7,9]/[3]/[5]/[7]/[9]
  :param dropout_prob:
  :return:
  """
  batch_size, max_text_len, dim = text_emb.get_shape().as_list()

  conv_features = []
  window_size = [int(size) for size in window_size.split(',')]
  num_filters = dim // len(window_size)
  for i, filter_size in enumerate(window_size):
    with tf.variable_scope('conv_{}'.format(filter_size)):
      conv_feat = tf.layers.conv1d(
        inputs=text_emb,
        filters=num_filters,
        kernel_size=filter_size,
        strides=1,
        padding='same',
        activation=tf.nn.relu
      ) # [B, S, num_filters]
      conv_features.append(conv_feat)

  conv_features = tf.concat(conv_features, axis=-1)
  text_cnn_emb = tf.nn.dropout(conv_features, 1.0 - dropout_prob)
  return text_cnn_emb


def text_encoder_4d(text_emb, text_mask, encoder_type,
                    name='text_encoder', reuse=False, position_emb=None,
                    dropout_prob=0.0, query=None,
                    **kwargs):
  batch_size, max_seq_len, max_text_len, dim = text_emb.get_shape().as_list()
  text_emb = tf.reshape(text_emb, [batch_size*max_seq_len, max_text_len, dim])
  text_mask = tf.reshape(text_mask, [batch_size*max_seq_len, max_text_len])
  if query is not None:
    query = tf.reshape(query, [batch_size*max_seq_len, 1, dim])

  text_enc_emb = text_encoder(text_emb, text_mask, encoder_type, name,
                              reuse=reuse, position_emb=position_emb,
                              dropout_prob=dropout_prob, query=query,
                              **kwargs)
  text_enc_emb = tf.reshape(text_enc_emb, [batch_size, max_seq_len, dim])
  return text_enc_emb


def text_encoder(text_emb, text_mask, encoder_type,
                 name='text_encoder', reuse=False, position_emb=None,
                 dropout_prob=0.0, query=None,
                 **kwargs):
  with tf.variable_scope(name, reuse=reuse):
    if position_emb is not None:
      position_emb = tf.expand_dims(position_emb, axis=0)
      text_emb += position_emb

    if encoder_type.startswith('cnn'):
      text_emb = text_cnn(text_emb, kwargs['window_size'])

    if encoder_type.endswith('avg'):
      text_enc_emb = average_pooling(text_emb, text_mask, avg_axis=1)
    elif encoder_type.endswith('max'):
      text_enc_emb = max_pooling(text_emb, text_mask, max_axis=1)
    elif encoder_type.endswith('multi_att'):
      text_enc_emb = additive_attention_layer(
        keys=text_emb, query=query,
        mask=text_mask, dropout_prob=dropout_prob,
        activation=kwargs['att_act'], att_weight_type='multi'
      )
    elif encoder_type.endswith('single_att'):
      text_enc_emb = additive_attention_layer(
        keys=text_emb, query=query,
        mask=text_mask, dropout_prob=dropout_prob,
        activation=kwargs['att_act'], att_weight_type='single'
      )
    elif encoder_type.endswith('dot_att'):
      text_enc_emb = dot_scale_attention_layer(
        keys=text_emb, mask=text_mask, dropout_prob=dropout_prob,
        num_attention_heads=kwargs['num_attention_heads'],
        use_fc=kwargs['dot_att_use_fc'], query=query
      )
      if query is None:
        text_enc_emb = tf.reduce_mean(
          text_enc_emb*tf.expand_dims(text_mask, axis=-1), axis=1)
    else:
      raise NotImplementedError

  return text_enc_emb


def seq_encoder(seq_emb, seq_mask, encoder_type,
                name='seq_encoder', reuse=False, query=None, dropout_prob=0.0,
                **kwargs):
  with tf.variable_scope(name, reuse=reuse):
    if encoder_type == 'avg':
      user_emb = average_pooling(seq_emb, seq_mask, 1)
    elif encoder_type == 'multi_att':
      user_emb = additive_attention_layer(
        keys=seq_emb, query=query, mask=seq_mask, dropout_prob=dropout_prob,
        activation=kwargs['att_act'], att_weight_type='multi'
      )
    elif encoder_type == 'single_att':
      user_emb = additive_attention_layer(
        keys=seq_emb, query=query, mask=seq_mask, dropout_prob=dropout_prob,
        activation=kwargs['att_act'], att_weight_type='single'
      )
    elif encoder_type == 'dot_att':
      user_emb = dot_scale_attention_layer(
        keys=seq_emb, query=query, mask=seq_mask, dropout_prob=dropout_prob,
        num_attention_heads=kwargs['num_attention_heads'],
        use_fc=kwargs['dot_att_use_fc']
      )
      if query is None:
        user_emb = tf.reduce_mean(
          user_emb * tf.expand_dims(seq_mask, axis=-1), axis=1)
    elif encoder_type == 'trm_att':
      user_emb = transformer_attention_layer(
        keys=seq_emb, mask=seq_mask, query=query, dropout_prob=dropout_prob,
        num_attention_heads=kwargs['num_attention_heads']
      )
      if query is None:
        user_emb = tf.reduce_mean(
          user_emb * tf.expand_dims(seq_mask, axis=-1), axis=1)
    elif encoder_type == 'trm':
      user_emb = transformer_layer(
        keys=seq_emb, mask=seq_mask, dropout_prob=dropout_prob,
        num_attention_heads=kwargs['num_attention_heads']
      )
    else:
      raise NotImplementedError

    return user_emb


def transformer_attention_layer(keys,
                                mask,
                                query=None,
                                dropout_prob=0.0,
                                num_attention_heads=4,
                                return_attention_weight=False):
  batch_size, to_seq_len, emb_dim = keys.get_shape().as_list()

  if query is None:
    mask = bert.create_attention_mask_from_input_mask(keys, mask)
    query = keys
    from_seq_len = to_seq_len
  else:
    mask = bert.create_attention_mask_from_input_mask(query, mask)
    _, from_seq_len, _ = query.get_shape().as_list()

  user_emb = bert.attention_layer(
    from_tensor=query,
    to_tensor=keys,
    attention_mask=mask,
    num_attention_heads=num_attention_heads,
    size_per_head=emb_dim // num_attention_heads,
    attention_probs_dropout_prob=dropout_prob,
    initializer_range=0.02,
    do_return_2d_tensor=False,
    batch_size=batch_size,
    from_seq_length=from_seq_len,
    to_seq_length=to_seq_len,
    return_attention_weight=return_attention_weight
  )
  # [B, from_seq_len, dim]
  if from_seq_len == 1:
    user_emb = tf.squeeze(user_emb)
  return user_emb


def transformer_layer(keys,
                      mask,
                      dropout_prob=0.0,
                      num_attention_heads=4):
  batch_size, to_seq_len, emb_dim = keys.get_shape().as_list()
  trm_mask = bert.create_attention_mask_from_input_mask(keys, mask)
  user_emb = bert.transformer_model(
    input_tensor=keys,
    attention_mask=trm_mask,
    hidden_size=emb_dim,
    num_hidden_layers=1,
    num_attention_heads=num_attention_heads,
    intermediate_size=emb_dim * 4,
    intermediate_act_fn=gelu,
    hidden_dropout_prob=dropout_prob,
    attention_probs_dropout_prob=dropout_prob,
    initializer_range=0.02,
    do_return_all_layers=False
  )
  user_emb = tf.reduce_mean(user_emb * tf.expand_dims(mask, axis=-1), axis=1)
  return user_emb


def mixed_seq_encoder(content_seq_emb,
                      content_seq_mask,
                      product_seq_emb,
                      product_seq_mask,
                      query,
                      domain_transfer_type,
                      seq_encoder_type,
                      dropout_prob=0.0,
                      **kwargs):
  if domain_transfer_type == 'early':
    seq_emb = tf.concat([content_seq_emb, product_seq_emb], axis=1)
    seq_mask = tf.concat([content_seq_mask, product_seq_mask], axis=1)
    user_emb = seq_encoder(
      seq_emb=seq_emb, seq_mask=seq_mask, encoder_type=seq_encoder_type,
      name='', reuse=False, query=query, dropout_prob=dropout_prob,
      **kwargs
    )
  else:
    if kwargs['seq_is_share_params']:
      content_seq_encoder_name = 'seq_encoder'
      product_seq_encoder_name = content_seq_encoder_name
      product_seq_encoder_reuse = True
    else:
      content_seq_encoder_name = 'content_seq_encoder'
      product_seq_encoder_name = 'product_seq_encoder'
      product_seq_encoder_reuse = False

    if domain_transfer_type == 'cross':
      content_seq_query = average_pooling(
        product_seq_emb[:, :10], product_seq_mask[:, :10], 1
      )  # [B, D]
      product_seq_query = average_pooling(
        content_seq_emb[:, :10], content_seq_mask[:, :10], 1
      )  # [B, D]
      content_seq_query = tf.expand_dims(content_seq_query, axis=1)
      product_seq_query = tf.expand_dims(product_seq_query, axis=1)
    else:
      content_seq_query = query
      product_seq_query = query

    content_user_emb = seq_encoder(
      seq_emb=content_seq_emb, seq_mask=content_seq_mask,
      encoder_type=seq_encoder_type,
      name=content_seq_encoder_name, reuse=False,
      query=product_seq_query, dropout_prob=dropout_prob,
      **kwargs
    )
    product_user_emb = seq_encoder(
      seq_emb=product_seq_emb, seq_mask=product_seq_mask,
      encoder_type=seq_encoder_type,
      name=product_seq_encoder_name, reuse=product_seq_encoder_reuse,
      query=content_seq_query, dropout_prob=dropout_prob,
      **kwargs
    )
    if domain_transfer_type == 'concat':
      user_emb = tf.concat([content_user_emb, product_user_emb], axis=1)
    elif domain_transfer_type == 'add':
      user_emb = content_user_emb + product_user_emb
    elif domain_transfer_type == 'att':
      user_emb = domain_attention_network(query, content_user_emb,
                                          product_user_emb)
    elif domain_transfer_type == 'gate':
      user_emb = gate_unit(query, content_user_emb, product_user_emb)
    else:
      raise NotImplementedError

  return user_emb


def domain_attention_network(query, key1, key2):
  """
  :param query: [B,1,D]
  :param key1: [B,D]
  :param key2: [B,D]
  :return:
  """
  dim = query.get_shape().as_list()[-1]
  key = tf.concat(
    [tf.expand_dims(key1, axis=1), tf.expand_dims(key2, axis=1)],
    axis=1
  )
  temp = query * key
  att_weight = tf.layers.dense(temp, dim, name='domain_transfer_attention')
  att_weight = tf.nn.softmax(att_weight, axis=1)
  result = tf.reduce_sum(att_weight * key, axis=1)
  return result


def gate_unit(query, key1, key2):
  """
  :param query: [B,1,D]
  :param key1: [B,D]
  :param key2: [B,D]
  :return:
  """
  dim = query.get_shape().as_list()[-1]
  query = tf.squeeze(query, axis=1)
  gate_weight = tf.layers.dense(query*key1+query*key2, dim, name='gate_unit')
  att_weight = tf.nn.sigmoid(gate_weight)
  result = att_weight * key1 + (1.0 - att_weight) * key2
  return result


def seq_encoder_return_att_weight(seq_emb, seq_mask, encoder_type,
                                  name='seq_encoder', reuse=False,
                                  query=None, dropout_prob=0.0,
                                  **kwargs):
  with tf.variable_scope(name, reuse=reuse):
    if encoder_type == 'multi_att':
      user_emb, att_weight = additive_attention_layer(
        keys=seq_emb, query=query, mask=seq_mask, dropout_prob=dropout_prob,
        activation=kwargs['att_act'], att_weight_type='multi',
        return_attention_weight=True
      )
      att_weight = tf.reduce_mean(att_weight, axis=-1)
    elif encoder_type == 'single_att':
      user_emb, att_weight = additive_attention_layer(
        keys=seq_emb, query=query, mask=seq_mask, dropout_prob=dropout_prob,
        activation=kwargs['att_act'], att_weight_type='single',
        return_attention_weight=True
      )
    elif encoder_type == 'dot_att':
      user_emb, att_weight = dot_scale_attention_layer(
        keys=seq_emb, query=query, mask=seq_mask, dropout_prob=dropout_prob,
        num_attention_heads=kwargs['num_attention_heads'],
        use_fc=kwargs['dot_att_use_fc'],
        return_attention_weight=True
      )
      if query is None:
        user_emb = tf.reduce_mean(
          user_emb * tf.expand_dims(seq_mask, axis=-1), axis=1)
    elif encoder_type == 'trm_att':
      user_emb, att_weight = transformer_attention_layer(
        keys=seq_emb, mask=seq_mask, query=query, dropout_prob=dropout_prob,
        num_attention_heads=kwargs['num_attention_heads'],
        return_attention_weight=True
      )
      if query is None:
        user_emb = tf.reduce_mean(
          user_emb * tf.expand_dims(seq_mask, axis=-1), axis=1)
    else:
      raise NotImplementedError

    return user_emb, att_weight


def compute_loss(labels,
                 logits,
                 loss_type,
                 pos_weight=1.0,
                 time_weight=1.0):
  batch_size = logits.get_shape().as_list()[0]
  if loss_type == 'bce':
    # binary cross entropy loss
    loss = tf.reduce_sum(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    ) * (1.0 / batch_size)
  elif loss_type == 'wbce':
    # weighted bce
    loss = tf.reduce_mean(
      tf.nn.weighted_cross_entropy_with_logits(
        logits=logits, targets=labels, pos_weight=pos_weight)
    )
  elif loss_type == 'bwbce':
    # batch weighted bce
    num_pos = tf.reduce_sum(labels)
    num_neg = batch_size - num_pos
    is_zero_neg = tf.cast(num_neg > 0, tf.float32)
    is_zero_pos = tf.cast(num_pos > 0, tf.float32)
    pos_weight = is_zero_neg * is_zero_pos * (num_neg / (num_pos + 1e-8)) + \
                 (1.0 - is_zero_pos) * 1.0 + \
                 (1.0 - is_zero_neg) * 1.0

    loss = tf.reduce_mean(
      tf.nn.weighted_cross_entropy_with_logits(
        logits=logits, targets=labels, pos_weight=pos_weight)
    )
  else:
    raise ValueError('invalid loss type: {}'.format(loss_type))
  return loss


def compute_metric(labels, logits):
  ctr_probs = tf.nn.sigmoid(logits)
  roc_auc = tf.metrics.auc(labels=labels, predictions=ctr_probs)
  predictions = tf.cast(tf.round(ctr_probs), tf.int32)
  accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)
  # precision = tf.metrics.precision_at_thresholds(
  #   labels=self.labels, predictions=predictions, thresholds=[0.5])
  # recall = tf.metrics.recall_at_thresholds(
  #   labels=self.labels,  predictions=predictions, thresholds=[0.5])
  precision = tf.metrics.precision(labels=labels, predictions=predictions)
  recall = tf.metrics.recall(labels=labels, predictions=predictions)
  f1_func = lambda p, r: p * r / (p + r) * 2
  f1 = f1_func(precision[0], recall[0]), f1_func(precision[1], recall[1])
  return roc_auc, accuracy, precision, recall, f1


def collaborative_cross_networks(input1,
                                 input2,
                                 mlp_layers,
                                 activation,
                                 use_cross=True,
                                 dropout_prob=0.0):
  act_func = get_activation(activation)
  mlp_layers = [int(unit) for unit in mlp_layers.split(',')]

  for i in range(len(mlp_layers)):
    mlp_net1 = tf.layers.dense(input1, mlp_layers[i], name='fc1_{}'.format(i))
    if use_cross:
      mlp_net1 += tf.layers.dense(input2, mlp_layers[i],
                                  name='cross_layer_{}'.format(i))
    mlp_net1 = tf.nn.dropout(mlp_net1, 1.0-dropout_prob)
    mlp_net1 = act_func(mlp_net1, name='input1_{}'.format(i))
    mlp_net2 = tf.layers.dense(input2, mlp_layers[i], name='fc2_{}'.format(i))
    if use_cross:
      mlp_net2 += tf.layers.dense(input1, mlp_layers[i],
                                  name='cross_layer_{}'.format(i), reuse=True)
    mlp_net2 = tf.nn.dropout(mlp_net2, 1.0-dropout_prob)
    mlp_net2 = act_func(mlp_net2, name='input2_{}'.format(i))

    input1 = mlp_net1
    input2 = mlp_net2

  return input1, input2


def build_embeddings(emb_name,
                     vocab_size,
                     emb_dim,
                     initializer=create_initializer(0.02)):
  embeddings = tf.get_variable(
    name=emb_name,
    shape=[vocab_size, emb_dim],
    initializer=initializer
  )
  return embeddings


def build_columns(column_key,
                  vocab_file=None,
                  vocab_list=None,
                  emb_dim=64):
  if vocab_list is None:
    with open(vocab_file, 'r') as reader:
      vocab_list = [line.strip('\n') for line in reader]

  emb_columns = tf.feature_column.embedding_column(
    tf.feature_column.categorical_column_with_vocabulary_list(
      key=column_key,
      vocabulary_list=vocab_list,
      dtype=tf.string
    ), emb_dim
  )
  return [emb_columns]


def local_load_data(filename, select_cols, sep='\t'):
  # for CMF or CoNet
  # col_indices = {
  #   "user_id": 0,
  #   'product_id': 1,
  #   'product_click': 2,
  #   "content_id": 3,
  #   "content_click": 4,
  #   "content_title_input_ids": 5,
  #   "content_title_input_len": 6,
  #   "content_image_feat": 7,
  #   "product_title_input_ids": 8,
  #   "product_title_input_len": 9,
  #   "product_image_feat": 10
  #
  # }

  # For single-task methods
  col_indices = {
    "user_id": 0,
    "content_id": 1,
    "content_click": 2,
    "content_seq": 3,
    "product_seq": 4,
    "content_time_seq": 5,
    "product_time_seq": 6,
    "content_seq_len": 7,
    "product_seq_len": 8,
    "content_title_input_ids": 9,
    "content_title_input_len": 10,
    "content_title_seq_input_ids": 11,
    "content_title_seq_input_len": 12,
    "product_title_seq_input_ids": 13,
    "product_title_seq_input_len": 14,
    "content_image_feat": 15,
    "content_image_seq_feat": 16,
    "product_image_seq_feat": 17,
    "content_seq_time_ids": 18,
    "product_seq_time_ids": 19,
    "posb": 20
  }
  select_cols = select_cols.split(',')
  indices = [col_indices[col] for col in select_cols]

  data = []
  with open(filename, 'r') as reader:
    for line in reader:
      arrs = line.strip('\n').split(sep)
      selected_data = [arrs[i] for i in indices]
      data.append(selected_data)
  return data


def stat_mtl_eval_results(model_dir, save_name='eval_results.txt'):
  eval_dir = os.path.join(model_dir, 'eval/')
  print('eval_dir: ', eval_dir)
  eval_results = early_stopping.read_eval_metrics(eval_dir)

  step_list = []
  loss_list = []
  content_auc_list = []
  product_auc_list = []
  content_acc_list = []
  product_acc_list = []

  for step, metrics in eval_results.items():
    if step < 0:
      continue
    step_list.append(step)
    loss_list.append(metrics['loss'])
    content_auc_list.append(metrics['metric/test/content_roc_auc'])
    product_auc_list.append(metrics['metric/test/product_roc_auc'])
    content_acc_list.append(metrics['metric/test/content_accuracy'])
    product_acc_list.append(metrics['metric/test/product_accuracy'])
    print(step, metrics)

  if len(content_auc_list) <= 0:
    return

  best_content_auc = max(content_auc_list)
  best_product_auc = max(product_auc_list)
  best_content_auc_epoch = np.argsort(-np.asarray(content_auc_list))[0] + 1
  best_product_auc_epoch = np.argsort(-np.asarray(product_auc_list))[0] + 1

  save_path = os.path.join(model_dir, save_name)
  with gfile.GFile(save_path, mode='w') as writer:
    cols_name = 'epoch xx: content_auc, product_auc, content_acc, ' \
                'product_acc, loss,    step\n'
    writer.write(cols_name)
    print(cols_name, end='')
    for i in range(len(step_list)):
      line_to_write = 'epoch {:0>2}:   {:.4f},      {:.4f},      {:.4f},' \
                      '      {:.4f}, {:.5f}, {:0>10}\n'.format(
        i+1,
        content_auc_list[i],
        product_auc_list[i],
        content_acc_list[i],
        product_acc_list[i],
        loss_list[i],
        step_list[i]
      )
      print(line_to_write, end='')
      writer.write(line_to_write)
    writer.write('best content epoch: {}, best content auc: {:.4f}'.format(
      best_content_auc_epoch, best_content_auc))
    writer.write('best product epoch: {}, best product auc: {:.4f}'.format(
      best_product_auc_epoch, best_product_auc))


# https://github.com/zhougr1993/DeepInterestNetwork/blob/master/din/model.py
def din_attention(queries, keys, keys_length):
  '''
    queries:     [B, H]
    keys:        [B, T, H]
    keys_length: [B]
  '''
  queries_hidden_units = queries.get_shape().as_list()[-1]
  queries = tf.tile(queries, [1, tf.shape(keys)[1]])
  queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])
  din_all = tf.concat([queries, keys, queries-keys, queries*keys], axis=-1)
  d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid,
                                  name='f1_att', reuse=tf.AUTO_REUSE)
  d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid,
                                  name='f2_att', reuse=tf.AUTO_REUSE)
  d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None,
                                  name='f3_att', reuse=tf.AUTO_REUSE)
  d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])
  outputs = d_layer_3_all
  # Mask
  key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])   # [B, T]
  key_masks = tf.expand_dims(key_masks, 1) # [B, 1, T]
  paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
  outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]

  # Scale
  outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

  # Activation
  outputs = tf.nn.softmax(outputs)  # [B, 1, T]

  # Weighted sum
  outputs = tf.matmul(outputs, keys)  # [B, 1, H]

  return tf.squeeze(outputs, 1)


def neural_collaborative_filtering(user_emb,
                                   target_emb,
                                   emb_size,
                                   mlp_layers,
                                   mlp_activation,
                                   dropout_prob):
  """perform nueral collaborative filtering algorithms."""
  act_func = get_activation(mlp_activation)
  mlp_layers = [int(unit) for unit in mlp_layers.split(',')]

  with tf.variable_scope('mf'):
    mf_output = user_emb * target_emb

  with tf.variable_scope('mlp'):
    mlp_input = tf.concat([user_emb, target_emb], axis=1)
    mlp_net = mlp_input

    for i in range(len(mlp_layers) - 1):
      mlp_net = tf.layers.dense(mlp_net, mlp_layers[i])
      mlp_net = tf.nn.dropout(mlp_net, 1.0 - dropout_prob)
      if act_func is not None:
        mlp_net = act_func(mlp_net)
    mlp_net = tf.layers.dense(mlp_net, mlp_layers[-1])
    mlp_net = tf.nn.dropout(mlp_net, 1.0 - dropout_prob)
    mlp_output = mlp_net
  with tf.variable_scope('pred_layer'):
    pred_input = tf.concat([mf_output, mlp_output], axis=1)
    pred_net = tf.layers.dense(pred_input, emb_size)
    pred_net = tf.nn.dropout(pred_net, 1.0 - dropout_prob)
    if act_func is not None:
      pred_net = act_func(pred_net)
    logits = tf.layers.dense(pred_net, 1)

  return tf.squeeze(logits, axis=1)


def matrix_factorization(user_emb, target_emb):
  with tf.variable_scope('mf'):
    mf_output = user_emb * target_emb
    logits = tf.reduce_sum(mf_output, axis=1)
  return tf.squeeze(logits)

