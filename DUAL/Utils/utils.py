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
import math
from Utils import bert
from Utils import gradient_reversal_layer as grl
from functools import partial



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
  # [B, S, T, att_dim], [B, S, att_dim]
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
  return output


def dot_scale_attention_layer(keys,
                              query=None,
                              values=None,
                              mask=None,
                              dropout_prob=0.0,
                              num_attention_heads=4,
                              use_fc=True,
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
  return tf.reshape(context, [batch_size, from_seq_len, dim])


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

  for step, metrics in eval_results.items():
    if step < 0:
      continue
    eval_step_list.append(step)
    eval_loss_list.append(metrics['loss'])
    print(step, metrics)

  if len(eval_loss_list) <= 0:
    return

  best_loss = min(eval_loss_list)
  best_auc_epoch = np.argsort(np.asarray(eval_loss_list))[0] + 1

  save_path = os.path.join(model_dir, save_name)
  with gfile.GFile(save_path, mode='w') as writer:
    cols_name = 'epoch xx:   loss,  step\n'
    writer.write(cols_name)
    print(cols_name, end='')
    for i in range(len(eval_step_list)):
      line_to_write = 'epoch {:0>2}: {:.5f},{:0>10}\n'.format(
        i+1,
        eval_loss_list[i],
        eval_step_list[i]
      )
      print(line_to_write, end='')
      writer.write(line_to_write)
    writer.write('best epoch: {}, min loss: {:.4f}'.format(
      best_auc_epoch, best_loss))

  print('best epoch: {}, best auc: {:.4f}'.format(best_auc_epoch, best_loss))


def stat_eval_results_transfer(model_dir, save_name='eval_results.txt'):
  eval_dir = os.path.join(model_dir, 'eval/')
  print('eval_dir: ', eval_dir)
  eval_results = early_stopping.read_eval_metrics(eval_dir)
  #print(eval_results)

  eval_step_list = []
  eval_loss_list = []
  eval_content_auc_list = []
  eval_content_acc_list = []
  eval_content_precision_list = []
  eval_content_recall_list = []
  eval_content_f1_list = []
  eval_product_auc_list = []
  eval_product_acc_list = []
  eval_product_precision_list = []
  eval_product_recall_list = []
  eval_product_f1_list = []
  for step, metrics in eval_results.items():
    if step < 0:
      continue
    eval_step_list.append(step)
    eval_loss_list.append(metrics['loss'])
    eval_content_auc_list.append(metrics['metrics/test/content/roc_auc'])
    eval_product_auc_list.append(metrics['metrics/test/product/roc_auc'])
    eval_content_acc_list.append(metrics['metrics/test/content/accuracy'])
    eval_product_acc_list.append(metrics['metrics/test/product/accuracy'])
    eval_content_precision_list.append(metrics['metrics/test/content/precision'])
    eval_product_precision_list.append(metrics['metrics/test/product/precision'])
    eval_content_recall_list.append(metrics['metrics/test/content/recall'])
    eval_product_recall_list.append(metrics['metrics/test/product/recall'])
    eval_content_f1_list.append(metrics['metrics/test/content/f1'])
    eval_product_f1_list.append(metrics['metrics/test/product/f1'])

    #print(step, metrics)

  if len(eval_content_auc_list) <= 0:
    return

  best_auc = max(eval_content_auc_list)
  best_auc_epoch = np.argsort(-np.asarray(eval_content_auc_list))[0] + 1

  save_path = os.path.join(model_dir, save_name)
  with gfile.GFile(save_path, mode='w') as writer:

    cols_name = 'epoch xx:      auc,       precision,     recall,         f1,' \
                '           acc,      loss,     step\n'
    writer.write(cols_name)
    print(cols_name, end='')
    for i in range(len(eval_step_list)):
      line_to_write = 'epoch {:0>2}: {:.4f}/{:.4f},{:.4f}/{:.4f},' \
                      '{:.4f}/{:.4f},{:.4f}/{:.4f},' \
                      '{:.4f}/{:.4f},{:.5f},{:0>10}\n'.format(
        i+1,
        eval_content_auc_list[i],
        eval_product_auc_list[i],
        eval_content_precision_list[i],
        eval_product_precision_list[i],
        eval_content_recall_list[i],
        eval_product_recall_list[i],
        eval_content_f1_list[i],
        eval_product_f1_list[i],
        eval_content_acc_list[i],
        eval_product_acc_list[i],
        eval_loss_list[i],
        eval_step_list[i]
      )
      print(line_to_write, end='')
      writer.write(line_to_write)
    writer.write('best epoch: {}, best auc: {:.4f}'.format(
      best_auc_epoch, best_auc))

  print('best epoch: {}, best auc: {:.4f}'.format(best_auc_epoch, best_auc))


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


def compute_loss(labels,
                 logits,
                 loss_type,
                 pos_weight=1.0,
                 time_weight=1.0):
  batch_size = logits.get_shape().as_list()[0]
  if loss_type == 'bce':
    # binary cross entropy loss
    loss = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
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


def local_load_data(filename, indices, sep='\t'):
  data = []
  with open(filename, 'r') as reader:
    for line in reader:
      arrs = line.strip('\n').split(sep)
      selected_data = [arrs[i] for i in indices]
      data.append(selected_data)
  return data


def correlation_loss(source_samples, target_samples, weight, scope=None):
  """Adds a similarity loss term, the correlation between two representations.
  Args:
    source_samples: a tensor of shape [num_samples, num_features]
    target_samples: a tensor of shape [num_samples, num_features]
    weight: a scalar weight for the loss.
    scope: optional name scope for summary tags.
  Returns:
    a scalar tensor representing the correlation loss value.
  """
  with tf.name_scope('corr_loss'):
    source_samples -= tf.reduce_mean(source_samples, 0)
    target_samples -= tf.reduce_mean(target_samples, 0)

    source_samples = tf.nn.l2_normalize(source_samples, 1)
    target_samples = tf.nn.l2_normalize(target_samples, 1)

    source_cov = tf.matmul(tf.transpose(source_samples), source_samples)
    target_cov = tf.matmul(tf.transpose(target_samples), target_samples)

    corr_loss = tf.reduce_mean(tf.square(source_cov - target_cov)) * weight
  return corr_loss


def dann_loss(source_samples, target_samples, weight=1.0, scope='dann'):
  """Adds the domain adversarial (DANN) loss.
  Args:
    source_samples: a tensor of shape [num_samples, num_features].
    target_samples: a tensor of shape [num_samples, num_features].
    weight: the weight of the loss.
    scope: optional name scope for summary tags.
  Returns:
    a scalar tensor representing the correlation loss value.
  """
  with tf.variable_scope(scope):
    batch_size = tf.shape(source_samples)[0]
    samples = tf.concat(axis=0, values=[source_samples, target_samples])

    domain_selection_mask = tf.concat(
        axis=0, values=[tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))])

    # Perform the gradient reversal and be careful with the shape.
    global_step = tf.train.get_or_create_global_step()
    warmup_step = tf.constant(200000.0, tf.float32)
    step = tf.cast(global_step, tf.float32) / warmup_step
    grl_lambda = 2. / (1. + tf.exp(-10. * step)) - 1
    samples = grl.flip_gradient(samples, grl_lambda)
    tf.summary.scalar('grl_lambda', grl_lambda)

    logits = tf.layers.dense(samples, 1, name='grl_layer')

    domain_predictions = tf.sigmoid(logits)

  domain_loss = tf.losses.log_loss(
      domain_selection_mask, domain_predictions, weights=weight)

  domain_accuracy = tf.metrics.accuracy(
    tf.round(domain_predictions), domain_selection_mask)

  return domain_loss, domain_accuracy


def difference_loss(private_samples, shared_samples, weight=1.0, name=''):
  """Adds the difference loss between the private and shared representations.
  Args:
    private_samples: a tensor of shape [num_samples, num_features].
    shared_samples: a tensor of shape [num_samples, num_features].
    weight: the weight of the incoherence loss.
    name: the name of the tf summary.
  """
  private_samples -= tf.reduce_mean(private_samples, 0)
  shared_samples -= tf.reduce_mean(shared_samples, 0)

  private_samples = tf.nn.l2_normalize(private_samples, 1)
  shared_samples = tf.nn.l2_normalize(shared_samples, 1)

  correlation_matrix = tf.matmul(
      private_samples, shared_samples, transpose_a=True)

  cost = tf.reduce_mean(tf.square(correlation_matrix)) * weight
  cost = tf.where(cost > 0, cost, 0, name='value')

  return cost


def text_cnn(text_emb, window_size='3,5', dropout_prob=0.0):
  """
  perform text CNN with attention model
  :param text_emb: [B, S, D]
  :param text_mask: [B, S]
  :param window_size: [3,5]/[5,7]/[3,5,7,9]/[3]/[5]/[7]/[9]
  :param att_type: query-based attention model
  :param query: query is category word embeddings, [B, D]
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


def compute_pairwise_distances(x, y):
  """Computes the squared pairwise Euclidean distances between x and y.

  Args:
    x: a tensor of shape [num_x_samples, num_features]
    y: a tensor of shape [num_y_samples, num_features]

  Returns:
    a distance matrix of dimensions [num_x_samples, num_y_samples].

  Raises:
    ValueError: if the inputs do no matched the specified dimensions.
  """

  if not len(x.get_shape()) == len(y.get_shape()) == 2:
    raise ValueError('Both inputs should be matrices.')

  if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
    raise ValueError('The number of features should be the same.')

  norm = lambda x: tf.reduce_sum(tf.square(x), 1)

  # By making the `inner' dimensions of the two matrices equal to 1 using
  # broadcasting then we are essentially substracting every pair of rows
  # of x and y.
  # x will be num_samples x num_features x 1,
  # and y will be 1 x num_features x num_samples (after broadcasting).
  # After the substraction we will get a
  # num_x_samples x num_features x num_y_samples matrix.
  # The resulting dist will be of shape num_y_samples x num_x_samples.
  # and thus we need to transpose it again.
  return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))


def gaussian_kernel_matrix(x, y, sigmas):
  r"""Computes a Guassian Radial Basis Kernel between the samples of x and y.

  We create a sum of multiple gaussian kernels each having a width sigma_i.

  Args:
    x: a tensor of shape [num_samples, num_features]
    y: a tensor of shape [num_samples, num_features]
    sigmas: a tensor of floats which denote the widths of each of the
      gaussians in the kernel.
  Returns:
    A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
  """
  beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))

  dist = compute_pairwise_distances(x, y)

  s = tf.matmul(beta, tf.reshape(dist, (1, -1)))

  return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))



def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
  r"""Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.

  Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
  the distributions of x and y. Here we use the kernel two sample estimate
  using the empirical mean of the two distributions.

  MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
              = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },

  where K = <\phi(x), \phi(y)>,
    is the desired kernel function, in this case a radial basis kernel.

  Args:
      x: a tensor of shape [num_samples, num_features]
      y: a tensor of shape [num_samples, num_features]
      kernel: a function which computes the kernel in MMD. Defaults to the
              GaussianKernelMatrix.

  Returns:
      a scalar denoting the squared maximum mean discrepancy loss.
  """
  with tf.name_scope('MaximumMeanDiscrepancy'):
    # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
    cost = tf.reduce_mean(kernel(x, x))
    cost += tf.reduce_mean(kernel(y, y))
    cost -= 2 * tf.reduce_mean(kernel(x, y))

    # We do not allow the loss to become negative.
    cost = tf.where(cost > 0, cost, 0, name='value')
  return cost


def mmd_loss(source_samples, target_samples, weight=10.0, scope=None):
  """Adds a similarity loss term, the MMD between two representations.

  This Maximum Mean Discrepancy (MMD) loss is calculated with a number of
  different Gaussian kernels.

  Args:
    source_samples: a tensor of shape [num_samples, num_features].
    target_samples: a tensor of shape [num_samples, num_features].
    weight: the weight of the MMD loss.
    scope: optional name scope for summary tags.

  Returns:
    a scalar tensor representing the MMD loss value.
  """
  sigmas = [
      1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
      1e3, 1e4, 1e5, 1e6
  ]
  gaussian_kernel = partial(gaussian_kernel_matrix, sigmas=tf.constant(sigmas))

  loss_value = maximum_mean_discrepancy(
      source_samples, target_samples, kernel=gaussian_kernel)
  loss_value = tf.maximum(1e-4, loss_value) * weight

  return loss_value







