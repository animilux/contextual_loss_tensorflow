import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn import metrics

def contextual_loss(x, y, loss_type, h=0.5, channel_last=False) :
  if channel_last :
    x = tf.transpose(x, perm=[0,3,1,2])
    y = tf.transpose(y, perm=[0,3,1,2])

  if loss_type == 'cosine':
    dist = compute_cosine_distance(x, y)
  elif loss_type == 'l1':
    dist = compute_l1_distance(x, y)
  elif loss_type == 'l2':
    dist = compute_l2_distance(x, y)  

  d_min = K.min(dist, axis=2, keepdims=True)
  d_tilde = dist / (d_min + K.epsilon())
  w = K.exp((1 - d_tilde) / h)
  cx_ij = w / K.sum(w, axis=2, keepdims=True)
  cx = K.mean(K.max(cx_ij, axis=1), axis=1)

  cx_loss = K.mean(-K.log(cx))
  return cx_loss

def compute_l1_distance(x,y) :
  N, C, H, W = x.shape

  x_vec = tf.reshape(x, [N, C, -1])
  y_vec = tf.reshape(y, [N, C, -1])

  dist = tf.expand_dims(x_vec, axis=2) - tf.expand_dims(y_vec, axis=3)
  dist = K.sum(K.abs(dist), axis=1)
  dist = tf.transpose(dist, perm=[0,2,1])

  dist = K.clip(dist, min_value=0., max_value=100000)

  return dist

def compute_l2_distance(x,y) :
  N, C, H, W = x.shape

  x_vec = tf.reshape(x, [N, C, -1])
  y_vec = tf.reshape(y, [N, C, -1])

  x_s = K.sum(x_vec ** 2, axis=1, keepdims=True)
  y_s = K.sum(y_vec ** 2, axis=1, keepdims=True)

  A = tf.transpose(y_vec, perm=[0,2,1]) @ x_vec
  B = tf.transpose(x_s, perm=[0,2,1])

  dist = y_s - 2 * A + B
  dist = tf.transpose(dist, perm=[0,2,1])

  dist = K.clip(dist, min_value=0., max_value=100000)

  return dist

def compute_cosine_distance(x,y) :
  N, C, H, W = x.shape

  y_mu = K.mean(y, axis=[0,2,3], keepdims=True)

  x_centered = x - y_mu
  y_centered = y - y_mu

  x_normalized = x_centered / tf.norm(x_centered, ord=2, axis=1, keepdims=True)
  y_normalized = y_centered / tf.norm(y_centered, ord=2, axis=1, keepdims=True)

  x_normalized = tf.reshape(x_normalized, [N,C,-1])
  y_normalized = tf.reshape(y_normalized, [N,C,-1])

  x_normalized = tf.transpose(x_normalized, perm=[0,2,1])

  cosine_sim = tf.matmul(x_normalized, y_normalized)

  dist = 1 - cosine_sim

  return dist