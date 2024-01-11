# Copyright (c) 2023, Google Inc.
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# 3. Neither the name of Google Inc. nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Custom metrics and losses."""

from typing import Callable, Mapping, Optional, Tuple, Union

import tensorflow as tf

from polisher.make_images import encoding
from polisher.models import data_providers


def _grab_at_index(x: tf.Tensor, index: int) -> tf.Tensor:
  y = tf.gather(x, [index], axis=1)
  y = tf.squeeze(y, axis=1)
  return y


def _split_diploid(x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
  x1 = _grab_at_index(x, index=0)
  x2 = _grab_at_index(x, index=1)
  return x1, x2


class PerExampleAccuracy(tf.keras.metrics.Accuracy):
  """Computes per-example accuracy."""

  def __init__(
      self, name: str = 'per_example_accuracy', ploidy: int = 1, **kwargs
  ):
    self.ploidy = ploidy
    super().__init__(name=name, **kwargs)

  def update_state(
      self,
      y_true: tf.Tensor,
      y_pred_scores: tf.Tensor,
      sample_weight: Optional[tf.Tensor] = None,
  ) -> None:
    """Accumulates running per-example accuracy."""
    if self.ploidy == 2:
      y_true, _ = _split_diploid(y_true)
      y_pred_scores, _ = _split_diploid(y_pred_scores)

    del sample_weight  # We use the mask calculated here instead.

    # Left shift the label and prediction and compare.
    y_true = tf.cast(
        left_shift_sequence(y_true), data_providers.get_tf_data_type()
    )
    # Convert pred scores and left shift.
    y_pred = tf.cast(
        tf.argmax(y_pred_scores, axis=-1), data_providers.get_tf_data_type()
    )
    y_pred = left_shift_sequence(y_pred)

    # Count matching positions per row.
    y_pred_matches = tf.math.count_nonzero(tf.equal(y_true, y_pred), axis=-1)
    # Count total positions per row.
    y_true_counts = tf.math.count_nonzero(tf.ones_like(y_pred), axis=-1)
    # Calculate accuracy where matching positions == total by row.
    super().update_state(y_pred_matches, y_true_counts)


class PerClassAccuracy(tf.keras.metrics.Accuracy):
  """Compute per-position accuracy for the given class."""

  def __init__(self, class_value: int, name: Optional[str] = None, **kwargs):
    self.class_value = class_value
    super().__init__(name=name, **kwargs)

  def update_state(
      self,
      y_true: tf.Tensor,
      y_pred_scores: tf.Tensor,
      sample_weight: Optional[tf.Tensor] = None,
  ) -> None:
    """Accumulates running per-position accuracy for the given class."""
    del sample_weight  # We use the mask calculated here instead.
    y_pred = tf.cast(
        tf.argmax(y_pred_scores, axis=-1), data_providers.get_tf_data_type()
    )
    mask = tf.cast(
        tf.equal(y_true, self.class_value), data_providers.get_tf_data_type()
    )
    super().update_state(y_true, y_pred, sample_weight=mask)


@tf.function
def left_shift_sequence(y_true: tf.Tensor) -> tf.int32:
  """Removes internal gaps and shifts labels to the left.

  Args:
    y_true: Label tensor.

  Returns:
    left shifted y_true
  """
  gap_token = encoding.get_gap_or_pad_encoding()
  shape = tf.shape(y_true)
  seq_length = shape[1]

  ixs = tf.broadcast_to(tf.range(seq_length), shape)
  # Sorting is performed in 2 stages. Sort internal gaps back by increasing
  # an index by the seq length, perform sort, then subtract to return
  # original index.
  sort_order = tf.sort(tf.where(y_true != gap_token, ixs, seq_length + ixs))
  sort_order = tf.where(
      sort_order < seq_length, sort_order, sort_order - seq_length
  )
  y_true_left_aligned = tf.gather(y_true, sort_order, axis=1, batch_dims=-1)
  return y_true_left_aligned


# Type aliases to represent "pointwise" cost functions for alignment loss.
SubsCostFn = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
InsCostFn = Callable[[tf.Tensor], tf.Tensor]


def xentropy_subs_cost_fn(
    y_true: tf.Tensor, y_pred: tf.Tensor, eps: float = 1e-7
) -> tf.Tensor:
  """Pointwise cross-entropy substitution cost function for alignment loss.

  Args:
    y_true: A tf.Tensor<int>[batch, m, n_tokens] representing the one-hot
      encoded ground-truth sequences.
    y_pred: A tf.Tensor<float>[batch, n, n_tokens] representing the scores for
      for predicted sequences. It is assumed that y_pred[b][l] lies in a k-dim
      probability simplex.
    eps: A small positive float. All scores in y_pred will be clipped to [eps, 1
      - eps] for numerical stability.

  Returns:
    A tf.Tensor<float>[batch, m, n] such that out[b][l1][l2] represents the
    (sparse) cross-entropy loss between y_true[b][l1] and y_pred[b][l2].
  """
  y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)
  y_true, y_pred = tf.expand_dims(y_true, 2), tf.expand_dims(y_pred, 1)
  return -tf.reduce_sum(tf.math.xlogy(y_true, y_pred), axis=-1)


def accuracy_subs_cost_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
  """Pointwise accuracy substitution cost function for alignment metric.

  Args:
    y_true: A tf.Tensor<int>[batch, m, n_tokens] representing the one-hot
      encoded ground-truth sequences.
    y_pred: A tf.Tensor<float>[batch, n, n_tokens] representing the scores for
      for predicted sequences. It is assumed that y_pred[b][l] lies in a k-dim
      probability simplex.

  Returns:
    A tf.Tensor<float>[batch, m, n] such that out[b][l1][l2] has value 1.0 if
    argmax(y_true[b][l1]) equals argmax(y_pred[b][l2]) being 0.0 otherwise.
  """
  dtype = y_pred.dtype
  y_true, y_pred = tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1)
  y_true, y_pred = tf.expand_dims(y_true, 2), tf.expand_dims(y_pred, 1)
  return tf.cast(y_true == y_pred, dtype)


def pbmm2_subs_cost_fn(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    *,
    matching_score: tf.Tensor,
    mismatch_penalty: tf.Tensor,
) -> tf.Tensor:
  """Pointwise match/mismatch substitution cost function for alignment metric.

  Args:
    y_true: A tf.Tensor<int>[batch, m] representing the ground-truth sequences.
    y_pred: A tf.Tensor<float>[batch, n] representing the (argmax-decoded)
      predicted sequences.
    matching_score: The score with which to reward matches.
    mismatch_penalty: The penalty with which to punish mismatches.

  Returns:
    A tf.Tensor<float>[batch, m, n] such that out[b][l1][l2] has value
    matching_score if y_true[b][l1] equals y_pred[b][l2] being -mismatch_penalty
    otherwise.
  """
  y_true, y_pred = tf.expand_dims(y_true, 2), tf.expand_dims(y_pred, 1)
  return tf.where(y_true == y_pred, matching_score, -mismatch_penalty)


def xentropy_ins_cost_fn(y_pred: tf.Tensor, eps: float = 1e-7) -> tf.Tensor:
  """Pointwise cross-entropy insertion cost function for alignment loss.

  Args:
    y_pred: A tf.Tensor<float>[batch, n, n_tokens] representing the scores for
      for predicted sequences. It is assumed that y_pred[b][l] lies in a k-dim
      probability simplex.
    eps: A small positive float. All scores in y_pred will be clipped to [eps, 1
      - eps] for numerical stability.

  Returns:
    A tf.Tensor<float>[batch, n] such that out[b][l] represents the
    cross-entropy loss between gap or pad tokens and y_pred[b][l].
  """
  gap_token = encoding.get_gap_or_pad_encoding()
  ins_scores = tf.clip_by_value(y_pred[..., gap_token], eps, 1 - eps)
  return -tf.math.log(ins_scores)


def wavefrontify(tensor: tf.Tensor) -> tf.Tensor:
  """Rearranges batch of input 2D tensors for vectorized wavefront algorithm.

  Args:
    tensor: A tf.Tensor<dtype>[batch, len1, len2].

  Returns:
    A single tf.Tensor<dtype>[len1 + len2 - 1, len1, batch] satisfying
      out[k][i][n] = t[n][i][k - i]
    if the RHS is well-defined, and 0 otherwise.
    In other words, for each len1 x len2 matrix t[n], out[..., n] is a
    (len1 + len2 - 1) x len1 matrix whose rows correspond to antidiagonals of
    t[n].
  """
  b, l1, l2 = tf.shape(tensor)[0], tf.shape(tensor)[1], tf.shape(tensor)[2]
  n_pad, padded_len = l1 - 1, l1 + l2 - 1

  ta = tf.TensorArray(tensor.dtype, size=l1, clear_after_read=True)
  for i in tf.range(l1):
    row_i = tf.squeeze(tf.slice(tensor, [0, i, 0], [b, 1, l2]), axis=1)
    row_i = tf.pad(row_i, [[0, 0], [n_pad, n_pad]])
    row_i = tf.slice(row_i, [0, n_pad - i], [b, padded_len])
    ta = ta.write(i, row_i)  # row_i[b, padded_len]
  ta = ta.stack()  # ta[l1, b, padded_len]

  return tf.transpose(ta, (2, 0, 1))  # out[padded_len, l1, b]


def wavefrontify_vec(tensor: tf.Tensor, len1: int) -> tf.Tensor:
  """Rearranges batch of 1D input tensors for vectorized wavefront algorithm.

  Args:
    tensor: A tf.Tensor<dtype>[batch, len2].
    len1: An integer corresponding to the length of y_true plus one.

  Returns:
    A single tf.Tensor<dtype>[len1 + len2 - 1, len1, batch] satisfying
      out[k][i][n] = t[n][k - i]
    if the RHS is well-defined, and 0 otherwise.
  """
  b, len2 = tf.shape(tensor)[0], tf.shape(tensor)[1]
  n_pad, padded_len = len1 - 1, len1 + len2 - 1

  ta = tf.TensorArray(tensor.dtype, size=len1, clear_after_read=True)
  for i in tf.range(len1):
    row_i = tf.pad(tensor, [[0, 0], [n_pad, n_pad]])
    row_i = tf.slice(row_i, [0, n_pad - i], [b, padded_len])
    ta = ta.write(i, row_i)  # row_i[b, padded_len]
  ta = ta.stack()  # ta[len1, b, padded_len]

  return tf.transpose(ta, (2, 0, 1))  # out[padded_len, len1, b]


class AlignmentLoss(tf.keras.losses.Loss):
  r"""Implements a differentiable alignment loss.

  Attributes:
    subs_cost_fn: A (batched) function $\Delta^{B \times L_1} \times \Delta^{B
      \times L_2} \rightarrow \mathbb{R}_{+}^{B \times L_1 \times L_2}$
      computing the "outer product" per-position costs for a batch of B
      sequences `y_true` and their corresponding predictions `y_pred`. It is
      assumed that $L_2 \ge L_1$ and $\Delta$ represents the k-dimensional
      probability simplex.
    ins_cost_fn: A (batched_ function $\Delta^{B \times L} \rightarrow
      \mathbb{R}_{+}^{B \times L}$ computing the per-position insertion cost for
      a batch of B predictions `y_pred`. \Delta$ represents the k-dimensional
      probability simplex.
    del_cost: A float representing the (constant) cost of deletions.
    loss_reg: A float representing the regularization strength. Set to None to
      disable regularization (i.e. to compute hard alignments).
    width: An int representing the width of the alignement path. Set to None to
      remove this constraint.
    reduction: (Optional) type of `tf.keras.losses.Reduction` to apply to loss.
      Default value is `AUTO`. When used in custom training loops under the
      scope of `tf.distribute.Strategy`, must be set to `NONE` or `SUM`.
  """

  def __init__(
      self,
      subs_cost_fn: SubsCostFn = xentropy_subs_cost_fn,
      ins_cost_fn: InsCostFn = xentropy_ins_cost_fn,
      del_cost: float = 1.0,
      loss_reg: Optional[float] = 1.0,
      width: Optional[int] = None,
      reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.AUTO,
  ):
    super().__init__(reduction=reduction)
    self.subs_cost_fn = subs_cost_fn
    self.ins_cost_fn = ins_cost_fn
    self.del_cost = del_cost
    self.loss_reg = loss_reg
    self.width = width

  def preprocess_y_true(
      self,
      y_true: tf.Tensor,
      dtype: tf.DType = tf.float32,
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Applies AlignmentLoss-specific preprocessing to labels tensor.

    Args:
      y_true: A tf.Tensor<[float, int]>[batch, m] representing the ground-truth
        sequences.
      dtype: The dtype for the one-hot encoded output tensor of sequence labels.

    Returns:
      A tuple (y_true_oh, seq_lens) such that
        +  y_true_oh is a tf.Tensor<dtype>[batch, m, n_tokens], where n_tokens
           is the number of tokens in encoding excluding the gap token.
           It contains a one-hot representation of the input y_true, with gap
           tokens removed and extra gap tokens appended if necessary.
        +  seq_lens is a tf.Tensor<int>[batch] containing the length of each
           label sequence in y_true, excluding any pad and gap tokens.
    """
    # Ensures y_true is of integer type.
    y_true = tf.cast(y_true, tf.int32)
    # Removes internal gaps, shifting sequences left and adding padding when
    # necessary.
    y_true = left_shift_sequence(y_true)
    # Computes per-example label sequence length, excluding padding.
    pad_token = encoding.get_gap_or_pad_encoding()
    seq_lens = tf.reduce_sum(tf.cast(y_true != pad_token, y_true.dtype), -1)
    # Converts y_true to one-hot.
    n_tokens = encoding.get_vocab_size()
    y_true_oh = tf.one_hot(y_true, depth=n_tokens, dtype=dtype)
    return y_true_oh, seq_lens

  def preprocess_y_pred(self, y_pred: tf.Tensor) -> tf.Tensor:
    # Ensures predicted scores add to one.
    y_pred = y_pred / tf.reduce_sum(y_pred, axis=-1, keepdims=True)
    return y_pred

  def alignment(
      self,
      subs_costs: tf.Tensor,
      ins_costs: tf.Tensor,
      del_cost: float,
      seq_lens: tf.Tensor,
      inf: float,
      dtype: tf.DType,
  ) -> tf.Tensor:
    """Computes the alignment score values.

    Args:
      subs_costs: A tf.Tensor<float>[batch, len_1, len_2] input matrix of
        substitution costs.
      ins_costs: A tf.Tensor<float>[batch, len_1] input vector of insertion
        costs.
      del_cost: A float, the cost of deletion.
      seq_lens: A tf.Tensor<int>[batch] input matrix of true sequence lengths.
      inf: A float with very high value.
      dtype: the data type of y_pred

    Returns:
      A tf.Tensor<float>[batch] of values of the alignment scores.
    """
    # Gathers shape variables.
    b, m = tf.shape(subs_costs)[0], tf.shape(subs_costs)[1]
    n = tf.shape(subs_costs)[2]  # We assume tf.shape(y_pred)[0] equals b.
    # Computes and rearranges cost tensors for vectorized wavefront iterations.
    subs_costs = wavefrontify(subs_costs)
    ins_costs = wavefrontify_vec(ins_costs, m + 1)

    # Sets up reduction operators.
    if self.loss_reg is None:
      minop = lambda t: tf.reduce_min(t, 0)
    else:
      loss_reg = tf.convert_to_tensor(self.loss_reg, dtype)
      minop = lambda t: -loss_reg * tf.reduce_logsumexp(-t / loss_reg, 0)

    # Initializes recursion.
    v_opt = tf.fill([b], inf)
    v_p2 = tf.pad(tf.fill([m - 1, b], inf), [[1, 0], [0, 0]])
    v_p1 = tf.concat(
        [
            tf.slice(ins_costs[0], [0, 0], [1, b]),
            tf.fill([1, b], del_cost),
            tf.fill([m - 1, b], inf),
        ],
        0,
    )
    # Precomputes auxiliary (constant) tensors used during the recursion.
    i_range = tf.range(m + 1, dtype=tf.int32)
    k_end = seq_lens + n  # Indexes antidiagonal containing last entry, w/o pad.
    # Indexes last entries in "wavefrontified" slices, accounting for padding.
    nd_indices = tf.stack([seq_lens, tf.range(b, dtype=seq_lens.dtype)], -1)

    # Runs forward recursion.
    for k in tf.range(2, m + n + 1):
      # Masks invalid entries in "wavefrontified" value tensor.
      j_range = k - i_range
      inv_mask = tf.logical_and(j_range >= 0, j_range <= n)[:, tf.newaxis]

      o_m = v_p2 + subs_costs[k - 2]  # [m, b]
      o_i = v_p1 + ins_costs[k - 1]  # [m + 1, b]
      v_p2 = v_p1[:-1]
      o_d = v_p2 + del_cost  # [m, b]

      v_p1 = tf.concat(
          [tf.slice(o_i, [0, 0], [1, b]), minop(tf.stack([o_m, o_i[1:], o_d]))],
          0,
      )
      v_p1 = tf.where(inv_mask, v_p1, inf)
      v_opt = tf.where(k_end == k, tf.gather_nd(v_p1, nd_indices), v_opt)

    return v_opt

  def weave_band(self, input_v: tf.Tensor, inf: float) -> tf.Tensor:
    """Transforms a band around the diagonal of the matrix in a tall matrix.

    Args:
      input_v: A tf.Tensor<float>[batch, len, len] batch of square input
        matrices.
      inf: a very large float.

    Returns:
      A tf.Tensor<float>[batch, 2 * len - 1, 2 * width + 1] such that
      input_v[i, j] is returned in out[i + j, i - j + width]. With input matrix
       A B C D
       E F G H
       I J K L
       M N O P
      the function returns, for width=1
       0 A 0
       E 0 B
       0 F 0
       J 0 G
       0 K 0
       O 0 L
       0 P 0
    """
    batch = tf.shape(input_v)[0]
    len_v = tf.shape(input_v)[1]
    width = tf.cast(self.width, dtype=tf.int32)
    n_diag = 2 * width + 1
    diags = tf.linalg.diag_part(
        input_v, k=(-width, width), padding_value=inf, align='LEFT_LEFT'
    )
    weave = tf.reshape(
        tf.stack([diags, tf.fill(tf.shape(diags), inf)], -1),
        [batch, n_diag, -1],
    )
    woven_band_tr = inf * tf.ones((n_diag, batch, 2 * len_v))
    for diff in tf.range(-width, width + 1):
      i = diff + width
      abs_diff = tf.abs(diff)
      padded_weave = tf.roll(weave[:, n_diag - 1 - i], abs_diff, -1)
      woven_band_tr = tf.tensor_scatter_nd_update(
          woven_band_tr, [[i]], padded_weave[tf.newaxis, ...]
      )
    return tf.transpose(woven_band_tr, (1, 2, 0))[:, :-1, :]

  def index_ending_band(
      self, len_1: tf.Tensor, seq_lens: tf.Tensor
  ) -> tf.Tensor:
    """Computes the indices at which to fetch the solution of the program."""
    batch = tf.shape(seq_lens)[0]
    i = seq_lens
    j = len_1 - tf.nn.relu(len_1 - seq_lens - self.width)
    sum_index = i + j
    diff_index = j - i + self.width
    range_batch = tf.range(batch)
    return tf.concat(
        [
            range_batch[..., tf.newaxis],
            sum_index[..., tf.newaxis],
            diff_index[..., tf.newaxis],
        ],
        axis=-1,
    )

  def banded_alignment(
      self,
      subs_costs: tf.Tensor,
      ins_costs: tf.Tensor,
      del_cost: float,
      seq_lens: tf.Tensor,
      inf: float,
      dtype: tf.DType,
  ) -> tf.Tensor:
    """Computes the alignment score values, with a band-restriction on the path.

    Args:
      subs_costs: A tf.Tensor<float>[batch, len_1, len_2] input matrix of
        substitution costs.
      ins_costs: A tf.Tensor<float>[batch, len_1] input vector of insertion
        costs.
      del_cost: A float, the cost of deletion.
      seq_lens: A tf.Tensor<int>[batch] input matrix of true sequence lengths.
      inf: A float with very high value.
      dtype: The data type of y_pred.

    Returns:
      A tf.Tensor<float>[batch] of values of the alignment scores.
    """
    batch = tf.shape(subs_costs)[0]
    len_1 = tf.shape(subs_costs)[1]
    len_2 = tf.shape(subs_costs)[2]
    val_trans = tf.zeros((len_1 + 1, len_2 + 1, batch))
    updates = [
        del_cost
        * tf.tile(
            tf.range(len_1 + 1, dtype=tf.float32)[..., tf.newaxis],
            multiples=[1, batch],
        )
    ]
    val_trans = tf.tensor_scatter_nd_update(val_trans, [[0]], updates)
    for i in tf.range(1, len_1 + 1):
      previous_row = val_trans[i - 1, 0]
      val_trans = tf.tensor_scatter_nd_update(
          val_trans, [[i, 0]], [previous_row + ins_costs[:, i - 1]]
      )
      values = tf.transpose(val_trans, [2, 1, 0])
    input_band = self.weave_band(values, inf)
    subs_band = self.weave_band(subs_costs, inf)
    ins_costs_pad = tf.pad(ins_costs, [[0, 0], [1, 0]], constant_values=0.0)
    # TODO: uphere
    insert_expand = tf.tile(
        ins_costs_pad[:, tf.newaxis, :], multiples=[1, len_1 + 1, 1]
    )
    insert_band = self.weave_band(insert_expand, inf)
    length = tf.shape(input_band)[1]
    # Sets up reduction operators.
    if self.loss_reg is None:
      minop = lambda t: tf.reduce_min(t, axis=-1)
    else:
      loss_reg = tf.convert_to_tensor(self.loss_reg, dtype)
      minop = lambda t: -loss_reg * tf.reduce_logsumexp(-t / loss_reg, axis=-1)
    for k in tf.range(2, length):
      input_minus_one = tf.pad(
          input_band[..., :-1], [[0, 0], [0, 0], [1, 0]], constant_values=inf
      )
      input_plus_one = tf.pad(
          input_band[..., 1:], [[0, 0], [0, 0], [0, 1]], constant_values=inf
      )
      min_tens = tf.stack(
          [
              input_band[:, k - 2, :] + subs_band[:, k - 2, :],
              input_plus_one[:, k - 1, :] + del_cost,
              input_minus_one[:, k - 1, :] + insert_band[:, k, :],
          ],
          axis=-1,
      )
      insert_mins = minop(min_tens)
      input_trans = tf.tensor_scatter_nd_update(
          tf.transpose(input_band, [1, 0, 2]), [[k]], [insert_mins]
      )
      input_band = tf.transpose(input_trans, [1, 0, 2])
    fetch_indices = self.index_ending_band(len_1, seq_lens)
    return tf.gather_nd(input_band, fetch_indices)

  def eval(
      self,
      y_true: tf.Tensor,
      y_pred: tf.Tensor,
      return_matches: bool = False,
  ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
    """Computes the alignment loss for a batch of sequences.

    Args:
      y_true: A tf.Tensor<[float, int]>[batch, m] representing the ground-truth
        sequences.
      y_pred: A tf.Tensor<float>[batch, n, n_tokens], (n >= m) representing the
        scores for predicted sequences.
      return_matches: If True, return an extra tensor representing the aligned
        base pairs.

    Returns:
      A tf.Tensor<float>[batch] with the value of the loss for each example. If
      return_matches is True, it also returns a tf.Tensor<float>[batch, m, n]
      whose entries represent the probability that y_true[b][i] is aligned to
      y_pred[b][j] according to the Gibbs distribution implied by the (soft)
      alignment model.
    """
    # Gathers type variables.
    dtype = y_pred.dtype
    # Defines an appropriate large positive float to represent "infinity".
    inf = tf.convert_to_tensor(1e9, dtype)  # TODO: float16 support?

    # Removes internal gaps, computes length excl. pad and converts to one-hot.
    y_true, seq_lens = self.preprocess_y_true(y_true)
    # Combines pad and gap tokens and ensures predicted scores add to be one.
    y_pred = self.preprocess_y_pred(y_pred)
    subs_costs = self.subs_cost_fn(y_true, y_pred)
    ins_costs = self.ins_cost_fn(y_pred)
    del_cost = tf.convert_to_tensor(self.del_cost, dtype)

    args = (subs_costs, ins_costs, del_cost, seq_lens, inf, dtype)
    fn = self.alignment if self.width is None else self.banded_alignment
    if return_matches:
      # TODO: replace by a custom backtracking implementation.
      with tf.GradientTape() as tape:
        tape.watch(subs_costs)
        loss_val = fn(*args)
      matches = tape.gradient(loss_val, subs_costs)
      return loss_val, matches
    else:
      return fn(*args)

  def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Computes the alignment loss for a batch of sequences.

    Args:
      y_true: A tf.Tensor<[float, int]>[batch, m] representing the ground-truth
        sequences.
      y_pred: A tf.Tensor<float>[batch, n, n_tokens], (n >= m) representing the
        scores for predicted sequences.

    Returns:
      A tf.Tensor<float>[batch] with the value of the loss for each example.
    """
    return self.eval(y_true, y_pred, return_matches=False)  # pytype: disable=bad-return-type  # dynamic-method-lookup


def preprocess_y_true_metric(y_true: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
  """Applies AlignmentMetric-specific preprocessing to labels tensor.

  Args:
    y_true: A tf.Tensor<[float, int]>[batch, m] representing the ground-truth
      sequences.

  Returns:
    A tuple (y_true, y_true_lens) such that
      +  y_true is a tf.Tensor<int>[batch, m]. It contains the input y_true,
          with gap tokens removed and extra gap tokens appended
          if necessary.
      +  y_true_lens is a tf.Tensor<int>[batch] containing the length of each
          label sequence in y_true, excluding any pad and gap tokens.
  """
  # Ensures y_true is of integer type.
  y_true = tf.cast(y_true, tf.int32)
  # Removes internal gaps, shifting sequences left and adding padding when
  # necessary.
  y_true = left_shift_sequence(y_true)
  # Computes per-example label sequence length, excluding padding.
  pad_token = encoding.get_gap_or_pad_encoding()
  y_true_lens = tf.reduce_sum(tf.cast(y_true != pad_token, y_true.dtype), -1)
  return y_true, y_true_lens


def preprocess_y_pred_metric(y_pred: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
  """Applies AlignmentMetric-specific preprocessing to predictions tensor.

  Args:
    y_pred: A tf.Tensor<[float, int]>[batch, m, n_tokens] representing the
      predicted tokens.

  Returns:
    A tuple (y_pred, y_pred_lens) such that
      +  y_pred is a tf.Tensor<int>[batch, m]. It contains the most likely
          token at each position of the input y_pred, with
          gap tokens removed and extra gap tokens appended
          if necessary.
      +  y_pred_lens is a tf.Tensor<int>[batch] containing the length of each
          predicted sequence in y_pred, excluding any pad and gap tokens.
  """
  # Find the most likely base per position.
  y_pred = tf.argmax(y_pred, axis=-1)
  # Ensures y_pred is of integer type.
  y_pred = tf.cast(y_pred, tf.int32)
  # Removes internal gaps, shifting sequences left and adding padding when
  # necessary.
  y_pred = left_shift_sequence(y_pred)
  # Computes per-example label sequence length, excluding padding.
  pad_token = encoding.get_gap_or_pad_encoding()
  y_pred_lens = tf.reduce_sum(tf.cast(y_pred != pad_token, y_pred.dtype), -1)
  return y_pred, y_pred_lens


class AlignmentMetric(tf.keras.metrics.Metric):
  r"""Implements an accelerator-friendly alignment metric.

  The metric is a (rough) approximation of PBMM2. At the moment, it is
  implemented as a Needleman-Wunsch (NW) alignment (i.e. global) with an affine
  gap penalty model.

  Differences:
    + PBMM2 uses a piece-wise affine gap penalty model min{o+ke,O+kE}, whereas
      AlignmentMetric so far implements a single component, i.e., o+ke.
    + TODO: double-check if PBMM2 is local / global and whether any
      heuristics are used instead of NW.

  Attributes:
    matching_score: The score of matches (-A).
    mismatch_penalty: The penalty of mismatches (-B).
    gap_open_penalty: The penalty of opening gaps (-o).
    gap_extend_penalty: The penalty of extending gaps (-e).
  """

  def __init__(
      self,
      matching_score: float = 2.0,
      mismatch_penalty: float = 5.0,
      gap_open_penalty: float = 5.0,
      gap_extend_penalty: float = 4.0,
      name: str = 'alignment_metric',
      **kwargs,
  ):
    super().__init__(name=name, **kwargs)
    self.matching_score = matching_score
    self.mismatch_penalty = mismatch_penalty
    # PBMM2 uses gap_open + gap_len * gap_extend but alignment routine used the
    # alterative definition gap_open + (gap_len - 1) * gap_extend.
    self.gap_open_penalty = gap_open_penalty + gap_extend_penalty
    self.gap_extend_penalty = gap_extend_penalty
    self._pid = tf.metrics.Mean()

  def alignment(
      self,
      y_true: tf.Tensor,
      y_pred: tf.Tensor,
  ) -> Tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
    """Computes the alignment loss for a batch of sequences.

    Args:
      y_true: A tf.Tensor<[float, int]>[batch, m] representing the ground-truth
        sequences.
      y_pred: A tf.Tensor<float>[batch, n, n_tokens], (n >= m) representing the
        scores for predicted sequences.

    Returns:
      A tuple (v_opt, paths, metric_values) such that:
      + v_opt is a tf.Tensor<float>[batch] with the optimal alignment score for
        each example.
      + paths is a tf.Tensor<float>[batch, m, n] whose entries represent the
        probability that y_true[b][i] is aligned to y_pred[b][j] according to
        the Gibbs distribution implied by the (soft) alignment model.
      + metric_values is a dict with the following (key: value) pairs
        - num_matches: tf.Tensor<int>[batch].
        - num_insertions: tf.Tensor<int>[batch].
        - num_deletions: tf.Tensor<int>[batch].
        - num_correct_matches: tf.Tensor<int>[batch].
        - alignment_length: tf.Tensor<int>[batch].
        - pid: tf.Tensor<float>[batch].
        These represent per-example alignment-derived metrics.
    """
    # Gathers type variables.
    dtype = y_pred.dtype
    # Gathers shape variables.
    b = tf.shape(y_true)[0]  # Equals tf.shape(y_pred)[0].
    # Note: the assert will not be executed on TPU.
    tf.debugging.assert_equal(
        x=b,
        y=tf.shape(y_pred)[0],
        message='y_true and y_pred must have the same batch size.',
    )
    m, n = tf.shape(y_true)[1], tf.shape(y_pred)[1]
    # Defines an appropriate large positive float to represent "infinity".
    inf = tf.convert_to_tensor(1e9, dtype)  # TODO: float16 support?
    # Convert parameters to tf.Tensor.
    matching_score = tf.convert_to_tensor(self.matching_score, dtype=dtype)
    mismatch_penalty = tf.convert_to_tensor(self.mismatch_penalty, dtype=dtype)
    gap_open = tf.convert_to_tensor(self.gap_open_penalty, dtype=dtype)
    gap_extend = tf.convert_to_tensor(self.gap_extend_penalty, dtype=dtype)

    # Removes internal gaps and computes length excluding padding.
    y_true, y_true_lens = preprocess_y_true_metric(y_true)
    # Applies argmax-decoding, removes internal gaps and computes length
    # excluding padding.
    y_pred, y_pred_lens = preprocess_y_pred_metric(y_pred)

    # Computes substitution costs for each pair of positions and rearranges the
    # tensor for vectorized wavefront iterations.
    subs_costs = pbmm2_subs_cost_fn(
        y_true,
        y_pred,
        matching_score=matching_score,
        mismatch_penalty=mismatch_penalty,
    )
    wavefrontified_subs_costs = wavefrontify(subs_costs)
    # Stacks gap penalties as tf.Tensor of shape [3, 1, 1] for broadcasting.
    gap_pens = tf.stack([gap_open, gap_open, gap_extend])[:, None, None]

    # Setups reduction operators.
    def reduce_max_with_argmax(
        t: tf.Tensor, axis: int = 0
    ) -> Tuple[tf.Tensor, tf.Tensor]:
      # Note(fllinares): I haven't yet managed to beat the performance of this
      # (wasteful) implementation with tf.argmax + tf.gather / tf.gather_nd :(
      t_max = tf.reduce_max(t, axis=axis)
      t_argmax = tf.argmax(t, axis=axis, output_type=tf.int32)
      return t_max, t_argmax

    # -------------------------------------------------------------------------
    # INITIALIZATION OF FORWARD RECURSION
    # -------------------------------------------------------------------------

    # Precomputes auxiliary (constant) tensors used during the recursion.
    i_range = tf.range(m + 1, dtype=tf.int32)
    # Indexes antidiagonal containing last entry, w/o pad.
    k_end = y_true_lens + y_pred_lens
    # Indexes last entries in "wavefrontified" slices, accounting for padding.
    samp_idx = tf.range(b, dtype=tf.int32)
    nd_indices = tf.stack([y_true_lens, samp_idx], axis=-1)

    # Initializes DP values for antidiagonal k = 0.
    # M: V[0][i, -i] = 0 if i = 0 else -inf.
    # I: V[1][i, -i] = -inf.
    # D: V[2][i, -i] = -inf.
    # v_all_p2 has shape [3, m, b] since the last col (i = m) is discarded.
    v_all_p2 = tf.concat(
        [
            tf.pad(
                tf.fill([1, m - 1, b], -inf),
                paddings=[[0, 0], [1, 0], [0, 0]],
                constant_values=tf.convert_to_tensor(0.0, dtype=dtype),
            ),
            tf.fill([2, m, b], -inf),
        ],
        axis=0,
    )
    # Initializes DP values for antidiagonal k = 1.
    # M: V[0][i, 1 - i] = -inf for all i.
    # I: V[1][i, 1 - i] = -gap_open if i = 0 else -inf.
    # D: V[2][i, 1 - i] = -gap_open if i = 1 else -inf.
    # v_all_p1 has shape [3, m + 1, b], for i = 0, 1, ..., m.
    v_all_p1 = tf.stack([
        tf.fill([m + 1, b], -inf),
        tf.pad(
            tf.fill([m, b], -inf),
            paddings=[[1, 0], [0, 0]],
            constant_values=-gap_open,
        ),
        tf.roll(
            tf.pad(
                tf.fill([m, b], -inf),
                paddings=[[1, 0], [0, 0]],
                constant_values=-gap_open,
            ),
            shift=1,
            axis=0,
        ),
    ])
    # Allocates memory for backtracking "directions" tensor.
    dir_all = tf.TensorArray(tf.int32, size=m + n + 1, clear_after_read=True)
    # Initializes backtracking "directions" for antidiagonal k = 0.
    # M: DIR[0][i, -i] = -1 if i = 0 else -2 (invalid).
    # I: DIR[1][i, -i] = -2 (invalid).
    # D: DIR[2][i, -i] = -2 (invalid).
    # d_p2 has shape [3, m + 1, b], for i = 0, 1, ..., m.
    dir_all_p2 = tf.concat(
        [
            tf.pad(
                tf.fill([1, m, b], -2),
                paddings=[[0, 0], [1, 0], [0, 0]],
                constant_values=tf.convert_to_tensor(-1, dtype=tf.int32),
            ),
            tf.fill([2, m + 1, b], -2),
        ],
        axis=0,
    )
    # Initializes backtracking "directions" for antidiagonal k = 1.
    dir_all = dir_all.write(0, tf.cast(dir_all_p2, tf.int32))
    # M: DIR[0][i, 1 - i] = -2 (invalid) for all i.
    # I: DIR[1][i, 1 - i] = 0 (M) if i = 0 else -2 (invalid).
    # D: DIR[2][i, 1 - i] = 0 (M) if i = 1 else -2 (invalid).
    # d_p1 has shape [3, m + 1, b], for i = 0, 1, ..., m.
    dir_all_p1 = tf.stack([
        tf.fill([m + 1, b], -2),
        tf.pad(
            tf.fill([m, b], -2),
            paddings=[[1, 0], [0, 0]],
            constant_values=tf.convert_to_tensor(0, dtype=tf.int32),
        ),
        tf.roll(
            tf.pad(
                tf.fill([m, b], -2),
                paddings=[[1, 0], [0, 0]],
                constant_values=tf.convert_to_tensor(0, dtype=tf.int32),
            ),
            shift=1,
            axis=0,
        ),
    ])
    dir_all = dir_all.write(1, tf.cast(dir_all_p1, tf.int32))

    # Initializes optimal score per sequence.
    # The initial value corresponds to the solution for the edge case where both
    # `y_true[idx]` and `y_pred[idx]` are "empty", i.e., consist only of gaps.
    # This is the only case in which `v_opt` and `m_opt` will *not* be updated
    # during the forward recursion.
    v_opt = tf.fill([b], tf.convert_to_tensor(0.0, dtype=dtype))
    # Initializes optimal last state (M/I/D) per sequence.
    m_opt = tf.fill([b], tf.convert_to_tensor(-1, dtype=tf.int32))

    # -------------------------------------------------------------------------
    # FORWARD RECURSION
    # -------------------------------------------------------------------------

    def maybe_update(
        k: Union[int, tf.Tensor],
        v_opt: tf.Tensor,
        m_opt: tf.Tensor,
        v_all_p1: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
      # Online computation of optimal alignment scores and final optimal state.
      v_opt_k, m_opt_k = reduce_max_with_argmax(v_all_p1, axis=0)
      # For each sequence, checks if the antidiagonal contains the entry
      # (y_true_lens[i], y_pred_lens[i]) for i = 1, ..., b.
      update_cond = k_end == k
      # If that's the case, fetches those entries from v_opt_k and m_opt_k and
      # stores them in v_opt and m_opt.
      v_opt = tf.where(update_cond, tf.gather_nd(v_opt_k, nd_indices), v_opt)
      m_opt = tf.where(update_cond, tf.gather_nd(m_opt_k, nd_indices), m_opt)
      return v_opt, m_opt

    # Performs out-of-loop updates for antidiagonal k = 1.
    # This covers the edge case `k_end[idx] = 1` for some pair `y_true[idx]`,
    # `y_pred[idx]`. That is, the case for which one of the two sequences
    # consists of only gaps and the other has exactly one base.
    v_opt, m_opt = maybe_update(1, v_opt, m_opt, v_all_p1)

    for k in tf.range(2, m + n + 1):
      # Masks invalid entries in "wavefrontified" value tensor.
      j_range = k - i_range
      inv_mask = tf.logical_and(j_range >= 0, j_range <= n)[None, :, None]

      o_match = v_all_p2 + wavefrontified_subs_costs[k - 2]  # [3, m, b].
      o_ins = v_all_p1[:2] - gap_pens[1:]  # [2, m + 1, b].
      # Assigns v_p2 <- v_p1 for next iteration, discarding the last column,
      # i.e., i = m - 1.
      v_all_p2 = v_all_p1[:, :-1]  # [3, m, b].
      o_del = v_all_p2 - gap_pens  # [3, m, b].

      v_match, dir_match = reduce_max_with_argmax(o_match, axis=0)  # [m, b].
      v_ins, dir_ins = reduce_max_with_argmax(o_ins, axis=0)  # [m + 1, b].
      v_del, dir_del = reduce_max_with_argmax(o_del, axis=0)  # [m, b].

      # Adds sentinel values to (0, i) entries of V[0] (M) and V[2] (D).
      # Only V[1] (I) has finite values for (0, 1) entries, which are equal to
      # -(gap_open + (i + 1) * gap_extend).
      v_match = tf.pad(v_match, [[1, 0], [0, 0]], constant_values=-inf)
      v_del = tf.pad(v_del, [[1, 0], [0, 0]], constant_values=-inf)
      # Adds sentinel values to (0, i) entries of DIR[0] (M) and DIR[2] (D).
      # Only DIR[1] (I) has valid values for (0, i) entries, which are equal to
      # 1 (previous state being also I) except (0, 1), which has value 0
      # (previous state being M by convention, see dir_p1 above).
      dir_match = tf.pad(dir_match, [[1, 0], [0, 0]], constant_values=-2)
      dir_del = tf.pad(dir_del, [[1, 0], [0, 0]], constant_values=-2)

      v_all_p1 = tf.where(inv_mask, tf.stack([v_match, v_ins, v_del]), -inf)
      dir_all = dir_all.write(k, tf.stack([dir_match, dir_ins, dir_del]))
      v_opt, m_opt = maybe_update(k, v_opt, m_opt, v_all_p1)

    # -------------------------------------------------------------------------
    # INITIALIZATION OF BACKWARD RECURSION
    # -------------------------------------------------------------------------

    # TODO: use an Enum to make these variables a bit more readable.

    # Creates auxiliary tensors to encode backtracking "actions".
    #   A match (0) sends us two antidiagonals up, one col left.
    #   An insert (1) sends us one antidiagonal up in the same col.
    #   A deletion (2) sends us one antidiagonal up, one col left.
    steps_k = tf.convert_to_tensor([-2, -1, -1], dtype=tf.int32)
    steps_i = tf.convert_to_tensor([-1, 0, -1], dtype=tf.int32)
    # Creates auxiliary tensor to encode alignment states.
    #   match state (1): {match, insert, deletion} -> {match}.
    #   insert open (2): {match, deletion} -> {insert}.
    #   insert extend (3): {insert} -> {insert}.
    #   delete open (4): {match, insert} -> {delete}.
    #   delete extend (5): {delete} -> {delete}.
    trans_enc = tf.constant(
        [[1, 1, 1], [2, 3, 2], [4, 4, 5]], dtype=tf.int32
    )  # [m_curr, m_prev]
    # Initializes additional backtracking variables.
    k_opt = k_end  # [b], next antidiagonal for backtracking.
    i_opt = y_true_lens  # [b], next col for backtracking.
    # Allocates memory for sparse representation of alignment.
    paths_sp = tf.TensorArray(tf.int32, size=m + n + 1, clear_after_read=True)

    # -------------------------------------------------------------------------
    # BACKWARD RECURSION
    # -------------------------------------------------------------------------

    for k in tf.range(m + n, -1, -1):
      # Safeguards against invalid indexing after stop condition is reached.
      safe_m_opt = tf.maximum(m_opt, 0)
      safe_i_opt = tf.maximum(i_opt, 0)
      # Computes tentative next indices for each alignment.
      k_opt_n = k_opt + tf.gather(steps_k, safe_m_opt)
      i_opt_n = i_opt + tf.gather(steps_i, safe_m_opt)
      # Computes tentative next state types for each alignment.
      m_opt_n_idx = tf.stack([safe_m_opt, safe_i_opt, samp_idx], -1)
      m_opt_n = tf.gather_nd(dir_all.read(k), m_opt_n_idx)
      # Computes tentative next sparse updates for paths tensor, safeguarding
      # against invalid indexing.
      safe_m_opt_n = tf.maximum(m_opt_n, 0)
      edges_n = tf.gather_nd(
          trans_enc, tf.stack([safe_m_opt, safe_m_opt_n], axis=-1)
      )
      paths_sp_n = tf.stack([samp_idx, i_opt, k_opt - i_opt, edges_n], -1)

      # Checks if start (0, 0) was reached during backtracking.
      reached_start = m_opt_n == -1
      # Indicates alignments to be updated in this iteration.
      cond = tf.logical_and(k_opt == k, tf.logical_not(reached_start))
      # Conditionally applies updates for each alignment.
      k_opt = tf.where(cond, k_opt_n, k_opt)
      i_opt = tf.where(cond, i_opt_n, i_opt)
      m_opt = tf.where(cond, m_opt_n, m_opt)
      paths_sp_k = tf.where(
          cond[:, None], paths_sp_n, tf.zeros([b, 4], tf.int32)
      )
      paths_sp = paths_sp.write(k, paths_sp_k)  # [0, 0, 0, 0] used as dummy up.

    # Applies sparse updates, building paths tensor.
    paths_sp = tf.reshape(paths_sp.stack(), [-1, 4])  # [((m + n + 1) * b), 4].
    paths_sp_idx, paths_sp_upd = paths_sp[:, :3], paths_sp[:, 3]
    paths = tf.scatter_nd(paths_sp_idx, paths_sp_upd, [b, m + 1, n + 1])

    # -------------------------------------------------------------------------
    # COMPUTE METRICS FROM ALIGNMENT
    # -------------------------------------------------------------------------
    matches_mask = paths == 1
    insertions_mask = tf.logical_or(paths == 2, paths == 3)
    deletions_mask = tf.logical_or(paths == 4, paths == 5)
    # Note: the following assumes that matching_score and mismatch_penalty are
    # both strictly greater than zero to avoid recomputing the pairwise comp.
    # between y_true and y_pred.
    correct_matches = tf.logical_and(matches_mask[:, 1:, 1:], subs_costs > 0)

    sum_positions = lambda t: tf.reduce_sum(tf.cast(t, tf.int32), axis=[1, 2])
    metric_values = {
        'num_matches': sum_positions(matches_mask),
        'num_insertions': sum_positions(insertions_mask),
        'num_deletions': sum_positions(deletions_mask),
        'num_correct_matches': sum_positions(correct_matches),
    }
    metric_values['alignment_length'] = (
        metric_values['num_matches']
        + metric_values['num_insertions']
        + metric_values['num_deletions']
    )
    # Computes percent identity (PID). PID is defined as 1.0 in the particular
    # case in which the ground-truth and predicted sequences are "empty", i.e.,
    # consist only of gap tokens.
    unsafe_pid = (
        metric_values['num_correct_matches'] / metric_values['alignment_length']
    )
    metric_values['pid'] = tf.where(
        metric_values['alignment_length'] > 0,
        tf.cast(unsafe_pid, dtype),
        tf.convert_to_tensor(1.0, dtype),
    )

    return v_opt, paths, metric_values

  def update_state(
      self,
      y_true: tf.Tensor,
      y_pred: tf.Tensor,
      sample_weight: Optional[tf.Tensor] = None,
  ):
    _, _, metric_values = self.alignment(y_true, y_pred)
    self._pid.update_state(metric_values['pid'], sample_weight=sample_weight)

  def result(self) -> tf.Tensor:
    return self._pid.result()

  def reset_states(self):
    self._pid.reset_states()


class DiploidAlignmentMetric(AlignmentMetric):
  """Finds highest percent identity for pairing 2 outputs with 2 labels."""

  def __init__(
      self,
      name: str = 'diploid_alignment_metric',
      take_min: bool = False,
      **kwargs,
  ):
    """Initialize diploid alignment metric.

    Args:
      name: Name for the metric for dashboards.
      take_min: If True, take the minimum (worst) alignment identity between the
        two possible pairings of two outputs with two labels. By default at
        False, take the max (best).
      **kwargs: Passed on to initialize parent AlignmentMetric.
    """
    super().__init__(name=name, **kwargs)
    self.take_min = take_min

  def update_state(
      self,
      y_true: tf.Tensor,
      y_pred: tf.Tensor,
      sample_weight: Optional[tf.Tensor] = None,
  ):
    label1, label2 = _split_diploid(y_true)
    y_pred1, y_pred2 = _split_diploid(y_pred)

    def alignment_pid(label, pred):
      _, _, metric_values = self.alignment(label, pred)
      return metric_values['pid']

    one_way = tf.math.add(
        alignment_pid(label1, y_pred1),
        alignment_pid(label2, y_pred2),
    )
    other_way = tf.math.add(
        alignment_pid(label1, y_pred2),
        alignment_pid(label2, y_pred1),
    )

    if self.take_min:
      # This is the reverse pair, to check how far this is from the best way.
      score = tf.math.minimum(one_way, other_way)
    else:
      # Best match is what we normally want for this metric:
      score = tf.math.maximum(one_way, other_way)

    self._pid.update_state(score, sample_weight=sample_weight)


class DistillationLoss(tf.keras.losses.Loss):
  """Computes the distillation loss between the student and teacher logits.

  Distillation loss is defined as the mean KL divergence between
  temperature-scaled softmax probabilities derived from the student and teacher
  models, with the mean taken over the positions in the window.

  Attributes:
    temperature: Temperature for softening probability distributions. Larger
      temperature gives softer distributions.
    reduction: (Optional) type of `tf.keras.losses.Reduction` to apply to loss.
      Default value is `AUTO`. When used in custom training loops under the
      scope of `tf.distribute.Strategy`, must be set to `NONE` or `SUM`.
  """

  def __init__(
      self,
      temperature: float = 1.0,
      reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.AUTO,
  ):
    super().__init__(reduction=reduction)
    self.temperature = temperature

  def call(
      self, teacher_logits: tf.Tensor, student_logits: tf.Tensor
  ) -> tf.Tensor:
    """Computes the distillation loss between student and teacher logits.

    Args:
      teacher_logits: A tf.Tensor<float>[batch, window_length, vocab_size]
        representing the logits produced by the teacher model.
      student_logits: A tf.Tensor<float>[batch, window_length, vocab_size]
        representing the logits produced by the student model.

    Returns:
      A tf.Tensor<float> with the value of the loss.
    """
    teacher_probs = tf.nn.softmax(teacher_logits / self.temperature, axis=-1)
    student_probs = tf.nn.softmax(student_logits / self.temperature, axis=-1)
    kl = tf.keras.losses.kld(teacher_probs, student_probs)
    # Take the mean across the positions in the sequence.
    return tf.math.reduce_mean(kl, axis=-1)


class DiploidAlignmentLoss(tf.keras.losses.Loss):
  """Computes alignment loss for two outputs versus two labels."""

  def __init__(
      self,
      reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.AUTO,
      **kwargs,
  ):
    super().__init__(reduction=reduction)

    # Very important to set `reduction=tf.keras.losses.Reduction.NONE` for
    # AlignmentLoss to get individual scores for each sequence, since we need to
    # be able to swap outputs to the labels they match best.
    self.alignment_loss = AlignmentLoss(
        **kwargs, reduction=tf.keras.losses.Reduction.NONE
    )

  def eval(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Computes the alignment loss for a batch of sequences.

    Args:
      y_true: A tf.Tensor<[float, int]>[batch, m] representing the ground-truth
        sequences.
      y_pred: A tf.Tensor<float>[batch, n, n_tokens], (n >= m) representing the
        scores for predicted sequences.

    Returns:
      A tf.Tensor<float>[batch] with the value of the loss for each example.
    """
    # Break y_true into label1 and label2.
    label1, label2 = _split_diploid(y_true)

    y_pred1, y_pred2 = _split_diploid(y_pred)

    # AlignmentLoss must have `reduction=tf.keras.losses.Reduction.NONE` to get
    # individual scores for each sequence instead of reducing across the batch,
    # since here we do an element-wise swap for identifying the best match.

    one_way = tf.math.add(
        self.alignment_loss(label1, y_pred1),
        self.alignment_loss(label2, y_pred2),
    )
    other_way = tf.math.add(
        self.alignment_loss(label1, y_pred2),
        self.alignment_loss(label2, y_pred1),
    )

    loss = tf.math.minimum(one_way, other_way)

    return loss

  def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return self.eval(y_true, y_pred)
