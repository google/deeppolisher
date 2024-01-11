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
"""Functions for yielding input arrays for models."""

from collections.abc import Callable
from typing import Optional, Union

import numpy as np
import tensorflow.compat.v2 as tf

from polisher.make_images import encoding

_TF_DATA_TYPE = tf.float32
_TF_DATA_READ_TYPE = tf.uint8
_BUFFER_SIZE = 1_000_000
_DEFAULT_PREFETCH_BUFFER_BYTES = 16_000_000


def get_tf_data_type() -> tf.dtypes.DType:
  """Returns the datatype set for tensorflow."""
  return _TF_DATA_TYPE


# Define base fields for TFRecords.
_PROTO_FEATURES_INFERENCE = {
    'name': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
    'contig': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
    'active_position': tf.io.FixedLenSequenceFeature(
        [], tf.int64, allow_missing=True
    ),
    'reference_positions': tf.io.FixedLenSequenceFeature(
        [], tf.int64, allow_missing=True
    ),
    'reference_indices': tf.io.FixedLenSequenceFeature(
        [], tf.int64, allow_missing=True
    ),
    'encoded_reference': tf.io.FixedLenSequenceFeature(
        [], tf.string, allow_missing=True
    ),
    'shape': tf.io.FixedLenFeature([3], tf.int64),
    'example': tf.io.FixedLenFeature([], tf.string),
}

_PROTO_FEATURES_WITH_LABEL = {
    'label': tf.io.FixedLenFeature([], tf.string)
} | _PROTO_FEATURES_INFERENCE


def get_total_rows() -> int:
  """Returns total rows in input tf.Examples.

  Returns:
    Total number of rows in the full example.
  """
  return sum(encoding.get_feature_depths().values())


def get_feature_indices() -> dict[str, tuple[int, int]]:
  """Return slices of each feature values.

  Returns:
    A dictionary containing feature_name as key and indices as values.
  """
  feature_depths = encoding.get_feature_depths()
  feature_indices = dict()
  row_index = 0
  for feature_name, feature_depth in feature_depths.items():
    feature_indices[feature_name] = (
        row_index,
        row_index + feature_depth,
    )
    row_index += feature_depth
  return feature_indices


def parse_example(
    proto_string: dict[str, tf.Tensor],
    inference: bool = False,
) -> dict[str, tf.Tensor]:
  """Parses serialized Training or Inference TF.Examples.

  Args:
    proto_string: Unparsed protobuf string for a single example.
    inference: If true then label metadata will not be parsed.

  Returns:
    A parsed example.
  """

  proto_features = _PROTO_FEATURES_INFERENCE
  if not inference:
    proto_features = _PROTO_FEATURES_WITH_LABEL

  parsed_features = tf.io.parse_single_example(
      serialized=proto_string, features=proto_features
  )
  return parsed_features


def format_example(example: tf.Tensor) -> tf.Tensor:
  """Returns model input matrix formatted based on input args.

  Args:
    example: A parsed example tensor.

  Returns:
    A concatenated tensor of all features extracted from the example.
  """
  feature_indices_dict = get_feature_indices()
  all_rows = []
  for feature_indices in feature_indices_dict.values():
    feature_slice = slice(*feature_indices)
    feature_rows = example[feature_slice]
    all_rows.append(feature_rows)

  feature_rows_concatenated = tf.concat(all_rows, axis=0)
  feature_rows_concatenated.set_shape(
      (get_total_rows(), encoding.get_max_length(), 1)
  )
  return feature_rows_concatenated


def process_input(
    proto_string: Union[tf.Tensor, bytes], inference: bool, ploidy: int = 1
) -> dict[str, tf.Tensor]:
  """Parses a serialized tf.Example to return an input, label, and metadata.

  Args:
    proto_string: A tensor containing the serialized tf.Example string.
    inference: Whether to parse tf.Examples for inference or training.
    ploidy: 1 for haploid, 2 for diploid applications.

  Returns:
    A dictionary with all attributes of an example.
  """
  content = parse_example(proto_string=proto_string, inference=inference)
  name = content['name']
  contig = content['contig']
  active_position = content['active_position']
  reference_positions = content['reference_positions']
  reference_indices = content['reference_indices']
  encoded_reference = tf.io.decode_raw(
      content['encoded_reference'], _TF_DATA_READ_TYPE
  )

  example = tf.io.decode_raw(content['example'], _TF_DATA_READ_TYPE)
  example = tf.reshape(example, content['shape'])
  example_formatted = format_example(example)
  example_formatted = tf.cast(example_formatted, _TF_DATA_TYPE)

  attributes = {
      'name': name,
      'contig': contig,
      'active_position': active_position,
      'encoded_reference': encoded_reference,
      'reference_positions': reference_positions,
      'reference_indices': reference_indices,
      'example': example_formatted,
  }

  if not inference:
    label = tf.io.decode_raw(content['label'], _TF_DATA_READ_TYPE)
    if ploidy == 1:
      label.set_shape((encoding.get_max_length()))
    elif ploidy == 2:
      # label shape is e.g. (2 sequences, 100 bp)
      label = tf.reshape(label, [2, encoding.get_max_length()])
    else:
      raise ValueError('Only ploidy values of 1 or 2 are supported.')
    label = tf.cast(label, _TF_DATA_TYPE)
  else:
    label = tf.convert_to_tensor(np.array([]))
  attributes['label'] = label
  return attributes


def tf_example_to_training_tuple(
    tf_example: dict[str, tf.Tensor]
) -> tuple[tf.Tensor, tf.Tensor]:
  """Return only example and read.

  Args:
   tf_example: A parsed tf_example dictionary.

  Returns:
   A tuple of (example, label). All other metadata are dropped.
  """
  example = tf_example['example']
  label = tf_example['label']
  return (example, label)


def get_dataset(
    file_pattern: str,
    inference: bool = False,
    ploidy: int = 1,
    num_epochs: Optional[int] = None,
    batch_size: int = 32,
    limit: int = -1,
    drop_remainder: bool = True,
    example_label_tuple: bool = False,
    reshuffle_each_iteration: bool = True,
    shuffle_dataset: bool = True,
) -> tf.data.Dataset:
  """Parses TFRecords and return a dataset.

  Args:
    file_pattern: File path(s).
    inference: Whether to parse tf.Examples for inference or training.
    ploidy: 1 for haploid, 2 for diploid applications.
    num_epochs: How many epochs for which to repeat. None to avoid repeating.
    batch_size: How many examples should be in each batch.
    limit: Max number of examples to get. Set to -1 for no limit.
    drop_remainder: Passed to TFRecordDataset.batch
    example_label_tuple: If True, output simplified format for training/eval as
      (rows, label)
    reshuffle_each_iteration: If True, data will be re-shuffled for each epoch.
    shuffle_dataset: If True, dataset will be shuffled.

  Returns:
    A dataset for batches for training and evaluation with the neural network.
  """

  def _process_input_helper(proto_string: tf.Tensor) -> dict[str, tf.Tensor]:
    return process_input(
        proto_string=proto_string, inference=inference, ploidy=ploidy
    )

  file_patterns = tf.io.gfile.glob(file_pattern)
  ds = tf.data.TFRecordDataset(
      file_patterns,
      buffer_size=_DEFAULT_PREFETCH_BUFFER_BYTES,
      compression_type='GZIP',
  )

  if shuffle_dataset:
    ds = ds.shuffle(
        buffer_size=_BUFFER_SIZE,
        reshuffle_each_iteration=reshuffle_each_iteration,
    )
  if num_epochs and num_epochs > 1:
    ds = ds.repeat(num_epochs)

  ds = ds.map(
      map_func=_process_input_helper, num_parallel_calls=tf.data.AUTOTUNE
  )

  # When training, we can only pick the label and example.
  if example_label_tuple:
    ds = ds.map(
        tf_example_to_training_tuple,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
  ds = ds.batch(batch_size=batch_size, drop_remainder=drop_remainder).prefetch(
      tf.data.AUTOTUNE
  )
  if limit >= 1:
    ds = ds.take(limit)
  return ds


def create_input_fn(
    file_pattern: str,
    mode: str,
    batch_size: int,
    num_epochs: int,
    limit: int = -1,
    drop_remainder: bool = True,
    ploidy: int = 1,
) -> Callable[[], tf.data.Dataset]:
  """Returns an input function that will return a tfrecord based dataset.

  This function is used during training to create an input function.
  Example usage:
  train_input_fn = data_providers.create_input_fn(...)
  eval_input_fn = data_providers.create_input_fn(...)
  train_dataset = strategy.experimental_distribute_dataset(train_input_fn())
  eval_dataset = strategy.experimental_distribute_dataset(eval_input_fn())

  Args:
    file_pattern: Input file pattern.
    mode: Mode from which the function is called in, must be train or eval.
    batch_size: Batch size to use for the dataset.
    num_epochs: Number of epochs to repeat.
    limit: Max number of examples to get. Set to -1 for no limit.
    drop_remainder: Passed to TFRecordDataset.batch.
    ploidy: 1 for haploid, 2 for diploid applications.

  Returns:
    Input function that returns tfrecord based dataset.
  """
  # This function is only called during training, so make sure we are either
  # in model training or model evaluation mode.
  if mode not in ['train', 'eval']:
    raise ValueError(
        f'create_input_fn mode must be train or eval but found {mode}'
    )

  def input_fn() -> tf.data.Dataset:
    """Prepares a dataset for training or evaluation."""
    # Only shuffle the dataset if in train mode.
    shuffle_dataset = mode == 'train'
    return get_dataset(
        file_pattern=file_pattern,
        num_epochs=num_epochs,
        batch_size=batch_size,
        inference=False,
        ploidy=ploidy,
        limit=limit,
        drop_remainder=drop_remainder,
        example_label_tuple=True,
        reshuffle_each_iteration=True,
        shuffle_dataset=shuffle_dataset,
    )

  return input_fn
