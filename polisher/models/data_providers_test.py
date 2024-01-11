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
"""Tests for google3.learning.genomics.polisher.models.data_providers."""

import json

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from polisher.make_images import test_utils
from polisher.models import data_providers


def get_test_dataset(inference: bool) -> tuple[str, int]:
  """Loads inference or training dataset and json summary."""
  if inference:
    dataset_path = 'tf_examples/haploid/inference/*.tfrecords.gz'
    summary_json = (
        'tf_examples/haploid/inference/make_images_inference.summary.json'
    )
    size_key = 'example_counter'
  else:
    dataset_path = 'tf_examples/haploid/train/*.tfrecords.gz'
    summary_json = 'tf_examples/haploid/train/make_images_training.summary.json'
    size_key = 'example_counter'
  file_pattern = test_utils.polisher_testdata(dataset_path)
  summary_json_path = test_utils.polisher_testdata(summary_json)
  summary = json.load(tf.io.gfile.GFile(summary_json_path))
  return file_pattern, summary[size_key]


class DataProvidersTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      dict(
          num_epochs=1,
          batch_size=1,
          inference=False,
          message='Test 1: batch size evenly divides # examples train.',
      ),
      dict(
          num_epochs=5,
          batch_size=1,
          inference=False,
          message='Test 2: multiple epochs train',
      ),
      dict(
          num_epochs=5,
          batch_size=10,
          inference=False,
          message='Test 3: batch size does not evenly divide # examples train',
      ),
      dict(
          num_epochs=1,
          batch_size=1,
          inference=True,
          message='Test 4: batch size evenly divides # examples inference',
      ),
      dict(
          num_epochs=5,
          batch_size=1,
          inference=True,
          message='Test 5: multiple epochs inference',
      ),
      dict(
          num_epochs=5,
          batch_size=10,
          inference=True,
          message='Test 6: batch does not evenly divide # examples inference',
      ),
  )
  def test_get_dataset(self, num_epochs, batch_size, inference, message):
    """Checks that batches are of expected size and all examples yielded."""

    file_pattern, dataset_size = get_test_dataset(inference)
    dataset = data_providers.get_dataset(
        file_pattern=file_pattern,
        num_epochs=num_epochs,
        batch_size=batch_size,
        inference=inference,
        drop_remainder=False,
        example_label_tuple=True,
    )

    total = 0
    for rows, label in dataset.as_numpy_iterator():
      # Last batch may contain fewer examples.
      if not inference:
        self.assertLen(rows, len(label), msg=message)
      self.assertLessEqual(len(rows), batch_size, msg=message)
      total += len(rows)
    self.assertEqual(total, num_epochs * dataset_size, msg=message)

  @parameterized.parameters(
      dict(
          reshuffle_each_iteration=True,
          message='Test 1: shuffle per epoch true',
      ),
      dict(
          reshuffle_each_iteration=False,
          message='Test 2: shuffle per epoch false',
      ),
  )
  def test_get_dataset_shuffle(self, reshuffle_each_iteration, message):
    """Checks that batches are of expected size and all examples yielded."""
    num_epochs = 2
    file_pattern, dataset_size = get_test_dataset(inference=True)
    dataset = data_providers.get_dataset(
        file_pattern=file_pattern,
        num_epochs=2,
        batch_size=dataset_size,
        inference=True,
        drop_remainder=False,
        example_label_tuple=True,
        reshuffle_each_iteration=reshuffle_each_iteration,
    )
    data_batches = [rows for rows, _ in dataset.as_numpy_iterator()]
    self.assertLen(
        data_batches, num_epochs, msg='Dataset shuffle: batch length.'
    )
    if reshuffle_each_iteration:
      self.assertNotAllEqual(data_batches[0], data_batches[1], msg=message)
    else:
      self.assertAllEqual(data_batches[0], data_batches[1], msg=message)

  @parameterized.parameters(
      dict(
          num_epochs=1,
          batch_size=1,
          inference=False,
          message='Test 1: batch size evenly divides # examples train',
      ),
      dict(
          num_epochs=5,
          batch_size=1,
          inference=False,
          message='Test 2: multiple epochs train',
      ),
      dict(
          num_epochs=5,
          batch_size=10,
          inference=False,
          message='Test 3: batch size does not evenly divide # examples train',
      ),
      dict(
          num_epochs=1,
          batch_size=1,
          inference=True,
          message='Test 4: batch size evenly divides # examples inference',
      ),
      dict(
          num_epochs=5,
          batch_size=1,
          inference=True,
          message='Test 5: multiple epochs inference',
      ),
      dict(
          num_epochs=5,
          batch_size=10,
          inference=True,
          message='Test 6: batch does not evenly divide # examples inference',
      ),
  )
  def test_dataset_metadata(self, num_epochs, batch_size, inference, message):
    """Checks that batches are of expected size and all examples yielded."""
    # Dataset sizes computed using gqui. Currently, eval set is empty because
    # the testdata only contains one molecule, which is added to training set
    # based on end position.
    file_pattern, dataset_size = get_test_dataset(inference)
    dataset = data_providers.get_dataset(
        file_pattern=file_pattern,
        num_epochs=num_epochs,
        batch_size=batch_size,
        inference=inference,
        drop_remainder=False,
        example_label_tuple=False,
    )

    total = 0
    for tf_example in dataset.as_numpy_iterator():
      name = tf_example['name']
      contig = tf_example['contig']
      active_position = tf_example['active_position']
      reference_positions = tf_example['reference_positions']
      reference_indices = tf_example['reference_indices']
      encoded_reference = tf_example['encoded_reference']
      image = tf_example['example']
      label = tf_example['label']
      name = tf_example['name']
      # Last batch may contain fewer examples.
      actual_batch_size = len(image)
      if not inference:
        self.assertLen(label, actual_batch_size, msg=f'{message}: Label.')
      self.assertLen(name, actual_batch_size, msg=f'{message}: Name.')
      self.assertLen(contig, actual_batch_size, msg=f'{message}: Contig.')
      self.assertLen(
          active_position,
          actual_batch_size,
          msg=f'{message}: Active position.',
      )
      self.assertLen(
          reference_positions,
          actual_batch_size,
          msg=f'{message}: Reference positions.',
      )
      self.assertLen(
          reference_indices,
          actual_batch_size,
          msg=f'{message}: Refernece indices.',
      )
      self.assertLen(
          encoded_reference,
          actual_batch_size,
          msg=f'{message}: Encoded reference.',
      )
      self.assertLessEqual(
          actual_batch_size, batch_size, msg=f'{message}: Actual batch size.'
      )
      total += actual_batch_size
    self.assertEqual(total, num_epochs * dataset_size)

  @parameterized.parameters(
      dict(
          num_epochs=1,
          batch_size=1,
          inference=False,
          message='Test 1: batch size evenly divides # examples train',
      ),
      dict(
          num_epochs=5,
          batch_size=1,
          inference=False,
          message='Test 2: multiple epochs train',
      ),
  )
  def test_example_indices(self, num_epochs, batch_size, inference, message):
    """Checks that batches are of expected size and all examples yielded."""
    file_pattern, _ = get_test_dataset(inference)
    dataset = data_providers.get_dataset(
        file_pattern=file_pattern,
        num_epochs=num_epochs,
        batch_size=batch_size,
        inference=inference,
        drop_remainder=False,
        example_label_tuple=False,
    )

    total_expected_rows = data_providers.get_total_rows()
    for data in dataset.as_numpy_iterator():
      single_example = data['example']
      feature_indices_dictionary = data_providers.get_feature_indices()
      rows_observed = 0
      # Iterate over each encoded features and see if the features are
      # not empty.
      for feature_name, feature_indices in feature_indices_dictionary.items():
        encoded_rows = single_example[:, slice(*feature_indices), :, :]
        self.assertNotEmpty(encoded_rows, msg=f'{message}: {feature_name}')
        rows_observed += encoded_rows.shape[1]
      self.assertEqual(
          rows_observed, total_expected_rows, msg=f'{message}: Total row length'
      )

  @parameterized.parameters(
      dict(
          inference=False,
          message='Test 1: processing training input',
      ),
      dict(
          inference=True,
          message='Test 2: processing inference input',
      ),
  )
  def test_process_input(self, inference, message):
    """Checks that batches are of expected size and all examples yielded."""
    file_pattern, _ = get_test_dataset(inference)
    file_patterns = tf.io.gfile.glob(file_pattern)
    ds = tf.data.TFRecordDataset(file_patterns, compression_type='GZIP')
    for data in ds.as_numpy_iterator():
      tf_example = data_providers.process_input(data, inference)
      self.assertNotEmpty(tf_example['name'], msg=f'{message}: Name')
      self.assertNotEmpty(tf_example['contig'], msg=f'{message}: contig')
      self.assertNotEmpty(
          tf_example['active_position'], msg=f'{message}: active_position'
      )
      self.assertNotEmpty(
          tf_example['encoded_reference'], msg=f'{message}: encoded_reference'
      )
      self.assertNotEmpty(
          tf_example['reference_positions'],
          msg=f'{message}: reference_positions',
      )
      self.assertNotEmpty(
          tf_example['reference_indices'], msg=f'{message}: reference_indices'
      )
      self.assertNotEmpty(tf_example['example'], msg=f'{message}: example')
      if inference:
        self.assertEmpty(tf_example['label'], msg=f'{message}: label inference')
      else:
        self.assertNotEmpty(tf_example['label'], msg=f'{message}: label train')

  @parameterized.parameters(
      dict(
          inference=False,
          message='Test 1: processing training input',
      ),
      dict(
          inference=True,
          message='Test 2: processing inference input',
      ),
  )
  def test_training_tuples(self, inference, message):
    """Checks that batches are of expected size and all examples yielded."""
    file_pattern, _ = get_test_dataset(inference)
    file_patterns = tf.io.gfile.glob(file_pattern)
    ds = tf.data.TFRecordDataset(file_patterns, compression_type='GZIP')
    for data in ds.as_numpy_iterator():
      tf_example = data_providers.process_input(data, inference)
      example, label = data_providers.tf_example_to_training_tuple(tf_example)
      if inference:
        self.assertEmpty(label, msg=f'{message}: label inference empty check.')
      else:
        self.assertNotEmpty(label, msg=f'{message}: label train empty check.')
      self.assertNotEmpty(example, msg=f'{message}: example train empty check.')
      self.assertAllEqual(
          tf_example['example'],
          example,
          msg=f'{message}: example equality test.',
      )
      self.assertAllEqual(
          tf_example['label'],
          label,
          msg=f'{message}: label equality test.',
      )

  @parameterized.parameters(
      dict(
          mode='train',
          batch_size=1,
          limit=5,
          message='Test 1: create_input_fn with train mode.',
      ),
      dict(
          mode='eval',
          batch_size=1,
          limit=10,
          message='Test 2: create_input_fn with eval mode.',
      ),
  )
  def test_create_input_fn(self, mode, batch_size, limit, message):
    """Checks that create_input_fn can be used to get dataset."""
    file_pattern, _ = get_test_dataset(inference=False)
    input_fn = data_providers.create_input_fn(
        file_pattern=file_pattern,
        mode=mode,
        batch_size=batch_size,
        limit=limit,
        drop_remainder=False,
        num_epochs=1,
    )

    dataset = input_fn()
    total = 0
    for rows, label in dataset.as_numpy_iterator():
      # Last batch may contain fewer examples.
      self.assertLen(rows, len(label), msg=f'{message}: limit size test.')
      self.assertLessEqual(
          len(rows), batch_size, msg=f'{message}: limit size test.'
      )
      total += len(rows)
    self.assertEqual(total, limit, msg=f'{message}: limit size test.')

  @parameterized.parameters(
      dict(
          mode='test',
          batch_size=1,
          limit=5,
          exception_msg=(
              'create_input_fn mode must be train or eval but found test'
          ),
          message='Test 1: Pass mode=test to create_input_fn.',
      ),
      dict(
          mode='evaluation',
          batch_size=1,
          limit=10,
          exception_msg=(
              'create_input_fn mode must be train or eval but found evaluation'
          ),
          message='Test 2: Pass mode=evaluation to create_input_fn.',
      ),
  )
  def test_create_input_fn_exception(
      self, mode, batch_size, limit, exception_msg, message
  ):
    """Test get_dataset_fn exceptions when mode is wrong."""
    with self.assertRaisesRegex(ValueError, exception_msg, msg=message):
      file_pattern, _ = get_test_dataset(inference=False)
      data_providers.create_input_fn(
          file_pattern=file_pattern,
          mode=mode,
          batch_size=batch_size,
          limit=limit,
          drop_remainder=False,
          num_epochs=1,
      )


if __name__ == '__main__':
  absltest.main()
