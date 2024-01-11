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
r"""Inference binary for polisher.

Example usage:

# Haploid:
CHECKPOINT=<internal>
INPUT=${PWD}/learning/genomics/polisher/testdata/tf_examples/haploid/inference
OUTPUT=/tmp/haploid_inference

# Diploid:
CHECKPOINT=<internal>
INPUT=${PWD}/learning/genomics/polisher/testdata/tf_examples/diploid/inference/
OUTPUT=/tmp/diploid_inference

REF=${PWD}/learning/genomics/polisher/testdata/GRCh38_chr20_0_200000.fa

mkdir -p $OUTPUT

time blaze run -c opt \
//learning/genomics/polisher/inference:inference -- \
  --input_dir "${INPUT}" \
  --out_dir "${OUTPUT}" \
  --checkpoint "${CHECKPOINT}" \
  --reference_file "${REF}" \
  --sample_name "HG002" \
  --alsologtostderr 2>&1 | tee "${OUTPUT}"/log
"""

import collections
import dataclasses
import multiprocessing
import os
import random

from absl import flags
from absl import logging
import ml_collections
from ml_collections.config_flags import config_flags
import numpy as np
import pysam
import tensorflow as tf

from polisher.inference import inference_utils
from polisher.inference import vcf_writer
from polisher.models import data_providers
from polisher.models import model_utils
from absl import app




_PARAMS = config_flags.DEFINE_config_file(
    'params', None, 'Training configuration.'
)

_INPUT_DIR = flags.DEFINE_string(
    'input_dir', None, 'Path to input directory with example images.'
)
_CHECKPOINT = flags.DEFINE_string(
    'checkpoint', None, 'Path to checkpoint that will be loaded in.'
)
_OUT_DIR = flags.DEFINE_string(
    'out_dir', None, 'Output path for logs and model predictions.'
)
_REF_FASTA = flags.DEFINE_string(
    'reference_file', None, 'Path to reference fasta file.'
)
_SAMPLE_NAME = flags.DEFINE_string('sample_name', None, 'Name of the sample.')
_LIMIT = flags.DEFINE_integer(
    'limit',
    -1,
    'Limit to N records per train/tune dataset. -1 will evaluate all examples.',
)
_CPUS = flags.DEFINE_integer(
    'cpus',
    multiprocessing.cpu_count(),
    'Number of worker processes to use. Use 0 to disable parallel processing. '
    'Minimum of 2 CPUs required for parallel processing.',
)
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size',
    512,
    'Batch size to use during inference. Default: 512.',
)
_WRITE_EVERY_N_BATCH = flags.DEFINE_integer(
    'write_every_n_batch',
    500,
    'Write predictions after write_every_n_batch batches. Default: 500.',
)


def register_required_flags():
  flags.mark_flags_as_required([
      'out_dir',
      'checkpoint',
  ])


@dataclasses.dataclass
class PostProcessExample:
  """Parameters for post_processing method.

  ref_file: Path to reference or assembly fasta file.
  example: Dictrionary containing example.
  y_preds: Prediction vectors from the neural network
  quality_scores: Quality scores from the neural network.
  thread_id: A thread_id to discriminate which sequences to process.
  """

  ref_file: str
  example: dict[str, tf.Tensor]
  y_preds: np.ndarray
  quality_scores: np.ndarray
  thread_id: int


def post_processing(
    post_process_example: PostProcessExample,
) -> list[inference_utils.Variant]:
  """Iterate over each prediction, find candidates, and write to output VCF.

  Args:
    post_process_example: Example information for one batch.

  Returns:
    A list of variants found in the prediction for the batch.
  """
  fasta_file = pysam.FastaFile(post_process_example.ref_file)

  all_variants = []
  counters = collections.Counter()
  # Iterate over each prediction window and report variants.
  for example_i, _ in enumerate(post_process_example.example['contig']):
    variants = inference_utils.variants_from_example(
        example_i=example_i,
        batch=post_process_example.example,
        batch_predictions=post_process_example.y_preds,
        batch_quality_scores=post_process_example.quality_scores,
        fasta_file=fasta_file,
    )
    if variants:
      counters['total variants'] += len(variants)
      counters['examples with variants'] += 1
      all_variants.extend(variants)
    else:
      counters['examples without variants'] += 1
  print('Postprocessed a batch:', counters)
  return all_variants


def run_inference(
    model: tf.keras.Model,
    out_dir: str,
    ref_file: str,
    sample_name: str,
    limit: int = -1,
) -> None:
  """Runs inference with given model and dataset and writes out results.

  Args:
    model: A trained keras model used to run inference with.
    out_dir: Path to output directory.
    ref_file: Path to reference or assembly fasta file.
    sample_name: Name of sample to be used in the VCF.
    limit: Limit used to limit the dataset, -1 means all data will be processed.
  """
  # Create output summary and vcf file.
  run_summaries = {'total_windows': 0, 'total_variants': 0}
  vcf_file_writer = vcf_writer.VCFWriter(
      reference_file_path=ref_file,
      sample_name=sample_name,
      output_dir=out_dir,
      filename='polisher_output.unsorted',
  )
  logs_file_path = os.path.join(out_dir, 'inference' + '.log')
  logs_file = tf.io.gfile.GFile(logs_file_path, 'w')

  data_set_path = _INPUT_DIR.value + '/*tfrecords.gz'

  dataset = data_providers.get_dataset(
      file_pattern=data_set_path,
      num_epochs=1,
      batch_size=_BATCH_SIZE.value,
      limit=limit,
      drop_remainder=False,
      inference=True,
      example_label_tuple=False,
      shuffle_dataset=False,
  )
  # Start post-processing process
  predicted_batches = []
  variants = []
  for batch_counter, example in enumerate(dataset):
    logging.info(
        'Inference: Prediction completed on %d batches.',
        batch_counter + 1,
    )
    run_summaries['total_windows'] += len(example['example'])
    # Do model prediction and get the output from the model.
    softmax_output = model.predict_on_batch(example['example'])
    y_preds = np.argmax(softmax_output, axis=-1)
    error_prob = 1 - np.max(softmax_output, axis=-1)
    quality_scores = -10 * np.log10(error_prob)
    quality_scores = np.round(quality_scores, decimals=0)
    quality_scores = quality_scores.astype(dtype=np.int32).tolist()
    post_process_example = PostProcessExample(
        ref_file=ref_file,
        example=example,
        y_preds=y_preds,
        quality_scores=quality_scores,
        thread_id=batch_counter % _CPUS.value,
    )
    predicted_batches.append(post_process_example)
    # When we have processed _WRITE_EVERY_N_BATCH batches, we will post-process
    # them so the memory consumption remains within the limit.
    if len(predicted_batches) >= _WRITE_EVERY_N_BATCH.value:
      logging.info(
          'Inference: Starting post-processing on %d batches.',
          len(predicted_batches),
      )
      with multiprocessing.Pool(processes=_CPUS.value) as pool:
        variants_list = list(pool.map(post_processing, predicted_batches))
        pool.close()
        pool.join()
      logging.info(
          'Inference: Post-processing finished on %d batches.',
          len(predicted_batches),
      )
      # Flatten the list.
      for v in variants_list:
        variants.extend(v)
      predicted_batches = []

  if predicted_batches:
    logging.info(
        'Inference: Starting post-processing on %d batches.',
        len(predicted_batches),
    )
    with multiprocessing.Pool(processes=_CPUS.value) as pool:
      variants_list = list(pool.map(post_processing, predicted_batches))
      pool.close()
      pool.join()
    logging.info(
        'Inference: Post-processing finished on %d batches.',
        len(predicted_batches),
    )
    # Flatten the list.
    for v in variants_list:
      variants.extend(v)

  logging.info('Inference: writing variants.')
  if variants:
    variants.sort(key=lambda x: (x.contig, x.position_start))
    vcf_file_writer.write_vcf_records(variants)
    run_summaries['total_variants'] += len(variants)

  logging.info('Inference: Finished inference.')
  for key, count in run_summaries.items():
    logging.info('Total count: %s: %r', key, count)
  logs_file.write(
      'Total windows processed: %d.\n' % run_summaries['total_windows']
  )
  logs_file.write(
      'Total variants found: %d.\n' % run_summaries['total_variants']
  )


def setup_inference(
    out_dir: str,
    params: ml_collections.ConfigDict,
    checkpoint_path: str,
    ref_file: str,
    sample_name: str,
    limit: int,
) -> None:
  """Sets up run_inference by loading models.

  Args:
    out_dir: Path to output directory.
    params: All parameters to be used.
    checkpoint_path: Path to a trained keras model used to run inference with.
    ref_file: Path to reference or assembly fasta file.
    sample_name: Name of sample to be used in the VCF.
    limit: Limit used to limit the dataset, -1 means all data will be processed.
  """
  model_utils.modify_params(
      params, speedy=True, max_length=params.max_length, is_training=False
  )

  # Set seed for reproducibility.
  random.seed(params.seed)
  tf.random.set_seed(params.seed)
  strategy = tf.distribute.MirroredStrategy()

  with strategy.scope():
    model = model_utils.get_model(params)
    row_size = data_providers.get_total_rows()
    input_shape = (1, row_size, params.max_length, params.num_channels)
    model_utils.print_model_summary(model, input_shape)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(
        checkpoint_path
    ).expect_partial().assert_existing_objects_matched()

    run_inference(
        model=model,
        out_dir=out_dir,
        ref_file=ref_file,
        sample_name=sample_name,
        limit=limit,
    )


def main(unused_args=None):
  """Set up parallel processing and run inference."""
  if not _PARAMS.value:
    params = model_utils.read_params_from_json(
        checkpoint_path=_CHECKPOINT.value
    )
  else:
    params = _PARAMS.value

  # If ploidy is not set then we assume the model is haploid.
  if 'ploidy' not in params.keys():
    params.ploidy = 1

  # Set eval path in parameter.
  params.eval_path = _INPUT_DIR.value + '/*tfrecords.gz'

  # Create the output directory if doesn't exist.
  if not tf.io.gfile.isdir(_OUT_DIR.value):
    tf.io.gfile.makedirs(_OUT_DIR.value)

  setup_inference(
      out_dir=_OUT_DIR.value,
      params=params,
      checkpoint_path=_CHECKPOINT.value,
      ref_file=_REF_FASTA.value,
      sample_name=_SAMPLE_NAME.value,
      limit=_LIMIT.value,
  )


if __name__ == '__main__':
  logging.use_python_logging()
  app.run(main)
