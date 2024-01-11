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
r"""Training binary for all neural network models using a custom training loop.

To use this binary for training a specific model, the corresponding config file
should be specified as input. Example usage:

CONFIG="//learning/genomics/polisher/models/model_configs.py:transformer_learn_values+HG002"
OUT_DIR=/tmp

time blaze run -c opt \
//learning/genomics/polisher/models:model_train_custom_loop -- \
  --params ${CONFIG} \
  --xm_runlocal \
  --alsologtostderr
"""

import datetime
import logging
import os
import random
from typing import Optional

from absl import app
from absl import flags
import ml_collections
from ml_collections.config_flags import config_flags
import tensorflow as tf

from polisher.models import convert_to_saved_model
from polisher.models import model_utils


os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE'] = 'False'
_MAIN_EVAL_METRIC_NAME = 'per_example_accuracy'


def get_main_eval_metric_name() -> str:
  return _MAIN_EVAL_METRIC_NAME


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file('params', None, 'Training configuration.')
_OUT_DIR = flags.DEFINE_string(
    'out_dir', None, 'Output path for logs and model checkpoints.'
)
_TPU = flags.DEFINE_string(
    'tpu',
    None,
    (
        'Name of the TPU to use. This gets '
        'populated automatically when using XManager.'
    ),
)
_TPU_TOPOLOGY = flags.DEFINE_string('tpu_topology', None, 'Tpu topology.')
_DEBUG = flags.DEFINE_bool(
    'debug', False, 'Enables dumping debug info for TensorBoard Debugger V2.'
)
_WRITE_CHECKPOINT_METRICS = flags.DEFINE_bool(
    'write_checkpoint_metrics',
    False,
    'Whether to write eval metrics for each checkpoint during training.',
)
_EVAL_AND_LOG_EVERY_STEP = flags.DEFINE_bool(
    'eval_and_log_every_step',
    False,
    (
        'Eval and log after every step. '
        'Use this e.g. for testing training and inspecting metrics locally.'
    ),
)
_EAGER = flags.DEFINE_bool(
    'eager',
    False,
    'Enable eager execution for tensorflow. Use this only for local debugging.',
)


def train_model(
    out_dir: str,
    params: ml_collections.ConfigDict,
    strategy: tf.distribute.Strategy,
    write_checkpoint_metrics: bool,
) -> None:
  """Trains the model under the given strategy and params."""
  # Freeze config dict here to ensure it is hashable.
  params = ml_collections.FrozenConfigDict(params)

  if out_dir is None:
    raise ValueError('--out_dir must be defined.')

  model_utils.save_params_as_json(out_dir, params)
  train_dataset, eval_dataset = model_utils.get_datasets(params, strategy)
  train_iterator = iter(train_dataset)
  eval_iterator = iter(eval_dataset)

  steps_per_epoch, steps_per_eval = model_utils.get_step_counts(
      params, _EVAL_AND_LOG_EVERY_STEP.value
  )
  logging.info(
      'Steps per epoch: %s, steps per eval: %s', steps_per_epoch, steps_per_eval
  )
  # Number of steps this model will train for.
  total_train_steps = steps_per_epoch * params['num_epochs']
  logging.info('Total training steps = %s', total_train_steps)

  with strategy.scope():
    logging.info('Building model.')
    if FLAGS.checkpoint:
      model = convert_to_saved_model.initialize_model(FLAGS.checkpoint)
      if model is None:
        raise ValueError(
            'Could not load model from checkpoint ', FLAGS.checkpoint
        )
    else:
      model = model_utils.get_model(params)
    logging.info('Done building model.')
    eval_checkpoint = os.path.join(out_dir, 'eval_checkpoint.txt')

    # Calculate the number of steps to decay the learning rate over.
    # Usually this number is the total training steps. However, since we train
    # the model for more epochs to obtain the final model, decay_steps is based
    # on the total training steps taken during final training.
    decay_steps = steps_per_epoch * params['num_epochs_for_decay']
    optimizer = model_utils.create_optimizer(params, decay_steps)
    train_loss = tf.keras.metrics.Mean(name='loss')
    train_metrics = model_utils.get_metrics(
        name_prefix='', ploidy=params['ploidy']
    )
    eval_loss = tf.keras.metrics.Mean(name='loss')
    eval_metrics = model_utils.get_metrics(
        name_prefix='', ploidy=params['ploidy']
    )
    loss_object = model_utils.get_loss(
        params, reduction=tf.keras.losses.Reduction.NONE
    )

    def compute_loss(labels, predictions):
      per_example_loss = loss_object(labels, predictions)
      # We divide per-replica losses by global batch size and sum this value
      # across all replicas to compute average loss scaled by global batch size.
      return tf.nn.compute_average_loss(
          per_example_loss, global_batch_size=params.batch_size
      )

    # model, optimizer, and checkpoint must be created under `strategy.scope`.
    # pytype: disable=wrong-arg-types  # typed-keras
    checkpoint, initial_epoch, initial_step_train = (
        model_utils.get_checkpoint_and_initial_epoch(
            model, optimizer, out_dir, eval_checkpoint
        )
    )
    # pytype: enable=wrong-arg-types  # typed-keras

  # Create summary writers
  train_writer = tf.summary.create_file_writer(os.path.join(out_dir, 'train'))
  eval_writer = tf.summary.create_file_writer(os.path.join(out_dir, 'eval'))

  def train_step(inputs):
    """Training StepFn."""
    logging.info('Within train step.')
    features, labels = inputs
    with tf.GradientTape() as tape:
      predictions = model(features)
      loss = compute_loss(labels, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss.update_state(loss)

    for metric in train_metrics:
      metric.update_state(labels, predictions)
    return loss

  def eval_step(inputs):
    """Eval StepFn."""
    logging.info('Within eval step.')
    features, labels = inputs
    predictions = model(features)
    loss = compute_loss(labels, predictions)
    eval_loss.update_state(loss)

    for metric in eval_metrics:
      metric.update_state(labels, predictions)

    return loss

  @tf.function
  def distributed_train_step(iterator):
    per_replica_losses = strategy.run(train_step, args=(next(iterator),))
    return strategy.reduce(
        tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
    )

  @tf.function
  def distributed_eval_step(iterator):
    per_replica_losses = strategy.run(eval_step, args=(next(iterator),))
    return strategy.reduce(
        tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
    )

  log_train_steps = 100
  log_eval_steps = 1000
  if _EVAL_AND_LOG_EVERY_STEP.value:
    log_train_steps = 1
    log_eval_steps = 1

  # Decide the best checkpoint using main eval metric.
  max_main_eval_metric = 0.0
  # From a list of eval metrics get the main eval metric.
  main_eval_metric = next(
      (
          metric
          for metric in eval_metrics
          if metric.name == get_main_eval_metric_name()
      ),
      None,
  )
  if not main_eval_metric:
    raise ValueError('No eval metric found.')

  for epoch in range(initial_epoch, params['num_epochs']):
    logging.info('Starting to run epoch: %s', epoch)
    train_time_start = datetime.datetime.now()
    for step_train in range(initial_step_train, steps_per_epoch):
      with tf.profiler.experimental.Trace('train', step_num=step_train, _r=1):
        distributed_train_step(train_iterator)
        # Log and reset train metrics.
        if optimizer.iterations % log_train_steps == 0:
          train_time_end = datetime.datetime.now()
          train_steps_per_second = (
              log_train_steps
              / (train_time_end - train_time_start).total_seconds()
          )
          with train_writer.as_default():
            model_utils.log_and_save_metrics(
                epoch=epoch,
                num_epochs=params['num_epochs'],
                step=step_train,
                total_steps=steps_per_epoch,
                optimizer=optimizer,
                metrics=[train_loss] + train_metrics,
                training=True,
                steps_per_second=train_steps_per_second,
            )
            train_time_start = datetime.datetime.now()
      # Log eval metrics, save checkpoint, and reset eval metrics every
      # log_eval_steps and at the end of training.
      if (optimizer.iterations % log_eval_steps == 0) or (
          optimizer.iterations == total_train_steps
      ):
        # Run evalution on the whole eval dataset and collect metrics.
        eval_time_start = datetime.datetime.now()
        for step_eval in range(1, steps_per_eval + 1):
          with tf.profiler.experimental.Trace('eval', step_num=step_eval, _r=1):
            distributed_eval_step(eval_iterator)
        eval_time_end = datetime.datetime.now()
        eval_steps_per_second = (
            steps_per_eval / (eval_time_end - eval_time_start).total_seconds()
        )
        # Save checkpoint.
        checkpoint_name = model_utils.save_checkpoint(
            checkpoint,
            out_dir,
            [eval_loss] + eval_metrics,
            write_checkpoint_metrics,
        )
        with tf.io.gfile.GFile(eval_checkpoint, 'w') as f:
          f.write(f'{checkpoint_name}\t{epoch}\t{step_train}')

        # Record the best checkpoint based on the main eval metric.
        main_eval_metric_val = float(main_eval_metric.result())
        if main_eval_metric_val >= max_main_eval_metric:
          max_main_eval_metric = main_eval_metric_val
          with tf.io.gfile.GFile(
              os.path.join(out_dir, 'best_checkpoint.txt'), 'w'
          ) as f:
            f.write(os.path.basename(checkpoint_name))
        # Log metrics on the eval set, this must be done at the end since
        # log_and_save_metrics will reset the eval metrics values.
        with eval_writer.as_default():
          model_utils.log_and_save_metrics(
              epoch=epoch,
              num_epochs=params['num_epochs'],
              step=step_eval,
              total_steps=steps_per_eval,
              optimizer=optimizer,
              metrics=[eval_loss] + eval_metrics,
              training=False,
              steps_per_second=eval_steps_per_second,
          )
        # Reset timer
        train_time_start = datetime.datetime.now()

    initial_step_train = 0


def train(
    out_dir: str,
    params: ml_collections.ConfigDict,
    tpu: Optional[str],
    tpu_topology: Optional[str],
    write_checkpoint_metrics: bool,
    debug: Optional[bool] = False,
):
  """Run the model training and return evaluation output."""
  model_utils.modify_params(params, tpu=tpu, tpu_topology=tpu_topology)
  random.seed(params.seed)
  tf.random.set_seed(params.seed)
  os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE'] = 'False'
  while True:
    try:
      if tpu is not None:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
      elif debug:
        strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')
      else:
        strategy = tf.distribute.MirroredStrategy()
      train_model(out_dir, params, strategy, write_checkpoint_metrics)
      break
    except tf.errors.UnavailableError:
      continue


def main(unused_args=None):
  if _EAGER.value:
    tf.config.experimental_run_functions_eagerly(True)
  train(
      _OUT_DIR.value,
      FLAGS.params,
      _TPU.value,
      _TPU_TOPOLOGY.value,
      _WRITE_CHECKPOINT_METRICS.value,
      _DEBUG.value,
  )


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'params',
  ])
  app.run(main)
