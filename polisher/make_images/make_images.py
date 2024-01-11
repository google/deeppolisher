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
r"""Generates tensor examples (images) from read alignments to the reference.

This script takes read alignments to a reference or assembly and creates
tensor examples (tensorflow records) of the read alignment to the reference.
Currently the method is desigend to work with haplotype 1 and haplotype 2 reads
independently and only works on diploid human genomes.

Description:
This script takes in haplotagged reads from a diploid sample aligned to a
reference and creates examples for each haplotype independently. The expectation
is the reads to be haplotagged only with HP:0, HP:1 and HP:2. The method works
in this way:
1) Bin reads into two haplotype sets {read_hp1 and read_hp2}, reads that are
haplotagged as HP:0 will be put into both HP1 and HP2 bins as their haplotag
was undetermined.
2) Calculate how much padding is required due to insertions observed in reads.
3) Encode reference and reads with features with padding. The features that we
currently use: {reference, read_bases, match/mismatch, base_quality,
mapping_quality}
4) Slide through the entire example to create chunks of examples so we can run
a DNN model.
"""

import collections
import multiprocessing
from typing import Sequence

from absl import flags
from absl import logging

from polisher.make_images import region_processor
from polisher.make_images import utils_make_images
from absl import app



_BAM_FILE = flags.DEFINE_string(
    'bam', None, 'Input BAM containing reads to reference alignment. '
    'It is expected that the reads in the BAM are haplotagged. '
    'The assumtion is that the sample is dioloid so examples'
    ' from HP:1 and HP:2 will be generated.')
_FASTA_FILE = flags.DEFINE_string(
    'fasta', None, 'Input FASTA file of the reference/assembly.')
_TRAINING_MODE = flags.DEFINE_boolean(
    'training_mode', False, 'If set then training mode is enabled.'
    'Required inputs during training: '
    'truth_to_ref, region_bed.')
_TRUTH_TO_REF = flags.DEFINE_string(
    'truth_to_ref', None, 'Input truth haplotype alignment to reference.')
_REGION_BED = flags.DEFINE_string(
    'region_bed', None,
    'Input region bedfile containing high-confidence regions.')
_REGION = flags.DEFINE_string(
    'region', None,
    'Comma separated list of regions in format contig_name:start-stop'
    'or simply contig_name. Only providing the name of the contig will process'
    ' the entire contig.')
_OUTPUT = flags.DEFINE_string(
    'output',
    None,
    (
        'Path to output directory and a prefix. '
        'For example: --output=/path/to/examples '
        'Where "/path/to/" is the directory and "examples" is the prefix.'
    ),
)
_CPUS = flags.DEFINE_integer(
    'cpus', multiprocessing.cpu_count(),
    'Number of worker processes to use. Use 0 to disable parallel processing. '
    'Minimum of 2 CPUs required for parallel processing.')
_INTERVAL_SIZE = flags.DEFINE_integer(
    'interval_size', 20000, 'Interval size used for processing the bam file.')

# Experimental for diploid:
_PLOIDY = flags.DEFINE_integer(
    'ploidy',
    1,
    (
        '[EXPERIMENTAL] Number of output sequences desired, e.g. 1 for '
        'haploid polishing and 2 for diploid variant calling.'
    ),
)


def register_required_flags():
  flags.mark_flags_as_required([
      'bam',
      'fasta',
      'output',
  ])


def main(argv: Sequence[str]) -> None:
  del argv
  if _CPUS.value == 0:
    raise ValueError('Must set cpus to >=1 for processing.')

  logging.info('MAKE IMAGES TOTAL CPUs: %d', _CPUS.value)

  is_training = _TRAINING_MODE.value
  if is_training:
    # In training mode, these values must be passed:
    missing_parameters = []
    if _TRUTH_TO_REF.value is None:
      missing_parameters.append('truth_to_ref')
    if _REGION_BED.value is None:
      missing_parameters.append('region_bed')
    if missing_parameters:
      missing_parameters_str = ', '.join(missing_parameters)
      error_msg = ('Missing required parameters for training mode:'
                   f' {missing_parameters_str!r}.')
      raise ValueError(error_msg)
    logging.info('Make images in mode: Training')
  else:
    logging.info('Make images in mode: Inference')

  all_intervals = utils_make_images.get_contig_regions(
      bam_file=_BAM_FILE.value,
      fasta_file=_FASTA_FILE.value,
      region=_REGION.value,
      interval_length=_INTERVAL_SIZE.value,
  )
  bed_regions_by_contig = dict()
  if _REGION_BED.value:
    bed_regions_by_contig = utils_make_images.read_bed(_REGION_BED.value)

  arguments = []
  for process_id in range(0, _CPUS.value):
    options = region_processor.OptionsForProcess(
        bam_file=_BAM_FILE.value,
        fasta_file=_FASTA_FILE.value,
        truth_to_ref=_TRUTH_TO_REF.value,
        bed_regions_by_contig=bed_regions_by_contig,
        all_intervals=all_intervals,
        train_mode=is_training,
        output_filename=_OUTPUT.value,
        process_id=process_id,
        cpus=_CPUS.value,
        ploidy=_PLOIDY.value,
    )
    arguments.append((options,))

  with multiprocessing.Pool(processes=_CPUS.value) as pool:
    run_summaries = list(pool.starmap(region_processor.run_process, arguments))
    pool.close()
    pool.join()

  total_summary_counts = collections.defaultdict(lambda: 0)
  for process_id, process_summary in enumerate(run_summaries):
    logging.info(
        'Process %r: Total %r intervals processed and generated %r examples.',
        process_id,
        process_summary['interval_counter'],
        process_summary['example_counter'],
    )

    for key, value in process_summary.items():
      if key.endswith('_counter'):
        total_summary_counts[key] += value

  for key, count in total_summary_counts.items():
    logging.info('Total count across processes: %s: %r', key, count)

  flags_for_summary = [
      _BAM_FILE, _FASTA_FILE, _TRUTH_TO_REF, _TRAINING_MODE, _REGION_BED,
      _REGION, _OUTPUT, _CPUS, _INTERVAL_SIZE
  ]
  utils_make_images.write_summary(
      total_summary_counts, _OUTPUT.value, is_training, flags_for_summary
  )


if __name__ == '__main__':
  logging.use_python_logging()
  app.run(main)
