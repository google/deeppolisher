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
"""Tests for make_images."""

import json
import os

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized

from polisher.make_images import make_images
from polisher.make_images import test_utils
from polisher.models import data_providers
from absl import app


FLAGS = flags.FLAGS


class MakeImagesE2ETest(parameterized.TestCase):
  """E2E tests for make images."""

  @parameterized.parameters(
      dict(
          expected_total_examples=1,
          flag_values={
              "bam": "HG002_chr20_0_200000_hifi_2_GRCh38.bam",
              "fasta": "GRCh38_chr20_0_200000.fa",
              "region_bed": (
                  "HG002_GRCh38_1_22_v4.2.1_benchmark.chr20_0_200000.bed"
              ),
              "truth_to_ref": "HG002_chr20_0_200000_GIABv42_2_GRCh38.hap1.bam",
              "is_training": True,
              "cpus": 1,
              "region": "chr20:85625-85832",
              "interval_size": 1000,
              "ploidy": 1,
          },
          expected_summary_counters={
              "interval_counter": 1,
              "example_counter": 1,
              "skipped_outside_bed_counter": 0,
              "skipped_too_few_truth_reads_counter": 0,
              "skipped_too_many_truth_reads_counter": 0,
              "skipped_too_many_reads_counter": 0,
          },
      ),
      dict(
          expected_total_examples=0,
          flag_values={
              "bam": "HG002_chr20_0_200000_hifi_2_GRCh38.bam",
              "fasta": "GRCh38_chr20_0_200000.fa",
              "region_bed": (
                  "HG002_GRCh38_1_22_v4.2.1_benchmark.chr20_0_200000.bed"
              ),
              "truth_to_ref": "HG002_chr20_0_200000_GIABv42_2_GRCh38.hap1.bam",
              "is_training": True,
              "cpus": 1,
              "region": "chr20:101754-101852",
              "interval_size": 1000,
              "ploidy": 1,
          },
          expected_summary_counters={
              "interval_counter": 1,
              "example_counter": 0,
              "skipped_outside_bed_counter": 0,
              "skipped_too_few_truth_reads_counter": 0,
              "skipped_too_many_truth_reads_counter": 0,
              "skipped_too_many_reads_counter": 0,
          },
      ),
      dict(
          expected_total_examples=0,
          flag_values={
              "bam": "HG002_chr20_0_200000_hifi_2_GRCh38.bam",
              "fasta": "GRCh38_chr20_0_200000.fa",
              "region_bed": (
                  "HG002_GRCh38_1_22_v4.2.1_benchmark.chr20_0_200000.bed"
              ),
              "truth_to_ref": "HG002_chr20_0_200000_GIABv42_2_GRCh38.hap1.bam",
              "is_training": True,
              "cpus": 1,
              "region": "chr20:0-100",
              "interval_size": 1000,
              "ploidy": 1,
          },
          expected_summary_counters={
              "interval_counter": 0,
              "example_counter": 0,
              "skipped_outside_bed_counter": 1,
              "skipped_too_few_truth_reads_counter": 0,
              "skipped_too_many_truth_reads_counter": 0,
              "skipped_too_many_reads_counter": 0,
          },
      ),
      dict(
          expected_total_examples=4,
          flag_values={
              "bam": "HG002_chr20_0_200000_hifi_2_GRCh38.bam",
              "fasta": "GRCh38_chr20_0_200000.fa",
              "region_bed": (
                  "HG002_GRCh38_1_22_v4.2.1_benchmark.chr20_0_200000.bed"
              ),
              "truth_to_ref": "HG002_chr20_0_200000_GIABv42_2_GRCh38.hap1.bam",
              "is_training": True,
              "cpus": 4,
              "region": "chr20:85000-86000",
              "interval_size": 100,
              "ploidy": 1,
          },
          expected_summary_counters={
              "interval_counter": 10,
              "example_counter": 4,
              "skipped_outside_bed_counter": 0,
              "skipped_too_few_truth_reads_counter": 0,
              "skipped_too_many_truth_reads_counter": 0,
              "skipped_too_many_reads_counter": 0,
          },
      ),
      dict(
          expected_total_examples=3,
          flag_values={
              "bam": "HG002.pfda_challenge.grch38.phased.chr20_0-200000.bam",
              "fasta": "GRCh38_chr20_0_200000.fa",
              "region_bed": (
                  "HG002_GRCh38_1_22_v4.2.1_benchmark.chr20_0_200000.bed"
              ),
              # Diploid requires a different truth_to_ref because there are two
              # sequences expected per locus while haploid has only 1.
              "truth_to_ref": "truth_to_ref.chr20_0-200000.bam",
              "is_training": True,
              "cpus": 1,
              "region": "chr20:100000-101000",
              "interval_size": 100,
              "ploidy": 2,
          },
          expected_summary_counters={
              "interval_counter": 10,
              "example_counter": 3,
              "skipped_outside_bed_counter": 0,
              "skipped_too_few_truth_reads_counter": 0,
              "skipped_too_many_truth_reads_counter": 0,
              "skipped_too_many_reads_counter": 0,
          },
      ),
  )
  @flagsaver.flagsaver
  def test_make_images_golden(
      self, expected_total_examples, flag_values, expected_summary_counters
  ):
    """Test make_images method."""
    # Set flag values.
    FLAGS.bam = test_utils.polisher_testdata(flag_values["bam"])
    FLAGS.fasta = test_utils.polisher_testdata(flag_values["fasta"])
    FLAGS.region_bed = test_utils.polisher_testdata(flag_values["region_bed"])
    FLAGS.truth_to_ref = test_utils.polisher_testdata(
        flag_values["truth_to_ref"]
    )
    FLAGS.region = flag_values["region"]
    tmp_dir = self.create_tempdir()
    output = os.path.join(tmp_dir, "make_examples")
    FLAGS.output = output
    FLAGS.interval_size = flag_values["interval_size"]
    FLAGS.training_mode = flag_values["is_training"]
    FLAGS.cpus = flag_values["cpus"]
    FLAGS.ploidy = flag_values["ploidy"]

    # Run make_images with the flags above:
    make_images.main([])

    # Read output tfrecords:
    dataset = data_providers.get_dataset(
        file_pattern=f"{output}_*.tfrecords.gz",
        inference=not flag_values["is_training"],
        ploidy=flag_values["ploidy"],
        batch_size=32,
        drop_remainder=False,
        num_epochs=1,
        shuffle_dataset=False,
    )
    batches = list(dataset)

    if expected_total_examples == 0:
      self.assertEmpty(batches)
    else:
      self.assertLen(
          batches,
          1,
          msg=(
              "Check there's at most one batch, "
              "assuming num examples < batch_size."
          ),
      )

      observed_num_examples = batches[0]["name"].shape[0]
      self.assertEqual(
          observed_num_examples,
          expected_total_examples,
          msg="Counting examples from tfrecord output files.",
      )

      for field, value in batches[0].items():
        self.assertEqual(
            value.shape[0],
            expected_total_examples,
            msg=(
                f"field '{field}' has the number of examples as the first"
                " dimension."
            ),
        )

    # Read summary stats from the output JSON file:
    summary_name = "training" if flag_values["is_training"] else "inference"
    summary_path = f"{output}_{summary_name}.summary.json"
    summary = json.load(open(summary_path, "r"))
    self.assertEqual(
        expected_total_examples,
        summary["example_counter"],
        msg="Checking example count from summary JSON output file.",
    )

    summary_counters = {
        key: value for key, value in summary.items() if key.endswith("_counter")
    }
    self.assertEqual(
        expected_summary_counters,
        summary_counters,
        msg=f"Checking counters for region {FLAGS.region}",
    )


if __name__ == "__main__":
  g3_multiprocessing.handle_test_main(absltest.main)
