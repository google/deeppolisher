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
"""Tests for google3.learning.genomics.deepconsensus.preprocess.preprocess_utils."""

import collections
import dataclasses
import json
import os
from typing import Any, List, Tuple

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import pysam

from polisher.make_images import haplotype
from polisher.make_images import test_utils
from polisher.make_images import utils_make_images


@dataclasses.dataclass
class Flag:
  name: str
  value: Any


@dataclasses.dataclass
class ReadAttributes:
  """Represents attributes of a read used for test.

  Attributes:
    bam_file: Path to an alignment bam file to get optional read attributes.
    query_name: Name of the read.
    query_sequence: Nucleotide sequence of read.
    contig: Contig name where read is aligned.
    start: Position where read is aligned.
    base_qualities: Read base qualities.
    mapping_quality: Mapping quality of the read.
    cigartuples: Cigartuple describing the alignment of the read.
    tags: Auxiliary tags for the reads.
  """
  bam_file: str
  query_name: str
  query_sequence: str
  contig: str
  start: int
  base_qualities: List[int]
  mapping_quality: int
  cigartuples: List[Tuple[int, int]]
  tags: List[Tuple[str, Any]]


class Test(parameterized.TestCase):
  """Tests for make images component of the polisher."""

  @parameterized.parameters(
      dict(
          region_string="chr20:0-1000",
          expected_record=utils_make_images.RegionRecord("chr20", 0, 1000),
          message="Test 1: valid region string parsing"),
      dict(
          region_string="chr20",
          expected_record=utils_make_images.RegionRecord("chr20", 0, 199999),
          message="Test 2: valid region with contig only, no range provided."),
      dict(
          region_string="chr20:0-210000",
          expected_record=utils_make_images.RegionRecord("chr20", 0, 199999),
          message="Test 2: Region that overflows contig length."))
  @flagsaver.flagsaver
  def test_process_region_string(self, region_string, expected_record, message):
    """Test process region string method that parses a region string."""
    ref_file = "GRCh38_chr20_0_200000.fa"
    fasta = test_utils.polisher_testdata(ref_file)
    common_contigs = ["chr20"]
    returned_record = utils_make_images.process_region_string(
        region_string, fasta, common_contigs)
    self.assertEqual(
        expected_record.contig, returned_record.contig, msg=message)
    self.assertEqual(expected_record.start, returned_record.start, msg=message)
    self.assertEqual(expected_record.stop, returned_record.stop, msg=message)

  @parameterized.parameters(
      dict(
          region_string="chr20:1000-0",
          exception_msg="Malformed region string 'chr20:1000-0'.",
          message="Test 3: Invalid contig region, stop is smaller than start."),
      dict(
          region_string="chr20:0::-::10:0:0",
          exception_msg="Malformed region string 'chr20:0::-::10:0:0'.",
          message="Test 4: Invalid range"))
  @flagsaver.flagsaver
  def test_process_region_string_exeptions(self, region_string, exception_msg,
                                           message):
    """Test process region string method that parses a region string."""
    ref_file = "GRCh38_chr20_0_200000.fa"
    fasta = test_utils.polisher_testdata(ref_file)
    common_contigs = ["chr20"]
    with self.assertRaisesRegex(ValueError, exception_msg, msg=message):
      utils_make_images.process_region_string(region_string, fasta,
                                              common_contigs)

  @parameterized.parameters(
      dict(
          input_list=[
              utils_make_images.RegionRecord("chr20", 0, 1000),
              utils_make_images.RegionRecord("chrX", 0, 1000)
          ],
          interval_length=500,
          expected_list=[
              utils_make_images.RegionRecord("chr20", 0, 500),
              utils_make_images.RegionRecord("chr20", 500, 1000),
              utils_make_images.RegionRecord("chrX", 0, 500),
              utils_make_images.RegionRecord("chrX", 500, 1000)
          ],
          message="Test1: Valid input with ranges that end without division"),
      dict(
          input_list=[
              utils_make_images.RegionRecord("chr20", 0, 1100),
              utils_make_images.RegionRecord("chrX", 0, 1098)
          ],
          interval_length=500,
          expected_list=[
              utils_make_images.RegionRecord("chr20", 0, 500),
              utils_make_images.RegionRecord("chr20", 500, 1000),
              utils_make_images.RegionRecord("chr20", 1000, 1100),
              utils_make_images.RegionRecord("chrX", 0, 500),
              utils_make_images.RegionRecord("chrX", 500, 1000),
              utils_make_images.RegionRecord("chrX", 1000, 1098)
          ],
          message="Test 2: End is not divible by interval length"),
      dict(
          input_list=[
              utils_make_images.RegionRecord("chr20", 0, 358),
              utils_make_images.RegionRecord("chrX", 0, 457)
          ],
          interval_length=1000,
          expected_list=[
              utils_make_images.RegionRecord("chr20", 0, 358),
              utils_make_images.RegionRecord("chrX", 0, 457)
          ],
          message="Test 3: Ranges where contig length is much smaller."))
  @flagsaver.flagsaver
  def test_split_regions_in_intervals(self, input_list, interval_length,
                                      expected_list, message):
    """Test split regions method that divides a region by interval length."""
    returned_list = utils_make_images.split_regions_in_intervals(
        input_list, interval_length)
    for returned_record, expected_record in zip(returned_list, expected_list):
      self.assertEqual(
          expected_record.contig, returned_record.contig, msg=message)
      self.assertEqual(
          expected_record.start, returned_record.start, msg=message)
      self.assertEqual(expected_record.stop, returned_record.stop, msg=message)

  @parameterized.parameters(
      dict(
          region="chr20:0-1000",
          expected_list=[
              utils_make_images.RegionRecord("chr20", 0, 500),
              utils_make_images.RegionRecord("chr20", 500, 1000)
          ],
          interval_length=500,
          message="Test 1: With a region."),
      dict(
          region="chr20:1324-2000",
          expected_list=[utils_make_images.RegionRecord("chr20", 1324, 2000)],
          interval_length=1000,
          message="Test 2: Region does not start at 0, length is less than 1000"
      ),
      dict(
          region="chr20:67-123",
          expected_list=[utils_make_images.RegionRecord("chr20", 67, 123)],
          interval_length=1000,
          message="Test 3: Region does not start at 0, length is less than 1000"
      ),
      dict(
          region="chr20:67-123,chr20:1324-2000",
          expected_list=[
              utils_make_images.RegionRecord("chr20", 67, 123),
              utils_make_images.RegionRecord("chr20", 1324, 2000)
          ],
          interval_length=1000,
          message="Test 4: Multiple regions"),
      dict(
          region="",
          expected_list=[
              utils_make_images.RegionRecord("chr20", 0, 100000),
              utils_make_images.RegionRecord("chr20", 100000, 199999)
          ],
          interval_length=100000,
          message="Test 4: No region provided, get from bam and fasta."))
  @flagsaver.flagsaver
  def test_get_contig_regions(self, region, expected_list, interval_length,
                              message):
    """Test get contig regions that produces a list of regions."""
    ref_file = "GRCh38_chr20_0_200000.fa"
    fasta = test_utils.polisher_testdata(ref_file)
    bam = "HG002_chr20_0_200000_hifi_2_GRCh38.bam"
    bam_file = test_utils.polisher_testdata(bam)
    returned_list = utils_make_images.get_contig_regions(
        bam_file, fasta, region, interval_length)
    self.assertEqual(len(expected_list), len(returned_list))
    for returned_record, expected_record in zip(returned_list, expected_list):
      self.assertEqual(
          expected_record.contig, returned_record.contig, msg=message)
      self.assertEqual(
          expected_record.start, returned_record.start, msg=message)
      self.assertEqual(expected_record.stop, returned_record.stop, msg=message)

  @parameterized.parameters(
      dict(
          region="chr404",
          interval_length=1000,
          exception_msg="Contig not found: contig 'chr404'",
          message="Test 5: Contig not found in FASTA."),
      dict(
          region="chr404:0-400",
          interval_length=1000,
          exception_msg="Contig not found: contig 'chr404'",
          message="Test 5: Contig not found in FASTA."),
      dict(
          region="chr20:13-24-2000",
          interval_length=1000,
          exception_msg="Malformed region string 'chr20:13-24-2000'.",
          message="Test 6: Malformed string."),
      dict(
          region="chr20,chr404",
          interval_length=1000,
          exception_msg="Contig not found: contig 'chr404'",
          message="Test 7: One contig is invalid."),
      dict(
          region="chr20:0-1000,chr404:0-100",
          interval_length=1000,
          exception_msg="Contig not found: contig 'chr404'",
          message="Test 8: One contig is invalid."))
  @flagsaver.flagsaver
  def test_get_contig_regions_exeptions(self, region, exception_msg,
                                        interval_length, message):
    """Test process region string method that parses a region string."""
    ref_file = "GRCh38_chr20_0_200000.fa"
    fasta = test_utils.polisher_testdata(ref_file)
    bam = "HG002_chr20_0_200000_hifi_2_GRCh38.bam"
    bam_file = test_utils.polisher_testdata(bam)
    with self.assertRaisesRegex(ValueError, exception_msg, msg=message):
      utils_make_images.get_contig_regions(bam_file, fasta, region,
                                           interval_length)

  def test_read_bed(self):
    """Test read bed file function."""
    bed = "HG002_GRCh38_1_22_v4.2.1_benchmark.chr20_0_200000.bed"
    bed_file = test_utils.polisher_testdata(bed)
    returned_dict = utils_make_images.read_bed(bed_file)
    expected_record = utils_make_images.RegionRecord("chr20", 81335, 101267)
    self.assertIn("chr20", returned_dict, msg="Bed file read test.")
    self.assertLen(returned_dict["chr20"], 21, msg="Bed file read test.")
    self.assertEqual(
        expected_record.contig,
        returned_dict["chr20"][0].contig,
        msg="Bed file read test.")
    self.assertEqual(
        expected_record.start,
        returned_dict["chr20"][0].start,
        msg="Bed file read test.")
    self.assertEqual(
        expected_record.stop,
        returned_dict["chr20"][0].stop,
        msg="Bed file read test.")
    # test invalid bed file
    invalid_bed = "HG002_GRCh38_1_22_v4.2.1_benchmark.chr20.invalid.bed"
    invalid_bed_file = test_utils.polisher_testdata(invalid_bed)
    with self.assertRaisesRegex(ValueError, "Invalid entry in BED file."):
      utils_make_images.read_bed(invalid_bed_file)

  @parameterized.parameters(
      dict(
          all_intervals=[
              utils_make_images.RegionRecord("chr20", 0, 10),
              utils_make_images.RegionRecord("chr20", 1, 11),
              utils_make_images.RegionRecord("chr20", 2, 12),
              utils_make_images.RegionRecord("chr20", 3, 13),
              utils_make_images.RegionRecord("chr20", 4, 14),
          ],
          process_id=0,
          total_cpus=2,
          expected_list=[
              utils_make_images.RegionRecord("chr20", 0, 10),
              utils_make_images.RegionRecord("chr20", 2, 12),
              utils_make_images.RegionRecord("chr20", 4, 14),
          ],
          message="Test 1: Test of even process id."),
      dict(
          all_intervals=[
              utils_make_images.RegionRecord("chr20", 0, 10),
              utils_make_images.RegionRecord("chr20", 1, 11),
              utils_make_images.RegionRecord("chr20", 2, 12),
              utils_make_images.RegionRecord("chr20", 3, 13),
              utils_make_images.RegionRecord("chr20", 4, 14),
          ],
          process_id=1,
          total_cpus=2,
          expected_list=[
              utils_make_images.RegionRecord("chr20", 1, 11),
              utils_make_images.RegionRecord("chr20", 3, 13),
          ],
          message="Test 2: Test of odd process id"),
      dict(
          all_intervals=[
              utils_make_images.RegionRecord("chr20", 0, 10),
              utils_make_images.RegionRecord("chr20", 1, 11),
          ],
          process_id=0,
          total_cpus=1,
          expected_list=[
              utils_make_images.RegionRecord("chr20", 0, 10),
              utils_make_images.RegionRecord("chr20", 1, 11),
          ],
          message="Test 3: Test with only one process."))
  @flagsaver.flagsaver
  def test_get_process_intervals(self, all_intervals, process_id, total_cpus,
                                 expected_list, message):
    """Test get process intervals."""
    returned_list = list(
        utils_make_images.get_process_intervals(all_intervals, process_id,
                                                total_cpus))

    self.assertListEqual(expected_list, returned_list, msg=message)

  @parameterized.parameters(
      dict(
          all_intervals=[
              utils_make_images.RegionRecord("chr20", 0, 10),
              utils_make_images.RegionRecord("chr20", 1, 11),
          ],
          process_id=1,
          total_cpus=1,
          exception_msg="Pre-processing error: process-id: 1 "
          "must be less than total available cpus: 1.",
          message="Test 1: Test invalid process_id."))
  @flagsaver.flagsaver
  def test_get_process_intervals_exeptions(self, all_intervals, process_id,
                                           total_cpus, exception_msg, message):
    """Test get process intervals execptions."""
    with self.assertRaisesRegex(ValueError, exception_msg, msg=message):
      _ = list(
          utils_make_images.get_process_intervals(all_intervals, process_id,
                                                  total_cpus))

  def test_get_reads_from_bam(self):
    """Test get_reads_from_bam method."""
    bam = "HG002_chr20_0_200000_hifi_2_GRCh38.bam"
    bam_file = test_utils.polisher_testdata(bam)
    interval = utils_make_images.RegionRecord("chr20", 100000, 101000)
    all_reads = utils_make_images.get_reads_from_bam(bam_file, interval)
    read_names = [read.query_name for read in all_reads]
    self.assertLen(all_reads, 42)
    self.assertIn("m64012_190921_234837/117574370/ccs", read_names)

  @parameterized.parameters(
      dict(
          interval=utils_make_images.RegionRecord("chr20", 100000, 100010),
          expected_sequence="TTGAGCATGC",
          message="Test 1: Simple sequence fetch."),
      dict(
          interval=utils_make_images.RegionRecord("chr20", 100000, 100009),
          expected_sequence="TTGAGCATG",
          message="Test 2: Simple sequence fetch."),
      dict(
          interval=utils_make_images.RegionRecord("chr20", 100001, 100009),
          expected_sequence="TGAGCATG",
          message="Test 3: Simple sequence fetch."),
      dict(
          interval=utils_make_images.RegionRecord("chr20", 10000001, 10000009),
          expected_sequence="",
          message="Test 3: Empty sequence fetch."))
  @flagsaver.flagsaver
  def test_get_reference_sequence_from_fasta(self, interval, expected_sequence,
                                             message):
    """Test get_reference_sequence_from_fasta method."""
    fasta = "GRCh38_chr20_0_200000.fa"
    padding = 0
    fasta_file = test_utils.polisher_testdata(fasta)
    sequence = utils_make_images.get_reference_sequence_from_fasta(
        fasta_file, interval, padding)
    self.assertEqual(sequence, expected_sequence, msg=message)

  @parameterized.parameters(
      dict(
          interval=utils_make_images.RegionRecord("chr20", 0, 1000),
          bed_list=[utils_make_images.RegionRecord("chr20", 0, 2000)],
          expected_list=[utils_make_images.RegionRecord("chr20", 0, 1000)],
          message="Test 1: Easy range intersect."),
      dict(
          interval=utils_make_images.RegionRecord("chr20", 0, 1000),
          bed_list=[utils_make_images.RegionRecord("chr20", 50, 2000)],
          expected_list=[utils_make_images.RegionRecord("chr20", 50, 1000)],
          message="Test 2: Starts to the left."),
      dict(
          interval=utils_make_images.RegionRecord("chr20", 0, 1000),
          bed_list=[utils_make_images.RegionRecord("chr20", 202, 555)],
          expected_list=[utils_make_images.RegionRecord("chr20", 202, 555)],
          message="Test 3: Contained regions."),
      dict(
          interval=utils_make_images.RegionRecord("chr20", 0, 1000),
          bed_list=[
              utils_make_images.RegionRecord("chr20", 73, 500),
              utils_make_images.RegionRecord("chr20", 600, 5000)
          ],
          expected_list=[
              utils_make_images.RegionRecord("chr20", 73, 500),
              utils_make_images.RegionRecord("chr20", 600, 1000)
          ],
          message="Test 4: Regions with two breakages."),
      dict(
          interval=utils_make_images.RegionRecord("chr20", 501, 599),
          bed_list=[
              utils_make_images.RegionRecord("chr20", 0, 500),
              utils_make_images.RegionRecord("chr20", 600, 1000)
          ],
          expected_list=[],
          message="Test 5: Region not contained."),
      dict(
          interval=utils_make_images.RegionRecord("chr20", 501, 599),
          bed_list=[],
          expected_list=[utils_make_images.RegionRecord("chr20", 501, 599)],
          message="Test 6: No bed region."),
      dict(
          interval=utils_make_images.RegionRecord("chr20", 0, 1000),
          bed_list=[utils_make_images.RegionRecord("chr2", 0, 2000)],
          expected_list=[],
          message="Test 7: Contigs are different."),
  )
  @flagsaver.flagsaver
  def test_range_intersect(self, interval, bed_list, expected_list, message):
    """Test range intersect."""
    returned_list = utils_make_images.range_intersect(interval, bed_list)

    self.assertEqual(len(expected_list), len(returned_list))
    for returned_record, expected_record in zip(returned_list, expected_list):
      self.assertEqual(
          expected_record.contig, returned_record.contig, msg=message)
      self.assertEqual(
          expected_record.start, returned_record.start, msg=message)
      self.assertEqual(expected_record.stop, returned_record.stop, msg=message)

  def set_read_attributes(self, read):
    """Set attributes of reads."""
    read.is_duplicate = False
    read.is_qcfail = False
    read.is_secondary = False
    read.is_unmapped = False
    read.is_supplementary = False
    read.mapping_quality = 60

  def get_pysam_read_object(self, read_attributes):
    """Creates a pysam read object given attributes."""
    bam_file = test_utils.polisher_testdata(read_attributes.bam_file)
    pysam_bam = pysam.AlignmentFile(bam_file)
    read = pysam.AlignedSegment(pysam_bam.header)
    self.set_read_attributes(read)
    read.query_name = read_attributes.query_name
    read.query_sequence = read_attributes.query_sequence
    read.reference_name = read_attributes.contig
    read.reference_start = read_attributes.start
    read.query_qualities = read_attributes.base_qualities
    read.cigartuples = read_attributes.cigartuples
    read.mapping_quality = read_attributes.mapping_quality
    if read_attributes.tags:
      for tag, value in read_attributes.tags:
        read.set_tag(tag, value)
    return read

  @parameterized.parameters(
      dict(
          reads=[
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="read_1",
                  query_sequence="ACGTACGTACA",
                  contig="chr20",
                  start=0,
                  base_qualities=[60] * 11,
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 11)],
                  tags=[("HP", 1)])
          ],
          expected_reads_in_bin_by_index={1: [0]},
          message="Test 1: Simple binning."),
      dict(
          reads=[
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="read_1",
                  query_sequence="ACGTACGTACA",
                  contig="chr20",
                  start=0,
                  base_qualities=[60] * 11,
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 11)],
                  tags=[("HP", 1)]),
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="read_2",
                  query_sequence="ACGTACGTACA",
                  contig="chr20",
                  start=0,
                  base_qualities=[60] * 11,
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 11)],
                  tags=[("HP", 2)])
          ],
          expected_reads_in_bin_by_index={
              1: [0],
              2: [1]
          },
          message="Test 2: Binning two reads."),
      dict(
          reads=[
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="read_1",
                  query_sequence="ACGTACGTACA",
                  contig="chr20",
                  start=0,
                  base_qualities=[60] * 11,
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 11)],
                  tags=None),
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="read_2",
                  query_sequence="ACGTACGTACA",
                  contig="chr20",
                  start=0,
                  base_qualities=[60] * 11,
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 11)],
                  tags=[("HP", 2)])
          ],
          expected_reads_in_bin_by_index={
              0: [0],
              2: [1]
          },
          message="Test 3: Binning two reads, one missing hp tag."))
  @flagsaver.flagsaver
  def test_bin_reads_by_halotype(self, reads, expected_reads_in_bin_by_index,
                                 message):
    """Tests for read binning based on haplotypes."""
    all_reads = []
    for read_attr in reads:
      all_reads.append(self.get_pysam_read_object(read_attr))

    binned_reads = utils_make_images.bin_reads_by_haplotype(all_reads)

    # convert the expected bin by index to expected bin with read
    expected_read_bin = collections.defaultdict(list)
    for hp_tag, read_index_list in expected_reads_in_bin_by_index.items():
      for read_index in read_index_list:
        expected_read_bin[hp_tag].append(all_reads[read_index])
    self.assertDictEqual(binned_reads, expected_read_bin, msg=message)

  @parameterized.parameters(
      dict(
          reads=[
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="read_1",
                  query_sequence="ACGTACGTACA",
                  contig="chr20",
                  start=0,
                  base_qualities=[60] * 11,
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 11)],
                  tags=[("HP", 3)])
          ],
          exception_message="Found HP tag 3 in read 'read_1'",
          message="Test 1: Haplotype violation test 1."),
      dict(
          reads=[
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="read_2",
                  query_sequence="ACGTACGTACA",
                  contig="chr20",
                  start=0,
                  base_qualities=[60] * 11,
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 11)],
                  tags=[("HP", -1)])
          ],
          exception_message="Found HP tag -1 in read 'read_2'",
          message="Test 1: Haplotype violation test 2."))
  @flagsaver.flagsaver
  def test_bin_reads_by_halotype_exceptions(self, reads, exception_message,
                                            message):
    """Tests for exceptions in read binning based on haplotypes."""
    all_reads = []
    for read_attr in reads:
      all_reads.append(self.get_pysam_read_object(read_attr))

    with self.assertRaisesRegex(ValueError, exception_message, msg=message):
      utils_make_images.bin_reads_by_haplotype(all_reads)

  @parameterized.parameters(
      dict(
          intervals=[utils_make_images.RegionRecord("chr20", 0, 10)],
          position=5,
          expected_value=True,
          message="Test 1: Test where position is within region."),
      dict(
          intervals=[utils_make_images.RegionRecord("chr20", 0, 10)],
          position=0,
          expected_value=True,
          message="Test 2: Test where position is equal to start."),
      dict(
          intervals=[utils_make_images.RegionRecord("chr20", 0, 10)],
          position=10,
          expected_value=True,
          message="Test 3: Test where position is equal to end."),
      dict(
          intervals=[utils_make_images.RegionRecord("chr20", 0, 10)],
          position=11,
          expected_value=False,
          message="Test 4: Test where position is not within region."),
      dict(
          intervals=[
              utils_make_images.RegionRecord("chr20", 0, 10),
              utils_make_images.RegionRecord("chr20", 20, 30)
          ],
          position=15,
          expected_value=False,
          message="Test 5: Test with two intervals and not contained."),
      dict(
          intervals=[],
          position=15,
          expected_value=False,
          message="Test 6: No intervals provided."),
  )
  @flagsaver.flagsaver
  def test_check_if_position_is_within_regions(self, intervals, position,
                                               expected_value, message):
    """Test get process intervals."""
    returned_value = utils_make_images.check_if_position_is_within_regions(
        position, intervals)

    self.assertEqual(returned_value, expected_value, msg=message)

  @parameterized.parameters(
      dict(
          reads=[
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="read_1",
                  query_sequence="ACGTACGTACA",
                  contig="chr20",
                  start=0,
                  base_qualities=[60] * 11,
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 11)],
                  tags=[("HP", 1)])
          ],
          expected_reads_in_bin_by_index={1: [0]},
          message="Test 1: Simple binning."),
      dict(
          reads=[
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="read_1",
                  query_sequence="ACGTACGTACA",
                  contig="chr20",
                  start=0,
                  base_qualities=[60] * 11,
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 11)],
                  tags=[("HP", 1)]),
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="read_2",
                  query_sequence="ACGTACGTACA",
                  contig="chr20",
                  start=0,
                  base_qualities=[60] * 11,
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 11)],
                  tags=[("HP", 2)])
          ],
          expected_reads_in_bin_by_index={
              1: [0],
              2: [1]
          },
          message="Test 2: Binning two reads."),
      dict(
          reads=[
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="read_1",
                  query_sequence="ACGTACGTACA",
                  contig="chr20",
                  start=0,
                  base_qualities=[60] * 11,
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 11)],
                  tags=None),
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="read_2",
                  query_sequence="ACGTACGTACA",
                  contig="chr20",
                  start=0,
                  base_qualities=[60] * 11,
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 11)],
                  tags=[("HP", 2)])
          ],
          expected_reads_in_bin_by_index={
              0: [0],
              1: [0],
              2: [1, 0]
          },
          message="Test 3: Binning two reads, one missing hp tag."))
  @flagsaver.flagsaver
  def test_add_non_haplotype_reads_to_bins(self, reads,
                                           expected_reads_in_bin_by_index,
                                           message):
    """Tests for adding non-haplotyped reads to haplotype bins."""
    all_reads = []
    for read_attr in reads:
      all_reads.append(self.get_pysam_read_object(read_attr))

    binned_reads = utils_make_images.bin_reads_by_haplotype(all_reads)
    non_haplotype_added_bin = haplotype.add_non_haplotype_reads_to_bins(
        binned_reads)

    # convert the expected bin by index to expected bin with read
    expected_read_bin = collections.defaultdict(list)
    for hp_tag, read_index_list in expected_reads_in_bin_by_index.items():
      for read_index in read_index_list:
        expected_read_bin[hp_tag].append(all_reads[read_index])
    self.assertDictEqual(
        non_haplotype_added_bin, expected_read_bin, msg=message)

  @parameterized.parameters(
      dict(
          is_duplicate=False,
          is_qcfail=False,
          is_secondary=False,
          is_unmapped=False,
          is_supplementary=False,
          mapping_quality=60,
          is_filtered=False,
          allow_supplementary=False,
          message="Test 1: Read filter test, good read, no filtering."),
      dict(
          is_duplicate=False,
          is_qcfail=True,
          is_secondary=False,
          is_unmapped=False,
          is_supplementary=False,
          mapping_quality=60,
          is_filtered=True,
          allow_supplementary=False,
          message="Test 2: Read filter test, filtered because read is qcfail."),
      dict(
          is_duplicate=False,
          is_qcfail=False,
          is_secondary=True,
          is_unmapped=True,
          is_supplementary=False,
          mapping_quality=60,
          is_filtered=True,
          allow_supplementary=False,
          message="Test 3: Read filter test, secondary and unmapped flags on."),
      dict(
          is_duplicate=False,
          is_qcfail=False,
          is_secondary=False,
          is_unmapped=False,
          is_supplementary=True,
          mapping_quality=60,
          is_filtered=True,
          allow_supplementary=False,
          message="Test 4: Read filter test, supplementary flags on."),
      dict(
          is_duplicate=False,
          is_qcfail=False,
          is_secondary=False,
          is_unmapped=False,
          is_supplementary=False,
          mapping_quality=0,
          is_filtered=True,
          allow_supplementary=False,
          message="Test 5: Read filter test, poor mapping quality."),
      dict(
          is_duplicate=False,
          is_qcfail=False,
          is_secondary=False,
          is_unmapped=False,
          is_supplementary=True,
          mapping_quality=60,
          is_filtered=False,
          allow_supplementary=True,
          message="Test 6: Read filter test, not filtering supplementary."),
      dict(
          is_duplicate=False,
          is_qcfail=False,
          is_secondary=False,
          is_unmapped=False,
          is_supplementary=True,
          mapping_quality=60,
          is_filtered=True,
          allow_supplementary=False,
          message="Test 7: Read filter test, filtering supplementary."),
      dict(
          is_duplicate=False,
          is_qcfail=False,
          is_secondary=False,
          is_unmapped=True,
          is_supplementary=False,
          mapping_quality=60,
          is_filtered=True,
          allow_supplementary=False,
          message="Test 8: Read filter test, filtered because is unmapped."),
  )
  @flagsaver.flagsaver
  def test_filter_reads(self, is_duplicate, is_qcfail, is_secondary,
                        is_unmapped, is_supplementary, mapping_quality,
                        is_filtered, allow_supplementary, message):
    read_attr = ReadAttributes(
        bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
        query_name="read_1",
        query_sequence="ACGTACGTACA",
        contig="chr20",
        start=0,
        base_qualities=[60] * 11,
        mapping_quality=60,
        cigartuples=[(pysam.CMATCH, 11)],
        tags=None)
    read = self.get_pysam_read_object(read_attr)
    read.mapping_quality = mapping_quality
    if is_duplicate:
      read.is_duplicate = True
    if is_qcfail:
      read.is_qcfail = True
    if is_secondary:
      read.is_secondary = True
    if is_unmapped:
      read.is_unmapped = True
    if is_supplementary:
      read.is_supplementary = True

    mapping_quality_threshold = 60
    filtered_reads = utils_make_images.filter_reads([read],
                                                    mapping_quality_threshold,
                                                    allow_supplementary)
    if is_filtered:
      self.assertEmpty(filtered_reads, msg=message)
    else:
      self.assertEqual(filtered_reads, [read], msg=message)

  @parameterized.parameters(
      dict(
          run_summary={},
          output_filename="make_examples",
          is_training=True,
          flags=[],
          expected_summary={},
          message="Test 1: Write summary with empty attributes.",
      ),
      dict(
          run_summary={},
          output_filename="make_examples",
          is_training=True,
          flags=[
              Flag(name="bam", value="bam_value"),
              Flag(name="contig", value="contig_value"),
          ],
          expected_summary={"bam": "bam_value", "contig": "contig_value"},
          message="Test 2: Write summary with flags only.",
      ),
      dict(
          run_summary={"example_counter": 1000, "interval_counter": 2},
          output_filename="make_examples",
          is_training=True,
          flags=[],
          expected_summary={"example_counter": 1000, "interval_counter": 2},
          message="Test 3: Write summary with run summary only.",
      ),
      dict(
          run_summary={"example_counter": 2000, "interval_counter": 5},
          output_filename="make_examples",
          is_training=True,
          flags=[
              Flag(name="bam", value="bam_value"),
              Flag(name="contig", value="contig_value"),
          ],
          expected_summary={
              "bam": "bam_value",
              "contig": "contig_value",
              "example_counter": 2000,
              "interval_counter": 5,
          },
          message="Test 4: Write summary with flags and run_summary.",
      ),
      dict(
          run_summary={"example_counter": 3000, "interval_counter": 6},
          output_filename="make_examples",
          is_training=False,
          flags=[
              Flag(name="bam", value="bam_value"),
              Flag(name="contig", value="contig_value"),
          ],
          expected_summary={
              "bam": "bam_value",
              "contig": "contig_value",
              "example_counter": 3000,
              "interval_counter": 6,
          },
          message="Test 5: Write summary with is_training false.",
      ),
  )
  @flagsaver.flagsaver
  def test_write_summary(self, run_summary, output_filename, is_training, flags,
                         expected_summary, message):
    tmp_dir = self.create_tempdir()
    output_filename = os.path.join(tmp_dir, output_filename)

    utils_make_images.write_summary(run_summary, output_filename, is_training,
                                    flags)

    summary_name = "training" if is_training else "inference"
    summary_path = f"{output_filename}_{summary_name}.summary.json"
    saved_summary = json.load(open(summary_path, "r"))

    self.assertDictEqual(saved_summary, expected_summary, msg=message)


if __name__ == "__main__":
  absltest.main()
