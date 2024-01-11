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
"""Utility functions being used for make images."""

import collections
import dataclasses
import json
import os
from typing import Dict, Iterable, Iterator, List, Sequence, Any

from absl import logging

import pysam
import tensorflow as tf

from polisher.make_images import haplotype


@dataclasses.dataclass
class RegionRecord:
  """Represents a genomics region.

  Attributes:
    contig: Name of a contig or chromosome.
    start: start position of the region.
    stop: end position of the region.
  """
  contig: str
  start: int
  stop: int

  def __str__(self):
    return '[REGION: Contig= %s, Start= %d, Stop= %d]' % (self.contig,
                                                          self.start, self.stop)


def get_contig_length(fasta_file: str, contig: str,
                      common_contigs: List[str]) -> int:
  """Returns length of a contig found in the fasta file.

  Args:
    fasta_file: Path to a fasta file.
    contig: Name of a contig.
    common_contigs: Common contigs found between bam and fasta files.

  Returns:
    Length of a contig from the fasta file.

  Raises:
    ValueError if contig not found in Fasta
  """
  with pysam.FastaFile(fasta_file) as fasta_reader:
    if contig not in common_contigs:
      raise ValueError(
          f'Contig not found: contig {contig!r} not found in fasta.')
    # the pysam query is 1-based, so using -1 to make it 0-based.
    contig_length = fasta_reader.get_reference_length(contig) - 1

  return contig_length


def parse_region_string(region_string: str, fasta_file: str,
                        common_contigs: List[str]) -> RegionRecord:
  """Takes string in "contig:start-stop" format and returns a RegionRecord.

  This function parses a string in format "contig:start-stop" and creates a
  RegionRecord(contig, start, stop).

  If the region string is malformed then error is raised.
  Args:
    region_string: User specified region string in format contig:start-stop.
    fasta_file: Path to the reference fasta file.
    common_contigs: Common contigs found in bam and fasta.

  Returns:
    Parsed RegionRecord from the region_string.

  Raises:
    ValueError: If region string passed by the user is malformed.
  """
  if len(region_string.split(':')) != 2:
    raise ValueError(f'Malformed region string {region_string!r}. '
                     'Expected format contig:start-stop.')

  contig, start_stop = region_string.split(':')
  if len(start_stop.split('-')) != 2:
    raise ValueError(f'Malformed region string {region_string!r}. '
                     'Expected format contig:start-stop.')

  start, stop = start_stop.split('-')
  # check if stop is overflowing the length of the contig
  contig_end = get_contig_length(fasta_file, contig, common_contigs)
  stop = min(int(stop), int(contig_end))
  region_record = RegionRecord(contig, int(start), int(stop))

  if region_record.start > region_record.stop:
    raise ValueError(f'Malformed region string {region_string!r}.'
                     ' start value must be greater than stop value.')

  return region_record


def parse_contig_string(region_string: str, fasta_file: str,
                        common_contigs: List[str]) -> RegionRecord:
  """Takes string in format "contig" and returns a RegionRecord.

  This function parses a string in format "contig" and creates a
  RegionRecord(contig, 0, len(contig)). The len(contig) is fetched from the
  fasta file.

  If the contig is not found in fasta file then error is raised.

  Args:
    region_string: User specified region string in format contig.
    fasta_file: Path to the reference fasta file.
    common_contigs: Common contigs found between bam and fasta.

  Returns:
    Parsed RegionRecord from the region_string.

  Raises:
    ValueError: If region string passed by the user is malformed.
  """
  contig = region_string
  contig_end = get_contig_length(fasta_file, contig, common_contigs)
  region_record = RegionRecord(contig, 0, int(contig_end))

  return region_record


def process_region_string(region_string: str, fasta_file: str,
                          common_contigs: List[str]) -> RegionRecord:
  """Takes region string and returns a RegionRecord.

  User can define the region in two ways:
  1) "contig_name:start-stop":
    - In this case the string is parsed to return
    RegionRecord(contig, start, stop)
  2) "contig_name"
    - When no start-stop is defined, the entire contig/chromosome will be
    processed so RegionRecord(contig, 0, len(contig)) is returned
    where len(contig) is captured by querying the fasta file.

  Args:
    region_string: User specified region string.
    fasta_file: Path to the reference fasta file.
    common_contigs: Common contigs found between bam and fasta.

  Returns:
    Parsed RegionRecord from the region_string.
  """
  if ':' in region_string:
    # user has passed a string in the format "contig:start-stop"
    # it is guaranteed that the fasta file will have no contig names with ':'
    # in it.
    return parse_region_string(region_string, fasta_file, common_contigs)
  else:
    # user has passed a string the format "contig"
    return parse_contig_string(region_string, fasta_file, common_contigs)


def read_bed(bed_file: str) -> Dict[str, List[RegionRecord]]:
  """Reads in bed file and returns a dictionary of RegionRecord lists.

  This function reads in a bed file and creates a dictionary where the keys are
  contigs like {chr1, chr2, chr3} and values for each key would be the regions
  observed in the bed file with that contig.

  For example: Bed file contains:
  chr1 50 60
  chr2 60 70
  chr2 90 100
  This function will return:
  {'chr1': [(chr1, 50, 60)], 'chr2': [(chr2, 60, 70), (chr2, 90, 100)]}

  Args:
    bed_file: Path to a valid bed file.

  Returns:
    A dictionary where key is the contig and value is a list of regions observed
    in the bed file with the associated contig value.
  Raises:
    ValueError: If bed file has an invalid entry.
  """
  region_records_by_contig = dict()
  with open(bed_file, 'r') as bedfile:
    for line in bedfile:
      if len(line.strip().split('\t')) < 3:
        raise ValueError('Invalid entry in BED file.')

      contig, start, stop = line.strip().split('\t')[:3]
      if contig in region_records_by_contig:
        region_records_by_contig[contig].append(
            RegionRecord(contig, int(start), int(stop)))
      else:
        region_records_by_contig[contig] = [
            RegionRecord(contig, int(start), int(stop))
        ]

  return region_records_by_contig


def split_regions_in_intervals(regions: List[RegionRecord],
                               region_length: int) -> List[RegionRecord]:
  """Splits each region into intervals of region length.

  Given a list of region: [(chr20, 0, 1000)] and an region_length=500 this
  function will return [(chr20, 0, 500), (chr20, 500, 1000)].

  Args:
    regions: List of regions.
    region_length: Length of each intervals.

  Returns:
    A list of regions of maximum length of region_length.
  """
  all_intervals = []
  for region in regions:
    for pos in range(region.start, region.stop, region_length):
      interval_start = max(region.start, pos)
      interval_end = min(region.stop, pos + region_length)
      interval_record = RegionRecord(region.contig, interval_start,
                                     interval_end)
      all_intervals.append(interval_record)
  return all_intervals


def get_contig_regions(bam_file: str, fasta_file: str, region: str,
                       interval_length: int) -> List[RegionRecord]:
  """Creates a list of regions for processing.

  Reads contig names from bam and fasta file and creates a list of regions
  for processing. If the region string is not present then the intervals are
  created by taking the name of the contigs that are common between bam and
  fasta file.

  Args:
    bam_file: Path to an alignment bam file.
    fasta_file: Path to a fasta file.
    region: A region string in the format contig:start-end.
    interval_length: The length of intervals.

  Returns:
    A list of region intervals.
  Raises:
    ValueError: If a contig is not found in both BAM and FASTA.
  """
  with pysam.AlignmentFile(bam_file) as bam_reader, pysam.FastaFile(
      fasta_file) as fasta_reader:
    bam_contigs = bam_reader.references
    fasta_contigs = fasta_reader.references

    bam_fasta_common_contigs = list(set(fasta_contigs) & set(bam_contigs))
    regions_to_process = []

    if region:
      # if region parameter has been set and needs to be parsed
      if ',' in region:
        # user has provided comma separated values of a list of regions
        # process them individually
        parsed_regions = region.split(',')
        for parsed_region in parsed_regions:
          region_record = process_region_string(parsed_region, fasta_file,
                                                bam_fasta_common_contigs)
          regions_to_process.append(region_record)
      else:
        # user has provided a single region value.
        region_record = process_region_string(region, fasta_file,
                                              bam_fasta_common_contigs)
        regions_to_process.append(region_record)
    else:
      # user didn't provide any region, process the entire genome.
      for contig in bam_fasta_common_contigs:
        contig_end = get_contig_length(fasta_file, contig,
                                       bam_fasta_common_contigs)
        region_record = RegionRecord(contig, 0, contig_end)
        regions_to_process.append(region_record)

    region_intervals = split_regions_in_intervals(regions_to_process,
                                                  interval_length)

  return region_intervals


def range_intersect(interval: RegionRecord,
                    bed_intervals: List[RegionRecord]) -> List[RegionRecord]:
  """Intersect a given interval with intervals we get from a bed file.

  Takes an interval and creates an intersection with the intervals contained in
  bed files. This function assumes that the bed file is "not" sorted. So, we
  check every interval present in the bed file.

  Args:
    interval: A given interval.
    bed_intervals: A list of intervals from the same contig found in bed file.

  Returns:
    A list of intervals representing the intersected intervals.
  """
  if not bed_intervals:
    return [interval]

  intersected_regions = []
  for bed_interval in bed_intervals:
    if bed_interval.contig != interval.contig:
      continue
    if bed_interval.stop < interval.start or bed_interval.start > interval.stop:
      continue
    intersected_regions.append(
        RegionRecord(bed_interval.contig, max(bed_interval.start,
                                              interval.start),
                     min(bed_interval.stop, interval.stop)))

  return intersected_regions


def get_process_intervals(all_intervals: Sequence[RegionRecord],
                          process_id: int,
                          total_cpus: int) -> Iterator[RegionRecord]:
  """Generates a list of intervals for a process with process_id.

  Takes a list of intervals that needs to be processed by a total_cpus processes
  and generates a list of intervals for the given process with an id process_id.
  The process_id value must be less than total_cpus.

  For example, if we have 5 intervals as such = [0, 1, 2, 3, 4, 5] with
  total_cpus=2 then for the process with process_id 0 we will get [0, 2, 4] and
  for process_id 1 we will get [1, 3, 5].

  Reasoning for not simply grouping the intervals into n groups/chunks:
  Genomic regions vary significantly in complexity. It starts with repetitive
  telomeric sequence, goes into short/long arm and hits the centromere.
  If we give a single process to process the entire centromere by grouping
  nearby regions into n chunks, that process will take much longer to finish and
  the load will not be balanced. Hence, we scatter the regions so continuous
  intervals get scattered between different processes.

  Args:
    all_intervals: A list of all intervals that needs to be processed.
    process_id: ID of a given process.
    total_cpus: Total number of processes available to process all intervals.

  Yields:
    A list of intervals assigned to the given process.
  """
  if process_id >= total_cpus:
    raise ValueError(
        f'Pre-processing error: process-id: {process_id!r} must be less than'
        f' total available cpus: {total_cpus!r}.')

  for interval_no, interval in enumerate(all_intervals):
    if interval_no % total_cpus == process_id:
      yield interval


def get_reads_from_bam(bam_file: str,
                       interval: RegionRecord) -> List[pysam.AlignedSegment]:
  """Get a list of reads from a bam file within a region.

  Args:
    bam_file: Path to an alignment bam file.
    interval: A genomic interval to extract reads from.

  Returns:
    A list of aligned reads within the interval.
  """
  aligned_reads = []
  with pysam.AlignmentFile(bam_file) as bam_reader:
    aligned_reads = list(
        bam_reader.fetch(interval.contig, interval.start, interval.stop))
  return aligned_reads


def get_reference_sequence_from_fasta(fasta_file: str, interval: RegionRecord,
                                      padding: int) -> str:
  """Get the reference sequence from fasta file.

  Args:
    fasta_file: Path to the fasta file.
    interval: A genomic interval to extract reference sequence from.
    padding: Added padding to fetch the reference sequence.

  Returns:
    A reference sequence of the given interval.
  """
  reference_sequence = ''
  with pysam.FastaFile(fasta_file) as fasta_reader:
    reference_sequence = fasta_reader.fetch(interval.contig, interval.start,
                                            interval.stop + padding)
  return reference_sequence


def bin_reads_by_haplotype(
    reads: Iterable[pysam.AlignedSegment]
) -> Dict[int, List[pysam.AlignedSegment]]:
  """Bin reads by their associated haplotype tag.

  The human genome is diploid (we have two chromosomes, one inherited from
  mother and other one from father). This program assumes that the diploid
  assumption is always valid.

  Upstream phasing algorithms use {0, 1, 2} as haplotype tags. Where 0 means
  the read could not be haplotyped, 1 means reads belong to haplotype one and
  2 means haplotype two.

  Args:
    reads: A list of reads with haplotype tags.

  Returns:
    A dictionary where key is the haplotype tag and value is a list of reads
    with that haplotype value.

  Raises:
    ValueError: If HP-tag is not found in the allowed hp-tag list then error
    is raised.
  """
  read_haplotype_dictionary = collections.defaultdict(list)

  for read in reads:
    read_haplotype_tag = haplotype.get_read_haplotype_tag(read)
    read_haplotype_dictionary[read_haplotype_tag].append(read)

  return read_haplotype_dictionary


def check_if_position_is_within_regions(position: int,
                                        intervals: List[RegionRecord]) -> bool:
  """Checks if a position is within given bed intervals.

  This method is used to see if active_positions are contained within provided
  intervals. If there are no intervals sent then the parameter for intervals
  is not set so the return should be True. If there is a set of intervals then
  true only if is contained within one of the intervals.

  Args:
    position: A value representing a genomic position.
    intervals: A list of close-ended genomic intervals.

  Returns:
    True if position is contained within any of the interval in the list or
    there are no intervals to check against.

    False if there are intervals but the position is not within any of the
    interval.
  """
  for interval in intervals:
    # bed interval ends at the left or starts at right (no overlap)
    if interval.start <= position <= interval.stop:
      return True

  return False


def filter_reads(
    reads: List[pysam.AlignedSegment],
    mapping_quality_threshold: int,
    allow_supplementary: bool = False) -> List[pysam.AlignedSegment]:
  """Filter reads and split them  haplotype tag.

  Reads are filtered if they are duplicate, qc_failed, secordary, unmapped
  or has mapping quality below a set threshold. Supplementary reads are filtered
  based on the `allow_supplementary` parameter.

  Args:
    reads: A list of reads with haplotype tags.
    mapping_quality_threshold: Threshold for a mapping quality filter.
    allow_supplementary: If true, the supplementary reads are not filtered.

  Returns:
    A list of filtered reads.
  """
  filtered_reads = []
  for read in reads:
    if read.is_duplicate or read.is_qcfail:
      continue

    if read.is_secondary or read.is_unmapped:
      continue

    if read.mapping_quality < mapping_quality_threshold:
      continue

    if read.is_supplementary and not allow_supplementary:
      continue

    filtered_reads.append(read)
  return filtered_reads


def write_tf_record(tf_examples: List[bytes], tf_writer: tf.io.TFRecordWriter):
  """Writes tf examples to a file."""
  for tf_example in tf_examples:
    tf_writer.write(tf_example)
  tf_writer.flush()


def write_summary(run_summary: Dict[str, Any], output_filename: str,
                  is_training: bool, flags: Sequence[Any]):
  """Writes a json file given a set of flags and summary of the run.

  Args:
    run_summary: A dictionary that contains the run summary.
    output_filename: Name of the output file.
    is_training: True if training mode is true.
    flags: List of flags to dump in the summary json.
  """
  # Write the summary of the run.
  summary_name = 'training' if is_training else 'inference'
  # Replace the output extension to json.
  summary_filename = f'{output_filename}_{summary_name}.summary.json'

  logging.info('Writing %s.', summary_filename)
  tf.io.gfile.makedirs(os.path.dirname(summary_filename))
  with tf.io.gfile.GFile(summary_filename, 'w') as summary_file:
    summary = {flag.name: flag.value for flag in flags}
    summary.update(dict(run_summary.items()))
    json_summary = json.dumps(summary, indent=True)
    summary_file.write(json_summary)
