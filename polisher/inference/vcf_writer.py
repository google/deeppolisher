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
"""Module for writing a VCF file."""
import pysam
from polisher.inference import inference_utils


class VCFWriter:
  """Pysam-based VCF writing."""

  def __init__(self, reference_file_path, sample_name, output_dir, filename):
    """Constructor for VCFWriter.

    Args:
      reference_file_path: Path to the reference FASTA file.
      sample_name: Sample name to put in the output VCF file.
      output_dir: Path to the output directory.
      filename: Filename for the VCF.
    """
    # Fetch the contigs and their lengths to put in the header.
    with pysam.FastaFile(reference_file_path) as fasta_reader:
      fasta_contigs = fasta_reader.references
      contig_lengths = {
          contig: fasta_reader.get_reference_length(contig)
          for contig in fasta_contigs
      }

    # Get the VCF header.
    self.vcf_header = self.get_vcf_header(
        sample_name, fasta_contigs, contig_lengths
    )
    # Make sure the output_dir has / at the end.
    self.output_dir = output_dir if output_dir[-1] == '/' else output_dir + '/'

    # Get output filename.
    self.output_filename = self.output_dir + filename + '.vcf.gz'

    # Create the output VCF file.
    self.vcf_file = pysam.VariantFile(
        self.output_filename, 'w', header=self.vcf_header
    )

  def write_vcf_records(
      self, variants_list: list[inference_utils.Variant]
  ) -> int:
    """Write variants to the output file.

    Args:
      variants_list: List of variants to write in self.vcf_file.

    Returns:
      Total variants.
    """
    total_variants = 0
    # Iterate over each variant.
    for variant in variants_list:
      # Create the reference and alt allele list.
      alleles = [variant.ref_base, variant.alt_base]
      if variant.alt_tuple:
        alleles = (variant.ref_base, variant.alt_tuple[0], variant.alt_tuple[1])
      if variant.genotype:
        genotype = variant.genotype
      else:
        genotype = [0, 1]
      # Create pysam vcf record.
      # Currently we report everything unfiltered, if we want to put in logics
      # for filtering, it can be implemented here.
      vcf_record = self.vcf_file.new_record(
          contig=variant.contig,
          start=variant.position_start,
          stop=variant.position_end,
          id='.',
          qual=variant.quality,
          filter='PASS',
          alleles=alleles,
          GT=genotype,
          GQ=variant.quality,
      )
      # Write the vcf record to file.
      self.vcf_file.write(vcf_record)
      total_variants += 1

    return total_variants

  def get_vcf_header(
      self, sample_name: str, contigs: list[str], contig_lengths: dict[str, int]
  ) -> pysam.VariantHeader:
    """Get the VCF header.

    Args:
      sample_name: Name of the sample to use in the VCF header.
      contigs: List of contig names.
      contig_lengths: List of contig lengths.

    Returns:
      A pysam.VariantHeader containing the header information.
    """
    header = pysam.VariantHeader()
    # Filter PASS, we use this for default.
    items = [('ID', 'PASS'), ('Description', 'All filters passed')]
    header.add_meta(key='FILTER', items=items)
    # If refCall, this can be used instead of PASS.
    # Currently not used but useful for future.
    items = [('ID', 'refCall'), ('Description', 'Call is homozygous')]
    header.add_meta(key='FILTER', items=items)
    # If genotype quality is too low, this can be used insted of PASS.
    # Currently not used but useful for future.
    items = [('ID', 'lowGQ'), ('Description', 'Low genotype quality')]
    header.add_meta(key='FILTER', items=items)
    # If variant quality is too low, this can be used insted of PASS.
    # Currently not used but useful for future.
    items = [('ID', 'lowQUAL'), ('Description', 'Low variant call quality')]
    header.add_meta(key='FILTER', items=items)
    # Put the genotype of the record. This is a FORMAT, not a filter.
    items = [
        ('ID', 'GT'),
        ('Number', 1),
        ('Type', 'String'),
        ('Description', 'Genotype'),
    ]
    header.add_meta(key='FORMAT', items=items)
    items = [
        ('ID', 'GQ'),
        ('Number', 1),
        ('Type', 'Float'),
        ('Description', 'Genotype Quality'),
    ]
    header.add_meta(key='FORMAT', items=items)
    # The rest of the formats are depth, AD, VAF that we currently don't
    # use for polisher, but we may use in future.
    items = [
        ('ID', 'DP'),
        ('Number', 1),
        ('Type', 'Integer'),
        ('Description', 'Depth'),
    ]
    header.add_meta(key='FORMAT', items=items)
    items = [
        ('ID', 'AD'),
        ('Number', 'A'),
        ('Type', 'Integer'),
        ('Description', 'Allele depth'),
    ]
    header.add_meta(key='FORMAT', items=items)
    items = [
        ('ID', 'VAF'),
        ('Number', 'A'),
        ('Type', 'Float'),
        ('Description', 'Variant allele fractions.'),
    ]
    header.add_meta(key='FORMAT', items=items)

    # Add all contig names to the VCF.
    for contig_name in contigs:
      header.contigs.add(contig_name, length=contig_lengths[contig_name])

    # Add sample to the header.
    header.add_sample(sample_name)

    return header
