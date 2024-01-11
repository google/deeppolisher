# HG002 chr20 case study

In this case study we show how to polish HiFiasm diploid assembly with DeepPolisher.
We use PacBio HiFi reads to polish the assembly.

## Prerequisites

Software needed to run this case-study:

* [Docker container](https://docs.docker.com/engine/install/).
* [samtools, bcftools and htslib](https://www.htslib.org/download/).


## Input requirements

The input to this case study are:

* Diploid assembly of HG002 chr20 using [hifiasm 0.19.5](https://github.com/chhylp123/hifiasm/releases/tag/0.19.5).
* HG002 chr20 reads aligned to the assembly using [PHARAOH pipeline](https://github.com/miramastoras/hpp_production_workflows/blob/master/QC/wdl/workflows/PHARAOH.wdl).

The input bam needs to be **haploid**. Meaning, haplotype-specific reads are
aligned to the assembly specific for that haplotype. If you align all reads
against the assembly for a diploid sample, DeepPolisher will not be able to
identify errors accurately.

## Run DeepPolisher

Download input data and set up directories:

```bash
BASE="${HOME}/polisher_case_study"
INPUT_DIR="${BASE}/input"
OUTPUT_DIR="${BASE}/output"
mkdir -p ${INPUT_DIR}
mkdir -p ${OUTPUT_DIR}
HTTPDIR=https://storage.googleapis.com/brain-genomics/kishwar/share/polisher/polisher-case-study

curl ${HTTPDIR}/HG002.trio_hifiasm_0.19.5.DC_1.2_40x.mat.chr20.fasta > ${INPUT_DIR}/HG002.trio_hifiasm_0.19.5.DC_1.2_40x.mat.chr20.fasta
curl ${HTTPDIR}/HG002.trio_hifiasm_0.19.5.DC_1.2_40x.mat.chr20.fasta.fai > ${INPUT_DIR}/HG002.trio_hifiasm_0.19.5.DC_1.2_40x.mat.chr20.fasta.fai
curl ${HTTPDIR}/HG002.trio_hifiasm_0.19.5.DC_1.2_40x.pat.chr20.fasta > ${INPUT_DIR}/HG002.trio_hifiasm_0.19.5.DC_1.2_40x.pat.chr20.fasta
curl ${HTTPDIR}/HG002.trio_hifiasm_0.19.5.DC_1.2_40x.pat.chr20.fasta.fai > ${INPUT_DIR}/HG002.trio_hifiasm_0.19.5.DC_1.2_40x.pat.chr20.fasta.fai
curl ${HTTPDIR}/HG002.trio_hifiasm_0.19.5.DC_1.2_40x.dip.minimap2v2.26.PHARAOH.chr20.bam > ${INPUT_DIR}/HG002.trio_hifiasm_0.19.5.DC_1.2_40x.dip.minimap2v2.26.PHARAOH.chr20.bam
curl ${HTTPDIR}/HG002.trio_hifiasm_0.19.5.DC_1.2_40x.dip.minimap2v2.26.PHARAOH.chr20.bam.bai > ${INPUT_DIR}/HG002.trio_hifiasm_0.19.5.DC_1.2_40x.dip.minimap2v2.26.PHARAOH.chr20.bam.bai

MAT_BAM="${INPUT_DIR}/HG002.trio_hifiasm_0.19.5.DC_1.2_40x.dip.minimap2v2.26.PHARAOH.chr20.bam"
MAT_FASTA="${INPUT_DIR}/HG002.trio_hifiasm_0.19.5.DC_1.2_40x.mat.chr20.fasta"
PAT_BAM="${INPUT_DIR}/HG002.trio_hifiasm_0.19.5.DC_1.2_40x.dip.minimap2v2.26.PHARAOH.chr20.bam"
PAT_FASTA="${INPUT_DIR}/HG002.trio_hifiasm_0.19.5.DC_1.2_40x.pat.chr20.fasta"
MAT_IMAGE_OUTPUT_DIR="${OUTPUT_DIR}/mat_images"
PAT_IMAGE_OUTPUT_DIR="${OUTPUT_DIR}/pat_images"
MAT_VCF_OUTPUT_DIR="${OUTPUT_DIR}/mat_vcf"
PAT_VCF_OUTPUT_DIR="${OUTPUT_DIR}/pat_vcf"
mkdir -p ${MAT_IMAGE_OUTPUT_DIR}
mkdir -p ${PAT_IMAGE_OUTPUT_DIR}
mkdir -p ${MAT_VCF_OUTPUT_DIR}
mkdir -p ${PAT_VCF_OUTPUT_DIR}
```

Run DeepPolisher:

```bash
sudo docker pull google/deepconsensus:polisher_v0.1.0

# Make images MAT
sudo docker run -it -v ${INPUT_DIR}:${INPUT_DIR} -v ${OUTPUT_DIR}:${OUTPUT_DIR} \
google/deepconsensus:polisher_v0.1.0 \
polisher make_images \
--bam ${MAT_BAM} \
--fasta ${MAT_FASTA} \
--output ${MAT_IMAGE_OUTPUT_DIR}/polishing_mat \
--cpus $(nproc)

# Inference on MAT images to generate MAT VCFs
sudo docker run -it -v ${INPUT_DIR}:${INPUT_DIR} -v ${OUTPUT_DIR}:${OUTPUT_DIR} \
google/deepconsensus:polisher_v0.1.0 \
polisher inference \
--input_dir ${MAT_IMAGE_OUTPUT_DIR} \
--out_dir ${MAT_VCF_OUTPUT_DIR} \
--checkpoint /opt/models/pacbio_model/checkpoint \
--reference_file ${MAT_FASTA} \
--sample_name HG002

# Make images PAT
sudo docker run -it -v ${INPUT_DIR}:${INPUT_DIR} -v ${OUTPUT_DIR}:${OUTPUT_DIR} \
google/deepconsensus:polisher_v0.1.0 \
polisher make_images \
--bam ${PAT_BAM} \
--fasta ${PAT_FASTA} \
--output ${PAT_IMAGE_OUTPUT_DIR}/polishing_pat \
--cpus $(nproc)

# Inference on PAT images
sudo docker run -it -v ${INPUT_DIR}:${INPUT_DIR} -v ${OUTPUT_DIR}:${OUTPUT_DIR} \
google/deepconsensus:polisher_v0.1.0 \
polisher inference \
--input_dir ${PAT_IMAGE_OUTPUT_DIR} \
--out_dir ${PAT_VCF_OUTPUT_DIR} \
--checkpoint /opt/models/pacbio_model/checkpoint \
--reference_file ${PAT_FASTA} \
--sample_name HG002
```

This produces two VCF files:

* `${PAT_VCF_OUTPUT_DIR}/polisher_output.unsorted.vcf.gz`: Errors found in
paternal assembly.
* `${MAT_VCF_OUTPUT_DIR}/polisher_output.unsorted.vcf.gz`: Errors found in
maternal assembly.

The edits proposed in these two VCFs can be applied to the assembly to get a
polished assembly.

## Apply the variants to the assembly

We can use `bcftools consensus` module to apply the edits suggested by
DeepPolisher to the assembly to generate a polished assembly.

```bash
tabix -p vcf ${PAT_VCF_OUTPUT_DIR}/polisher_output.unsorted.vcf.gz
tabix -p vcf ${MAT_VCF_OUTPUT_DIR}/polisher_output.unsorted.vcf.gz

bcftools consensus \
-f ${PAT_FASTA} \
-H 2 \
${PAT_VCF_OUTPUT_DIR}/polisher_output.unsorted.vcf.gz > ${OUTPUT_DIR}/pat.polished.fasta

bcftools consensus \
-f ${MAT_FASTA} \
-H 2 \
${MAT_VCF_OUTPUT_DIR}/polisher_output.unsorted.vcf.gz > ${OUTPUT_DIR}/mat.polished.fasta
```

After this step we have `pat.polished.fasta` and `mat.polished.fasta` which are
the two polished haplotypes of the assembly.

## Assess the quality improvement

There are several ways to assess the quality improvement after polishing. For
this case study, we will use GIAB variants to see how many errors we have
removed.

This is done is two steps. First, run `dipcall` to project the assembly to 
GRCh38 and derive variants. Then use Hap.py to assess the quality of the
variants. If we improve the quality of the assembly, we are going to see the
total variants error drop after polishing.

## Run Dipcall

```bash
cd ${OUTPUT_DIR}
wget https://github.com/lh3/dipcall/releases/download/v0.3/dipcall-0.3_x64-linux.tar.bz2
tar -jxf dipcall-0.3_x64-linux.tar.bz2
cd -

# Download GRCh38 reference and GIAB truth
HTTPDIR=https://storage.googleapis.com/brain-genomics/kishwar/share/polisher/polisher-case-study
curl ${HTTPDIR}/GRCh38_no_alt_chr20.fa > ${INPUT_DIR}/GRCh38_no_alt_chr20.fa
curl ${HTTPDIR}/GRCh38_no_alt_chr20.fa.fai > ${INPUT_DIR}/GRCh38_no_alt_chr20.fa.fai
curl ${HTTPDIR}/HG002_GRCh38_1_22_v4.2.1_benchmark.dipcall_intersected.bed > ${INPUT_DIR}/HG002_GRCh38_1_22_v4.2.1_benchmark.dipcall_intersected.bed
curl ${HTTPDIR}/HG002_GRCh38_1_22_v4.2.1_benchmark.chr20.vcf.gz > ${INPUT_DIR}/HG002_GRCh38_1_22_v4.2.1_benchmark.chr20.vcf.gz
curl ${HTTPDIR}/HG002_GRCh38_1_22_v4.2.1_benchmark.chr20.vcf.gz.tbi > ${INPUT_DIR}/HG002_GRCh38_1_22_v4.2.1_benchmark.chr20.vcf.gz.tbi

DIPCALL_OUTPUT="${OUTPUT_DIR}/dipcall_output"
GRCH38_CHR20_FASTA="${INPUT_DIR}/GRCh38_no_alt_chr20.fa"
mkdir -p ${DIPCALL_OUTPUT}

# Run dipcall on the polished assembly
${OUTPUT_DIR}/dipcall.kit/run-dipcall \
${DIPCALL_OUTPUT}/HG002_polished_2_GRCh38 \
${GRCH38_CHR20_FASTA} \
${OUTPUT_DIR}/pat.polished.fasta \
${OUTPUT_DIR}/mat.polished.fasta > ${OUTPUT_DIR}/HG002_polished_2_GRCh38.mak

make -j2 -f ${OUTPUT_DIR}/HG002_polished_2_GRCh38.mak

# Run dipcall on the raw assembly
${OUTPUT_DIR}/dipcall.kit/run-dipcall \
${DIPCALL_OUTPUT}/HG002_raw_2_GRCh38 \
${GRCH38_CHR20_FASTA} \
${INPUT_DIR}/HG002.trio_hifiasm_0.19.5.DC_1.2_40x.pat.chr20.fasta \
${INPUT_DIR}/HG002.trio_hifiasm_0.19.5.DC_1.2_40x.mat.chr20.fasta > ${OUTPUT_DIR}/HG002_raw_2_GRCh38.mak

make -j2 -f ${OUTPUT_DIR}/HG002_raw_2_GRCh38.mak
```

This step gives us two VCF files:

* `${DIPCALL_OUTPUT}/HG002_raw_2_GRCh38.dip.vcf.gz`: Variants derived from
unpolished assembly.
* `${DIPCALL_OUTPUT}/HG002_polished_2_GRCh38.dip.vcf.gz`: Variants derived from
polished assembly.

We expect the set of variants derived from the polished assembly to be more accurate than the ones derived from the unpolished assembly.

## Run Hap.py

```bash
GIAB_TRUTH_VCF=${INPUT_DIR}/HG002_GRCh38_1_22_v4.2.1_benchmark.chr20.vcf.gz
GIAB_TRUTH_BED=${INPUT_DIR}/HG002_GRCh38_1_22_v4.2.1_benchmark.dipcall_intersected.bed
HAPPY_OUTPUT="${OUTPUT_DIR}/happy_outputs"
mkdir -p ${HAPPY_OUTPUT}

sudo docker run -it -v ${INPUT_DIR}:${INPUT_DIR} -v ${OUTPUT_DIR}:${OUTPUT_DIR} \
jmcdani20/hap.py:v0.3.12 /opt/hap.py/bin/hap.py \
${GIAB_TRUTH_VCF} \
${DIPCALL_OUTPUT}/HG002_raw_2_GRCh38.dip.vcf.gz \
-f ${GIAB_TRUTH_BED} \
-r ${GRCH38_CHR20_FASTA} \
-o ${HAPPY_OUTPUT}/HG002_chr20_raw \
-l chr20 \
--pass-only \
--no-roc \
--no-json \
--engine=vcfeval \
--threads=$(nproc)
```

Output:

This shows accuracy of the variants derived from the assembly which reflects
the quality of the unpolished assembly.

```bash
Benchmarking Summary:
Type Filter  TRUTH.TOTAL  TRUTH.TP  TRUTH.FN  QUERY.TOTAL  QUERY.FP  QUERY.UNK  FP.gt  FP.al  METRIC.Recall  METRIC.Precision  METRIC.Frac_NA  METRIC.F1_Score  TRUTH.TOTAL.TiTv_ratio  QUERY.TOTAL.TiTv_ratio  TRUTH.TOTAL.het_hom_ratio  QUERY.TOTAL.het_hom_ratio
INDEL    ALL        11217     11015       202        21364       175       9810    100     35       0.981992          0.984854        0.459184         0.983421                     NaN                     NaN                   1.565568                   2.091495
INDEL   PASS        11217     11015       202        21364       175       9810    100     35       0.981992          0.984854        0.459184         0.983421                     NaN                     NaN                   1.565568                   2.091495
  SNP    ALL        71094     70883       211        81722        18      10697      7      2       0.997032          0.999747        0.130895         0.998387                2.312573                2.150037                   1.715044                   1.777861
  SNP   PASS        71094     70883       211        81722        18      10697      7      2       0.997032          0.999747        0.130895         0.998387                2.312573                2.150037                   1.715044                   1.777861
```

```bash
sudo docker run -it -v ${INPUT_DIR}:${INPUT_DIR} -v ${OUTPUT_DIR}:${OUTPUT_DIR} \
jmcdani20/hap.py:v0.3.12 /opt/hap.py/bin/hap.py \
${GIAB_TRUTH_VCF} \
${DIPCALL_OUTPUT}/HG002_polished_2_GRCh38.dip.vcf.gz \
-f ${GIAB_TRUTH_BED} \
-r ${GRCH38_CHR20_FASTA} \
-o ${HAPPY_OUTPUT}/HG002_chr20_polished \
-l chr20 \
--pass-only \
--no-roc \
--no-json \
--engine=vcfeval \
--threads=$(nproc)
```

Output:

This shows accuracy of the variants derived from the polished assembly which
reflects the quality of the polished assembly.

```bash
Type Filter  TRUTH.TOTAL  TRUTH.TP  TRUTH.FN  QUERY.TOTAL  QUERY.FP  QUERY.UNK  FP.gt  FP.al  METRIC.Recall  METRIC.Precision  METRIC.Frac_NA  METRIC.F1_Score  TRUTH.TOTAL.TiTv_ratio  QUERY.TOTAL.TiTv_ratio  TRUTH.TOTAL.het_hom_ratio  QUERY.TOTAL.het_hom_ratio
INDEL    ALL        11217     11166        51        21515        60       9905     14     17       0.995453          0.994832        0.460376         0.995143                     NaN                     NaN                   1.565568                   2.207714
INDEL   PASS        11217     11166        51        21515        60       9905     14     17       0.995453          0.994832        0.460376         0.995143                     NaN                     NaN                   1.565568                   2.207714
  SNP    ALL        71094     70894       200        81734        14      10701      3      3       0.997187          0.999803        0.130925         0.998493                2.312573                 2.14993                   1.715044                   1.781072
  SNP   PASS        71094     70894       200        81734        14      10701      3      3       0.997187          0.999803        0.130925         0.998493                2.312573                 2.14993                   1.715044                   1.781072
```

This shows:

* Total INDEL errors of 377 (202 + 175) in the **unpolished assembly** has
reduced to 111 (51 + 60) in the **polished assembly**.
* Total SNP errors of 229 (211 + 18) in the **unpolished assembly** has reduced
to 214 (200 + 14) in the **polished assembly**.
