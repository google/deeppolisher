# Build with:
#   sudo docker build -t deeppolisher .
# For GPU:
#   sudo docker build --build-arg build_gpu=true --build-arg FROM_IMAGE=nvidia/cuda:11.3.1-cudnn8-runtime -t deeppolisher_gpu .



ARG FROM_IMAGE=continuumio/miniconda3
ARG VERSION=v0.1.0

FROM continuumio/miniconda3 as conda_setup
RUN conda config --add channels defaults && \
    conda config --add channels bioconda && \
    conda config --add channels conda-forge
RUN conda create -n bio \
                    python=3.9.15 \
                    parallel \
                    jq \
                    gcc \
                    memory_profiler \
                    pycocotools \
                    bioconda::seqtk \
                    bioconda::bedtools \
                    bioconda::pysam \
                    bioconda::samtools=1.15 \
    && conda clean -a

FROM ${FROM_IMAGE} as builder
COPY --from=conda_setup /opt/conda /opt/conda

ENV PATH=/opt/conda/envs/bio/bin:/opt/conda/bin:"${PATH}"
ENV LD_LIBRARY_PATH=/opt/conda/envs/bio/lib:/opt/mytools/lib/x86_64-linux-gnu:"${LD_LIBRARY_PATH}"

ARG VERSION
ENV VERSION=${VERSION}

WORKDIR /opt/models/pacbio_model
ADD https://storage.googleapis.com/brain-genomics-public/research/deeppolisher/models/${VERSION}/pacbio_model/checkpoint.data-00000-of-00001 .
ADD https://storage.googleapis.com/brain-genomics-public/research/deeppolisher/models/${VERSION}/pacbio_model/checkpoint.index .
ADD https://storage.googleapis.com/brain-genomics-public/research/deeppolisher/models/${VERSION}/pacbio_model/params.json .
RUN chmod -R +r /opt/models/pacbio_model/*

COPY . /opt/polisher
WORKDIR /opt/polisher

ARG build_gpu
RUN if [ "${_TAG_NAME}" = "*gpu" ] || [ "${build_gpu}" = "true" ]; then \
        echo "Installing polisher[gpu] version"; \
        pip install .[gpu]; \
    else \
        echo "Installing polisher[cpu] version"; \
        pip install .[cpu]; \
    fi

CMD ["polisher"]

