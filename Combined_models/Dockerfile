FROM continuumio/miniconda3

COPY environment.yaml/
RUN conda env create -f environment.yaml && conda clean -a
RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/miniconda/envs/env/bin:$PATH
