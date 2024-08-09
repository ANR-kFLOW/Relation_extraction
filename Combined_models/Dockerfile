FROM continuumio/miniconda3:24.1.2-0

ADD environment.yml environment.yml

RUN conda env create -n rel_extractor -f environment.yml
ENV PATH /opt/conda/envs/rel_extractor/bin:$PATH
#RUN /bin/bash -c "source activate rel_extractor"

ADD . .

#ENV CONDA_DEFAULT_ENV rel_extractor

EXPOSE 5004
CMD ["python", "app.py"]