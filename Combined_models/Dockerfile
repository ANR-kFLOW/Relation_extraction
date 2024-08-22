FROM continuumio/miniconda3:24.1.2-0

ARG ENV_FILE=environment.yml

ADD ${ENV_FILE} environment.yml

RUN conda env create -n rel_extractor -f environment.yml
ENV PATH=/opt/conda/envs/rel_extractor/bin:$PATH
ENV WANDB_MODE=disabled

ADD . .


EXPOSE 5004
CMD ["python", "app.py"]