FROM continuumio/miniconda3:24.1.2-0

ARG ENV_FILE=environment.yml

ADD ${ENV_FILE} environment.yml

RUN conda env create -n rel_extractor -f environment.yml

#CMD ["conda create --name rel_extractor"]
#CMD ['conda activate rel_extractor']
#CMD ['conda install anaconda::evaluate']
#CMD ['conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia']
#CMD ['conda install anaconda::scikit-learn']
#CMD ['conda install conda-forge::transformers']
#CMD ['conda install conda-forge::accelerate']
#CMD ['conda install conda-forge::langchain']
#CMD ['conda install conda-forge::python-dotenv']
#CMD ['conda install anaconda::flask']
#CMD ['conda install conda-forge::werkzeug=2.2']
#CMD ['conda install conda-forge::openai=0.28.1']
#CMD ['conda install conda-forge::huggingface_hub']
#CMD ['conda install conda-forge::flask-swagger-ui']
#CMD ['conda install conda-forge::flask-restful']
#CMD ['conda install conda-forge::streamlit']
#CMD ['conda install anaconda::st-annotated-text']
#CMD ['conda env export > env.yml']

ENV PATH=/opt/conda/envs/rel_extractor/bin:$PATH
ENV WANDB_MODE=disabled


ADD . .
EXPOSE 5004






CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "5004"]