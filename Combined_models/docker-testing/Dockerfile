# Start with the Miniconda3 base image
FROM continuumio/miniconda3

# Update conda to the latest version
RUN conda update -n base -c defaults conda

# Create a new conda environment with Python 3.11
RUN conda create -n myenv python=3.11 -y

# Activate the environment and install additional packages if needed
RUN conda run -n myenv conda install numpy pandas -c conda-forge -y

# Make sure the conda environment is activated by default
RUN echo "conda activate myenv" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Set the default command to start a shell
CMD ["bash"]