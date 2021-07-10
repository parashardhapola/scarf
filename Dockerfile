FROM ubuntu:20.04

RUN apt update -y && apt autoremove -y && apt clean -y && apt autoclean -y && apt upgrade -y
RUN apt install -y wget build-essential git nano

# The following is done to make sure that tzdata package doesnt prompt for timezone during installation
ARG TZ="Europe/Stockholm"
RUN DEBIAN_FRONTEND="noninteractive" TZ=$TZ apt-get -y install tzdata
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone &

#Installing dependencies for sgtsne
RUN apt install -y libmetis-dev libtbb-dev libfftw3-dev lib32gcc-7-dev libflann-dev libcilkrts5

# Installing lastest Miniconda3
RUN wget -O miniconda_inst "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" && \
	bash miniconda_inst -b && \
	rm miniconda_inst

# Exporting PATH and also saving it in bashrc for next session
# /workspace/bin is so that sgtsne can be found
RUN echo "export PATH=$PATH:/root/miniconda3/bin:/workspace/bin" >> /root/.bashrc
ENV PATH=$PATH:/root/miniconda3/bin:/workspace/bin

# Installing numpy and pybind11 beforehand because sometimes they don't install so well from requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -U numpy pybind11
RUN pip install --no-cache-dir -U dask<=2021.03.1

# Needed for dask
RUN conda install -c conda-forge 'fsspec>=0.3.3'

# This for interactive programming purposes
RUN pip install jupyterlab ipython-autotime
#RUN conda install nodejs

# For compiling docs
RUN pip install Sphinx sphinx-autodoc-typehints nbsphinx sphinx_rtd_theme

# For building vignettes
RUN conda install -y nodejs
RUN conda install -y -c conda-forge pandoc
RUN pip install jupytext
RUN jupyter lab build -y

RUN pip install scarf

# Building from Github repo
#RUN mkdir -p /tmp/scarf
#COPY . /tmp/scarf/
#RUN pip install -e /tmp/scarf/

RUN mkdir workspace && \
    echo "jupyter lab --port 9734 --ip=0.0.0.0 --allow-root --no-browser" > launch_jupyter.sh

# If you want to launch jupyter manually then feel free to comment this out.
CMD cd workspace && \
	bash ../launch_jupyter.sh
