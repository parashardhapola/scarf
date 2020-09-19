FROM ubuntu:20.04

RUN apt update -y && apt autoremove -y && apt clean -y && apt autoclean -y
RUN apt install -y wget build-essential git nano

# Installing lastest Miniconda3
RUN wget -O miniconda_inst "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" && \
	bash miniconda_inst -b && \
	rm miniconda_inst

# Exporting PATH and also saving it in bashrc for next session
RUN echo "export PATH=$PATH:/root/miniconda3/bin" >> /root/.bashrc
ENV PATH=/root/miniconda3/bin:$PATH

# Installing numpy and pybind11 beforehand because sometimes thery dont't install so well from requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir numpy pybind11

# Installing pcst_fast here because I still haven't figured out how to install a git repo from requirements.txt
RUN pip install git+https://github.com/fraenkel-lab/pcst_fast.git

# This for interactive programming purposes
RUN pip install jupyterlab ipython-autotime
#RUN conda install nodejs

# For compiling docs
RUN pip install Sphinx sphinx-autodoc-typehints nbsphinx

# Setting up git for deve purposes. For example pushing commits or making pull requests.
# You can change this to your github credentials.
RUN git config --global user.name "parashardhapola"
RUN git config --global user.email parashar.dhapola@gmail.com

# Install Scarf directly from github repo. Comment this out if you want to install from pypi manually
# using `pip install scarf`. Alternatively you can also fork the repo and provide your username in link below.
RUN pip install git+https://github.com/parashardhapola/scarf.git

RUN mkdir workspace && \
    echo "jupyter lab --port 9734 --ip=0.0.0.0 --allow-root --no-browser" > /workspace/launch_jupyter.sh

# If you want to launch jupyter manually then feel free to comment this out.
CMD cd workspace && \
	bash launch_jupyter.sh
