FROM ubuntu:22.04

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

RUN pip install scarf
