FROM pytorchlightning/pytorch_lightning:base-conda-py3.8-torch1.7

RUN conda install python numpy pandas matplotlib scikit-learn tqdm jupyter 
RUN conda install -yc pytorch pytorch torchvision cudatoolkit=10.1
RUN conda install -yc conda-forge pytorch-lightning 
RUN conda install -yc conda-forge black flake8 isort loguru

COPY ./setup/requirements.txt /etc/requirements.txt
RUN pip install --user --no-cache-dir -r /etc/requirements.txt

COPY ./dockerfiles/.bashrc /root/.bashrc 

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user

ENTRYPOINT /bin/bash
WORKDIR /home/wilds-project