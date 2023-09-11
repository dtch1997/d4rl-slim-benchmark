FROM dtch1997/ml-ubuntu-cuda:ubuntu-22.04-python3.10-latest

COPY ./requirements /tmp/requirements
RUN python3 -m pip --no-cache-dir install -U pip wheel setuptools \
    && python3 -m pip --no-cache-dir install -r /tmp/requirements/base.txt \
    && rm -r /tmp/requirements

WORKDIR /workdir
USER $USERNAME