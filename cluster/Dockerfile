FROM dtch1997/ml-ubuntu-cuda:ubuntu-22.04-python3.10-latest

# Install MujocoPy dependencies
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -q -y --no-install-recommends \
    cuda-nvcc-11-8 \
    curl \
    git \
    ffmpeg \
    libegl1 \
    libgl1 \
    libgl1-mesa-glx \
    libgles2 \
    libglew2.2 \
    libglfw3 \
    libglvnd0 \
    libglx0 \
    libosmesa6-dev \
    # Extra deps for downstream
    # unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MujocoPy
ENV MUJOCO_DIR=/opt/mujoco
ENV MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210
RUN mkdir -p ${MUJOCO_DIR} && \ 
    wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz && \ 
    tar -C ${MUJOCO_DIR} -xvzf mujoco210-linux-x86_64.tar.gz
ENV LD_LIBRARY_PATH ${MUJOCO_PY_MUJOCO_PATH}/bin:${LD_LIBRARY_PATH}

# Install requirements
COPY ./requirements /tmp/requirements
RUN python3 -m pip --no-cache-dir install -U pip wheel setuptools \
    && python3 -m pip --no-cache-dir install -r /tmp/requirements/d4rl-slim.txt \
    && rm -r /tmp/requirements

WORKDIR /workdir
USER $USERNAME
