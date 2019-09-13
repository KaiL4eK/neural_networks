FROM ubuntu:16.04
ENV http_proxy $HTTP_PROXY
ENV https_proxy $HTTP_PROXY
ARG DOWNLOAD_LINK=http://registrationcenter-download.intel.com/akdlm/irc_nas/15792/l_openvino_toolkit_p_2019.2.275.tgz
ARG UID=1000
ARG INSTALL_DIR=/opt/intel/openvino
ARG TEMP_DIR=/tmp/openvino_installer
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    cpio \
    sudo \
    tree \
    autoconf \
    automake \
    libtool \
    unzip \
    python3-pip \
    lsb-release && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p $TEMP_DIR && cd $TEMP_DIR
COPY l_openvino_toolkit_p_2019.2.275.tgz .
RUN tar xf l_openvino_toolkit*.tgz && \
    cd l_openvino_toolkit* && \
    sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh -s silent.cfg && \
    rm -rf $TEMP_DIR
# RUN tree $INSTALL_DIR -L 2
# RUN $INSTALL_DIR/install_dependencies/install_cv_sdk_dependencies.sh
RUN $INSTALL_DIR/install_dependencies/install_openvino_dependencies.sh
# build Inference Engine samples
RUN mkdir $INSTALL_DIR/deployment_tools/inference_engine/samples/build && cd $INSTALL_DIR/deployment_tools/inference_engine/samples/build && \
    /bin/bash -c "source $INSTALL_DIR/bin/setupvars.sh && cmake .. && make -j`nproc --all`"

# For NCS
RUN cd /tmp/ && \
   wget https://github.com/libusb/libusb/archive/v1.0.22.zip && \
   unzip v1.0.22.zip && cd libusb-1.0.22 && \
   ./bootstrap.sh && \
   ./configure --disable-udev --enable-shared && \
   make -j`nproc --all` && make install && \
   rm -rf /tmp/*

RUN mkdir -p /home/developer && \
    echo "developer:x:$UID:$UID:Developer,,,:/home/developer:/bin/bash" >> /etc/passwd && \
    echo "developer:x:$UID:" >> /etc/group && \
    echo "developer ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/developer && \
    chmod 0440 /etc/sudoers.d/developer && \
    chown developer:developer -R /home/developer && \
    chown root:root /usr/bin/sudo && chmod 4755 /usr/bin/sudo

RUN apt install -y python3-setuptools python3
RUN pip3 install --no-cache-dir requests numpy networkx
RUN pip3 install --no-cache-dir tensorflow defusedxml test-generator==0.1.1

USER developer
ENV HOME /home/developer
ENV MO_PATH /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py
WORKDIR /home/developer/ws

RUN echo "source /opt/intel/openvino/bin/setupvars.sh" >> /home/developer/.bashrc
