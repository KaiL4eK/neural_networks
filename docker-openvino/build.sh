#!/bin/bash

wget -N http://registrationcenter-download.intel.com/akdlm/irc_nas/15792/l_openvino_toolkit_p_2019.2.275.tgz
docker build -t docker-openvino --build-arg UID=`id -u` .
