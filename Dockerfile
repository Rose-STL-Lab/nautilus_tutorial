FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

##############################################################################
# Temporary Installation Directory
##############################################################################
ENV STAGE_DIR=/tmp
RUN mkdir -p ${STAGE_DIR}
