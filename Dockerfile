ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:20.12-py3
FROM $BASE_IMAGE

RUN pip install colour==0.1.5
RUN pip install efficientnet-pytorch==0.7.1
RUN pip install grad-cam==1.3.7
RUN pip install matplotlib==3.3.4
RUN pip install numpy
RUN pip install opencv-python==4.5.5.62
RUN pip install pandas==1.1.5
RUN pip install scikit-learn==0.24.2
RUN pip install scipy==1.5.4
RUN pip install seaborn==0.11.2
RUN pip install termcolor
RUN pip install tqdm==4.62.3
RUN pip3 install torch==1.8.2 torchvision==0.9.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
