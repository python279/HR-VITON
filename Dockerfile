FROM python:3.8-buster

RUN echo "deb https://mirrors.aliyun.com/debian/ buster main contrib non-free" > /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/debian/ buster-updates main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/debian/ buster-backports main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/debian-security/ buster/updates main contrib non-free" >> /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends zip unzip libgl1-mesa-glx \
    && apt-get clean

RUN pip3 install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip3 install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

WORKDIR /HR-VITON
COPY . /HR-VITON

RUN pip3 install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN mkdir -p eval_models/weights/v0.1 \
    && curl -o eval_models/weights/v0.1/mtviton.pth "http://mirrors.uat.enflame.cc/enflame.cn/maas/HR-VITON/eval_models/weights/v0.1/mtviton.pth" \
    && curl -o eval_models/weights/v0.1/gen.pth "http://mirrors.uat.enflame.cc/enflame.cn/maas/HR-VITON/eval_models/weights/v0.1/gen.pth"

ENTRYPOINT ["python3", "server.py"]