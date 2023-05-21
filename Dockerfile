FROM python:3.9-slim-buster

# Disable downloads caching.
ARG PIP_DISABLE_PIP_VERSION_CHECK=1
ARG PIP_NO_CACHE_DIR=1

# Install base dependencies.
RUN apt-get update
RUN apt-get install -y g++ git
RUN pip install torch==1.10.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install torchvision==0.11.1 --index-url https://download.pytorch.org/whl/cpu

# Install model packages.
RUN pip install openmim==0.1.5
RUN pip install mmcv-full==1.3.16 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.10.0/index.html
RUN pip install mmdet==2.18.0
RUN pip install mmpose==0.20.0
RUN pip install anime-face-detector>=0.09 --no-dependencies

# Force headless version.
RUN yes | pip uninstall opencv-python
RUN pip install opencv-python-headless==4.7.0.72

# Load model in advance.
RUN python3 -c "from anime_face_detector import create_detector; create_detector('yolov3', device='cpu')"

# Web server dependencies.
RUN pip install fastapi "uvicorn[standard]"
RUN pip install python-multipart

ADD ./main.py ./

CMD exec uvicorn --host 0.0.0.0 --port $PORT main:app
