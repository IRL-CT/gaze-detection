FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV YOLO_CONFIG_DIR=/root/.config/Ultralytics

# 1. system deps
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ros-humble-cv-bridge \
    && rm -rf /var/lib/apt/lists/*

# 2. python deps
RUN pip3 install --no-cache-dir \
    flask \
    numpy \
    ultralytics

# 4. workspace
WORKDIR /ros2_ws
COPY . /ros2_ws

CMD ["bash", "-c", "source /opt/ros/humble/setup.bash && python3 -u pub.py"]