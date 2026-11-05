# PPE detector inference image (YOLO26s fine-tuned on SH17, TensorRT FP16 built on first start).
# NOTE: written for an x86-64 NVIDIA GPU host with the NVIDIA Container Toolkit. It has not been
# built or run on the author's PC, which has no Docker. For Jetson, start from
# ultralytics/ultralytics:8.4.202-jetson-jetpack6 instead (also untested).
# The weights are not in the repository: run `python run_all.py` first, which writes models/.
#
#   docker build -f docker/Dockerfile -t ppe-detector:0.1 .
#   docker run --rm --gpus all -v ./samples:/data/in -v ./out:/data/out -v ppe-trt:/app/cache \
#       ppe-detector:0.1 --source /data/in --out /data/out
FROM ultralytics/ultralytics:8.4.202

ENV YOLO_AUTOINSTALL=False \
    PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir tensorrt-cu12==10.16.1.11 --extra-index-url https://pypi.nvidia.com

WORKDIR /app
COPY models/ppe_yolo26s_sh17_group.pt /app/models/
COPY results/group/operating_points.csv /app/operating_points.csv
COPY docker/serve.py /app/serve.py

VOLUME ["/app/cache"]
ENTRYPOINT ["python", "/app/serve.py"]
CMD ["--source", "/data/in", "--out", "/data/out"]
