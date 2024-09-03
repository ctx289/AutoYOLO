from ultralytics.utils.benchmarks import benchmark

# Benchmark on GPU
model = '/youtu/xlab-team4/ryanwfu/training/ultralytics/yolov8n_default/train/weights/best.pt'
data = '/youtu/xlab-team4/ryanwfu/28_ultralytics/ultralytics-plus/ultralyticsp/cfg/datasets/yz_bottlecap1_online_crop.yaml'
benchmark(model=model, data=data, imgsz=800, half=False, device=0)
