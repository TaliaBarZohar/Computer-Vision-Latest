# ADA-VAD - Setup & Training Instructions

## 🧩 Environment Setup
1. Created conda environment: `VAD`
2. Installed packages including:
   - `torch==2.2.2`, `torchvision==0.17.2`, `torchaudio==2.2.2`
   - `opencv-python==4.7.0.72`, `pybind11==2.9.2`, `ninja==1.10.2`
   - CUDA extensions compiled: `resample2d`, `correlation`, `channelnorm`

## 🔧 Custom CUDA Extension Compilation
```bash
cd pre_process/flownet_networks/resample2d_package
python setup.py build_ext --inplace

cd ../correlation_package
python setup.py build_ext --inplace

cd ../channelnorm_package
python setup.py build_ext --inplace
```

## 🎞️ Frame Extraction from Video
```bash
mkdir -p data/avenue/training/frames/01
ffmpeg -i data/avenue/training_videos/01.avi data/avenue/training/frames/01/%04d.jpg
```

## 📦 Bounding Boxes Extraction
```bash
PYTHONPATH=/mnt/d/ADA-VAD:/mnt/d/ADA-VAD/pre_process/flownet_networks/resample2d_package:/mnt/d/ADA-VAD/pre_process/flownet_networks/correlation_package:/mnt/d/ADA-VAD/pre_process/flownet_networks/channelnorm_package \ 
python pre_process/extract_bboxes.py --proj_root=/mnt/d/ADA-VAD --dataset_name=avenue --mode=train
```

## 🧪 Training Sample Extraction
```bash
PYTHONPATH=. python pre_process/extract_samples.py --proj_root=/mnt/d/ADA-VAD --dataset_name=avenue --mode=train
```

## 🚀 Model Training
```bash
python unetonlyInfomaxAnypredictStage1.py -f cfgs/unetonlyAnypredictStage1/unetOnly_Stage1_ave_2.yaml
```