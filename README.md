[Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/abs/1911.08947)

### Installation

```
conda create -n PyTorch python=3.8
conda activate PyTorch
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
pip install opencv-python==4.5.5.64
pip install pyyaml
pip install timm
pip install tqdm
pip install shapely
pip install pyclipper
```

### Train

* Configure your dataset path in `main.py` for training
* Run `bash main.sh $ --train` for training, `$` is number of GPUs

### Test

* Configure your path in `main.py` for testing
* Run `python main.py --test` for testing

### Demo

* Configure your path in `main.py` for demo
* Run `python main.py --demo` for demo

### Results

| Version | Epochs | Precision | Recall |   F1 |  Backbone | 
|:-------:|:------:|----------:|-------:|-----:|----------:|
|   DB    |  1200  |      83.1 |   79.3 | 81.2 | ResNet-18 | 
|   DB*   |  1200  |      82.9 |   75.9 | 79.2 | ResNet-18 | 

* `*` means that the model is trained from original repo, see reference

<img src="./demo/demo.jpg"  alt="">

#### Reference

* https://github.com/MhLiao/DB
