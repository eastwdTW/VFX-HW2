# VFX HW2 Image Stitching

## Team 23
* R13943001 賴宗佑

## Dependencies
* Python 3.12.3
    * Numpy==2.2.4
    * opencv-python==4.11.0.86
    * tqdm==4.67.1

* To set up the environment:
```
python -m venv HW2
source HW2/bin/activate
pip install numpy==2.2.4
pip install opencv-python==4.11.0.86
pip install tqdm==4.67.1
```

## Run Code
* Run with default setting
    * Dataset-Sample0
```
cd code
python main.py
```

* Change dataset
```
python main.py -s {0,1}
```

* For more advanced usage
```
python main.py -h
```