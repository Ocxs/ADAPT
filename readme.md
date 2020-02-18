
Our implementations for ADAPT, which has two key components: DUAL and ADEN.

Since the limited space in GitHub, we upload the demo data in [google drive](https://drive.google.com/drive/folders/1lg1CjDV6UCyv5ktXEHSXPySufJC1N70R?usp=sharing) and [BaiduYun]( https://pan.baidu.com/s/17wHRx8Ywi-BSgv9CIzn0aw) (password: sad8) for running our code. Download the demo data, and put them into `ADEN/Files/`. 

## Environment
python 2.7, TensorFlow 1.12

## Usage
### DUAL
DUAL is proposed to pre-train feature extractors of visual and textual features. The commands are as follows:
```
cd DUAL;
python main_dual.py
```

### ADEN

ADEN is proposed for storytelling recommendation by leveraging the cross-domain sequential behavior. The commands are as follows:
```
cd ADEN;
python main_aden_ft.py; 
```
Note that `python main_aden_ft.py` uses the feature extractors pre-trained by DUAL model, but `python main_aden.py` is without using the pre-trained feature extractors. Our pre-trained models of feature extractors are in `ADEN/Files/output/DUAL`.