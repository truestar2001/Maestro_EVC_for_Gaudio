# Maestro-EVC

## Setting
```bash
git clone https://github.com/truestar2001/Maestro_EVC_for_Gaudio
cd Maestro_EVC_for_Gaudio
conda create -n mevc python=3.11
conda activate mevc
pip install -r requirements.txt
mkdir -p checkpoint data inference/output
```

https://drive.google.com/file/d/1CVveXsWfEruRH5ALrVx9g7qf_EiOTnfJ/view?usp=drive_link 

- 위 링크에서 checkpoint 다운 후 checkpoint 폴더 안에 넣기

- 사용할 wav파일들 모두 data폴더 안에 넣기 (폴더 안에 들어있어도 무관)

## Extract features
```bash
python extractors/f0.py
python extractors/energy.py
python extractors/hubert.py
python extractors/emotion_diarization_embedding.py
python extractors/speaker_embedding.py
```
## Make inference pairs

- inference 폴더 안에 gaudio_pairs.txt 만들기

- gaudio_pairs.txt는 다음과 같이 구성되어야 함.

```bash
<content wav url>|<speaker wav url>|<emotion wav url>
```

- speaker는 학습 때 매우 적은 화자 수(10명)으로 학습했기 때문에 generalization 안됨. git에 있는 data/EVC Speaker/0020_000006.wav 를 speaker wav로 고정하고 test.

ex)
```bash
/home/Maestro_EVC/data/Deepdub Voice Clone/경도_001_00-00-00_000_00-00-11_564_EN.wav|/home/Maestro_EVC/data/EVC Speaker/0020_000006.wav|/home/Maestro_EVC/data/Original/경도_001_00-00-00_000_00-00-11_564_이쯤되니.wav
```

## Inference
```bash
python inference.py
```
