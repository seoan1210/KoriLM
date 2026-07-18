# KoriLM 🇰🇷

> A small Korean Language Model built from scratch with PyTorch.

KoriLM은 PyTorch를 기반으로 직접 구현한 초소형 한국어 언어 모델 프로젝트입니다.

대규모 상용 LLM을 사용하는 것이 아니라,
Tokenizer부터 Transformer 구조, 학습 시스템, 체크포인트 관리까지 직접 구현하며
한국어 LLM이 만들어지는 과정을 연구하는 프로젝트입니다.

---

## 🚀 Features

* ✅ PyTorch 기반 Transformer 구현
* ✅ SentencePiece 한국어 Tokenizer 적용
* ✅ 한국어 Corpus Pretraining
* ✅ Checkpoint 저장 및 이어 학습 지원
* ✅ Text Generation
* 🚧 대화형 Fine-tuning
* 🚧 Instruction Tuning

---

# 🧠 Model Architecture

KoriLM은 GPT 계열 Decoder-only Transformer 구조를 사용합니다.

```
Korean Text Dataset

        ↓

SentencePiece Tokenizer

        ↓

Token IDs

        ↓

Transformer Decoder

        ↓

Language Model Head

        ↓

Generated Text
```

---

# ⚙️ Current Model Configuration

| Parameter           |  Value |
| ------------------- | -----: |
| Parameters          |   ~41M |
| Layers              |      8 |
| Attention Heads     |      8 |
| Embedding Dimension |    512 |
| Context Length      |    256 |
| Vocabulary Size     | 16,000 |

---

# 📚 Training

## Dataset

현재 모델은 한국어 Wikipedia 기반 Corpus를 사용하여 사전학습됩니다.

예:

```
wiki/
 ├ wiki_000
 ├ wiki_001
 ├ wiki_002
 └ ...
```

데이터를 전처리한 뒤 모델 학습에 사용합니다.

---

## Train

```
python KoriLM.py train
```

학습 중 일정 step마다 checkpoint가 저장됩니다.

```
checkpoint.pt
```

학습 완료 후:

```
KoriLM.pt
```

파일이 생성됩니다.

---

# 💬 Generation

학습된 모델 테스트:

```
python KoriChat.py
```

예시:

```
너 : 카폰 마그로

KoriLM :
카폰 마그로는 이탈리아의 전통 음식으로 ...
```

---

# 📂 Project Structure

```
KoriLM/

├── KoriLM.py          # Model training
├── KoriChat.py        # Text generation
├── prepare_data.py    # Dataset preprocessing

├── data.txt           # Training corpus

├── kori_tokenizer.model

├── checkpoint.pt

└── KoriLM.pt          # Trained model
```

---

# 🛠 Development Roadmap

## Completed

* [x] Transformer architecture implementation
* [x] SentencePiece tokenizer
* [x] Korean dataset preprocessing
* [x] Language model training pipeline
* [x] Checkpoint save / resume

## Planned

* [ ] Korean conversational fine-tuning
* [ ] Instruction tuning
* [ ] Coding dataset training
* [ ] Larger parameter experiments
* [ ] Improved Korean language understanding

---

# 🎯 Goal

KoriLM의 목표는 대형 상용 LLM을 따라잡는 것이 아닙니다.

이 프로젝트의 목적은:

* 한국어 LLM이 어떻게 만들어지는지 이해하기
* 작은 모델이 언어를 학습하는 과정 연구하기
* 직접 만든 모델을 발전시키는 경험 쌓기

입니다.

---

# License

MIT License
