Toward Grounded and Time-Aware Music Language Models
=====================================================

Code accompanying project Winter 2026 COMP 767 report.

Overview
--------
A structured supervision framework for grounded, time-aware music-language
modeling. Audio is aligned with timestamped musical attributes (tempo, key,
chord, instruments), free-form descriptions, and grounding QA pairs in a
unified XML-structured training format. The model is LLaMA3-8B + LoRA with a
frozen MERT encoder and a trainable linear projector (768 -> 4096).

Directory Structure
-------------------
  preprocess.py             End-to-end data preprocessing pipeline (Steps 1-5)
  train.py                  Model training (LLaMA3-8B + LoRA/QLoRA)
  inference.py              Run model on test data
  evaluate.py               MIR evaluation (tempo, key, genre, instrument, chord)

  data/
    preprocess/             Dataset-specific parsers (AAM, GiantSteps, MLPMF, ...)
    feature_solver.py       MERT feature extraction -> .h5 files
    create_dataset.py       Converts raw annotations into XML training examples
    data_generator.py       Caption and segment generation utilities
    madmom_extract.py       Madmom-based tempo/key/chord augmentation
    add_natural_language.py Free-form descriptions via LLaMA-3-8B-Instruct
    build_grounding.py      Grounding QA pairs from timestamp-aligned attributes

  model/
    music_encoder.py        MusicEncoder: linear projector (768 -> 4096)
    music_llama.py          MusicLlamaForCausalLM: injects MERT features at <|x|>
    dataset.py              MusicDataset: audio features + structured captions

Model
-----
  Backbone:     LLaMA3-8B (NousResearch/Meta-Llama-3-8B)
  Adaptation:   LoRA r=16, alpha=32, dropout=0.05; 4-bit NF4 (QLoRA)
  Audio:        MERT encoder (frozen, 768-dim) + nn.Linear(768, 4096) (trainable)
  Training:     6 epochs, batch=16, grad_accum=2, lr=2e-4, FP16
  Special token: <|x|> (ID 128256) — replaced by projected MERT embeddings

Evaluation
----------
  Tempo:      GiantSteps Tempo, Acc2 (4% tolerance incl. 2x/0.5x)
  Key:        GiantSteps Key, MIREX score
  Genre:      MedleyDB, exact match accuracy
  Instrument: MedleyDB, F1 (multi-label)
  Chord:      AAM test split, frame-wise recall (100ms tolerance)
  Grounding:  mIoU for attribute localization; tempo/key interval prediction

Dependencies
------------
  transformers, peft, trl, torch, h5py, numpy, mir_eval, madmom, librosa

Usage
-----

1. Preprocess:
     python preprocess.py
   Optional augmentation / grounding:
     python data/madmom_extract.py
     python data/build_grounding.py

2. Train:
     python train.py

3. Inference:
     python inference.py

4. Evaluate:
     python evaluate.py
