"""
Data preprocessing pipeline for time-aware music-language modeling.

Steps:
  1. preprocess_raw_dataset()  -- parse raw audio datasets into unified JSON format
  2. extract_feature()         -- extract MERT features, saved as .h5 files
  3. split_data()              -- split into train/test JSON files
  4. generate_training_data()  -- build XML-structured training examples
  5. add_natural_language()    -- (optional) augment with free-form descriptions
"""

import os
import numpy as np


# ---------------------------------------------------------------------------
# Step 1: Dataset-specific preprocessors
# ---------------------------------------------------------------------------

def aam():
    from data.preprocess.aam import process
    root_folder = "dataset/download/AAM"
    output_folder = "dataset/raw_dataset/AAM"
    os.makedirs(output_folder, exist_ok=True)
    process(root_folder, output_folder)


def giantstep_key():
    from data.preprocess.giantstep_key import process
    root_folder = "dataset/download/giant_steps"
    output_folder = "dataset/raw_dataset/giant_steps_key"
    os.makedirs(output_folder, exist_ok=True)
    process(root_folder, output_folder)


def giantstep_tempo():
    from data.preprocess.giantstep_tempo import process
    root_folder = "dataset/download/giant_steps"
    output_folder = "dataset/raw_dataset/giant_steps_tempo"
    os.makedirs(output_folder, exist_ok=True)
    process(root_folder, output_folder)


def mlpmf():
    from data.preprocess.mlpmf import process
    root_folder = "dataset/download/MLPMF"
    output_folder = "dataset/raw_dataset/MLPMF"
    os.makedirs(output_folder, exist_ok=True)
    process(root_folder, output_folder)


def irmas():
    from data.preprocess.irmas import process
    root_folder = "dataset/download/IRMAS"
    output_folder = "dataset/raw_dataset/IRMAS"
    os.makedirs(output_folder, exist_ok=True)
    process(root_folder, output_folder)


def fsld():
    from data.preprocess.fsld import process
    root_folder = "dataset/download/FSLD"
    output_folder = "dataset/raw_dataset/FSLD"
    os.makedirs(output_folder, exist_ok=True)
    process(root_folder, output_folder)


def ballroom():
    from data.preprocess.ballroom import process
    root_folder = "dataset/download/Ballroom"
    output_folder = "dataset/raw_dataset/Ballroom"
    os.makedirs(output_folder, exist_ok=True)
    process(root_folder, output_folder)


def gtzan():
    from data.preprocess.gtzan import process
    root_folder = "dataset/download/GTZAN"
    output_folder = "dataset/raw_dataset/GTZAN"
    os.makedirs(output_folder, exist_ok=True)
    process(root_folder, output_folder)


def guitarset():
    from data.preprocess.guitarset import process
    root_folder = "dataset/download/guitarset"
    output_folder = "dataset/raw_dataset/guitarset"
    os.makedirs(output_folder, exist_ok=True)
    process(root_folder, output_folder)


def preprocess_raw_dataset():
    """Run dataset preprocessors. Uncomment datasets as needed."""
    # mlpmf()
    # irmas()
    # fsld()
    # ballroom()
    # gtzan()
    # guitarset()
    # giantstep_key()
    # giantstep_tempo()
    aam()


# ---------------------------------------------------------------------------
# Step 2: MERT feature extraction
# ---------------------------------------------------------------------------

def extract_feature():
    from data.feature_solver import save_feature
    file_path = "dataset/raw_dataset"
    output_folder = "dataset/new_dataset/encodec_feature"
    save_feature(file_path, output_folder, selected_dataset='AAM')

    # For additional datasets:
    # save_feature(file_path, output_folder, suffix='mtg0', selected_dataset='MTG', re_generate='0')
    # save_feature(file_path, output_folder, suffix='', selected_dataset='MLPMF')


# ---------------------------------------------------------------------------
# Step 3: Train/test split
# ---------------------------------------------------------------------------

def split_data(suffix="", selected_datasets=None):
    from data.create_dataset import split_dataset
    root_folder = "dataset/raw_dataset"
    output_folder = "dataset/new_dataset/splited_dataset"
    split_dataset(root_folder=root_folder, output_folder=output_folder,
                  suffix=suffix, selected_datasets=selected_datasets)


# ---------------------------------------------------------------------------
# Step 4: Build XML-structured training examples
# ---------------------------------------------------------------------------

def generate_training_data(suffix="", selected_keys=None):
    from data.create_dataset import create_caption
    root_folder = "dataset/new_dataset/splited_dataset"
    output_folder = "dataset/new_dataset/formatted_dataset"
    rng = np.random.RandomState(1234)
    create_caption(with_comparison=False,
                   root_folder=root_folder, output_folder=output_folder,
                   split=f'train{suffix}', rng=rng, fps=75,
                   selected_keys=selected_keys, rearrange=True, grounding_param=-1)


# ---------------------------------------------------------------------------
# Step 5 (optional): Add free-form natural language descriptions
# ---------------------------------------------------------------------------

def add_natural_language(suffix=""):
    from data.create_dataset import add_natural_language as _add_nl
    root_folder = "dataset/new_dataset/formatted_dataset"
    output_folder = "dataset/new_dataset/formatted_dataset"
    _add_nl(root_folder=root_folder, output_folder=output_folder, suffix=suffix)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    suffix = "_1023"
    selected_datasets = ['AAM']
    # selected_datasets = ['AAM', 'FSLD', 'MLPMF', 'IRMAS', 'Ballroom']
    # selected_datasets += ['MTG_mtg0', ..., 'MTG_mtg9']
    # selected_datasets += ['FMA_fma0', 'FMA_fma1', 'FMA_fma2']

    selected_keys = ['tempo', 'key', 'instruments', 'chord']

    preprocess_raw_dataset()
    extract_feature()
    split_data(suffix, selected_datasets)
    generate_training_data(suffix, selected_keys)
    # add_natural_language(suffix)
