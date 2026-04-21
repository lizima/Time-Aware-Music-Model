import librosa
from madmom.features.beats import RNNBeatProcessor
from madmom.features.chords import (
    CNNChordFeatureProcessor,
    CRFChordRecognitionProcessor,
)
from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor
from madmom.features.key import CNNKeyRecognitionProcessor, key_prediction_to_label
from madmom.features.tempo import TempoEstimationProcessor
from madmom.processors import SequentialProcessor
import os
import json
import time

# audio_dir = '/datapool/data2/home/ruihan/storage/debug/all_m4m/m4m_dataset/dataset/download/MLPMF/audio'
# audio_dir = '/datapool/data2/home/ruihan/storage/debug/all_m4m/revising/m4m_dataset/dataset/download/IRMAS/IRMAS-TestingData-Part1/Part1'
# audio_dir = '/datapool/data2/home/ruihan/storage/debug/all_m4m/revising/m4m_dataset/dataset/download/IRMAS/IRMAS-TestingData-Part2/IRTestingData-Part2'
audio_dir = '/datapool/data2/home/ruihan/storage/debug/all_m4m/revising/m4m_dataset/dataset/download/Ballroom/BallroomData'
dataset_name = audio_dir.split('/download/')[-1].split('/')[0]

# tempo
fps = 100
beat_proc = RNNBeatProcessor()
tempo_proc = TempoEstimationProcessor(fps=fps)

# key
key_proc = CNNKeyRecognitionProcessor()

# chord
featproc = CNNChordFeatureProcessor()
decode = CRFChordRecognitionProcessor(fps=10)
chordrec = SequentialProcessor([featproc, decode])

# downbeat
beats_per_bar = [3, 4]
fps = 100
downbeat_decode = DBNDownBeatTrackingProcessor(
    beats_per_bar=beats_per_bar, fps=fps
)
downbeat_process = RNNDownBeatProcessor()
downbeat_rec = SequentialProcessor([downbeat_process, downbeat_decode])

audios = os.listdir(audio_dir)

# audios = audios[1900:2400]
idx = 0

all_augment_data = []
cnt = 0
start = time.time()

for audio_subdir in audios:
    # continue when audio_subdir is not a directory
    subdir = os.path.join(audio_dir, audio_subdir)
    if not os.path.isdir(subdir):
        continue

    audio_files = os.listdir(subdir)
    for audio in audio_files:
        if not audio.endswith('.wav'):
            continue
        # print(audio)
        cnt += 1
        if cnt % 100 == 0:
            print('#############')
            print(cnt)
            print(time.time() - start)

        path = os.path.join(subdir, audio)
        print(path)
        # continue
        augment_data = {}
        augment_data['filename'] ='/dataset/download/' + path.split('/dataset/download/')[-1]

        try:
            # tempo
            beat_acts = beat_proc(path)
            tempo_acts = tempo_proc(beat_acts)
            tempo_est = round(tempo_acts[0][0], 1)
            augment_data['tempo mean'] = tempo_est

            # key
            key_acts = key_proc(path)
            key_est = key_prediction_to_label(key_acts)
            key_est = key_est.replace(" ", "")
            augment_data['key mode'] = key_est

            # chord
            chord_est = chordrec(path)

            chord_est = [
                [
                    round(x[0], 2),
                    x[2].replace(":maj", "maj").replace(":min", "min")
                    if x[2] != "N"
                    else "no chord",
                ]
                for x in chord_est.tolist()
            ]
            # chord_est = [[x[0], x[1]] for x in chord_est if x[1] != 'no chord']
            augment_data['chord progression'] = chord_est

            # beat
            downbeats_est = downbeat_rec(path)
            # downbeats_est = [{"time": x[0], "beat_number": int(x[1])} for x in downbeats_est.tolist()]
            downbeats_est = [[str(x[0]), str(int(x[1]))] for x in downbeats_est.tolist()]
            augment_data['beats'] = downbeats_est

            all_augment_data.append(augment_data)
            with open(f'augment_data_{dataset_name}{idx}.json', 'w') as f:
                json.dump(all_augment_data, f, indent=2)
        except:
            print('error')
            continue

