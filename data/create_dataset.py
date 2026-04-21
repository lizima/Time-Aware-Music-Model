import json
import os
import numpy as np
from .utils import format_props, crop_props, format_duration, divide_into_more_segs, crop_props_revise, get_pairs, get_comparison
import re
# from .utils import format_props_revise, format_props2_revise


def filter_data(data, key_words=None):
    new_data = []
    for d in data:
        flag = False
        for x in d["segments"]:
            if key_words is None:
                flag = True
                break
            else:
                for key in key_words:
                    if key in x and x[key] not in ["none", "", "None"]:
                        flag = True
                        break

        if flag:
            new_data.append(d)
    return new_data


def split_dataset(root_folder, output_folder, suffix = "", selected_datasets=None):
    splits = {}
    for dataset in os.listdir(root_folder):
        if dataset not in selected_datasets:
            continue
        dataset_folder = os.path.join(root_folder, dataset)
        metadata = os.path.join(dataset_folder, "metadata.json")
        with open(metadata, "r") as f:
            data = json.load(f)
        for d in data:
            d["dataset"] = dataset

        splits[dataset] = data

    ratio = 0.9
    train = []
    test = []
    for dataset in splits:
        data = filter_data(splits[dataset])
        np.random.shuffle(data)
        data_len = len(data)

        training_num = int(data_len * ratio)
        if training_num > 0:
            train += data[:training_num]
            if training_num < data_len:
                test += data[training_num:]

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)

    with open(os.path.join(output_folder, f"train{suffix}.json"), "w") as f:
        json.dump(train, f, indent=2)

    with open(os.path.join(output_folder, f"test{suffix}.json"), "w") as f:
        json.dump(test, f, indent=2)


def segs2caption(segs, onset, feature_token, rng, drop_out, eot, eos, max_n_tokens, fps):

    props = ""
    offsets = []
    tempo_dt = rng.randint(0, 3) - 2 if drop_out else 0
    tempo_dt = tempo_dt * 1. / 100


    tag_idx = 0
    marks = []
    keys = [k for k in segs[0] if k not in ["mark", "onset", "offset", "timestamp"]]
    rng.shuffle(keys)

    if len(segs) == 1 and segs[0]["mark"] == "M":
        segs = divide_into_more_segs(segs[0], rng)

    if len(segs) > 1 and drop_out and rng.rand() > .8:
        drop_idx = rng.randint(0, len(segs))
    else:
        drop_idx = -1

    for i, seg in enumerate(segs):
        if float(seg["offset"]) - float(seg["onset"]) < 1:
            drop_idx = -1
            continue
        if drop_idx == i:
            continue
        offset = float(seg["offset"])

        offsets.append(float(seg["offset"]))
        timestamp = format_duration(float(seg["onset"]) - onset) + "-" + format_duration(offset - onset)
        audio_tag = chr(tag_idx + ord('A'))
        marks.append(f"{audio_tag}({timestamp})")
        tag = f"<timestamp>{timestamp}</timestamp>"
        tag += "".join(
            [f"<{k}>{crop_props(k, seg[k], float(seg['onset']), offset, onset, aug=drop_out, tempo_dt=tempo_dt)}</{k}>"
             for k in keys if offset - float(seg['onset']) > 0.5])
        out_keys = ["timestamp"] + keys
        if props == "":
            tag = eot + tag
        props += f"<{audio_tag} {' '.join(out_keys)}>{tag}</{audio_tag}>"
        tag_idx += 1
    desc = f"<music {' '.join(marks)}>{props}</music>"
    if len(offsets) == 0:
        return None
    max_offset = max(offsets)

    dur = format_duration(max_offset - onset)
    n_tokens_st = int(onset * fps)
    n_tokens_ed = min(max_n_tokens, int(max_offset * fps)) if max_n_tokens is not None else int(max_offset * fps)
    feature = "".join([feature_token] * (n_tokens_ed - n_tokens_st))
    head = f"<audio duration feature><duration>{dur}</duration><feature>{feature}</feature></audio>"
    return head + desc + eos, n_tokens_st, n_tokens_ed, onset, max_offset

def segs2caption_revise(segs, onset, feature_token, rng, drop_out, eot, eos, max_n_tokens, fps, with_comparison=False):

    props = ""
    offsets = []
    tempo_dt = rng.randint(0, 3) - 2 if drop_out else 0
    tempo_dt = tempo_dt * 1. / 100


    tag_idx = 0
    marks = []
    comp_dic = {}
    keys = [k for k in segs[0] if k not in ["mark", "onset", "offset", "timestamp"]]
    for key in keys:
        comp_dic[key] = {}
    rng.shuffle(keys)

    if len(segs) == 1 and segs[0]["mark"] == "M":
        segs = divide_into_more_segs(segs[0], rng)

    if len(segs) > 1 and drop_out and rng.rand() > .8:
        drop_idx = rng.randint(0, len(segs))
    else:
        drop_idx = -1

    for i, seg in enumerate(segs):
        if float(seg["offset"]) - float(seg["onset"]) < 1:
            drop_idx = -1
            continue
        if drop_idx == i:
            continue
        offset = float(seg["offset"])

        offsets.append(float(seg["offset"]))
        timestamp = format_duration(float(seg["onset"]) - onset) + "-" + format_duration(offset - onset)
        audio_tag = chr(tag_idx + ord('A'))
        marks.append(f"{audio_tag}({timestamp})")
        for key in keys:
            comp_dic[key][tag_idx] = seg[key]
        tag = f"<timestamp>{timestamp}</timestamp>"
        tag += "".join(
            [f"<{k}>{crop_props_revise(k, seg[k], float(seg['onset']), offset, onset, aug=drop_out, tempo_dt=tempo_dt)}</{k}>"
             for k in keys if offset - float(seg['onset']) > 0.5])

        out_keys = ["timestamp"] + keys
        if props == "":
            tag = eot + tag
        props += f"<{audio_tag} {' '.join(out_keys)}>{tag}</{audio_tag}>"
        tag_idx += 1

    desc = f"<music {' '.join(marks)}>{props}</music>"

    comp = ""
    if tag_idx > 1:
        pairs = get_pairs(tag_idx, rng)
        if len(pairs) > 0:
            comp_1 = get_comparison(pairs, keys, comp_dic)
            comp = f"<comparison {' '.join(pairs)}>{comp_1}</comparison>"
            
    if len(offsets) == 0:
        return None
    max_offset = max(offsets)

    dur = format_duration(max_offset - onset)
    n_tokens_st = int(onset * fps)
    n_tokens_ed = min(max_n_tokens, int(max_offset * fps)) if max_n_tokens is not None else int(max_offset * fps)
    # print('Start, End, dur', n_tokens_st, n_tokens_ed, dur)
    feature = "".join([feature_token] * (n_tokens_ed - n_tokens_st))
    head = f"<audio duration feature><duration>{dur}</duration><feature>{feature}</feature></audio>"
    if with_comparison:
        if with_comparison == 'only':
            # return head + f"<music {' '.join(marks)}></music>" + eot + comp + eos, n_tokens_st, n_tokens_ed, onset, max_offset
            if comp != "":
                first_part = comp_1.split('>')[0] + '>'
                second_part = '>'.join(comp_1.split('>')[1:])
                tmp = first_part + eot + second_part
                return head + f"<music {' '.join(marks)}></music>" + f"<comparison {' '.join(pairs)}>{tmp}</comparison>" + eos, n_tokens_st, n_tokens_ed, onset, max_offset
            else:
                return head + desc + comp + eos, n_tokens_st, n_tokens_ed, onset, max_offset # reaturn anything, will drop afterwards
            
        elif with_comparison == 'only1':
            if comp != "":
                return head + f"<music {' '.join(marks)}></music>" + eot + comp + eos, n_tokens_st, n_tokens_ed, onset, max_offset
            else:
                return head + desc + comp + eos, n_tokens_st, n_tokens_ed, onset, max_offset # reaturn anything, will drop afterwards
        elif with_comparison == 'only2':
            return head + desc + comp + eos, n_tokens_st, n_tokens_ed, onset, max_offset
        elif with_comparison in ['only3', 'only4', 'only5', 'only6']:
            if comp == "":
                return head + desc + comp + eos, n_tokens_st, n_tokens_ed, onset, max_offset
            else:
                if with_comparison == 'only3':
                    mask_rate = 0.1
                elif with_comparison == 'only4':
                    mask_rate = 0.2
                elif with_comparison == 'only5':
                    mask_rate = 0.5
                elif with_comparison == 'only6':
                    mask_rate = 1.0
                
                pattern = re.compile(r"<key>(.*?)</key>|<tempo>(.*?)</tempo>")
                # matches = pattern.findall(desc)
                matches_with_indices = []
                for match in re.finditer(pattern, desc):
                    matched_string = match.group(1) if match.group(1) else match.group(2)
                    start_idx = match.start(1) if match.group(1) else match.start(2)
                    matches_with_indices.append((matched_string, start_idx))

                idxs_to_replace = []
                for match, idx in matches_with_indices:
                    idxs_to_replace.append((idx, idx + len(match)))

                mask_idx = 0
                for tup in idxs_to_replace:
                    if rng.rand() < mask_rate:
                        # a = desc[tup[0]:tup[1]]
                        b = f'mask{mask_idx}' + '@'*(tup[1]-tup[0]-len(f'mask{mask_idx}'))
                        desc = desc[:tup[0]] + b + desc[tup[1]:]
                        mask_idx += 1

                desc = desc.replace("@", "")

                return head + desc + comp + eos, n_tokens_st, n_tokens_ed, onset, max_offset
        else:
            return head + desc + comp + eos, n_tokens_st, n_tokens_ed, onset, max_offset
    else:
        # print('without comparison')
        return head + desc + eos, n_tokens_st, n_tokens_ed, onset, max_offset

def segs2caption_revise_before(segs, onset, feature_token, rng, drop_out, eot, eos, max_n_tokens, fps):

    props = ""
    offsets = []
    tempo_dt = rng.randint(0, 3) - 2 if drop_out else 0
    tempo_dt = tempo_dt * 1. / 100


    tag_idx = 0
    marks = []
    keys = [k for k in segs[0] if k not in ["mark", "onset", "offset", "timestamp"]]
    rng.shuffle(keys)

    if len(segs) == 1 and segs[0]["mark"] == "M":
        segs = divide_into_more_segs(segs[0], rng)

    if len(segs) > 1 and drop_out and rng.rand() > .8:
        drop_idx = rng.randint(0, len(segs))
    else:
        drop_idx = -1

    for i, seg in enumerate(segs):
        if float(seg["offset"]) - float(seg["onset"]) < 1:
            drop_idx = -1
            continue
        if drop_idx == i:
            continue
        offset = float(seg["offset"])

        offsets.append(float(seg["offset"]))
        # timestamp = format_duration(float(seg["onset"]) - onset) + "-" + format_duration(offset - onset)
        tmp1 = float(seg["onset"]) - onset
        tmp2 = offset - onset
        # timestamp = f'{tmp1:.1f}' + '-' + f'{tmp2:.1f}'
        timestamp = f'{tmp1:.1f}' + ', ' + f'{tmp2:.1f}'
    
        audio_tag = chr(tag_idx + ord('A'))
        marks.append(f"{audio_tag}({timestamp})")
        tag = f"<timestamp>{timestamp}</timestamp>"
        tag += "".join(
            [f"<{k}>{crop_props_revise(k, seg[k], float(seg['onset']), offset, onset, aug=drop_out, tempo_dt=tempo_dt)}</{k}>"
             for k in keys if offset - float(seg['onset']) > 0.5])

        out_keys = ["timestamp"] + keys
        if props == "":
            tag = eot + tag
        props += f"<{audio_tag} {' '.join(out_keys)}>{tag}</{audio_tag}>"
        tag_idx += 1
    desc = f"<music {' '.join(marks)}>{props}</music>"
    if len(offsets) == 0:
        return None
    max_offset = max(offsets)

    # dur = format_duration(max_offset - onset)
    dur = max_offset - onset
    dur = f'{dur:.1f}'
    n_tokens_st = int(onset * fps)
    n_tokens_ed = min(max_n_tokens, int(max_offset * fps)) if max_n_tokens is not None else int(max_offset * fps)
    # print('Start, End, dur', n_tokens_st, n_tokens_ed, dur)
    feature = "".join([feature_token] * (n_tokens_ed - n_tokens_st))
    head = f"<audio duration feature><duration>{dur}</duration><feature>{feature}</feature></audio>"
    return head + desc + eos, n_tokens_st, n_tokens_ed, onset, max_offset

def segs2caption_0318(segs, onset, feature_token, rng, drop_out, eot, eos, max_n_tokens, fps):
    props = ""
    offsets = []
    tempo_dt = rng.randint(0, 3) - 2 if drop_out else 0
    tempo_dt = tempo_dt * 1. / 100

    tag_idx = 0
    marks = []
    keys = [k for k in segs[0] if k not in ["mark", "onset", "offset", "timestamp"]]
    rng.shuffle(keys)

    if len(segs) == 1 and segs[0]["mark"] == "M":
        segs = divide_into_more_segs(segs[0], rng)

    if len(segs) > 1 and drop_out and rng.rand() > .8:
        drop_idx = rng.randint(0, len(segs))
    else:
        drop_idx = -1

    for i, seg in enumerate(segs):
        if float(seg["offset"]) - float(seg["onset"]) < 1:
            drop_idx = -1
            continue
        if drop_idx == i:
            continue
        offset = float(seg["offset"])

        offsets.append(float(seg["offset"]))
        timestamp = format_duration(float(seg["onset"]) - onset) + "-" + format_duration(offset - onset)
        audio_tag = chr(tag_idx + ord('A'))
        marks.append(f"{audio_tag}({timestamp})")

        tag = f"<timestamp>{timestamp}</timestamp>"

        tempo_values = []
        key_values = []
        for seg_t in segs:
            seg_onset = float(seg_t["onset"])
            seg_offset = float(seg_t["offset"])
            
            if seg_onset < offset and seg_offset > float(seg["onset"]):
                tempo_values.append(f"{seg_t['tempo']} bpm ({format_duration(max(seg_onset, float(seg['onset'])) - onset)}, {format_duration(min(seg_offset, offset) - onset)})")
                key_values.append(f"{seg_t['key']} ({format_duration(max(seg_onset, float(seg['onset'])) - onset)}, {format_duration(min(seg_offset, offset) - onset)})")

        tempo_str = " - ".join(tempo_values)
        key_str = " - ".join(key_values)

        attributes = [
            f"<{k}>{crop_props_revise(k, seg[k], float(seg['onset']), offset, onset, aug=drop_out, tempo_dt=tempo_dt)}</{k}>"
            for k in keys if offset - float(seg['onset']) > 0.5
        ]
        
        if tempo_str:
            attributes.append(f"<tempo>{tempo_str}</tempo>")
        if key_str:
            attributes.append(f"<key>{key_str}</key>")

        tag += "".join(attributes)

        out_keys = ["timestamp"] + keys + (["tempo"] if tempo_str else []) + (["key"] if key_str else [])
        if props == "":
            tag = eot + tag
        props += f"<{audio_tag} {' '.join(out_keys)}>{tag}</{audio_tag}>"
        tag_idx += 1

    section_tags = [
        f"{chr(i + ord('A'))}({format_duration(float(segs[i]['onset']) - onset)}-{format_duration(float(segs[i]['offset']) - onset)})"
        for i in range(len(segs))
    ]
    desc = f"<music {' '.join(section_tags)}>{props}</music>"
    
    if len(offsets) == 0:
        return None
    max_offset = max(offsets)

    return desc + eos, onset, max_offset, onset, max_offset






def segs2caption_revise2(segs, onset, feature_token, rng, drop_out, eot, eos, max_n_tokens, fps):

    props = ""
    offsets = []
    tempo_dt = rng.randint(0, 3) - 2 if drop_out else 0
    tempo_dt = tempo_dt * 1. / 100


    tag_idx = 0
    marks = []
    keys = [k for k in segs[0] if k not in ["mark", "onset", "offset", "timestamp"]]
    rng.shuffle(keys)

    if len(segs) == 1 and segs[0]["mark"] == "M":
        segs = divide_into_more_segs(segs[0], rng)

    if len(segs) > 1 and drop_out and rng.rand() > .8:
        drop_idx = rng.randint(0, len(segs))
    else:
        drop_idx = -1

    for i, seg in enumerate(segs):
        if float(seg["offset"]) - float(seg["onset"]) < 1:
            drop_idx = -1
            continue
        if drop_idx == i:
            continue
        offset = float(seg["offset"])

        offsets.append(float(seg["offset"]))
        timestamp = format_duration(float(seg["onset"]) - onset) + "-" + format_duration(offset - onset)
        audio_tag = chr(tag_idx + ord('A'))
        marks.append(f"{audio_tag}({timestamp})")
        tag = f"<timestamp>{timestamp}</timestamp>"
        tag += "".join(
            [f"<{k}>{crop_props_revise(k, seg[k], float(seg['onset']), offset, onset, aug=drop_out, tempo_dt=tempo_dt)}</{k}>"
             for k in keys if offset - float(seg['onset']) > 0.5])

        out_keys = ["timestamp"] + keys
        if props == "":
            tag = eot + tag
        props += f"<{audio_tag} {' '.join(out_keys)}>{tag}</{audio_tag}>"
        tag_idx += 1
    desc = f"<music {' '.join(marks)}>{props}</music>"
    if len(offsets) == 0:
        return None
    max_offset = max(offsets)

    dur = format_duration(max_offset - onset)
    n_tokens_st = int(onset * fps)
    n_tokens_ed = min(max_n_tokens, int(max_offset * fps)) if max_n_tokens is not None else int(max_offset * fps)
    # print('Start, End, dur', n_tokens_st, n_tokens_ed, dur)
    feature = "".join([feature_token] * (n_tokens_ed - n_tokens_st))
    abstract_feature = "".join(['<|y|>' for _ in range(20)])
    head = f"<audio duration feature><duration>{dur}</duration><feature>{feature}</feature><abstract>{abstract_feature}</abstract></audio>"
    return head + desc + eos, n_tokens_st, n_tokens_ed, onset, max_offset


def song2segs(song, max_sec, feature_token, eot, eos, max_n_tokens, fps, rng, drop_out, overlapping_ratio):
    dur = 0.
    segs = []
    for seg in song:
        temp = {k: seg[k] for k in seg}
        offset = float(temp["offset"])
        while offset >= dur + max_sec:
            dt = max_sec + dur - temp["onset"]
            low = int(dt * overlapping_ratio / 2)
            up = int(dt * overlapping_ratio)
            sample_sec = rng.randint(low, up) if up > low else dt
            temp["offset"] = max_sec + dur
            segs.append(temp)
            assert temp["offset"] > temp["onset"]
            yield segs2caption(segs, dur, feature_token, rng, drop_out, eot, eos, max_n_tokens, fps)
            temp["onset"] = sample_sec + temp["onset"]
            temp["offset"] = offset
            dur = temp["onset"]
            assert dur < offset
            segs = []
        segs.append(temp)

    if len(segs) > 0:
        assert segs[-1]["offset"] > segs[-1]["onset"]
        yield segs2caption(segs, dur, feature_token, rng, drop_out, eot, eos, max_n_tokens, fps)

def song2segs_revise(song, max_sec, feature_token, eot, eos, max_n_tokens, fps, rng, drop_out, overlapping_ratio, with_comparison=False):
    dur = 0.
    segs = []
    for seg in song:
        temp = {k: seg[k] for k in seg}
        offset = float(temp["offset"])
        while offset >= dur + max_sec:
            dt = max_sec + dur - temp["onset"]
            low = int(dt * overlapping_ratio / 2)
            up = int(dt * overlapping_ratio)
            sample_sec = rng.randint(low, up) if up > low else dt
            temp["offset"] = max_sec + dur
            segs.append(temp)
            assert temp["offset"] > temp["onset"]
            # 0925 segs2caption_revise -> segs2caption_revise_before
            yield segs2caption_revise_before(segs, dur, feature_token, rng, drop_out, eot, eos, max_n_tokens, fps)
            temp["onset"] = sample_sec + temp["onset"]
            temp["offset"] = offset
            dur = temp["onset"]
            assert dur < offset
            segs = []
        segs.append(temp)

    if len(segs) > 0:
        assert segs[-1]["offset"] > segs[-1]["onset"]
        yield segs2caption_revise_before(segs, dur, feature_token, rng, drop_out, eot, eos, max_n_tokens, fps)


def song2segs_0318(song, max_sec, feature_token, eot, eos, max_n_tokens, fps, rng, drop_out, overlapping_ratio, r1=1.0, with_comparison=False):
    dur = 0.
    segs = []
    prev_segment = None
    for seg in song:
        temp = {k: seg[k] for k in seg}
        offset = float(temp["offset"])

        while offset >= dur + max_sec:
            dt = max_sec + dur - temp["onset"]
            low = int(dt * overlapping_ratio / 2)
            up = int(dt * overlapping_ratio)
            sample_sec = rng.randint(low, up) if up > low else dt

            temp["offset"] = max_sec + dur

            if prev_segment and rng.rand() < r1:
                segs.append(prev_segment)
                segs.append(temp)
            else:
                segs.append(temp)

            yield segs2caption_0318(segs, dur, feature_token, rng, drop_out, eot, eos, max_n_tokens, fps)

            temp["onset"] = sample_sec + temp["onset"]
            temp["offset"] = offset
            dur = temp["onset"]
            segs = []
        
        segs.append(temp)
        prev_segment = temp

    if len(segs) > 0:
        yield segs2caption_0318(segs, dur, feature_token, rng, drop_out, eot, eos, max_n_tokens, fps)



def song2segs_revise2(song, max_sec, feature_token, eot, eos, max_n_tokens, fps, rng, drop_out, overlapping_ratio):
    dur = 0.
    segs = []
    for seg in song:
        temp = {k: seg[k] for k in seg}
        offset = float(temp["offset"])
        while offset >= dur + max_sec:
            dt = max_sec + dur - temp["onset"]
            low = int(dt * overlapping_ratio / 2)
            up = int(dt * overlapping_ratio)
            sample_sec = rng.randint(low, up) if up > low else dt
            temp["offset"] = max_sec + dur
            segs.append(temp)
            assert temp["offset"] > temp["onset"]
            yield segs2caption_revise2(segs, dur, feature_token, rng, drop_out, eot, eos, max_n_tokens, fps)
            temp["onset"] = sample_sec + temp["onset"]
            temp["offset"] = offset
            dur = temp["onset"]
            assert dur < offset
            segs = []
        segs.append(temp)

    if len(segs) > 0:
        assert segs[-1]["offset"] > segs[-1]["onset"]
        yield segs2caption_revise2(segs, dur, feature_token, rng, drop_out, eot, eos, max_n_tokens, fps)


def create_caption(root_folder, output_folder, training_data=None, split="train", rng=None,
                   eos="<|end_of_text|>", eot="<|eot_id|>", feature_token="<|x|>",
                   max_sec=22, drop_out=False, overlapping_ratio=1,
                   save_dict=True, fps=75, selected_keys=None, with_comparison=False, rearrange=False, grounding_param=-1):
    
    # using song2segs_revise()
    if training_data is None:
        dataset_path = os.path.join(root_folder, split + ".json")
        with open(dataset_path, "r") as f:
            data = json.load(f)
    else:
        data = training_data

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
    key_mapping = {
        "tempo mean": "tempo",
        "tempo": "tempo",
        "key mode": "key",
        "onset": "onset",
        "offset": "offset",
        "mark": "mark",
        "instruments": "instruments",
        "beats": "beats",
        "predominant instruments": "instruments",
        "instrumentation": "instruments",
        "chord progression": "chord",
        "chords": "chord",
        "chords": "chord",
        "genre": "genre",
        "genres": "genre",
        "monophonic ?": "is-monophonic",
        "time signature": "time-signature",
        "loop": "loop",
        "is_loop": "loop",
        "tempo std": "tempo-std",
        "swing ?": "is-swing",
        "swing ratio median": "swing-ratio-median",
        "swing ratio iqr": "swing-ratio-iqr",
        "ternary ?": "is-ternary",
        "vocal part": "vocal-part",
        "vocal gender": "vocal-gender",
        "emotion": "emotion",
        "melodiousness": "melodiousness",
        "articulation": "articulation",
        "rhythmic stability": "rhythmic-stability",
        "rhythmic complexity": "rhythmic-complexity",
        "dissonance": "dissonance",
        "tonal stability": "tonal-stability",
        "modality": "modality",
    }

    results = []
    for d in data:
        song = []
        for seg in d["segments"]:
            onset = seg["onset"]
            offset = seg["offset"]

            contents = {
                # "timestamp": [onset, offset]
            }
            basic_keys = ['onset', 'offset', 'mark']
            if not selected_keys:
                selected_keys = ['instruments', 'chord', 'key', 'tempo', 'beats', 'genre']
            for key in seg:
                formatted_key = key_mapping[key]
                if formatted_key not in basic_keys and formatted_key not in selected_keys:
                    continue
                val = format_props(formatted_key, seg[key])
                contents[formatted_key] = val

            for selected_key in selected_keys:
                if selected_key in contents:
                    song.append(contents)
                    break
            
            # song.append(contents)


        for crop_song in song2segs_revise(song, max_sec, feature_token, eot, eos,
                                None, fps, rng, drop_out=drop_out,
                                overlapping_ratio=overlapping_ratio, with_comparison=with_comparison):
            if crop_song is None:
                continue
            desc, n_tokens_st, n_tokens_ed, onset, max_offset = crop_song

            dur = max_offset - onset
            dur = f'{dur:.1f}'
            results.append({
                "filename": d["filename"],
                "dataset": d["dataset"],
                "n_tokens_st": n_tokens_st,
                "n_tokens_ed": n_tokens_ed,
                "onset": onset,
                "offset": max_offset,
                # "duration": format_duration(max_offset - onset),
                "duration": dur,
                "caption": desc
            })

    # results = rearrange_single_data(results)
    # print('rearranging...')
    if rearrange:
        print('rearranging...')
        results = rearrange_0319(results, grounding_param)

    if save_dict:
        with open(os.path.join(output_folder, f"caption_{split}.json"), "w") as f:
            json.dump(results, f, indent=2)
    return results

def create_caption_0318(root_folder, output_folder, training_data=None, split="train", rng=None,
                   eos="<|end_of_text|>", eot="<|eot_id|>", feature_token="<|x|>",
                   max_sec=22, drop_out=False, overlapping_ratio=1,
                   save_dict=True, fps=75, selected_keys=None, with_comparison=False):
    
    # using song2segs_revise()
    if training_data is None:
        dataset_path = os.path.join(root_folder, split + ".json")
        with open(dataset_path, "r") as f:
            data = json.load(f)
    else:
        data = training_data

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
    key_mapping = {
        "tempo mean": "tempo",
        "tempo": "tempo",
        "key mode": "key",
        "onset": "onset",
        "offset": "offset",
        "mark": "mark",
        "instruments": "instruments",
        "beats": "beats",
        "predominant instruments": "instruments",
        "instrumentation": "instruments",
        "chord progression": "chord",
        "chords": "chord",
        "chords": "chord",
        "genre": "genre",
        "genres": "genre",
        "monophonic ?": "is-monophonic",
        "time signature": "time-signature",
        "loop": "loop",
        "is_loop": "loop",
        "tempo std": "tempo-std",
        "swing ?": "is-swing",
        "swing ratio median": "swing-ratio-median",
        "swing ratio iqr": "swing-ratio-iqr",
        "ternary ?": "is-ternary",
        "vocal part": "vocal-part",
        "vocal gender": "vocal-gender",
        "emotion": "emotion",
        "melodiousness": "melodiousness",
        "articulation": "articulation",
        "rhythmic stability": "rhythmic-stability",
        "rhythmic complexity": "rhythmic-complexity",
        "dissonance": "dissonance",
        "tonal stability": "tonal-stability",
        "modality": "modality",
    }

    results = []
    for d in data:
        song = []
        for seg in d["segments"]:
            onset = seg["onset"]
            offset = seg["offset"]

            contents = {
                # "timestamp": [onset, offset]
            }
            basic_keys = ['onset', 'offset', 'mark']
            if not selected_keys:
                selected_keys = ['instruments', 'chord', 'key', 'tempo', 'beats', 'genre']
            for key in seg:
                formatted_key = key_mapping[key]
                if formatted_key not in basic_keys and formatted_key not in selected_keys:
                    continue
                val = format_props(formatted_key, seg[key])
                contents[formatted_key] = val

            for selected_key in selected_keys:
                if selected_key in contents:
                    song.append(contents)
                    break
            
            # song.append(contents)


        for crop_song in song2segs_0318(song, max_sec, feature_token, eot, eos,
                                None, fps, rng, drop_out=drop_out,
                                overlapping_ratio=overlapping_ratio, with_comparison=with_comparison):
            if crop_song is None:
                continue
            desc, n_tokens_st, n_tokens_ed, onset, max_offset = crop_song

            results.append({
                "filename": d["filename"],
                "dataset": d["dataset"],
                "n_tokens_st": n_tokens_st,
                "n_tokens_ed": n_tokens_ed,
                "onset": onset,
                "offset": max_offset,
                "duration": format_duration(max_offset - onset),
                "caption": desc
            })

    # results = rearrange_single_data(results)
    # print('rearranging...')

    if save_dict:
        with open(os.path.join(output_folder, f"caption_{split}.json"), "w") as f:
            json.dump(results, f, indent=2)
    return results

def rearrange_single_data(data):

    new_data = []
    tmp_dic = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G'}
    for i in range(len(data)):
        d = data[i]
        caption = d['caption']
        attribute_dic = {}
        segments_part = caption.split('</audio><music ')[-1].split('>')[0]
        segments_num = len(segments_part.split(' '))
        # print(caption.split('</audio><music ')[-1])
        
        overall_attributes = set()

        for j in range(segments_num):
            attribute_dic[tmp_dic[j]] = {}
            one_segment_part = re.findall(rf'<{tmp_dic[j]} (.*?)</{tmp_dic[j]}>', caption)[0]
            one_segment_attributes = one_segment_part.split('>')[0].split(' ')
            for attribute in one_segment_attributes:
                if attribute == 'timestamp':
                    timestamp = re.findall(rf'<{attribute}>(.*?)</{attribute}>', one_segment_part)[0]
                    start = timestamp.split('-')[0]
                    start = float(start.split(':')[-1])
                    end = timestamp.split('-')[1]
                    end = float(end.split(':')[-1])
                    attribute_dic[tmp_dic[j]]['onset'] = start
                    attribute_dic[tmp_dic[j]]['offset'] = end
                else:
                    overall_attributes.add(attribute)
                    attribute_dic[tmp_dic[j]][attribute] = re.findall(rf'<{attribute}>(.*?)</{attribute}>', one_segment_part)[0]

        overall_dic = {}
        for attribute in overall_attributes:
            if attribute in ['chord', 'beats']:
                overall_dic[attribute] = ''
            elif attribute in ['key', 'tempo']:
                overall_dic[attribute] = []
            elif attribute in ['instruments']:
                overall_dic[attribute] = set()
            for j in range(segments_num):
                if attribute not in attribute_dic[tmp_dic[j]]:
                    continue
                else:
                    if attribute in ['chord', 'beats']:
                        if overall_dic[attribute] == '':
                            overall_dic[attribute] = attribute_dic[tmp_dic[j]][attribute]
                        else:
                            overall_dic[attribute] += ', ' + attribute_dic[tmp_dic[j]][attribute]
                    elif attribute in ['key', 'tempo']:
                        overall_dic[attribute].append((attribute_dic[tmp_dic[j]]['onset'], attribute_dic[tmp_dic[j]][attribute]))
                    elif attribute in ['instruments']:
                        ins_list = attribute_dic[tmp_dic[j]][attribute].split(', ')
                        for ins in ins_list:
                            overall_dic[attribute].add(ins)
            if attribute in ['tempo', 'key']:
                edit = overall_dic[attribute]
                overall_dic[attribute] = [edit[0]]
                for j in range(1, len(edit)):
                    if edit[j][1] == edit[j-1][1]:
                        continue
                    else:
                        overall_dic[attribute].append(edit[j])

        # if len(overall_dic['key']) != segments_num:
        #     print('look ', i)
        # print(overall_dic)
        keep = caption.split('</audio>')[0] + '</audio>'
        edit = caption.split('</audio>')[-1]
        edit = edit.split('</A>')[0] + '</A>'

        start = '00:00'
        end = attribute_dic[tmp_dic[segments_num-1]]['offset']
        end = format_duration(end)

        # <music A(00:00-00:08) B(00:08-00:22)>
        ori_first_part = edit.split('>')[0] + '>'
        tgt_first_part = f"<music A({start}-{end})>"
        edit = edit.replace(ori_first_part, tgt_first_part)

        # <timestamp>00:00-00:11</timestamp>
        ori_second_part = edit.split('<timestamp>')[1].split('</timestamp>')[0]
        ori_second_part = '<timestamp>' + ori_second_part + '</timestamp>'
        tgt_second_part = ''
        edit = edit.replace(ori_second_part, tgt_second_part)
        edit = edit.replace('timestamp ', '')

        for attribute in overall_attributes:
            ori = re.findall(rf'<{attribute}>(.*?)</{attribute}>', edit)
            tgt = overall_dic[attribute]
            if attribute in ['chord', 'beats']:
                tgt = tgt
            elif attribute in ['key', 'tempo']:
                tgt = str(tgt).replace("'", "")[1:-1]
            elif attribute in ['instruments']:
                tgt = ', '.join(tgt)
            edit = edit.replace(ori[0], tgt)
        
        # print('edit:', edit)
            

        d['caption'] = keep + edit + '</music>' + '<|end_of_text|>'
        new_data.append(d)
    return new_data

def get_case_prompt(s):
    # music_part = s.split('<music A B>')[-1]
    comparison_attributes_markers = re.findall(r'<comparison \((.*?)\)>', s)[0].split(' ')
    concept_map_source = re.findall(r'<concept (.*?)>', s)[0].split(' ')
    concept_map = {}
    for marker in concept_map_source:
        term = re.findall(rf'<{marker}(.*?)</{marker}>', s)[0]
        term = re.findall(rf'<term>(.*?)</term>', term)[0]
        concept_map[marker] = term

    comparison_attributes = concept_map.values() # tempo, blabla
    # print(comparison_attributes)
    a_attributes = re.findall(r'<A (.*?)>', s)[0].split(' ')
    b_attributes = re.findall(r'<B (.*?)>', s)[0].split(' ')
    a_part = re.findall(r'<A (.*?)</A>', s)[0]
    b_part = re.findall(r'<B (.*?)</B>', s)[0]
    dic_A = {}
    
    for attribute in a_attributes:

        if attribute not in concept_map.keys():
            continue


        value = re.findall(rf'<{attribute}>(.*?)</{attribute}>', a_part)[0]
        dic_A[concept_map[attribute]] = value

    
    dic_B = {}
    for attribute in b_attributes:
        if attribute not in concept_map.keys():
            continue
        value = re.findall(rf'<{attribute}>(.*?)</{attribute}>', b_part)[0]
        dic_B[concept_map[attribute]] = value

    prompt = f"A: {str(dic_A)}\nB: {str(dic_B)}"
    return prompt

def find_first_letter_position(text):
    match = re.search(r'[a-zA-Z]', text)
    if match:
        return match.start()
    return -1

def add_natural_language(root_folder, output_folder, suffix, split = 'train'):
    import torch
    from transformers import LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import time
    
    new_json_ls = []

    base_model = "NousResearch/Meta-Llama-3-8B-Instruct"

    if torch.cuda.get_device_capability()[0] >= 8:
        attn_implementation = "flash_attention_2"
        torch_dtype = torch.bfloat16
    else:
        attn_implementation = "eager"
        torch_dtype = torch.half

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=attn_implementation,
        local_files_only=True
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # model.resize_token_embeddings(len(tokenizer))
    model.eval()

    basic_prompt = "You are a musician. I have a pair of music A and B, \
    I would like you to compare the two pieces of music and summarize the comparison in natural language in 80 words. \
    There are some points:\
    1. You must mention the music attribute I gave you in each group. \
    2. Moreover, you can inference more music information not limited on the given attribute based on your musical knowledge.\
    3. You are also encouraged to explain the relationship of different music attributes in one song.\
    4. Provide a purely comparison without adding any explanatory phrases like 'I think' or 'Here is my answer.'"

    dataset_path = os.path.join(root_folder, f"caption_pair_{split}_pair{suffix}.json")

    with open(dataset_path, "r") as f:
        data = json.load(f)

    cnt = 0
    batch_size = 32 # 32
    prompts = []
    for i in range(len(data)):
        d = data[i]
        cnt += 1
        # comparison_attributes = re.findall(r'<comparison \((.*?)\)>', d['caption'].split('</audio>')[1])[0].split(' ')
        case_prompt = get_case_prompt(d['caption'].split('</audio>')[1])

        

        prompt = f'''
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        {basic_prompt}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {case_prompt}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        '''

        prompts.append(prompt)
        if len(prompts) == batch_size:
            start = time.time()

            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to('cuda')
            max_length = inputs['input_ids'].shape[1] + 200

            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=max_length, top_k=30, top_p=0.8, temperature=0.5)

            # raw_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
            raw_responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            responses = []
            for raw_response in raw_responses:
                try:
                    response = raw_response.split('assistant')[-1]
                except:
                    response = raw_response
                # responses.append(raw_response)
                # try:
                #     response = raw_response.split('<|end_header_id|>')[-1].split('<|eot_id|>')[0]
                # except:
                #     response = raw_response
                first_letter = find_first_letter_position(response)
                if first_letter != -1:
                    response = response[first_letter:].strip()
                else:
                    response = response
                
                responses.append(response)
                # print(response)
                # print('################')
            
            end = time.time()
            for j in range(batch_size):
                new_caption = data[(i-batch_size+j+1)]['caption'].replace('to-do', responses[j])
                data[(i-batch_size+j+1)]['caption'] = new_caption
                new_json_ls.append(data[(i-batch_size+j+1)])
                with open(os.path.join(output_folder, f"caption_pair_{split}_pair{suffix}_natural_language.json"), "w") as f:
                    json.dump(new_json_ls, f, indent=2)
            prompts = []

    if len(prompts) > 0:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to('cuda')
        max_length = inputs['input_ids'].shape[1] + 200

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=max_length, top_k=30, top_p=0.8, temperature=0.5)

        # raw_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        raw_responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        responses = []
        for raw_response in raw_responses:
            try:
                response = raw_response.split('assistant')[-1]
            except:
                response = raw_response
            first_letter = find_first_letter_position(response)
            if first_letter != -1:
                response = response[first_letter:].strip()
            else:
                response = response

            responses.append(response)
        
        end = time.time()
        for j in range(len(prompts)):
            new_caption = data[(i-len(prompts)+j+1)]['caption'].replace('to-do', responses[j])
            data[(i-len(prompts)+j+1)]['caption'] = new_caption
            new_json_ls.append(data[(i-len(prompts)+j+1)])
            with open(os.path.join(output_folder, f"caption_pair_{split}_pair{suffix}_natural_language.json"), "w") as f:
                json.dump(new_json_ls, f, indent=2)


def rearrange_0319(results, grounding_param = -1):
    import json
    import re
    from collections import Counter

    data = results

    cnt_all = 0
    cnt = 0

    timestamp_pattern = re.compile(r'<timestamp>([\d., ]+)</timestamp>')
    tempo_pattern = re.compile(r'<tempo>([\d]+ bpm)</tempo>')
    key_pattern = re.compile(r'<key>([^<]+)</key>')
    instruments_pattern = re.compile(r'<instruments>([^<]+)</instruments>')
    chord_pattern = re.compile(r'<chord>(.*?)</chord>', re.S)
    music_pattern = re.compile(r'<music.*?</music>', re.S)

    def extract_values(pattern, text):
        return pattern.findall(text)

    def extract_field_order(music_text):
        field_pattern = re.compile(r'<(\w+)>')
        field_order = field_pattern.findall(music_text)
        new_field_order = []
        for i in range(len(field_order)):
            if field_order[i] not in new_field_order:
                new_field_order.append(field_order[i])
        return new_field_order

    def merge_music_segments(text, grounding_param):
        # grounding_parma -1 or 0.0~1.0
        import random

        include_grounding = True if grounding_param >= 0 else False
        grounding_first_rate = grounding_param
        match = music_pattern.search(text)
        if not match:
            return text
        music_text = match.group(0)
        segment_labels = re.findall(r'<music ([A-Z]\(.*?\))(?: [A-Z]\(.*?\))*>', music_text)
        if not segment_labels:
            return text
        original_label = segment_labels[0]
        timestamps = extract_values(timestamp_pattern, music_text)
        tempos = extract_values(tempo_pattern, music_text)
        keys = extract_values(key_pattern, music_text)
        instruments_list = extract_values(instruments_pattern, music_text)
        chords = extract_values(chord_pattern, music_text)
        segment_durations = []
        total_duration = 0
        start_time = float(timestamps[0].split(", ")[0])
        for t in timestamps:
            start, end = map(float, t.split(", "))
            duration = end - start
            segment_durations.append((duration, tempos.pop(0), keys.pop(0)))
            total_duration += duration
        merged_label = f"A({start_time}, {start_time + total_duration})"
        tempo_counter = Counter()
        key_counter = Counter()
        # print(segment_durations)
        # print(timestamps)
        if include_grounding:
            grounding_part = add_grounding(segment_durations, timestamps, chords)
        else:
            grounding_part = ""
        for duration, tempo, key in segment_durations:
            tempo_counter[tempo] += duration
            key_counter[key] += duration

        final_tempo = tempo_counter.most_common(1)[0][0]
        final_key = key_counter.most_common(1)[0][0]

        all_instruments = set()
        for instruments in instruments_list:
            all_instruments.update(instruments.split(', '))
        final_instruments = ', '.join(sorted(all_instruments))
        all_chords = []
        for chord in chords:
            all_chords.extend(chord.split('), ('))
        # final_chords = '(' + '), ('.join(all_chords) + ')'
        final_chords = '), ('.join(all_chords)
        eot_match = re.search(r'(<\|eot_id\|>)', music_text)
        eot_text = eot_match.group(1) if eot_match else ""
        field_order = extract_field_order(music_text)
        new_music_segment = f"<music {merged_label}><A timestamp tempo key instruments chord>{eot_text}"
        for field in field_order:
            if field == "timestamp":
                new_music_segment += f"<timestamp>{start_time}, {start_time + total_duration}</timestamp>"
            elif field == "tempo":
                new_music_segment += f"<tempo>{final_tempo}</tempo>"
            elif field == "key":
                new_music_segment += f"<key>{final_key}</key>"
            elif field == "instruments":
                new_music_segment += f"<instruments>{final_instruments}</instruments>"
            elif field == "chord":
                new_music_segment += f"<chord>{final_chords}</chord>"
        new_music_segment += "</A></music>"
        text = text.replace(music_text, new_music_segment)
        if not include_grounding:
            return text

        audio_part = text.split('</audio>')[0] + '</audio>'
        music_part = text.split('</audio>')[1].split('<|end_of_text|>')[0]
        music_part1 = music_part.split('<|eot_id|>')[0]
        music_part2 = music_part.split('<|eot_id|>')[1]

        if random.random() < grounding_first_rate:
            text = audio_part + '<grounding>' + '<|eot_id|>' + grounding_part.split('<grounding>')[1] + music_part1 + music_part2 + '<|end_of_text|>'
        else:
            text = text.split('<|end_of_text|>')[0] + grounding_part + '<|end_of_text|>'
        return text

    def add_grounding(segment_durations, timestamps, chords):
        import random
        import math

        # [(5.4, '179 bpm', 'F#major'), (16.1, '179 bpm', 'F#minor')]
        # ['0.0, 5.4', '5.4, 21.5']
        qa_pairs = {}
        q1 = "Is there any tempo change in the music?"
        q2 = "Is there any key change in the music?"

        if len(set([x[1] for x in segment_durations])) == 1:
            qa_pairs[q1] = "No"
        else:
            qa_pairs[q1] = "Yes"

        if len(set([x[2] for x in segment_durations])) == 1:
            qa_pairs[q2] = "No"
        else:
            qa_pairs[q2] = "Yes"
        
        if qa_pairs[q1] == "Yes":
            # random select an index 0~len(timestamps)-1
            index = random.randint(0, len(timestamps)-1)
            target_tempo = segment_durations[index][1]
            q3 = f'Which section has a tempo of {target_tempo}?'
            a3 = []
            for i in range(len(timestamps)):
                if segment_durations[i][1] == target_tempo:
                    start = timestamps[i].split(", ")[0]
                    end = timestamps[i].split(", ")[1]
                    a3.append(f'{start} ~ {end}s')
            a3 = ', '.join(a3)
            qa_pairs[q3] = a3

            # another question
            index = random.randint(0, len(timestamps)-1)
            start = float(timestamps[index].split(", ")[0])
            end = float(timestamps[index].split(", ")[1])
            s = random.randint(math.ceil(start), math.floor(end))
            e = random.randint(s, math.floor(end))
            if e > s:
                q5 = f'What is the tempo of the music from {s}.0s to {e}.0s?'
                qa_pairs[q5] = segment_durations[index][1]
        else:
            pass

        if qa_pairs[q2] == "Yes":
            index = random.randint(0, len(timestamps)-1)
            target_key = segment_durations[index][2]
            q4 = f'Which section has a key of {target_key}?'
            a4 = []
            for i in range(len(timestamps)):
                if segment_durations[i][2] == target_key:
                    start = timestamps[i].split(", ")[0]
                    end = timestamps[i].split(", ")[1]
                    a4.append(f'{start}s ~ {end}s')
            a4 = ', '.join(a4)
            qa_pairs[q4] = a4

            # another question
            index = random.randint(0, len(timestamps)-1)
            start = float(timestamps[index].split(", ")[0])
            end = float(timestamps[index].split(", ")[1])
            s = random.randint(math.ceil(start), math.floor(end))
            e = random.randint(s, math.floor(end))
            if e > s:
                q6 = f'What is the key of the music from {s}.0s to {e}.0s?'
                qa_pairs[q6] = segment_durations[index][2]


        else:
            pass
        
        chords = ', '.join(chords)
        # print(chords)
        chord_pattern = re.findall(r"\(([\d.]+), ([A-G]#?[a-z]+)\)", chords)
        chords = [(float(time), chord) for time, chord in chord_pattern]
        if len(chords) > 0:
            min_time = math.ceil(float(chords[0][0]))
            max_time = math.floor(float(chords[-1][0]))
            s = random.randint(min_time, max_time)
            e = random.randint(s, max_time)

            extracted_chords = []
            for time, chord in chords:
                if s <= time <= e:
                    if not extracted_chords or extracted_chords[-1] != chord:
                        extracted_chords.append(chord)

                    if e > s:
                        q7 = f'What are the chords from {s}.0s to {e}.0s?'
                        a7 = ', '.join(extracted_chords)
                        qa_pairs[q7] = a7
        
        # print(s,e,extracted_chords)

        # <grounding>|Q|q1|A|a1|Q|q1|A|a1...</grounding>
        if len(qa_pairs) >= 1:
            grounding = "<grounding>"
            for q, a in qa_pairs.items():
                grounding += f"|Q|{q}|A|{a}"
            grounding += "</grounding>"
        else:
            grounding = ''

        return grounding

    res = []
    for d in data:
        caption = d['caption']
        # caption = re.sub(r'(<\|x\|>)+', '*', caption)

        cnt_all += 1
        try:
            new_caption = merge_music_segments(caption, grounding_param)
        except:
            new_caption = caption
        d['caption'] = new_caption
        res.append(d)

    return res

