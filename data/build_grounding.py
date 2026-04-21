import json
import re
from collections import Counter

json_path = '/datapool/data2/home/ruihan/storage/debug/all_m4m/revising/m4m_dataset/dataset/new_dataset/formatted_dataset/caption_train_debug.json'

with open(json_path, 'r') as f:
    data = json.load(f)

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

def merge_music_segments(text):
    import random
    include_grounding = True
    grounding_first_rate = 0.0
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

for d in data[0:5]:
    caption = d['caption']
    caption = re.sub(r'(<\|x\|>)+', '*', caption)
    # print(caption)
    print('----------------------')
    cnt_all += 1
    new_caption = merge_music_segments(caption)
    print(new_caption)
    # print(new_caption)
    print('======================')
