import re
import torch
import json
import os
import argparse

def find_first_letter_position(text):
    match = re.search(r'[a-zA-Z]', text)
    if match:
        return match.start()
    return -1

dic_root = {
    'Ab': 'G#',
    'Bb': 'A#',
    'Cb': 'B',
    'Db': 'C#',
    'Eb': 'D#',
    'Fb': 'E',
    'Gb': 'F#',
}


def format_props(key, val):
    if key == "tempo":
        return int(float(val))
    if key == "key":
        replace_dict = {":": "", "major": "maj", "minor": "min"}
        for name in replace_dict:
            val = str.replace(val, name, replace_dict[name])
        val = str.replace(val, "maj", " major")
        val = str.replace(val, "min", " minor")
        val = str.replace(val, " ", "")
        val = val.split("major")
        scale, mode = [val[0].split("minor")[0], "minor"] if len(val) == 1 else [val[0], "major"]
        # convert all b to #
        if 'b' in scale:
            scale = dic_root[scale]
        return f"{scale}{mode}"
    if key == "chord":
        ls = []
        if type(val) is list:
            for tup in val:
                replace_dict = {":": "", "major": "maj", "minor": "min"}
                for name in replace_dict:
                    tup[1] = str.replace(tup[1], name, replace_dict[name])
                tup[1] = str.replace(tup[1], "maj", "major")
                tup[1] = str.replace(tup[1], "min", "minor")
                ls.append([float(tup[0]), tup[1]])
            # return str(ls)[1:-1]
        # print(ls)
        return ls
    
    if key == "beats":
        ls = []
        if type(val) is list:
            for tup in val:
                if int(tup[1]) < 0:
                    continue
                ls.append([float(tup[0]),tup[1]])
        return ls

    if key == "instruments":
        if type(val) is dict:
            val = [k for k in val if val[k]]
        return val

    return val

def add_natural_language(path):
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

    # basic_prompt = "You are a musician. I have a piece of music, \
    # I would like you to describe it in natural language in 80 words. \
    # There are some points:\
    # 1. You must mention the music attribute I gave you in each group. \
    # 2. Moreover, you can inference more music information not limited on the given attribute based on your musical knowledge.\
    # 3. You are also encouraged to explain the relationship of different music attributes in one song.\
    # 4. Provide a purely comparison without adding any explanatory phrases like 'I think' or 'Here is my answer.'"

    basic_prompt = "You are a musician. I have a piece of music, \
    I would like you to describe it in natural language in 150 words. \
    There are some points:\
    1. You must mention the music attribute I gave you in each group. \
    2. Provide a purely comparison without adding any explanatory phrases like 'I think' or 'Here is my answer.'"

    # dataset_path = os.path.join(root_folder, f"caption_pair_{split}_pair{suffix}.json")
    dataset_path = path

    with open(dataset_path, "r") as f:
        data = json.load(f)

    cnt = 0
    batch_size = 32 # 32
    prompts = []
    for i in range(len(data)):
        d = data[i]
        cnt += 1
        # comparison_attributes = re.findall(r'<comparison \((.*?)\)>', d['caption'].split('</audio>')[1])[0].split(' ')
        filename = d['filename']
        dic = d['segments'][0]
        case_prompt = []
        for k, v in dic.items():
            # if 'tempo' in k or 'instrument' in k or 'key' in k or 'genre' in k:
            #     case_prompt.append(f"{k}: {v}")

            if 'tempo' in k:
                v = format_props("tempo", v)
                case_prompt.append(f"tempo: {v}")

            if 'key' in k:
                v = format_props("key", v)
                case_prompt.append(f"key: {v}")

            if 'chord' in k:
                v = format_props("chord", v)
                case_prompt.append(f"chord: {v}")

            if 'genre' in k:
                v = format_props("genre", v)
                case_prompt.append(f"genre: {v}")

            if 'instrument' in k:
                v = format_props("instruments", v)
                case_prompt.append(f"instruments: {v}")
        case_prompt = ', '.join(case_prompt)
        # print(case_prompt)
        

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

            try:
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_length=max_length, top_k=30, top_p=0.8, temperature=0.5)

                # raw_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
                raw_responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            except:
                raw_responses = ['' for _ in range(len(prompts))]
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
                new_data_point = {}
                new_data_point['filename'] = data[(i-len(prompts)+j+1)]['filename']
                new_data_point['segments'] = responses[j]
                # new_caption = data[(i-(prompts)+j+1)]['caption'].replace('to-do', responses[j])
                # data[(i-len(prompts)+j+1)]['caption'] = new_caption
                # new_json_ls.append(data[(i-len(prompts)+j+1)])
                new_json_ls.append(new_data_point)
                new_path = path.replace('.json', '_new2.json')
                with open(new_path, "w") as f:
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
            new_data_point = {}
            new_data_point['filename'] = data[(i-len(prompts)+j+1)]['filename']
            new_data_point['segments'] = responses[j]
            # new_caption = data[(i-len(prompts)+j+1)]['caption'].replace('to-do', responses[j])
            # data[(i-len(prompts)+j+1)]['caption'] = new_caption
            # new_json_ls.append(data[(i-len(prompts)+j+1)])
            new_json_ls.append(new_data_point)
            new_path = path.replace('.json', '_new2.json')
            with open(new_path, "w") as f:
                json.dump(new_json_ls, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate natural language descriptions for music data.")
    parser.add_argument("path", type=str, help="Path to the JSON file containing music data")
    args = parser.parse_args()
    add_natural_language(args.path)
