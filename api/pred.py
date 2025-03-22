import os, csv, json
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset
import re
from openai import OpenAI
from transformers import AutoTokenizer
import tiktoken
import torch.multiprocessing as mp
import requests
import pickle
from concurrent.futures import Executor, ThreadPoolExecutor, as_completed

model_map = json.loads(open('config/model2path.json', encoding='utf-8').read())
maxlen_map = json.loads(open('config/model2maxlen.json', encoding='utf-8').read())

# URL = "http://127.0.0.1:8000/v1"
# URL = "http://127.0.0.1:30000/v1"
URL = "http://127.0.0.1:30000/generate"
# API_KEY = "token-abc123"
API_KEY = "EMPTY"
TIME_OUT = 60 * 60 # 1h
TIME_OUT *= 3 * 24

def query_llm(prompt, model, tokenizer, temperature=0.5, max_new_tokens=128, stop=None, top_p: float=1.0, top_k: int=-1, truncate: bool=False):
    # truncate
    if truncate:
        max_len = maxlen_map[model]
        if model in model_map:
            input_ids = tokenizer.encode(prompt)
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
                prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
        else:
            input_ids = tokenizer.encode(prompt, disallowed_special=())
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
                prompt = tokenizer.decode(input_ids)
    tries = 0
    while tries < 5:
        tries += 1
        try:
            # completion = client.chat.completions.create(
            #     model=model,
            #     messages=[{"role": "user", "content": prompt}],
            #     temperature=temperature,
            #     max_tokens=max_new_tokens,
            # )
            # return completion.choices[0].message.content
            response = requests.post(
                URL,
                json={
                    "text": prompt,
                    "sampling_params": {
                        "temperature": temperature,
                        "max_new_tokens": max_new_tokens,
                        "stop": stop,
                        "top_p": top_p,
                        "top_k": top_k,
                        "n": 1,
                    },
                },
                # timeout=TIME_OUT,
            )
            return response.json()['text']
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print("Error Occurs: \"%s\"        Retry ..."%(str(e)))
            time.sleep(1)
    else:
        print("Max tries. Failed.")
        return ''

def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None

# def get_pred(data, args, fout):
def get_pred(item, args, fout):
    model = args.model
    if "gpt" in model or "o1" in model:
        tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_map[model], trust_remote_code=True)
    # client = OpenAI(
    #     base_url=URL,
    #     api_key=API_KEY,
    #     timeout=TIME_OUT
    # )
    # for item in tqdm(data):
    prompt = item['input_text']
    output = query_llm(
        prompt,
        model,
        tokenizer,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
        truncate=args.truncate,
        stop=["<｜end▁of▁sentence｜>", "<｜begin▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>"], #, "<think>"]
    )
    response = output.strip()
    if response != "":
        item['output_text'] = response
        item["input_token_ids"] = tokenizer.encode(prompt)
        item["output_token_ids"] = tokenizer.encode(response)
        item["input_len"] = len(item["input_token_ids"])
        item["output_len"] = len(item["output_token_ids"])
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        fout.flush()

def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    out_file = os.path.join(args.save_dir, 'pred.jsonl')

    with open(args.data, 'rb') as fid:
        data_all = [
            {"_id": i, "input_text": text}
            for i, text in enumerate(pickle.load(fid))
        ]

    # cache
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}
    fout = open(out_file, 'a', encoding='utf-8')
    data = []
    for item in data_all:
        if item["_id"] not in has_data:
            data.append(item)

    with ThreadPoolExecutor(max_workers=args.n_proc) as executor:
        # 提交任务到线程池。
        futures = [
            executor.submit(get_pred, item, args, fout) for item in data
        ]
        # 按完成顺序获取结果。
        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()
    # data_subsets = [data[i::args.n_proc] for i in range(args.n_proc)]
    # processes = []
    # for rank in range(args.n_proc):
    #     p = mp.Process(target=get_pred, args=(data_subsets[rank], args, fout))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, required=True)
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--n_proc", "-n", type=int, default=16)
    parser.add_argument("--temperature", "-t", type=float, default=0.1)
    parser.add_argument("--max_new_tokens", "-mt", type=int, default=32768)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--truncate", action="store_true")
    args = parser.parse_args()
    main()
