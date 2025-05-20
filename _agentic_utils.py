import re
import torch
from transformers.cache_utils import DynamicCache
from tiger_utils import dedup_list
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaConfig, LlamaForCausalLM
from transformers import (
  AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig
)

from block_attention import apply_pkv_rotary_position_embeddings
from utils import get_prompt

def get_qa_prompt(q, relevant_objects):
  user_str = []

  user_str.append('Do not use your general knowledge. Do not assume the existence of external knowledge. Do not make any guesses.\nYou are provided with a user question, and information that might be relevant to the user question.\n\nYour task consists of the following steps:\n1. From the provided information, extract facts that is relevant to the user question\n\n2. Based on the provided information only, determine if you have sufficient information to answer the user question\n- If you can determine the answer, output a short answer (in a few words) to the user question. The short answer must be wrapped in <ans></ans>.\n- If you cannot determine the answer, output some keywords that can help you retrieve new information. The keywords must be wrapped in <keywords></keywords>.')

  user_str.append(f'Here is the information:\n{relevant_objects}')
    
  user_str.append(f'Here is the user question:\n{q["question"]}')

  user_str = '\n\n'.join(user_str)
  
  # default output from apply_chat_template:
  # <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nhello world<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n

  user_str = f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_str}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nTo answer the user question: {q["question"]}\n\n'

  return user_str


def get_reasoning_prompt(dataset: str, q, reasoning):
  choice_map = {
    0: '(A)',
    1: '(B)',
    2: '(C)',
    3: '(D)',
    4: '(E)',
    5: '(F)',
    6: '(G)',
    7: '(H)',
    8: '(I)',
    9: '(J)',
    10: '(K)'
  }

  question = q['question']

  if dataset in ['gpqa']:
    choices = q['choices']
  elif dataset == 'mmlupro':
    choices = q['options']

  choices = '\n'.join([f'{choice_map[i]} {choice}' for i, choice in enumerate(choices)])

  prompt = ['You are provided with a multi-choice question. Your task consists of the following steps:\n1. From the provided information, extracts the key insights helpful for solving the user question\n\n2. Break down and solve the question step by step, without relying on the provided answer choices\n\n3. Based on your analysis, determine if you have sufficient information to identify the single most probable answer\n- If you can identify the answer, output the answer as the letter corresponding to the answer choice, placed inside parentheses and wrapped in <ans></ans> (e.g., <ans>(A)</ans>).\n- If you cannot identify the answer, output sub-questions that, if solved, can lead to new information. The sub-questions must be wrapped in <subquestion></subquestion>.']

  if reasoning != '':
    prompt.append(f'Here is the information:\n{reasoning}')

  prompt.append(f'Here is the user question:\n{question}')
  prompt.append(f'Here are the multiple-choice answers:\n{choices}')

  prompt = '\n\n'.join(prompt)

  prompt = get_prompt(prompt)

  return prompt

def extract_query_qa(output: str):
  if '<keywords>' in output and '</keywords>' in output:
    query = re.findall(r'<keywords>(.*?)</keywords>', output)[-1]
  else:
    query = output
  
  return query

def extract_query_reasoning(output: str):
  # extract all, and de-duplicate
  if '<subquestion>' in output and '</subquestion>' in output:
    query = re.findall(r'<subquestion>(.*?)</subquestion>', output)
    query = [x.strip() for x in query]
    query = ' '.join(dedup_list(query))
  else:
    query = output
  
  return query

def merge_log_kv(log_ids, logs_kv, emb):
  kv_all = DynamicCache()

  key_caches_all, value_caches_all = [], []

  kvs = [logs_kv[log_id] for log_id in log_ids]

  # cross-reference with the source code (def update)
  for layer_idx in range(len(kvs[0])):
    key_caches = [kv.key_cache[layer_idx] for kv in kvs]
    value_caches = [kv.value_cache[layer_idx] for kv in kvs]
    
    key_caches_all.append(torch.concat(key_caches, dim=2))
    value_caches_all.append(torch.concat(value_caches, dim=2))
  
  seen_tokens = [kv._seen_tokens for kv in kvs]

  kv_all._seen_tokens = sum(seen_tokens)
  kv_all.key_cache = key_caches_all
  kv_all.value_cache = value_caches_all

  # print(kv_all.key_cache[0].shape)
  # print(kv_all.value_cache[0].shape)
  # print(kv_all._seen_tokens)

  kv_all = apply_pkv_rotary_position_embeddings(pkv=kv_all, emb=emb)

  return kv_all

def get_generation_config(tokenizer):
  return GenerationConfig(
    do_sample=False,
    temperature=1.0,
    repetition_penalty=1.0,
    num_beams=1,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=1024
  )

def load_model():
  # model_id = 'Qwen/Qwen2.5-7B-Instruct' if instruct else 'Qwen/Qwen2.5-7B'
  model_id = 'meta-llama/Llama-3.1-8B-Instruct'
  print(model_id)
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
    model_id, device_map='auto', torch_dtype=torch.bfloat16,
    attn_implementation='flash_attention_2'
  )
  model.eval()
  config: LlamaConfig = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_id)
  emb: LlamaRotaryEmbedding = LlamaRotaryEmbedding(config=config).to(device=model.device, dtype=torch.float32)
  emb.eval()

  return tokenizer, model, emb