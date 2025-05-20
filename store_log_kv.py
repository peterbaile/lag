from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.cache_utils import DynamicCache
import torch
from tiger_utils import read_json, write_pickle
from tqdm import tqdm
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaConfig
import re
import argparse

from block_attention import apply_pkv_rerotary_position_embeddings
from utils import STORE_PREFIX

def get_log_kv(text, tokenizer, model, emb):
  input_ids = tokenizer.encode(text, return_tensors='pt', add_special_tokens=False).to(model.device)

  with torch.no_grad():
    outputs = model(
      input_ids=input_ids,
      past_key_values=DynamicCache(),
      use_cache=True,
      output_attentions=False,
      output_hidden_states=False
    )

    # revert the KV value back
    kv = apply_pkv_rerotary_position_embeddings(pkv=outputs.past_key_values, emb=emb)
  
  kv.key_cache = [x.cpu() for x in kv.key_cache]
  kv.value_cache = [x.cpu() for x in kv.value_cache]

  return kv

def get_reasoning_kv(log, tokenizer, model, emb, last_num: int):
  assert (len(log) % 2 == 0) and len(log) >= 2

  # only use the assistant messages
  assistant_logs = [x for i, x in enumerate(log) if i % 2 == 1]

  assistant_logs_except_last = '\n\n'.join(assistant_logs[:-last_num]).replace('<|eot_id|>', '')
  
  assistant_logs_last = '\n\n'.join(assistant_logs[-last_num:]).replace('<|eot_id|>', '') + '\n\n'

  input_ids = tokenizer.encode(assistant_logs_except_last + '\n\n' + assistant_logs_last, return_tensors='pt', add_special_tokens=False)
  input_ids_except_last = tokenizer.encode(assistant_logs_except_last + '\n\n', return_tensors='pt', add_special_tokens=False)
  input_ids_last = tokenizer.encode(assistant_logs_last, return_tensors='pt', add_special_tokens=False)

  assert input_ids.shape[1] == input_ids_except_last.shape[1] + input_ids_last.shape[1]

  start_idx = input_ids_except_last.shape[1]

  kv = get_log_kv(assistant_logs_except_last + '\n\n' + assistant_logs_last, tokenizer, model, emb)

  assert kv.key_cache[0].shape[2] == input_ids.shape[1]

  kv._seen_tokens = kv._seen_tokens - start_idx
  num_layers = len(kv)
  for layer_idx in range(num_layers):
    # slicing only return view, not changing the original tensor
    # pickle saves the underlying storage
    kv.key_cache[layer_idx] = kv.key_cache[layer_idx][:, :, start_idx:, :].clone().contiguous()
    kv.value_cache[layer_idx] = kv.value_cache[layer_idx][:, :, start_idx:, :].clone().contiguous()

    assert kv.key_cache[layer_idx].shape == kv.value_cache[layer_idx].shape
    assert kv.key_cache[layer_idx].shape[2] == kv._seen_tokens == input_ids_last.shape[1]

  kv.key_cache = [x.cpu() for x in kv.key_cache]
  kv.value_cache = [x.cpu() for x in kv.value_cache]

  return kv

def store_reasoning_offline(dataset, last_num: int):
  '''The default storage strategy: Encode all reasoning traces, store the KV values corresponding to the last_num reasoning trace
  (e.g., if last_num = 2, store the KV values corresponding to tokens in the last two reasoning traces)
  '''

  model_id = 'meta-llama/Llama-3.1-8B-Instruct'
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.bfloat16)
  config: LlamaConfig = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_id)
  emb: LlamaRotaryEmbedding = LlamaRotaryEmbedding(config=config).to(device=model.device, dtype=torch.float32)

  print(model_id)

  logs = read_json(f'./data/{dataset}/preds/log.json')

  log_kvs = []

  for log in tqdm(logs):
    log = log[0]
    kv = get_reasoning_kv(log, tokenizer, model, emb, last_num)
    log_kvs.append(kv)

  write_pickle(log_kvs, f'{STORE_PREFIX}/kv/{dataset}/reasoning_last_{last_num}.pkl')

def store_reasoning_last_as_context(dataset):
  '''Encode the last reasoning trace, store the KV values corresponding to the last reasoning trace'''

  model_id = 'meta-llama/Llama-3.1-8B-Instruct'
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.bfloat16)
  config: LlamaConfig = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_id)
  emb: LlamaRotaryEmbedding = LlamaRotaryEmbedding(config=config).to(device=model.device, dtype=torch.float32)

  logs = read_json(f'./data/{dataset}/preds/log.json')

  log_kvs = []

  for log in tqdm(logs):
    log = log[0]
    assert (len(log) % 2 == 0) and len(log) >= 2

    assistant_logs_last = log[-1].replace('<|eot_id|>', '') + '\n\n'
    kv = get_log_kv(assistant_logs_last, tokenizer, model, emb)
    log_kvs.append(kv)

  write_pickle(log_kvs, f'{STORE_PREFIX}/kv/{dataset}/reasoning_last_as_context.pkl')

def store_reasoning_all(dataset):
  '''Encode all reasoning traces, store the KV values corresponding to all reasoning traces'''

  model_id = 'meta-llama/Llama-3.1-8B-Instruct'
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.bfloat16)
  config: LlamaConfig = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_id)
  emb: LlamaRotaryEmbedding = LlamaRotaryEmbedding(config=config).to(device=model.device, dtype=torch.float32)

  logs = read_json(f'./data/{dataset}/preds/log.json')

  log_kvs = []

  for log in tqdm(logs):
    log = log[0]
    assert (len(log) % 2 == 0) and len(log) >= 2

    assistant_logs = [x for i, x in enumerate(log) if i % 2 == 1]
    assistant_logs = '\n\n'.join(assistant_logs).replace('<|eot_id|>', '') + '\n\n'
    kv = get_log_kv(assistant_logs, tokenizer, model, emb)
    log_kvs.append(kv)

  write_pickle(log_kvs, f'{STORE_PREFIX}/kv/{dataset}/kv/reasoning_all.pkl')

# < > </ > tokens always seem to be independent in the token space
# extract the last action from the text
def extract_action(text: str):
  # if <ans> or <keywords> exist, then use them; otherwise, use the entire trace

  # search for an answer
  answer = None
  if '<ans>' in text and '</ans>' in text:
    _answer = re.findall(r'<ans>(.*?)</ans>', text)
    _answer = [x for x in _answer if x.strip() != '']
    if len(_answer) != 0:
      answer = f'<ans>{_answer[-1]}</ans>'
  
  if answer is not None:
    return answer
  
  # search for a keyword
  keywords = None
  if '<keywords>' in text and '</keywords>' in text:
    _keywords = re.findall(r'<keywords>(.*?)</keywords>', text)
    _keywords = [x for x in _keywords if x.strip() != '']
    if len(_keywords) != 0:
      keywords = f'<keywords>{_keywords[-1]}</keywords>'
  
  if keywords is not None:
    return keywords
  
  # TODO: this is only extracting the last subquestion, (maybe merge all subquestions?)
  question = None
  if '<subquestion>' in text and '</subquestion>' in text:
    _question = re.findall(r'<subquestion>(.*?)</subquestion>', text)
    _question = [x for x in _question if x.strip() != '']
    if len(_question) != 0:
      question = f'<subquestion>{_question[-1]}</subquestion>'
  
  if question is not None:
    return question

  return text

def sublist_indices(lst, sublst):
  sub_len = len(sublst)
  for i in range(len(lst) - sub_len + 1):
    if torch.equal(lst[i:i+sub_len], sublst):
      return i, i + sub_len  # start index, end index
  
  assert False

def store_reasoning_last_action(dataset: str):
  '''Encode all reasoning traces, store the KV values corresponding to the last agentic action in the reasoning trace'''

  model_id = 'meta-llama/Llama-3.1-8B-Instruct'
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.bfloat16)
  config: LlamaConfig = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_id)
  emb: LlamaRotaryEmbedding = LlamaRotaryEmbedding(config=config).to(device=model.device, dtype=torch.float32)

  print(model_id)

  logs = read_json(f'./data/{dataset}/preds/log.json')

  log_kvs = []

  for log in tqdm(logs):
    # use only assistant messages
    log = [x for i, x in enumerate(log[0]) if i % 2 == 1]

    action = extract_action(log[-1]).replace('<|eot_id|>', '')
    
    # rsplit so we are splitting on the last "last action" (if it was repeated multiple times)
    assert len(log[-1].rsplit(action, 1)) <= 2
    # if not the entire reasoning trace
    if action != log[-1].replace('<|eot_id|>', ''):
      log_last: str = log[-1].rsplit(action, 1)
      # remove the newlines right before and after action
      # (adding the newline right in front of action so it's possible to extract the action)
      # \n seems to be independent from <
      # but \n sticks to the end of >, sp >\n\n is a token
      log_last = log_last[0].rstrip() + '\n' + action + '\n\n' + log_last[1].lstrip()
    else:
      log_last = log[-1].replace(action, action + '\n\n')
    
    log_last = log_last.replace('<|eot_id|>', '')
    log_except_last = '\n\n'.join(log[:-1]).replace('<|eot_id|>', '')
    reasoning_trace = log_except_last + '\n\n' + log_last

    assert '<|eot_id|>' not in reasoning_trace

    kv = get_log_kv(reasoning_trace, tokenizer, model, emb)

    # find the token start and end_idx corresponding to the last action
    input_ids_action = tokenizer.encode(action + '\n\n', return_tensors='pt', add_special_tokens=False)
    input_ids_last = tokenizer.encode(log_last, return_tensors='pt', add_special_tokens=False)
    input_ids_except_last = tokenizer.encode(log_except_last + '\n\n', return_tensors='pt', add_special_tokens=False)

    start_idx, end_idx = sublist_indices(input_ids_last[0], input_ids_action[0])
    
    # from input_ids_last, find the start and end_idx corresponding to input_ids_action
    # then add it to the start_idx of input_ids_last
    last_start_idx = input_ids_except_last.shape[1]
    start_idx += last_start_idx
    end_idx += last_start_idx

    input_ids_reasoning_trace = tokenizer.encode(reasoning_trace, return_tensors='pt', add_special_tokens=False)
    assert tokenizer.decode(input_ids_reasoning_trace[0][start_idx:end_idx]) == action + '\n\n'
    assert '<|eot_id|>' not in action

    kv._seen_tokens = end_idx - start_idx
    num_layers = len(kv)
    for layer_idx in range(num_layers):
      # slicing only return view, not changing the original tensor
      # pickle saves the underlying storage
      kv.key_cache[layer_idx] = kv.key_cache[layer_idx][:, :, start_idx:end_idx, :].clone().contiguous()
      kv.value_cache[layer_idx] = kv.value_cache[layer_idx][:, :, start_idx:end_idx, :].clone().contiguous()

      assert kv.key_cache[layer_idx].shape == kv.value_cache[layer_idx].shape
      assert kv.key_cache[layer_idx].shape[2] == kv._seen_tokens == input_ids_action.shape[1]

    kv.key_cache = [x.cpu() for x in kv.key_cache]
    kv.value_cache = [x.cpu() for x in kv.value_cache]

    log_kvs.append(kv)
  
  write_pickle(log_kvs, f'{STORE_PREFIX}/kv/{dataset}/reasoning_last_action.pkl')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--dataset', type=str)
  args = parser.parse_args()

  # choose one encode-storage strategy from below
  # default is store_reasoning_offline(args.dataset, last_num=1)
  store_reasoning_offline(args.dataset, last_num=1)
  # store_reasoning_last_as_context(args.dataset)
  # store_reasoning_all(args.dataset)
  # store_reasoning_last_action(args.dataset)