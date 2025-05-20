from tiger_utils import (
  read_json, write_json, split_inputs_by_interval, dedup_list, read_pickle, merge
)
import re
from tqdm import tqdm
import argparse
import os
import torch.nn.functional as F
import torch
import numpy as np
from typing import Literal
from sentence_transformers import SentenceTransformer
from transformers import set_seed

from utils import embed, serialize_doc, STORE_PREFIX
from store_log_kv import get_reasoning_kv, extract_action
from _agentic_utils import (
  get_reasoning_prompt, get_qa_prompt,
  extract_query_reasoning, extract_query_qa,
  merge_log_kv,
  load_model, get_generation_config
)
from cheatsheet import get_cheatsheet

def retrieve(query, query_embed, corpus_embed, corpus_ids, embed_model, map_name, _k, state):
  if query_embed is None:
    query_embed = embed([query], 'snowflake', None, hide_progress=True, is_query=True, model=embed_model)

  scores = F.cosine_similarity(query_embed, corpus_embed, dim=-1)
  top_objects = corpus_ids[scores.topk(k=min(len(corpus_embed), _k)).indices]
  state[map_name][query] = [top_objects.tolist(), 0]

  return query_embed

def step(state, output, embed_model, object_embeds, object_ids, log_embeds, log_ids):
  if '<ans>' in output and '</ans>' in output:
    tmp_ans = re.findall(r'<ans>(.*?)</ans>', output)[-1]
    # only stop if the ans is not empty
    if tmp_ans.strip() != '':
      return state, True
  
  if TASK_TYPE == 'qa':
    query = extract_query_qa(output)
  elif TASK_TYPE == 'reasoning':
    query = extract_query_reasoning(output)

  query_embed = None

  # retrieve relevant documents
  if TASK_TYPE == 'qa':
    if query not in state['doc_map']:
      query_embed = retrieve(query, query_embed, object_embeds, object_ids, embed_model, 'doc_map', NUM_OBJECTS * NUM_STEPS, state)
    
    state['doc_map'][query][1] += NUM_OBJECTS

  # retrieve relevant logs
  # forces that logs are only retrieved in the first round
  # if len(state['log_map']) == 0:
  if query not in state['log_map']:
    if len(log_embeds) != 0:
      retrieve(query, query_embed, log_embeds, log_ids, embed_model, 'log_map', NUM_LOGS * NUM_STEPS, state)
  
  if query in state['log_map']:
    state['log_map'][query][1] += NUM_LOGS
    # TODO: force only a fixed number of log is used
    # state['log_map'][query][1] = NUM_LOGS
  
  return state, False

def format_objects(state, corpus_docs):
  # serialize objects
  relevant_objects = []
  doc_map = state['doc_map']
  for query in doc_map:
    objects, end_idx = doc_map[query]
    relevant_objects += objects[:end_idx]
  
  relevant_objects = dedup_list(relevant_objects)

  if dataset == 'musique':
    serialized_objects = [serialize_doc(doc=corpus_docs[o]) for o in relevant_objects]
  elif dataset == 'wikihop':
    serialized_objects = [serialize_doc(doc_title=o, doc_content=corpus_docs[o]) for o in relevant_objects]
  relevant_objects = '\n\n'.join(serialized_objects)

  return relevant_objects

def format_logs(q, state, tokenizer, model, emb, corpus_logs, corpus_logs_kv, corpus_logs_repr, mode: str):
  relevant_logs, relevant_logs_text, relevant_logs_kv, relevant_logs_input_ids = [], None, None, None
  cheatsheet_prompt, cheatsheet_output = None, None

  # serialize logs
  if len(state['log_map']) >= 1:
    log_map = state['log_map']
    for query in log_map:
      q_ids, end_idx = log_map[query]
      relevant_logs += q_ids[:end_idx]
    
    relevant_logs = dedup_list(relevant_logs)

    # concat all logs
    if mode.startswith('log') and mode != 'log_cheatsheet':
      relevant_logs_text = [corpus_logs_repr[q_id] for q_id in relevant_logs]
      relevant_logs_input_ids = [tokenizer.encode(x, return_tensors='pt', add_special_tokens=False) for x in relevant_logs_text]
      relevant_logs_input_ids = torch.hstack(relevant_logs_input_ids)
    # this is using the reflection output
    elif mode == 'log_cheatsheet':
      relevant_logs_input_ids, cheatsheet_prompt, cheatsheet_output = get_cheatsheet(q, relevant_logs, corpus_logs, corpus_logs_repr, tokenizer, model)
    
    if mode.startswith('log_kv'):
      relevant_logs_kv = merge_log_kv(relevant_logs, corpus_logs_kv, emb)
      assert relevant_logs_input_ids.shape[1] == relevant_logs_kv.key_cache[0].shape[-2]
  
  return relevant_logs_input_ids, relevant_logs_kv, cheatsheet_prompt, cheatsheet_output

def get_user_prompt(
  q, state, corpus_docs,
  corpus_logs, corpus_logs_kv, corpus_logs_repr,
  tokenizer, model, emb, instruction_input_ids, output1, mode, log_extract_output=None
):
  relevant_logs_input_ids, relevant_logs_kv, cheatsheet_prompt, cheatsheet_output = format_logs(
    q, state, tokenizer, model, emb,
    corpus_logs, corpus_logs_kv, corpus_logs_repr, mode
  )
  
  if TASK_TYPE == 'qa':
    relevant_objects = format_objects(state, corpus_docs)
    user_str = get_qa_prompt(q, relevant_objects)
  elif TASK_TYPE == 'reasoning':
    user_str = get_reasoning_prompt(dataset, q, output1)

  return user_str, relevant_logs_kv, relevant_logs_input_ids, cheatsheet_prompt, cheatsheet_output

# TODO: convert each log entry into a class
# def process_log_single(trace):
#   log_store.append(get_reasoning_kv(trace, tokenizer, model, emb))

def init_log(dataset, embed_model):
  log_fn = f'./data/{dataset}/preds/log.json'
  log_repr_fn=f'./data/{dataset}/preds/log.json'

  if mode == 'log_kv':
    log_kv_fn = f'{STORE_PREFIX}/kv/{dataset}/reasoning_last_{KV_LAST_NUM}.pkl'
  elif mode == 'log_kv_last_as_context':
    log_kv_fn = f'{STORE_PREFIX}/kv/{dataset}/reasoning_last_as_context.pkl'
  elif mode == 'log_kv_all':
    log_kv_fn = f'{STORE_PREFIX}/kv/{dataset}/reasoning_all.pkl'
  elif mode == 'log_kv_last_action':
    log_kv_fn = f'{STORE_PREFIX}/kv/{dataset}/reasoning_last_action.pkl'

  if not os.path.isfile(log_fn) or mode == 'base':
    return [], {}, {}, [], {}
  
  logs = read_json(log_fn)
  logs_repr = read_json(log_repr_fn)
  if mode.startswith('log_kv'):
    print(f'reading log KV from {log_kv_fn}')
    logs_kv = read_pickle(log_kv_fn)

  # filtered_logs_repr includes the actual tokens of the logs
  filtered_logs, filtered_logs_kv, filtered_logs_repr = {}, {}, {}

  # we do not need to exclude the last one (current question)
  # (we do it at the beginning of the question so log for the current question is not present)
  for log_idx, log in enumerate(logs):
    if log == '':
      continue
    
    # TODO: remove <eot_id> from reasoning trace in the future
    # log[0] is the trace, log[1] is state
    question_text = log[1]['question']
    if mode == 'log_text_all':
      reasoning_trace = [x for i, x in enumerate(log[0]) if i % 2 == 1]
      reasoning_trace = '\n\n'.join(reasoning_trace)
    else:
      reasoning_trace = log[0][-1]

    # use int as ID to allow indexing into the KV values
    filtered_logs[log_idx] = {'question': question_text, 'reasoning_trace': reasoning_trace}
    
    if mode.startswith('log_kv'):
      filtered_logs_kv[log_idx] = logs_kv[log_idx]
    
    filtered_logs_repr[log_idx] = [x for i, x in enumerate(logs_repr[log_idx][0]) if i % 2 == 1]
    # use the last assistant message(s)
    if mode == 'log_kv_last_action':
      filtered_logs_repr[log_idx] = extract_action(filtered_logs_repr[log_idx][-1]).replace('<|eot_id|>', '') + '\n\n'
    elif mode in ['log_text_all', 'log_kv_all']:
      filtered_logs_repr[log_idx] = '\n\n'.join(filtered_logs_repr[log_idx]).replace('<|eot_id|>', '') + '\n\n'
    else:
      filtered_logs_repr[log_idx] = '\n\n'.join(filtered_logs_repr[log_idx][-KV_LAST_NUM:]).replace('<|eot_id|>', '') + '\n\n'
  
  if len(filtered_logs) >= 1:
    filtered_logs_serialized = [f"{filtered_logs[q_id]['question']}\n{filtered_logs[q_id]['reasoning_trace']}" for q_id in filtered_logs]
    filtered_log_embeds = embed(filtered_logs_serialized, 'snowflake', None, hide_progress=True, model=embed_model)
  else:
    filtered_log_embeds = []
  
  print(f'#logs including answers/ #all logs: {len(filtered_logs)} / {len(logs)}')

  filtered_log_ids = np.array(list(filtered_logs.keys()))

  return filtered_log_embeds, filtered_logs, filtered_logs_kv, filtered_log_ids, filtered_logs_repr

def execute(
  dataset: str, model_name: str, partition, _fn: str,
  mode: Literal['base', 'log_text', 'log_kv', 'log_text_all', 'log_kv_last_as_context', 'log_cheatsheet', 'log_kv_last_action'],
  dynamic_store: bool, train: bool
):
  fn = f'{_fn}_{partition}.json'
  print(fn)

  if dynamic_store:
    log_store = []
  
  results = []
  if os.path.isfile(fn):
    results = read_json(fn)

  if train:
    print('train split')
    qs = read_json(f'./data/{dataset}/dev_train.json')
  else:
    print('test split')
    qs = read_json(f'./data/{dataset}/dev_test.json')

  if dynamic_store:
    qs = qs[:50]
  else:  
    qs = split_inputs_by_interval(qs, num_partitions, partition)

  corpus_embeds, corpus_docs, corpus_doc_ids = None, None, None
  if TASK_TYPE == 'qa':
    corpus_embeds = torch.from_numpy(np.load(f'./data/{dataset}/embeds/doc.npy'))
    corpus_docs = read_json(f'./data/{dataset}/dev_docs.json')
    corpus_doc_ids = np.array(list(corpus_docs.keys()))
  
  tokenizer, model, emb = load_model()
  generation_config = get_generation_config(tokenizer)
  embed_model = SentenceTransformer('Snowflake/snowflake-arctic-embed-m-v2.0', trust_remote_code=True).cuda()

  # static log store
  if not dynamic_store:
    log_embeds, corpus_logs, corpus_logs_kv, corpus_log_ids, corpus_logs_repr = init_log(dataset, embed_model)
  else:
    log_embeds, corpus_logs, corpus_logs_kv, corpus_log_ids, corpus_logs_repr = [], {}, {}, [], {}

  # instruction_input_ids = read_pickle(f'./data/{dataset}/kv/instruction_input_ids.pkl')
  instruction_input_ids = None

  for q_idx, q in enumerate(tqdm(qs)):
    if q_idx < len(results):
      continue
        
    state = {'doc_map': {}, 'log_map': {}, 'question': q['question']}
    # initialize state with some logs and documents (without tags, step just takes the entire input as query)
    state, _ = step(state, q['question'], embed_model, corpus_embeds, corpus_doc_ids, log_embeds, corpus_log_ids)
    num_steps, trace, cheatsheet_trace = 0, [], []

    # for reasoning task
    output1 = ''
    
    try:
      while num_steps < NUM_STEPS:
        user_prompt, kv_cache, kv_input_ids, cheatsheet_prompt, cheatsheet_output = get_user_prompt(
          q, state, corpus_docs, corpus_logs, corpus_logs_kv, corpus_logs_repr,
          tokenizer, model, emb, instruction_input_ids, output1, mode
        )

        input_ids = tokenizer.encode(user_prompt, return_tensors='pt', add_special_tokens=False).to(model.device)
        
        if mode.startswith('log'):
          kv_input_ids = kv_input_ids.to(model.device)
          input_ids = torch.hstack([kv_input_ids, input_ids])

        prompt1 = tokenizer.batch_decode(input_ids)[0]

        if not mode.startswith('log_kv'):
          output1 = model.generate(input_ids=input_ids, generation_config=generation_config, eos_token_id=[tokenizer.eos_token_id], tokenizer=tokenizer)
        else:
          kv_cache.key_cache = [x.to(device=model.device) for x in kv_cache.key_cache]
          kv_cache.value_cache = [x.to(device=model.device) for x in kv_cache.value_cache]
          output1 = model.generate(input_ids=input_ids, generation_config=generation_config, eos_token_id=[tokenizer.eos_token_id], tokenizer=tokenizer, past_key_values=kv_cache, use_cache=True)
        
        input_length = input_ids.size(-1)
        output1 = tokenizer.decode(output1[0][input_length:], skip_special_tokens=False)

        trace += [prompt1, output1]
        if mode == 'log_cheatsheet':
          cheatsheet_trace += [cheatsheet_prompt, cheatsheet_output]

        # input_tokens += get_num_tokens(prompt1, llama_tokenizer=tokenizer)
        # output_tokens += get_num_tokens(output1, llama_tokenizer=tokenizer)

        num_steps += 1

        state, terminate = step(state, output1, embed_model, corpus_embeds, corpus_doc_ids, log_embeds, corpus_log_ids)
        
        # break the loop (avoid adding objects of last round)
        if terminate or num_steps == NUM_STEPS:
          break

      results.append([trace, state, num_steps, 0, 0, cheatsheet_trace])
      
      if dynamic_store:
        # TODO: update all information about the logs
        process_log_single()
    except Exception as e:
      print(e)
      results.append('')

    write_json(results, fn)


if __name__ == '__main__':
  set_seed(1234)

  parser = argparse.ArgumentParser()
  parser.add_argument('-p', '--partition', type=int)
  parser.add_argument('-m', '--mode', type=str)
  parser.add_argument('-d', '--dataset', type=str)
  args = parser.parse_args()

  dataset = args.dataset

  assert dataset in ['musique', 'wikihop', 'gpqa', 'mmlupro']

  model_name = 'llama8'
  num_partitions = 8

  # at each step we retrieve top-2 documents, top-3 logs, and use KV values corresponding to the last reasoning trace
  NUM_OBJECTS, NUM_LOGS, KV_LAST_NUM = 2, 3, 1
  if dataset in ['musique', 'wikihop']:
    NUM_STEPS, TASK_TYPE = 8, 'qa'
  elif dataset in ['gpqa', 'mmlupro']:
    NUM_STEPS, TASK_TYPE = 3, 'reasoning'

  mode: str = args.mode
  if mode == 'base':
    fn = f'./data/{dataset}/preds/{mode}'
  else:
    fn = f'./data/{dataset}/preds/{mode}_l{NUM_LOGS}'
  
  # run inference
  execute(dataset, model_name, args.partition, fn, mode, dynamic_store=False, train=False)

  # merge inference outputs
  # merge(num_partitions, f'./data/{dataset}/preds/base', 'json')
  # merge(num_partitions, f'./data/{dataset}/preds/log_text_l{NUM_LOGS}', 'json')
  # merge(num_partitions, f'./data/{dataset}/preds/log_text_all_l{NUM_LOGS}', 'json')
  # merge(num_partitions, f'./data/{dataset}/preds/log_kv_l{NUM_LOGS}', 'json')
  # merge(num_partitions, f'./data/{dataset}/preds/log_kv_last_as_context_l{NUM_LOGS}', 'json')
  # merge(num_partitions, f'./data/{dataset}/preds/log_cheatsheet_l{NUM_LOGS}', 'json')
  # merge(num_partitions, f'./data/{dataset}/preds/log_kv_all_l{NUM_LOGS}', 'json')
  # merge(num_partitions, f'./data/{dataset}/preds/log_kv_last_action_l{NUM_LOGS}', 'json')