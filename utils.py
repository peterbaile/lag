from typing import List, Union
import torch
import os
import numpy as np
from tqdm import tqdm

STORE_PREFIX = f'/home/ubuntu/efs'

def serialize_doc(doc=None, doc_title=None, doc_content=None):
  if doc_title is None and doc_content is None:
    doc_title, doc_content = doc['title'], doc['content']
  return f'Document title: {doc_title}\nDocument content: {doc_content}'

def embed(texts: List[str], model_name: str, fn: Union[str, None], is_query=False, hide_progress=False, model=None):
  BATCH_SIZE = 200

  if fn is not None and os.path.isfile(fn):
    return torch.from_numpy(np.load(fn))

  from sentence_transformers import SentenceTransformer

  if model is None:
    model = SentenceTransformer('Snowflake/snowflake-arctic-embed-m-v2.0', trust_remote_code=True).cuda()

  embeds = []
  for i in tqdm(range((len(texts)//BATCH_SIZE) + 1), disable=hide_progress):
    _texts = texts[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

    if len(_texts) == 0:
      break

    assert len(_texts) >= 1

    if model_name == 'uae':
      vec = model.encode([Prompts.C.format(text=text) for text in _texts], show_progress_bar=False) if is_query else model.encode(_texts, show_progress_bar=False)
    elif model_name == 'gte':
      vec = model.encode(_texts, normalize_embeddings=True, show_progress_bar=False)
    elif model_name == 'snowflake':
      vec = model.encode(_texts, prompt_name='query', show_progress_bar=False) if is_query else model.encode(_texts, show_progress_bar=False)
    # elif model_name == 'stella':
    #   vec = model.encode(_texts, prompt_name='s2p_query') if is_query else model.encode(_texts)
    elif model_name == 'bge':
      vec = model.encode([instruction + text for text in _texts] if is_query else _texts, normalize_embeddings=True, show_progress_bar=False)

    embeds.append(vec)
  
  embeds = np.vstack(embeds)

  if fn is not None:
    np.save(fn, embeds)
  
  embeds = torch.from_numpy(embeds)

  return embeds

def get_prompt(user_str):
  return f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_str}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'