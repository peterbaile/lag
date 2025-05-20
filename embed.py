from tiger_utils import read_json

from utils import embed, serialize_doc

if __name__ == '__main__':  
  dataset = 'wikihop'

  # embed documents
  docs = read_json(f'./data/{dataset}/dev_docs.json')

  if dataset == 'musique':
    docs_serialized = [serialize_doc(doc=docs[doc_id]) for doc_id in docs]
  elif dataset == 'wikihop':
    docs_serialized = [serialize_doc(doc_title=doc_id, doc_content=docs[doc_id]) for doc_id in docs]
  
  embed(docs_serialized, 'snowflake', f'./data/{dataset}/embeds/doc.npy')

  # embed questions
  qs = read_json(f'./data/{dataset}/dev.json')
  qs = [q['question'] for q in qs]
  embed(qs, 'snowflake', f'./data/{dataset}/embeds/q.npy', is_query=True)

  # ----------obsolete-----------------
  # compute sim scores
  # doc_embeds = torch.from_numpy(np.load(f'./data/musique/embeds/doc.npy'))
  # q_embeds = torch.from_numpy(np.load(f'./data/musique/embeds/q.npy'))

  # sim_scores = cosine_sim(q_embeds, doc_embeds)
  
  # corpus_docs = read_json(f'./data/musique/dev_docs.json')
  # corpus_doc_ids = np.array(list(corpus_docs.keys()))

  # preds = corpus_doc_ids[sim_scores.topk(k=50, dim=-1).indices].tolist()
  # write_json(preds, f'./data/musique/embeds/preds.json')

