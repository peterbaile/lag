# 🔄 LAG: Log-Augmented Generation

<!-- If you find our code, or the paper helpful, please cite the paper

```

``` -->

## Overview
`agentic.py`: the main file that executes *LAG* and baselines

* **Existing standard agentic system**
  * `mode = base`: the standard ReAct-style agentic system that operates iteratively, deciding at each step whether to provide an answer or initiate a new reasoning action, which may include interactions with external tools
* **Existing reflection system**
  * `mode = log_cheatsheet`: an implementation of <a href="https://arxiv.org/pdf/2504.07952">Dynamic Cheatsheet</a>
* **Existing KV cache system**
  * `mode = log_kv_last_as_context`: an implementation of <a href="https://arxiv.org/abs/2409.15355">Block Attention</a> that encodes the last model response and stores the KV values corresponding to the tokens in that response
* **Ours: Log-augmented generation framework**
  * `mode = log_text_all`: log-augmented generation with logs represented as texts (all reasoning traces stored and provided to model)
  * `mode = log_text`: log-augmented generation with logs represented as texts (only last reasoning trace stored and provided to model)
  * `mode = log_kv` (default): log-augmented generation with logs represented as KV values (all reasoning traces encoded but only the last reasoning trace stored and provided to model)
  * `mode = log_kv_last_action`: log-augmented generation with logs represented as KV values (all reasoning traces encoded but only the last action stored and provided to model)


`store_log_kv.py`: encode and store the KV values with different strategies

`embed.py`: compute the embeddings of queries and documents for open-domain QA datasets


## Setup


#### File directory

We evaluated all methods on four datasets
* Knowledge-intensive setting: open-domain multi-hop QA datasets
  * Musique and 2WikiMultiHop
* Reasoning-intensive datasets: math and science QA datasets
  * GPQA and MMLU-Pro

A `data` directory is provided that includes one sub-directory for each dataset, structured as follows 
```
data/
  musique/
    dev.json # dev_train.json + dev_test.json
    dev_train.json # the split of questions used to construct the log store
    dev_test.json # the split of unseen questions
    (dev_docs.json) # only for multihop QA datasets (i.e., Musique and 2WikiMultiHop)
    preds/
      log.json # results of running standard agentic on dev_train.json
      train/
      test/
    (embeds/) # only for multihop QA datasets
      (doc.npy)
      (q.npy)
  wikihop/ # 2WikiMultiHop
    ...
  gpqa/
    ...
  mmlupro/
    ...
```

* `dev_train.json` and `dev_test.json` were constructed by doing a random 70/30 split from `dev.json`. The code can be found in `dataset.py`

#### Compute embeddings

If you are evaluating on the open-domain QA datasets (Musique and 2WikiMultiHop), you need to run

```
python embed.py
```

to obtain the embeddings for the documents (`doc.npy`) and questions (`q.npy`)

#### Obtain prior reasoning traces
Run the following to get `data/dataset_name/preds/base.json`

```
python agentic.py -m base -d dataset_name -p partition_index
```

Make sure `execute(..., train=True)` in the `main` function of `agentic.py` so inference is performed on `dev_train.py`

If you have multiple GPUs, you can partition tasks and run them in parallel. To do this, update `num_partitions` in `agentic.py` and run the above command for different partitions of the task.

```
CUDA_VISIBLE_DEVICES=0 python ... -p 0 & CUDA_VISIBLE_DEVICES=1 python ... -p 1 &
... &
CUDA_VISIBLE_DEVICES=num_partitions-1 python ... -p num_partitions - 1
```

Rename `data/dataset_name/preds/base.json` to `data/dataset_name/preds/log.json`

#### Obtain KV values of the logs
Choose a strategy in `store_log_kv.py` to obtain the KV values

```
python store_log_kv.py -d dataset_name
```

#### Log-augmented generation

Run the following to obtain results on log-augmented generation

```
python agentic.py -m log_kv -d dataset_name -p partition_index
```

* `execute(..., train=True)` in the `main` function of `agentic.py` to perform inference on the seen questions
* `execute(..., train=False)` to perform inference on the unseen questions

#### Work in progress

- [ ] Implement dynamic log store construction to build the log store on-the-fly rather than relying on a predefined split
- [ ] Support more open-source models

## Contact

If you have any questions or feedback, please send an email to peterbc@mit.edu.