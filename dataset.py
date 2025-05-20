import random
from tiger_utils import read_json, write_json

def split_dev(dataset):
  # 70 for train and 30 for test
  random.seed(0)
  qs = read_json(f'./data/{dataset}/dev.json')
  random.shuffle(qs)

  num_qs = len(qs)
  split = int(0.7 * num_qs)
  write_json(qs[:split], f'./data/{dataset}/dev_train.json')
  write_json(qs[split:], f'./data/{dataset}/dev_test.json')

if __name__ == '__main__':
  split_dev('mmlupro')