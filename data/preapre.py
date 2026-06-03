# this is the tokenize stage (text inn and stream of integers out)

import os
import pickle
import urllib.request

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
imput_file_path = os.path.join(here, "input.txt")

# downloading the dataset
if not os.path.exists(imput_file_path):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print("downloading Tiny shakespeare")
    urllib.request.urlretrieve(url, imput_file_path)

with open(input_file_path, "r") as f:
    data = f.read()
print(f"length of dataset in charcters: {len(data):,}")


# builddddddddddd the vocab for ever uniquue charcter sorted for determinismmmm
