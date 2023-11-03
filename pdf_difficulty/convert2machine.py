import os
import pickle

import numpy as np
import torch


def load_binary(name_file):
    data = None
    with open(name_file, 'rb') as fp:
        data = pickle.load(fp)
    return data

def save_binary(dictionary, name_file):
    with open(name_file, 'wb') as fp:
        pickle.dump(dictionary, fp, protocol=pickle.HIGHEST_PROTOCOL)

def convert2bootleg(path_score):
    if not os.path.exists("tmp"):
        os.mkdir(f"tmp")
    if not os.path.exists("pred"):
        os.mkdir(f"pred")
    tmp_path = "tmp/prediction.png"
    prediction_path = "pred/prediction.pickle"
    command = f"python3 pdf_difficulty/getBootleg.py '{path_score}' '{tmp_path}' '{prediction_path}' prediction.erlr"
    os.system(command)
    return prediction_path


def unhashfcn(numb):
    # Decodes bootleg array to int to reduce memory
    bootleg_array = []
    while numb != 0 and numb != 1:
        bootleg_array.append(numb % 2)
        numb = numb // 2
        if numb == 1:
            bootleg_array.append(1)
    bootleg_array.reverse()
    return [0 for _ in range(62 - len(bootleg_array))] + bootleg_array


def get_bootleg_matrix(bootleg_path):
    embedding_bootleg = load_binary(bootleg_path)
    embedding = torch.Tensor(
        np.array([unhashfcn(emb_note) for emb_page in embedding_bootleg for emb_note in emb_page])
    )
    return embedding


def convert2machine(path_score):
    bootleg_path = convert2bootleg(path_score)
    binary_matrix = get_bootleg_matrix(bootleg_path)
    os.remove(bootleg_path)
    return binary_matrix



