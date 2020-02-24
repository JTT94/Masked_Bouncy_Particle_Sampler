import pickle
import json


def pickle_obj(obj, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle_obj(file_path):
    with open(file_path, 'rb') as handle:
        obj = pickle.load(handle)
    return obj


def load_json(file_path):
    with open(file_path, 'rb') as handle:
        obj = json.load(handle)
    return obj


def save_json(dict, file_path):
    with open(file_path, 'w') as fp:
        json.dump(dict, fp)

