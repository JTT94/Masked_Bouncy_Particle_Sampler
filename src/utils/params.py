import json
import hashlib


class Params(object):
    def __init__(self, param_dict):
        # store again as class valuesParams
        for param, val in param_dict.items():
            self.__dict__[param] = val
        self.param_names = list(param_dict.keys())

    def param_dict(self):
        return self.__dict__

    def save_to_file(self, file_path):
        with open(file_path, 'w') as fp:
            json.dump(self.param_dict(), fp)

    def __repr__(self):
        return str(self.param_dict())

    def __str__(self):
        return str(self.param_dict())


def hash_dict(dic):
    bts = bytes(dic.__str__(), 'utf-8')
    result = hashlib.md5(bts)
    return result.hexdigest()


def check_dicts(dic1, dic2, comparison="both"):
    if comparison in ['both', 'left']:
        for key in dic1.keys():
            if not dic1[key] == dic2[key]:
                return False
    if comparison in ['both', 'right']:
        for key in dic2.keys():
            if not dic1[key] == dic2[key]:
                return False
    return True
