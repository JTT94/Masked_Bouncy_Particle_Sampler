import os, sys
import numpy as np
import pandas as pd
import re
from src.utils.serialize import unpickle_obj, load_json
from src.utils import print_verbose

def states_fp_constructor(iteration, output_dir):
    states_fp = os.path.join(output_dir, "states_iteration{0}.pickle".format(iteration))
    return states_fp

def output_chain_fp_constructor(iteration, group_id, output_dir):
    output_fp = os.path.join(output_dir,"sub_results_iter{0}_group{1}.pickle".format(iteration, group_id))
    return output_fp

class OutputReader(object):

    def __init__(self, masked_bps_obj):
        self.masked_bps_obj = masked_bps_obj

    def set_output_dir(self, directory):
        self.output_dir = directory

    def read_output(self, directory=None, verbose=False, inplace = True):

        print_verbose("Check directory", verbose=verbose)
        if directory is not None:
            self.set_output_dir(directory)

        if (directory is None) & (self.output_dir is None):
            print("Set directory first")
        print_verbose("Read output from directory: {0}".format(self.output_dir), verbose=verbose)

        mlbps = self.masked_bps_obj
        output_dir = self.output_dir

        all_output = {}
        for i in range(mlbps.d):
            all_output[i] = {}
            all_output[i]['x'] = []
            all_output[i]['v'] = []
            all_output[i]['t'] = []
            all_output[i]['mask'] = []
        all_output['fg'] = []
        all_output['refresh_times'] = []

        # sort files
        print_verbose("Check files", verbose=verbose)
        output_fps = [fp for fp in os.listdir(output_dir) if "group" in fp]
        iter_groups = [re.findall(r'\d+', fp) for fp in output_fps]
        iter_groups = pd.DataFrame([[int(iteration), int(group)] for iteration, group in iter_groups],
                                   columns=['iteration', 'group'])


        # get num iterations
        iterations = np.sort(iter_groups.iteration.unique())
        print_verbose("Number of parallel iterations: {0}".format(len(iterations)), verbose=verbose)

        # for each iteration grab files
        print_verbose("Read output files", verbose=verbose)
        for iteration in iterations:
            print_verbose("Iteration: {0}".format(iteration), verbose=verbose)
            states_fp = states_fp_constructor(iteration, output_dir)
            x, v, t, mask, fg_index = unpickle_obj(states_fp)
            factor_indices = mlbps.factor_graph_list[fg_index].factor_indices

            for i in range(mlbps.d):
                all_output[i]['x'].append(x[i])
                all_output[i]['v'].append(v[i])
                all_output[i]['t'].append(t[i])
                all_output[i]['mask'].append(mask[i])
            all_output['fg'].append(fg_index)
            all_output['refresh_times'].append(t[0])

            # get factor groups
            factor_groups = mlbps.split_mask_fn(factor_indices, mask)

            # sort param / group allocations
            params_by_group = [np.unique(np.concatenate([factor_indices[f] for f in group])) for group in factor_groups]
            #max_params_by_group = np.array(
            #    [np.max(np.concatenate([factor_indices[f] for f in group])) for group in factor_groups])
            #cut_points = np.array(
            #    [max_params_by_group[i] - max_params_by_group[i - 1] if i > 0 else max_params_by_group[i] for i in
            #     range(len(max_params_by_group))])
            #group_by_param = np.concatenate([np.repeat(i, cut_points[i]) for i in range(len(cut_points))])

            # get factor groups
            factor_groups = mlbps.split_mask_fn(factor_indices, mask)
            groups = np.sort(iter_groups.loc[iter_groups.iteration == iteration, 'group'].values)
            num_groups = len(factor_groups)
            num_iterations = 0
            # get output
            for group_id in range(num_groups):
                output_chain_fp = output_chain_fp_constructor(iteration, group_id, output_dir)
                res = unpickle_obj(output_chain_fp)
                num_iterations += np.shape(res)[0]
                active_params = np.where(mask[params_by_group[group_id]] == 1)[0]
                for reset_param_id in active_params:
                    param_id = params_by_group[group_id][reset_param_id]
                    all_output[param_id]['x'].extend(list(res[:, 0, reset_param_id]))
                    all_output[param_id]['v'].extend(list(res[:, 1, reset_param_id]))
                    all_output[param_id]['t'].extend(list(res[:, 2, reset_param_id]))
                    all_output[param_id]['mask'].extend(list(res[:, 3, reset_param_id]))

        if inplace:
            self.output = all_output
        else:
            return all_output, num_iterations

    def extract_params(self, param_id):
        x = np.array(self.output[param_id]['x'])
        v = np.array(self.output[param_id]['v'])
        t = np.array(self.output[param_id]['t'])
        mask = np.array(self.output[param_id]['mask'])

        return x,v,t,mask


