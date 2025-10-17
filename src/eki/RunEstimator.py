import numpy as np
import matplotlib as mpl
mpl.rcParams.update({'text.usetex': False, 'text.latex.preamble': '\\usepackage{gensymb}',})
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os
import sys
import time
import argparse
import yaml
import pickle
import Optimizer_EKI_np
import pandas as pd

# Functions
def _parse():
    parser = argparse.ArgumentParser(description='Run EKI')
    parser.add_argument('input_config', help='Check input_config')
    parser.add_argument('input_data', help='Name of input_data')
    return parser.parse_args()

def _read_file(input_config, input_data):
    with open(input_config, 'r') as config:
        input_config = yaml.load(config, yaml.SafeLoader)
    with open(input_data, 'r') as data:
        input_data = yaml.load(data, yaml.SafeLoader)
    return input_config, input_data

def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()
        file.write("\n")
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
        file.write("\n")
    file.flush()

def save_results(dir_out, results):
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    file_out = dir_out +'/'+dir_out + '.p'
    pickle.dump(results, open(file_out, 'wb'))


# #############################################################################
# # Main executable
# #############################################################################


if __name__ == "__main__":
    # Import shared memory loader (replaces YAML file parsing)
    from Model_Connection_np_Ensemble import load_config_from_shared_memory

    args = _parse()

    # Load configuration from shared memory instead of files
    (input_config, input_data) = load_config_from_shared_memory()

# Optimization repeat, Ensemble * Iteration Time chek
Checktime_List = []
Optimization_list = input_config['Optimizer_order']
for op in range(len(Optimization_list)):
    opt = Optimization_list[op]
    Checktime_list1 = []
    Checktime_list2 = []
    print('Start:' + opt)
    for e in range(input_config['nrepeat']):
        (input_config, input_data) = load_config_from_shared_memory()
        input_config['Optimizer'] = opt
        input_config['sample'] = input_config['sample_ctrl'] * (e+1)
        print('Sample:' + str(input_config['sample']))
        sample = input_config['sample']

        # Run and Plot results
        ave = []
        err = []
        Best_List = []
        Misfits_List=[]
        Discrepancy_bools_List=[]
        Residual_bools_List=[]
        Residuals_List=[]
        Misfit_List=[]
        Discrepancy_bool_List=[]
        Residual_bool_List=[]
        Residual_List=[]
        Noise_List=[]
        EnsXiter_List = []
        Diff_List = []
        Time_List = []
        Info_list = []
        receptor_range = input_data['nreceptor']
        t1 = time.time()
        for i in progressbar(range(1, receptor_range+1), "Computing: ", 40):
            (input_config, input_data) = load_config_from_shared_memory()
            input_config['Optimizer'] = opt
            input_config['sample'] = input_config['sample_ctrl'] * (e+1)
            posterior0 = None
            posterior_iter0 = None
            info_list = []
            misfit_list = []
            discrepancy_bool_list = []
            residual_bool_list = []
            residual_list = []
            noise_list = []
            ensXiter_list = []
            diff_list = []
            misfits_list = []
            discrepancy_bools_list = []
            residual_bools_list = []
            residuals_list = []

            t2i = time.time()
            if input_config['Receptor_Increment'] == 'Off':
                input_data['nreceptor'] = receptor_range
                print(f'receptor:', input_data['nreceptor'])
            elif input_config['Receptor_Increment'] == 'On':
                input_data['nreceptor'] = i
                print(f'receptor:', input_data['nreceptor'])
            else:
                print('Check the number of receptor')
                break

            # v1.0: Always use GPU (CuPy) for inverse model
            posterior0, posterior_iter0, info_list, misfit_list, discrepancy_bool_list, residual_bool_list, residual_list, noise_list, ensXiter_list, diff_list, misfits_list, discrepancy_bools_list, residual_bools_list, residuals_list = Optimizer_EKI_np.Run(input_config, input_data)

            Info_list=info_list
            posterior = posterior0.copy()
            Best_List.append(posterior_iter0.copy())
            Misfits_List.append(misfits_list.copy())
            Discrepancy_bools_List.append(discrepancy_bools_list.copy())
            Residual_bools_List.append(residual_bools_list.copy())
            Residuals_List.append(residuals_list.copy())
            Misfit_List.append(misfit_list.copy())
            Discrepancy_bool_List.append(discrepancy_bool_list.copy())
            Residual_bool_List.append(residual_bool_list.copy())
            Residual_List.append(residual_list.copy())
            Noise_List.append(noise_list.copy())
            EnsXiter_list = np.array(ensXiter_list.copy())
            EnsXiter = 0 if np.nonzero(EnsXiter_list)[0].size == 0 else EnsXiter_list[np.nonzero(EnsXiter_list)[0][0]]
            EnsXiter_List.append(EnsXiter)
            Diff_List.append(diff_list.copy())
            ave.append(np.mean(posterior,1))
            err.append(np.std(posterior,1))
            if input_config['Receptor_Increment'] == 'Off':
                Best_Iter0 = np.mean(np.array(Best_List[0]), axis=2)
                Best_Iter0_std = np.std(np.array(Best_List[0]), axis=2)
            elif input_config['Receptor_Increment'] == 'On':
                Best_Iter0 = None
                Best_Iter0_std = None
                Best_Iter0 = np.mean(np.array(Best_List[-1]), axis=2)
                Best_Iter0_std = np.std(np.array(Best_List[-1]), axis=2)
            else:   
                print('Check the number of receptor_increment')
                break

            if input_data['Source_location'] == 'Fixed':
                Best_Iter_reshape = None
                Best_Iter_std_reshape = None
                Best_Iter_reshape = Best_Iter0[-1].reshape([input_data['nsource'],int(input_data['time']*24/input_data['inverse_time_interval'])])
                Best_Iter_std_reshape = Best_Iter0_std[-1].reshape([input_data['nsource'],int(input_data['time']*24/input_data['inverse_time_interval'])])

            print(i/(receptor_range)*100)
            t2 = time.time()
            Time_List.append(t2-t2i)
            print(f"Time:", t2-t1)
            if input_config['Receptor_Increment'] == 'Off':
                break  # Exit receptor loop but continue the program
                # sys.exit()  # Don't exit the entire program!
        t3 = time.time()
        print(f"Time:", t3-t1)

        nreceptor = []
        seepoint = receptor_range
        for i in range(0, seepoint):
            nreceptor.append(i+1)

        if input_config['Receptor_Increment'] == 'Off':
            list_index = 0  # Only one element in list when Receptor_Increment='Off'
        else:
            list_index = receptor_range - 1  # Use last element when incrementing receptors

        receptorPoint = receptor_range
        iterations = [i+1 for i in range(input_config['iteration'])]

        # Check if Best_List has valid data before accessing
        if len(Best_List) == 0:
            print("[ERROR] Best_List is empty - no results to process")
            continue

        if input_config['Optimizer'] == 'EKI':
            Best_Iter = np.mean(np.array(Best_List[list_index]), axis=2)
            Best_Iter_std = np.std(np.array(Best_List[list_index]), axis=2)
            Residuals_Iter = np.array(Residuals_List[list_index][1:])
        else:
            Best_Iter = np.array(Best_List[list_index])
            Residuals_Iter = np.array(Residuals_List[list_index][1:])

        Checktime_list1.append([nreceptor, Time_List, EnsXiter_List])