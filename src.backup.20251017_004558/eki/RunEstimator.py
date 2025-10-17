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
# import Optimizer_EKI
import Optimizer_EKI_np
import datetime
import pandas as pd

import socket


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

# host = '127.0.0.1'  
# port = 65432        

# server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server_socket.bind((host, port))
# server_socket.listen(1)

# print(f"host {host}:{port} waiting")

# conn, addr = server_socket.accept()
# print(f"{addr}에서 연결되었습니다.")

# with conn:
#     while True:
#         data = conn.recv(1024)
#         if not data:
#             break
#         number = int(data.decode())
#         print(f"받은 숫자: {number}")

#         time.sleep(0.3)
#         number += 1
#         print(f"{number}를 클라이언트로 전송합니다.")
#         conn.sendall(str(number).encode())

# server_socket.close()

# #############################################################################
# # 소켓 종료
# #############################################################################


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
            if input_config['GPU_InverseModel'] == 'Off':
                print('Error')
                # posterior0, posterior_iter0, info_list, misfit_list, discrepancy_bool_list, residual_bool_list, residual_list, noise_list, ensXiter_list, diff_list, misfits_list, discrepancy_bools_list, residual_bools_list, residuals_list = Optimizer_EKI.Run(input_config, input_data)
            elif input_config['GPU_InverseModel'] == 'On':
                posterior0, posterior_iter0, info_list, misfit_list, discrepancy_bool_list, residual_bool_list, residual_list, noise_list, ensXiter_list, diff_list, misfits_list, discrepancy_bools_list, residual_bools_list, residuals_list = Optimizer_EKI_np.Run(input_config, input_data)
            else : print('Optimizer Error: Check GPU availability')
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
            elif input_data['Source_location'] == 'Single':
                Best_Iter_reshape = None
                Best_Iter_std_reshape = None
                Best_Iter_reshape_position = None
                Best_Iter_std_reshape_position = None
                Best_Iter_reshape = Best_Iter0[-1][3:].reshape([input_data['nsource'],int(input_data['time']*24/input_data['inverse_time_interval'])])
                Best_Iter_std_reshape = Best_Iter0_std[-1][3:].reshape([input_data['nsource'],int(input_data['time']*24/input_data['inverse_time_interval'])])
                Best_Iter_reshape_position = Best_Iter0[-1][:3]
                Best_Iter_std_reshape_position = Best_Iter0_std[-1][:3]
            elif input_data['Source_location'] == 'Multiple':
                Best_Iter_reshape = None
                Best_Iter_std_reshape = None
                Best_Iter_reshape = Best_Iter0[-1][:].reshape([input_data['nsource'],int(input_data['time']*24/input_data['inverse_time_interval'])+3])
                Best_Iter_std_reshape = Best_Iter0_std[-1][:].reshape([input_data['nsource'],int(input_data['time']*24/input_data['inverse_time_interval'])+3])

            def plot_iteration_graph(iterations, means, stds, file_path, csv_file_path):
                plt.figure(figsize=(10, 6))
                plt.plot(iterations, means, label='Mean', color='blue')
                plt.errorbar(iterations, means, yerr=stds, fmt='o', color='red', label='Standard Deviation')
                plt.xlabel('Iterations')
                plt.ylabel('Values')
                # plt.yscale('log')
                plt.title('Mean and Standard Deviation over Iterations')
                plt.legend()
                plt.grid(True)
                plt.savefig(file_path)
                plt.close()
                data = {
                        'Iterations': iterations,
                        'Mean': means,
                        'Standard Deviation': stds
                    }
                df = pd.DataFrame(data)
                df.to_csv(csv_file_path, index=False)

            def plot_time_range_graph(time_range, means, stds, file_path, csv_file_path):
                plt.figure(figsize=(10, 6))
                plt.plot(time_range, means, label='Mean', color='blue')
                plt.errorbar(time_range, means, yerr=stds, fmt='o', color='red', label='Standard Deviation')
                plt.xlabel('Time_range')
                plt.ylabel('Values')
                plt.title('Mean and Standard Deviation over time_range')
                plt.legend()
                plt.grid(True)
                plt.savefig(file_path)
                plt.close()
                data = {
                        'Time_range': time_range,
                        'Mean': means,
                        'Standard Deviation': stds
                    }
                df = pd.DataFrame(data)
                df.to_csv(csv_file_path, index=False)
            
            def plot_position_graph(means, stds, file_path, csv_file_path):
                plt.figure(figsize=(10, 6))
                plt.plot(['x','y','z'], means, label='Mean', color='blue')
                plt.errorbar(['x','y','z'], means, yerr=stds, fmt='o', color='red', label='Standard Deviation')
                plt.xlabel('Position')
                plt.ylabel('Values')
                plt.title('Mean and Standard Deviation over (x,y,z)')
                plt.legend()
                plt.grid(True)
                plt.savefig(file_path)
                plt.close()
                data = {
                        'Mean': means,
                        'Standard Deviation': stds
                    }
                df = pd.DataFrame(data)
                df.to_csv(csv_file_path, index=False)

            # output_path = f'./results_ens{sample}'
            # if not os.path.exists(output_path):
            #     os.makedirs(output_path)

            # DISABLED: Plot generation
            # current_nreceptor = input_data['nreceptor']
            # time_range = [input_data['time']*24*((i+1)/int(input_data['time']*24/input_data['inverse_time_interval'])) for i in range(int(input_data['time']*24/input_data['inverse_time_interval']))]
            # for s in range(np.array(Best_Iter0).shape[1]):
            #     iterations = [iter+1 for iter in range(len(Best_Iter0[:,s].tolist()))]
            #     plot_iteration_graph(iterations, Best_Iter0[:,s], Best_Iter0_std[:,s]*2.58/(len(Best_Iter0_std[:,s])**0.5), f'./results_ens{sample}/plot_receptor_{current_nreceptor}_all_source_{s}.png', f'./results_ens{sample}/plot_receptor_{current_nreceptor}_all_source_{s}.csv')
            # for s in range(input_data['nsource']):
            #     if input_data['Source_location'] == 'Fixed':
            #         plot_time_range_graph(time_range, Best_Iter_reshape[s,:], Best_Iter_std_reshape[s,:]*2.58/(len(Best_Iter_std_reshape[s,:])**0.5), f'./results_ens{sample}/plot_receptor_{current_nreceptor}_time_range_{s}.png', f'./results_ens{sample}/plot_receptor_{current_nreceptor}_time_range_{s}.csv')
            #     elif input_data['Source_location'] == 'Single':
            #         plot_time_range_graph(time_range, Best_Iter_reshape[s,:], Best_Iter_std_reshape[s,:]*2.58/(len(Best_Iter_std_reshape[s,:])**0.5), f'./results_ens{sample}/plot_receptor_{current_nreceptor}_time_range_{s}.png', f'./results_ens{sample}/plot_receptor_{current_nreceptor}_time_range_{s}.csv')
            #         plot_position_graph(Best_Iter_reshape_position, Best_Iter_std_reshape_position*2.58/(len(Best_Iter_std_reshape_position)**0.5), f'./results_ens{sample}/plot_receptor_{current_nreceptor}_position.png', f'./results_ens{sample}/plot_receptor_{current_nreceptor}_position.csv')
            #     elif input_data['Source_location'] == 'Multiple':
            #         plot_time_range_graph(time_range, Best_Iter_reshape[s,3:], Best_Iter_std_reshape[s,3:]*2.58/(len(Best_Iter_std_reshape[s,3:])**0.5), f'./results_ens{sample}/plot_receptor_{current_nreceptor}_time_range_{s}.png', f'./results_ens{sample}/plot_receptor_{current_nreceptor}_time_range_{s}.csv')
            #         plot_position_graph(Best_Iter_reshape[s,:3], Best_Iter_std_reshape[s,:3]*2.58/(len(Best_Iter_std_reshape[s,:3])**0.5), f'./results_ens{sample}/plot_receptor_{current_nreceptor}_position_{s}.png', f'./results_ens{sample}/plot_receptor_{current_nreceptor}_position_{s}.csv')
            
            print(i/(receptor_range)*100)
            t2 = time.time()
            Time_List.append(t2-t2i)
            print(f"Time:", t2-t1)
            if input_config['Receptor_Increment'] == 'Off':
                break  # Exit receptor loop but continue the program
                # sys.exit()  # Don't exit the entire program!
        t3 = time.time()
        print(f"Time:", t3-t1)

        # DISABLED: Directory and results saving
        # dir_str = input_config['Optimizer'] + '_nsource' + str(input_data['nsource']) + '_nreceptor' + str(input_data['nreceptor']) + '_sample'+str(input_config['sample'])+'_iteration'+str(input_config['iteration']) +'_err' + str(input_data['nreceptor_err'])
        # dir_out= 'results_{}'.format(dir_str)
        # os.makedirs(dir_out, exist_ok=True)

        # results = ave, err, Best_List, Misfits_List, Discrepancy_bools_List, Residual_bools_List, Residuals_List, Misfit_List, Discrepancy_bool_List, Residual_bool_List, Residual_List, Noise_List, EnsXiter_List, Diff_List, Time_List, Info_list
        # save_results(dir_out, results)

        nreceptor = []
        seepoint = receptor_range
        for i in range(0, seepoint):
            nreceptor.append(i+1)

        # FIXED: When Receptor_Increment='Off', only one element in Best_List
        # So we should use index 0, not receptorPoint-1
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