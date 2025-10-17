import numpy as np
import GaussianPuffClass_Rev_20240514 as Gpuff
import concurrent.futures

# Gaussian puff model
class Model(object):
    def __init__(self, input_config, input_data):
        self.name = 'gaussian_puff_model'
        self.input_data = input_data
        self.sample = input_config['sample']
        self.nsource = input_data['nsource']
        self.nreceptor = input_data['nreceptor']
        self.nreceptor_err = input_data['nreceptor_err']
        self.nreceptor_MDA = input_data['nreceptor_MDA']
        if input_data['receptor_position'] == []:
            input_data['receptor_position'] = [list(np.random.randint(low=[input_data['xreceptor_min'], input_data['yreceptor_min'], input_data['zreceptor_min']], high=[input_data['xreceptor_max']+1, input_data['yreceptor_max']+1, input_data['zreceptor_max']+1]).astype(float)) for _ in range(self.nreceptor)]

        # read the real data of sources
        self.real_state_init_list = []
        self.real_decay_list = []
        self.real_source_location_list = []
        self.real_dosecoeff_list = []
        self.total_real_state_list = []

        for s in range(self.nsource):
            actual_source = "Source_{0}".format(s+1)
            self.real_decay_list.append(input_data[actual_source][0])
            self.real_dosecoeff_list.append(input_data[actual_source][1])
            self.real_source_location_list.append(input_data[actual_source][2])
            self.real_state_init_list.append(input_data[actual_source][3])
            self.total_real_state_list.append(input_data[actual_source][2] + input_data[actual_source][3])
       
        self.real_decay = np.array(self.real_decay_list)
        self.real_source_location = np.array(self.real_source_location_list).T
        self.real_dosecoeff = np.array(self.real_dosecoeff_list)
        
        # read the prior data of sources
        self.state_init_list = []
        self.source_location_list = []
        self.state_std_list = []
        self.source_location_std_list = []
        self.decay_list = []
        self.total_state_list = []
        self.total_state_std_list = []
        for s in range(self.nsource):
            source = "Prior_Source_{0}".format(s+1)
            self.source_location_list.append(input_data[source][2][0])
            self.state_init_list.append(input_data[source][3][0])
            self.source_location_std_list.append((np.array(input_data[source][2][0])*input_data[source][2][1]).tolist())
            self.state_std_list.append((np.array(input_data[source][3][0])*input_data[source][3][1]).tolist())
            self.decay_list.append(input_data[source][0])
            self.total_state_list.append(input_data[source][2][0] + input_data[source][3][0])
            self.total_state_std_list.append((np.array(input_data[source][2][0])*input_data[source][2][1]).tolist() + (np.array(input_data[source][3][0])*input_data[source][3][1]).tolist())

        if input_data['Source_location'] == 'Fixed':
            self.real_state_init = np.hstack(self.real_state_init_list)
            self.state_init = np.hstack(self.state_init_list)
            self.state_std = np.hstack(self.state_std_list)
            self.nstate_partial = np.array(self.state_init_list).shape[1]
            self.source_location_case = 0
        elif input_data['Source_location'] == 'Single':
            self.real_state_init = np.hstack(self.real_source_location_list[0]+self.real_state_init_list)
            self.state_init = np.hstack(self.source_location_list[0]+self.state_init_list)
            self.state_std = np.hstack(self.source_location_std_list[0]+self.state_std_list)
            self.nstate_partial = np.array(self.state_init_list).shape[1]
            self.source_location_case = 1
        elif input_data['Source_location'] == 'Multiple':
            self.total_real_state = np.array(self.total_real_state_list).reshape(-1)
            self.total_state = np.array(self.total_state_list).reshape(-1)
            self.total_state_std = np.array(self.total_state_std_list).reshape(-1)
            self.nstate_partial = np.array(self.total_state_list).shape[1]
            self.real_state_init = self.total_real_state
            self.state_init = self.total_state
            self.state_std = self.total_state_std
            self.nstate_partial = self.nstate_partial
            self.source_location_case = 2
        else:
            print('Check the Source_location_info')

        
        self.decay = self.real_decay
        self.nstate = len(self.state_init)

        gpuff = Gpuff.PuffSimulation(self.input_data)
        gpuff.create_context()
        a = gpuff.run_simulations(self.input_data)
        self.obs = np.array(gpuff.run_simulations(self.input_data)).reshape(-1)
        gpuff.destroy_context()
        del gpuff

        self.obs_err = np.diag((np.floor(self.obs * 0) + np.ones([len(self.obs)])*self.nreceptor_err))  # nreceptor_err (percentage), needed to square later
        self.obs_MDA = np.diag((np.floor(self.obs * 0) + np.ones([len(self.obs)])*self.nreceptor_MDA))   # nreceptor_err (percentage), needed to square later

        # read the real data of sources's boundary for PSO
        self.lowerbounds_list = []
        self.upperbounds_list = []
        for s in range(self.nsource):
            real_source = "real_source{0}_boundary".format(s+1)
            for r in range(int(input_data['time']*24)):
                self.lowerbounds_list.append(input_data[real_source][0])
                self.upperbounds_list.append(input_data[real_source][1])
        self.bounds = np.array([self.lowerbounds_list, self.upperbounds_list]).T
                        
    def __str__(self):
        return self.name
    
    def make_ensemble(self):
        state = np.empty([self.nstate, self.sample])
        for i in range(self.nstate):
            state[i, :] = np.random.normal(self.state_init[i], self.state_std[i], self.sample)
        return state
    
    def state_to_ob(self, state):
        model_obs_list = []
        gpuff = Gpuff.PuffSimulation(self.input_data)
        gpuff.create_context()

        #self.gpuff.create_context()
        for ens in range(state.shape[1]):
            # gpuff.create_context()
            if self.source_location_case == 0:
                tmp_state = state[:,ens].reshape(-1, self.nstate_partial)
            elif self.source_location_case == 1:
                tmp_state = state[3:,ens].reshape(-1, self.nstate_partial) # except position elements
            elif self.source_location_case == 2:
                tmp_state = state[:,ens].reshape(-1, self.nstate_partial)
            else:
                print('Check the Source_location_info1')
            for s in range(self.nsource):
                actual_source = "Source_{0}".format(s+1)
                if self.source_location_case == 0:
                    self.input_data[actual_source][3] = tmp_state[s][:].tolist()
                elif self.source_location_case == 1:
                    self.input_data[actual_source][2] = state[:3,ens]
                    self.input_data[actual_source][3] = tmp_state[s][:].tolist()
                elif self.source_location_case == 2:
                    self.input_data[actual_source][2] = tmp_state[s][:3].tolist()
                    self.input_data[actual_source][3] = tmp_state[s][3:].tolist()
                else:
                    print('Check the Source_location_info2')
            model_obs_list.append(np.asarray(gpuff.run_simulations(self.input_data)).reshape(-1))
        gpuff.destroy_context()
        del gpuff
        #self.gpuff.destroy_context()
        model_obs = np.array(model_obs_list).T
        return model_obs
    
    def get_ob(self, time):
        self.obs_err = (self.obs * self.obs_err + self.obs_MDA)**2   # [obs_rel_std(rate) * true_obs + obs_abs_std]**2
        return self.obs, self.obs_err
    
    def predict(self, state, time):
        state = np.zeros([self.nstate, self.sample])
        return state