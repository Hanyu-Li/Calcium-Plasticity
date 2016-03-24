from brian_simulator import *
from brian_utils import *

params = {
    'cpre_0':0.1,
    'cpost_0':0.1,
    'rho_0':0.5,
    'c':0.2,
    'dummy':0.2,
    'Ipre':0,
    'Ipost':0,
    'w0':0.5,
    'w_EE':0.2,
    'w_IE':0.1,
    'w_II':-0.4,
    'w_EI':-0.4,
    #LIF specific constants,
    'tau_lif':26, #*ms
    'V_init':-60,
    'V_rest':-70,
    'V_reset':-70,
    'V_threshold':-50,
    'CM':0.001,
    'RM':20.0,
    'refrac':0,
    #Synapse model specific constants,
    'rho_init':0.019,
    'ca_initial':0,
    'ca_delay':4.61, #ms
    'Cpre':0.56175,
    'Cpost':1.23964,
    'eta':0,
    'tau_ca':22.6936,
    'theta_D':1,
    'theta_P':1.3,
    'gamma_D':331.909,
    'gamma_P':725.085,
    'taurho':346361, #*ms
    'taurho_fast':10, #*ms # dummy,
    'taupre':22,
    'taupost':22,
    'tau_ca':22, #*ms
    'rho_star':0.5,
    'D':4.6098,
    'baseline_I_ext_E':15,
    'baseline_I_ext_I':35,
    'mean_I_ext_E':21,
    'mean_I_ext_I':50,
    'sigma':20,
    'familiar_individual_sigma':5.3}



# additively applied to params
param_diffs = {
    'cpre_0':0,
    'cpost_0':0,
    'rho_0':0,
    'c':0,
    'dummy':0,
    'Ipre':0,
    'Ipost':0,
    'w0':0,
    'w_EE':0,
    'w_IE':0,
    'w_II':0,
    'w_EI':0,
    #LIF specific constants,
    'tau_lif':0, #*ms
    'V_init':0,
    'V_rest':0,
    'V_reset':0,
    'V_threshold':0,
    'CM':0,
    'RM':0,
    'refrac':0,
    #Synapse model specific constants,
    'rho_init':0,
    'ca_initial':0,
    'ca_delay':0, #ms
    'Cpre':0,
    'Cpost':0,
    'eta':0,
    'tau_ca':0,
    'theta_D':0,
    'theta_P':0,
    'gamma_D':0,
    'gamma_P':0,
    'taurho':0, #*ms
    'taurho_fast':0, #*ms # dummy,
    'taupre':0,
    'taupost':0,
    'tau_ca':0, #*ms
    'rho_star':0,
    'D':0,
    'baseline_I_ext_E':0,
    'baseline_I_ext_I':0,

    #'mean_I_ext_E':0,
    #'mean_I_ext_I':0,
    #'sigma': 0,
    #'familiar_individual_sigma':0}

    'mean_I_ext_E':0,
    'mean_I_ext_I':0,
    'sigma':[-10, 0],
    'familiar_individual_sigma':0}
    


# Control variables
simulation_length = 1000
stair_length = 500
N_E = 1000
N_I = 1
sample = 10
debug = False

sim = Brian_Simulator(simulation_length=simulation_length, stair_length=stair_length,N_E=N_E,N_I=N_I,sample=sample,
         params=params, debug=debug)


#12 with all excitatory, 15 with E:I=4:1 

# input pattern candidates

#input_flag = 'stair'
#input_flag = 'stable'
#input_flag = 'stable_with_bias'
#input_flag = '4_phase'
#input_flag = '4_phase_with_bias'
input_flag = '_fam_fam_nov_'
if input_flag == 'stair':
    resets = simulation_length / stair_length
else:
    resets = 1


# result variables

#spike_dict = build_spike_dict(param_diffs) # store spike trains for each parameter set

spike_dict = build_multivar_spike_dict(param_diffs) # store spike trains for each parameter set
spike_dict = build_real_value_spike_dict(params,param_diffs) # store spike trains for each parameter set

#print spike_dict
param_trial_num = len(spike_dict)

binned_rate_E = np.zeros((simulation_length * 10, param_trial_num))
binned_rate_I = np.zeros((simulation_length * 10, param_trial_num))
#rho = np.zeros((sample, simulation_length, param_trial_num))
rho = np.zeros((N_E, simulation_length, param_trial_num))
mean_rate_shift =np.zeros((param_trial_num,1))
#print spike_dict

if param_trial_num == -1:
    mode = 'cpp_standalone'
else:
    mode = 'cython'


t = arange(simulation_length)

#for i in arange(param_trial_num):
for ind, key in enumerate(spike_dict):
    cpp_directory = 'output_'+str(ind)

    (I_ext_E, I_ext_I, binned_rate_E[:,ind], binned_rate_I[:,ind], rho[:,:,ind], spike_dict[key]) = sim.run(key, mode=mode, input_flag=input_flag, resets=resets, cpp_directory=cpp_directory)
    #call(['rm','-r',cpp_directory])


# analysis

if input_flag == '4_phase_with_bias':
    analyse_all_parameter_sets(t, I_ext_E, spike_dict,real_data) 
    #for key in spike_dict:
    #    print "Key:",key
    #    analyse_spikes_phasewise(t, I_ext_E,key, spike_dict[key], real_data)
elif input_flag == '_fam_fam_nov_':
    #ref_data = sio.loadmat('data/nov_stim_rates.mat')
    #raw_data = sio.loadmat('data/Data_Sheinberg_Neuron2012_FiringRates.mat')
    (R_fam_E, R_fam_I, R_nov_E, R_nov_I) = unpack_raw_data('data/Data_Sheinberg_Neuron2012_FiringRates.mat')

    #real_data = ref_data['rnov']
    #lognormal_fit(real_data)
    analyse_all_parameter_sets(t, I_ext_E, spike_dict,R_fam_E,R_nov_E, params) 
    #analyse_all_parameter_sets(t, I_ext_I, spike_dict,real_data, params) 
    #for key in spike_dict:
        #print "Key:",key
        #analyse_spikes_phasewise(t, I_ext_E,key, spike_dict[key], real_data)

else:
    for key in spike_dict:
        #analyse_spikes_phasewise(t, I_ext_E,val)
        analyse_spikes(key, spike_dict[key])

#snapshot = simulation_length / 4 + 1
#visualize_I_ext(I_ext_E[snapshot,:])
visualize_all(I_ext_E, binned_rate_E, rho, t, resets, spike_dict.keys(), input_flag)
show()


