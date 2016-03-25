from brian_utils import *
#get_ipython().magic(u'matplotlib inline')

class Brian_Simulator:
    def __init__(self, simulation_length, stair_length, N_E, N_I,sample, params, debug):
        self.simulation_length = simulation_length
        self.stair_length = stair_length
        self.N_E = N_E
        self.N_I = N_I
        #self.I_ext_E = I_ext_E
        #self.I_ext_I = I_ext_I
        self.sample = sample
        self.params = params
        self.debug = debug

    def build_7_phase_input(self, baseline_I_ext_E=None, baseline_I_ext_I=None, mean_I_ext_E=None, mean_I_ext_I=None, familiar_individual_sigma=None):


        individual_sigmas_E_familiar = np.random.normal(0,familiar_individual_sigma,self.N_E)
        individual_sigmas_I_familiar = np.random.normal(0,familiar_individual_sigma,self.N_I)

        individual_sigmas_E_novel = np.random.normal(0,familiar_individual_sigma,self.N_E)
        individual_sigmas_I_novel = np.random.normal(0,familiar_individual_sigma,self.N_I)

        individual_sigmas_E = np.vstack((individual_sigmas_E_familiar, individual_sigmas_E_novel))
        individual_sigmas_I = np.vstack((individual_sigmas_I_familiar, individual_sigmas_I_novel))



        I_ext_E= build_input([0,1,0,1,0,2,0], [0, 0.4,0.5,0.6,0.7,0.8, 0.9, 1], self.simulation_length, self.N_E)
        individual_sigmas_I = 0*individual_sigmas_I
        I_ext_I= build_input([0,1,0,1,0,2,0], [0, 0.4,0.5,0.6,0.7,0.8, 0.9, 1], self.simulation_length, self.N_I)

        #I_ext_E= build_input([0,1,0,1,0,2,0], [0, 0.4,0.8,0.9,0.925,0.95, 0.975, 1], self.simulation_length, self.N_E)
        #individual_sigmas_I = 0*individual_sigmas_I
        #I_ext_I= build_input([0,1,0,1,0,2,0], [0, 0.4,0.8,0.9,0.925,0.95, 0.975, 1], self.simulation_length, self.N_I)

        I_ext_E= add_bias_phasewise(I_ext_E, baseline_I_ext_E, mean_I_ext_E, individual_sigmas_E)
        I_ext_I= add_bias_phasewise(I_ext_I, baseline_I_ext_I, mean_I_ext_I, individual_sigmas_I)
        return (I_ext_E, I_ext_I)



    def run(self, param_diff, mode='cython',input_flag='_fam_fam_nov_', resets=1,cpp_directory='output_0'):
        ## cpp mode
        if mode == 'cpp_standalone':
            set_device('cpp_standalone')
            #prefs.devices.cpp_standalone.openmp_threads = 8
        elif mode == 'cython':
        ## cython mode
            prefs.codegen.target = 'cython'

        # control parameters
        observe_window = 100
        E_record_id = range(self.N_E)
        E_sample_id = range(self.sample)
        I_record_id = range(self.N_I)

        #Unpack Variables used in brian code
        #print param_diff
        for key in self.params.keys():
            exec_str = key + " = self.params['" + key + "']"
            #print exec_str
            if "tau" in key:
                exec_str = exec_str+"*ms"
            exec(exec_str)
        if param_diff != 'Baseline':
            for key in param_diff:
                #print  key
                exec_str = key[0] + " = "+ key[0] +"+"+str(key[1])
                print exec_str
                exec(exec_str)
        
        if input_flag =='_fam_fam_nov_':
            I_ext_E, I_ext_I = self.build_7_phase_input(baseline_I_ext_E, baseline_I_ext_I, mean_I_ext_E, mean_I_ext_I, familiar_individual_sigma)

        elif input_flag == 'stair':
            I_ext_E = build_increasing_input(0, 40, self.stair_length, self.simulation_length, self.N_E)
            I_ext_I = build_increasing_input(0, 160, self.stair_length, self.simulation_length, self.N_I)
            resets = self.simulation_length / self.stair_length
        
        '''
        for key in self.params.keys():
            if key == param_diff[0]:
                exec_str = key + " = self.params['" + key + "']+" + "param_diff[1]"
            else:
                exec_str = key + " = self.params['" + key + "']"
            #print exec_str
            if "tau" in key:
                exec_str = exec_str+"*ms"
            exec(exec_str)

        
        '''
        start_scope()
        stim_E = TimedArray(I_ext_E, dt=1*ms)
        stim_I = TimedArray(I_ext_I, dt=1*ms)
        lif_eqs_E = '''
        dv/dt = (- (v+70) + stim_E(t,i)) / tau_lif + sigma*xi*tau_lif**-0.5 : 1
        '''
        lif_eqs_I = '''
        dv/dt = (- (v+70) + stim_I(t,i)) / tau_lif + sigma*xi*tau_lif**-0.5 : 1
        '''


        G_E = NeuronGroup(self.N_E, lif_eqs_E, threshold='v>V_threshold', reset='v = V_reset')
        G_E.v = V_init

        G_I = NeuronGroup(self.N_I, lif_eqs_I, threshold='v>V_threshold', reset='v = V_reset')
        G_I.v = V_init




        # plastic models
        synaptic_model_plastic = '''
                    w : 1 
                    dcpre/dt = -cpre / taupre : 1
                    dcpost/dt = -cpost / taupost : 1
                    c = cpre + cpost + eta*cpre*cpost : 1
                    drho/dt = (-rho*(1-rho)*(0.5-rho) + gamma_P*(1-rho)*(c>theta_P) - gamma_D*rho*(c>theta_D)) / taurho : 1
                    '''
        
        pre_model_E_plastic = '''
                    v_post += rho * w
                    cpre += Cpre              
                    '''
        post_model_E_plastic = '''
                    cpost += Cpost
                    
                    '''
        
        # static models
        synaptic_model_static = '''
                    w : 1
                    ''' 
        pre_model_E_static = '''
                    v_post += w
                    '''
        
        pre_model_I_static = '''
                    v_post += w                 
                    '''
        
        # no post_model_static




        # use convention S_[to][from]

        S_EE = Synapses(G_E, G_E, model = synaptic_model_plastic, pre = pre_model_E_plastic, post = post_model_E_plastic)
        S_IE = Synapses(G_E, G_I, model = synaptic_model_static, pre = pre_model_E_static, post = None)
        S_EI = Synapses(G_I, G_E, model = synaptic_model_static, pre = pre_model_I_static, post = None)
        S_II = Synapses(G_I, G_I, model = synaptic_model_static, pre = pre_model_I_static, post = None)
        
        
        prob = 0.05
        S_EE.connect('i!=j', p=prob)
        S_IE.connect(True, p=prob)
        S_EI.connect(True, p=prob)
        # remove II connection in this experiment
        S_II.connect('i!=j', p=0.0)
        

        
        # scaling J_EE connections wrt Higgins 2014 paper
        k_E = self.N_E * prob / (1000 * 0.05)
        k_I = self.N_I * prob / (1000 * 0.05)
        S_EE.cpre = cpre_0
        S_EE.cpost= cpost_0
        S_EE.rho = rho_0
        S_EE.w = w_EE / k_E
        S_EE.pre.delay = D*ms
        S_EE.post.delay = D*ms
        
        S_IE.w = w_IE / k_E
        S_EI.w = w_EI / k_I
        S_II.w = w_II / k_I
        


        # set up monitors
        statemon_S_EE = StateMonitor(S_EE, ['rho'], record = E_record_id, dt=1*ms)
        spikemon_G_E = SpikeMonitor(G_E)
        spikemon_G_I = SpikeMonitor(G_I)
        popratemon_G_E = PopulationRateMonitor(G_E)
        popratemon_G_I = PopulationRateMonitor(G_I)


        if mode == 'cython':
            if resets > 1:
                for r in arange(resets):
                    #run(self.simulation_length/resets*ms, report='stdout', report_period=10*second)
                    run(self.simulation_length/resets*ms)
                    print double(r)/double(resets) * 100, '%'
                    S_EE.cpre = cpre_0
                    S_EE.cpost= cpost_0
                    S_EE.rho = rho_0
            else:
                run(self.simulation_length*ms, report='stdout', report_period=1*second)
                    
            #run((self.simulation_length-observe_window)*ms, report='stdout', report_period=1*second)

            # Recorded simulation
            #run(observe_window*ms, report='stdout', report_period=1*second)

        if mode == 'cpp_standalone':
            if resets > 1:
                device.insert_code('main', '''
                    int _i = 0;
                    for(_i=%s;_i>0;_i--){
                        cout << _i << endl;
                    ''' % (resets))
                #for r in arange(resets):
                    #run(self.simulation_length/resets*ms, report='stdout', report_period=10*second)
                run(self.simulation_length/resets*ms)

                #print double(r)/double(resets) * 100, '%'
                S_EE.cpre = cpre_0
                S_EE.cpost= cpost_0
                S_EE.rho = rho_0
                device.insert_code('main','''
                    }
                ''')
                    

            else:
                run(self.simulation_length*ms, report='stdout', report_period=1*second)
            device.build(directory=cpp_directory, compile=True, run=True, debug=False)


        # Simulation ends

        # In[4]:
        # calculate binned firing rate
        window = 20*ms
        window_length = int(window/defaultclock.dt)
        #padding = np.zeros((window_length,1)).tolist()
        padding = [0] * window_length
        cumsum = numpy.cumsum(numpy.insert(popratemon_G_E.rate, 0, padding))
        binned_rate_E = (cumsum[window_length:] - cumsum[:-window_length]) / window_length

        cumsum = numpy.cumsum(numpy.insert(popratemon_G_I.rate, 0, padding))
        binned_rate_I = (cumsum[window_length:] - cumsum[:-window_length]) / window_length

        # plot if debug flag is true

        if self.debug:
            std_size = (10, 5)
            stretch_size = (50, 50)
            #unpack_connectivity(S_EE)
            print I_ext_E.shape
            baseline = I_ext_E[0,0]
            phase_split = np.where(diff(I_ext_E[:,0] - baseline) != 0)
            phase_split = np.squeeze(phase_split)
            snapshot_fam = phase_split[0] + 1
            snapshot_nov = phase_split[4] + 1
            visualize_I_ext(I_ext_E[snapshot_fam,:], "familiar pattern")
            visualize_I_ext(I_ext_E[snapshot_nov,:], "novel pattern")
            unpack_EI_connectivity(S_EE, S_IE, S_EI, S_II, self.N_E, self.N_I, statemon_S_EE.rho)

            figure(figsize=std_size)
            plot(statemon_S_EE.t, transpose(statemon_S_EE.rho[E_sample_id]))
            xlabel('time/s')
            ylabel('synaptic efficacy rho')
            savefig('results/a.png')
            
            figure(figsize=stretch_size)
            subplot(211)
            title('excitatory', fontsize=16)
            plot(spikemon_G_E.t, spikemon_G_E.i, '.k', markersize=2)
            xlim([0, self.simulation_length/1000])
            subplot(212)
            title('inhibitory', fontsize=16)
            plot(spikemon_G_I.t, spikemon_G_I.i, '.k', markersize=2)
            xlim([0, self.simulation_length/1000])
            savefig('results/b.png')

            figure(figsize=std_size)
            subplot(211)
            title('excitatory spike rate', fontsize=16)
            #plot(popratemon_G_E.t[window_length-1:]/ms, binned_rate_E)
            plot(popratemon_G_E.t/ms, binned_rate_E)

            #plot(self.I_ext_E[:,0], binned_rate) 
            
            subplot(212)
            title('inhibitory spike rate', fontsize=16)

            #plot(popratemon_G_I.t[window_length-1:]/ms, binned_rate_I)
            plot(popratemon_G_I.t/ms, binned_rate_I)
            savefig('results/c.png')
            draw()
            
            
        else:
            #print statemon_S_EE.rho
            print "Simulation Complete"
        #Analysis
        #print spikemon_G_E.i

        


        #Save results


        np_rho = np.array(statemon_S_EE.rho)
        stable_rho = np.mean(np_rho, axis=1)

        f2 = open('results/stable_rho.csv', 'w')
        writer = csv.writer(f2)
        writer.writerow(stable_rho)
        f2.close()

        f3 = open('results/spikes.csv','w')
        writer = csv.writer(f3)
        writer.writerow(spikemon_G_E.i)
        f3.close()

        return (I_ext_E, I_ext_I, binned_rate_E, binned_rate_I, np_rho, spikemon_G_E.all_values())
        #return (I_ext_E, I_ext_I, binned_rate_E, binned_rate_I, np_rho, spikemon_G_E.all_values(), statemon_EE,)
        #p = pyplot.plot(stable_rho)
        #savefig('results/rho.png')


