

from brian2 import *
import csv
import pylab
from subprocess import call

#get_ipython().magic(u'matplotlib inline')

def visualize_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    figure(1, figsize=(10, 4))
    subplot(121)
    plot(zeros(Ns), arange(Ns), 'ok', ms=10)
    plot(ones(Nt), arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plot([0, 1], [i, j], '-k')
    xticks([0, 1], ['Source', 'Target'])
    ylabel('Neuron index')
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    subplot(122)
    plot(S.i, S.j, 'ok')
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel('Source neuron index')
    ylabel('Target neuron index')
def visualize_all(I=None, F=None, rho=None, t=None, resets=None):
    fig = figure(figsize=(50, 30))
    plot_num = 4

    ax1 = fig.add_subplot(plot_num, 1, 1)
    visualize_tI_curve(I, t, sub=True,ax=ax1)

    ax2 = fig.add_subplot(plot_num, 1, 2)
    visualize_IF_curve(I, F, t, sub=True, ax=ax2)

    ax3 = fig.add_subplot(plot_num, 1, 3)
    ax4 = fig.add_subplot(plot_num, 1, 4)
    visualize_F_rho_curve(F, rho, I, t, resets, sub=True, ax_a=ax3, ax_b=ax4)

    draw()
    savefig('results/all.png')


def visualize_tI_curve(I=None, t=None, sub=False, ax=None):
    if not sub:
        figure(figsize=(10,8))
        ax = plt.gca()
    title('I ext')
    print t.shape, I[:,0].shape
    ax.plot(t, I[:,0])

def visualize_IF_curve(I=None, F=None, t=None, sub=False, ax=None):
    interp = F.shape[0] / I.shape[0]
    if len(F.shape) == 1:
        sample = 1
    else:
        sample = F.shape[1]

    a = F.reshape((-1,10,sample))
    avg_F = np.mean(a,axis=1)
    
    if not sub:
        figure(figsize=(10,8))
        ax = plt.gca()
    title('IF_curve')
    print t.shape
    for i in arange(sample):
        #avg_F[:,i] = np.mean(F[:,i].reshape(-1, interp), axis=1)
        #print I.shape, avg_F.shape
        print t.shape, avg_F[:,i].shape
        #plot(I[:,0], avg_F[:,i])
        label = 'sigma = '+str(5*(i+1))
        ax.plot(t, avg_F[:,i], label=label)
    ax.legend()
    #savefig('results/e.png')
    draw()
def visualize_F_rho_curve(F=None, rho=None, I=None, t=None, resets=None, sub=False, ax_a=None, ax_b=None):
    if len(F.shape) == 1:
        sample = 1
    else:
        sample = F.shape[1]
    a = F.reshape((-1,10,sample))
    avg_F = np.ndarray((F.shape[0]/10, sample), dtype=float64)
    avg_F = np.mean(a, axis=1)
    #avg_rho = np.ndarray((avg_F.shape[0], sample), dtype=float64)
    '''
    if rho.shape[0] == 1:
        avg_rho = rho
    else:
        avg_rho = np.mean(rho, axis=0)
    '''
    avg_rho = np.mean(rho, axis=0)
    if not sub:
        figure(figsize=(10,8))
        ax_a = plt.gca()
    ax_a.set_title('t rho curve')
    print avg_F.shape, avg_rho.shape
    for i in arange(sample):
        ax_a.plot(t, avg_rho[:,i])
        #plot(avg_F[:,i], avg_rho[:,i])


    #if resets > 1:
    avg_avg_rho = np.mean(avg_rho.reshape((resets, -1, sample)), axis=1)
    print "avgavg", avg_avg_rho.shape
    if not sub:
        figure(figsize=(10,8))
        ax_b = plt.gca()
    ax_b.set_title('average rho curve')
    for i in arange(sample):
        label = 'sigma = '+str(5*(i+1))
        ax_b.plot(avg_avg_rho[:,i], label=label)
    ax_b.legend()
    #savefig('results/f.png')
    draw()
        


def build_input(amplitude, cumulative_share, length, N):
    I = np.zeros((length, N), dtype=float64)
    splits = np.multiply(cumulative_share, length)
    for i in arange(len(amplitude)):
        I[splits[i]:splits[i+1],:] = amplitude[i]
    return I

def build_increasing_input(min_v, max_v, stair_length, length, N):
    stair_step = np.float64(length) / np.float64(stair_length)
    stair_height = (max_v - min_v)/stair_step
    a = np.zeros((length, N), dtype=float64)
    ind = np.where(np.remainder(np.arange(length), stair_length) == 0)
    a[ind,:] = stair_height
    b = np.cumsum(a, axis=0)
    return b

class Brian_Simulator:
    def __init__(self, simulation_length, N_E, N_I,sample, I_ext_E, I_ext_I, params, debug):
        self.simulation_length = simulation_length
        self.N_E = N_E
        self.N_I = N_I
        self.I_ext_E = I_ext_E
        self.I_ext_I = I_ext_I
        self.sample = sample
        self.params = params
        self.debug = debug

    def run(self, param_diffs, mode='cython', resets=1, cpp_directory='output_0'):
        ## cpp mode
        if mode == 'cpp_standalone':
            set_device('cpp_standalone')
            #prefs.devices.cpp_standalone.openmp_threads = 8
        elif mode == 'cython':
        ## cython mode
            prefs.codegen.target = 'cython'

        start_scope()

        #control parameters
        observe_window = 100
        E_record_id = range(self.sample)
        I_record_id = range(self.sample)

        #Unpack Variables used in brian code
        for key in self.params.keys():
            exec_str = key + " = (self.params['" + key + "']+" + "param_diffs['"+key+"'])"
            if "tau" in key:
                exec_str = exec_str+"*ms"
            exec(exec_str)

        
        stim_E = TimedArray(self.I_ext_E, dt=1*ms)
        stim_I = TimedArray(self.I_ext_I, dt=1*ms)
        lif_eqs_E = '''
        dv/dt = (- (v+70) + stim_E(t,i)) / tau_lif + sigma*xi*tau_lif**-0.5 : 1
        '''
        lif_eqs_I = '''
        dv/dt = (- (v+70) + stim_I(t,i)) / tau_lif + sigma*xi*tau_lif**-0.5 : 1
        '''
        #P = PoissonGroup(N, rates=F)
        # = NeuronGroup(2, eqs, threshold='v>vt', reset='v = vr')
        #S = Synapses(G, G, pre='v+=1*mV', connect='i==0 and j==1')


        G_E = NeuronGroup(self.N_E, lif_eqs_E, threshold='v>V_threshold', reset='v = V_reset')
        G_E.v = V_init

        G_I = NeuronGroup(self.N_I, lif_eqs_I, threshold='v>V_threshold', reset='v = V_reset')
        G_I.v = V_init




        # plastic models
        synaptic_model_plastic = '''
                    w : 1 
                    dcpre/dt = -cpre / taupre : 1
                    dcpost/dt = -cpost / taupost : 1
                    c = cpre + cpost : 1
                    dummy = (c>theta_D) : 1
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
        
        # no post_model_I




        # use convention S_[to][from]

        S_EE = Synapses(G_E, G_E, model = synaptic_model_plastic, pre = pre_model_E_static, post = post_model_E_plastic)
        S_IE = Synapses(G_E, G_I, model = synaptic_model_static, pre = pre_model_E_static, post = None)
        S_EI = Synapses(G_I, G_E, model = synaptic_model_static, pre = pre_model_I_static, post = None)
        S_II = Synapses(G_I, G_I, model = synaptic_model_static, pre = pre_model_I_static, post = None)
        
        
        S_EE.connect('i!=j', p=0.1)
        S_IE.connect(True, p=0.1)
        S_EI.connect(True, p=0.1)
        S_II.connect(True, p=0.1)

        #tmp = ((np.arange(len(S))+1) * 4).tolist()
        #S.delay = tmp*ms
        #S.delay = [4, 40, 400, 4000]*ms
        
        S_EE.cpre = cpre_0
        S_EE.cpost= cpost_0
        S_EE.rho = rho_0
        S_EE.w = w_EE
        S_EE.pre.delay = D*ms
        S_EE.post.delay = D*ms
        
        S_IE.w = w_IE
        S_EI.w = w_EI
        S_II.w = w_II
        
       
        
        #G_E.I_ext = self.I_ext_E
        #G_E.I_ext = [25] * self.N_E
        #G_E.I_ext[0] = 50


        # Unrecorded simulation
        #statemon_S = StateMonitor(S, ['rho'], record = [0,1], dt=0.1*ms)

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
                run(self.simulation_length*ms)
                    
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
                run(self.simulation_length*ms)
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

            figure(figsize=std_size)
            plot(statemon_S_EE.t, transpose(statemon_S_EE.rho[E_record_id]))
            xlabel('time/s')
            ylabel('synaptic efficacy \rho')
            savefig('results/a.png')
            
            figure(figsize=stretch_size)
            subplot(211)
            title('excitatory')
            plot(spikemon_G_E.t, spikemon_G_E.i, '.k')
            xlim([0, self.simulation_length/1000])
            subplot(212)
            title('inhibitory')
            plot(spikemon_G_I.t, spikemon_G_I.i, '.k')
            xlim([0, self.simulation_length/1000])
            savefig('results/b.png')

            figure(figsize=std_size)
            subplot(211)
            title('excitatory spike rate')
            #plot(popratemon_G_E.t[window_length-1:]/ms, binned_rate_E)
            plot(popratemon_G_E.t/ms, binned_rate_E)

            #plot(self.I_ext_E[:,0], binned_rate) 
            
            subplot(212)
            title('inhibitory spike rate')

            #plot(popratemon_G_I.t[window_length-1:]/ms, binned_rate_I)
            plot(popratemon_G_I.t/ms, binned_rate_I)
            savefig('results/c.png')
            draw()
            
            
        else:
            #print statemon_S_EE.rho
            print "skip plotting"
        #Analysis
        #print spikemon_G_E.i

        


        #Save results


        np_rho = np.array(statemon_S_EE.rho)
        stable_rho = np.mean(np_rho, axis=1)
        print stable_rho.shape

        f2 = open('results/stable_rho.csv', 'w')
        writer = csv.writer(f2)
        writer.writerow(stable_rho)
        f2.close()

        f3 = open('results/spikes.csv','w')
        writer = csv.writer(f3)
        writer.writerow(spikemon_G_E.i)
        f3.close()

        return (binned_rate_E, binned_rate_I, np_rho)
        #p = pyplot.plot(stable_rho)
        #savefig('results/rho.png')



def main():
    params = {
        'cpre_0':0.1,
        'cpost_0':0.1,
        'rho_0':0.6,
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
        'sigma':5,
        'refrac':0,
        #Synapse model specific constants,
        'rho_init':0.019,
        'ca_initial':0,
        'ca_delay':4.61, #ms
        'Cpre':0.56175,
        'Cpost':1.23964,
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
        'D':4.6098}

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
        'sigma':0,
        'refrac':0,
        #Synapse model specific constants,
        'rho_init':0,
        'ca_initial':0,
        'ca_delay':0, #ms
        'Cpre':0,
        'Cpost':0,
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
        'D':0}


    # control variables
    simulation_length = 10000
    stair_length = 1000
    resets = simulation_length / stair_length

    N_E = 500
    N_I = 1
    sample = 3
    
    # input current candidates
    I_ext_E_increasing = build_increasing_input(0, 40, stair_length, simulation_length, N_E)
    I_ext_I_increasing = build_increasing_input(0, 40, stair_length, simulation_length, N_I)
    I_ext_E_stable = build_input([12], [0,1], simulation_length, N_E)
    I_ext_I_stable = build_input([12], [0,1], simulation_length, N_I)
    I_ext_E_4_phase = build_input([0,1,0,1], [0, 0.25,0.5,0.75, 1], simulation_length, N_E)
    I_ext_I_4_phase = build_input([0,0,0,0], [0, 0.25,0.5,0.75, 1], simulation_length, N_I)

    input_flag = 'stair'
    #input_flag = 'stable'
    #input_flag = '4_phase'
    

    if input_flag == 'stair':
        I_ext_E = I_ext_E_increasing
        I_ext_I = I_ext_I_increasing
    elif input_flag == 'stable':
        I_ext_E = I_ext_E_stable
        I_ext_I = I_ext_I_stable
        resets = 1
    elif input_flag == '4_phase':
        I_ext_E = I_ext_E_4_phase
        I_ext_I = I_ext_I_4_phase

    debug = False
    mode = 'cpp_standalone'
    param_trial_num = 1


    # result variables
    binned_rate_E = np.zeros((simulation_length * 10, param_trial_num))
    binned_rate_I = np.zeros((simulation_length * 10, param_trial_num))
    rho = np.zeros((sample, simulation_length, param_trial_num))
    '''
    sim = Brian_Simulator(simulation_length=simulation_length, N_E=N_E,N_I=N_I,sample=sample, 
                      I_ext_E=I_ext_E_increasing, I_ext_I=I_ext_I, params=params, debug=debug)
    (binned_rate_E, binned_rate_I, rho) = sim.run()
    '''



    t = arange(simulation_length)
    
    sim = Brian_Simulator(simulation_length=simulation_length, N_E=N_E,N_I=N_I,sample=sample,
            I_ext_E=I_ext_E, I_ext_I=I_ext_I, params=params, debug=debug)
    for i in arange(param_trial_num):
        cpp_directory = 'output_'+str(i)
        param_diffs['sigma'] = 5*(i+1)
        #params['sigma'] = i * 5
        (binned_rate_E[:,i], binned_rate_I[:,i], rho[:,:,i]) = sim.run(param_diffs, mode=mode, resets=resets, cpp_directory=cpp_directory)
        print 't ', t.shape
        #call(['rm','-r',cpp_directory])

    visualize_all(I_ext_E, binned_rate_E, rho,t,  resets)
    #visualize_tI_curve(I_ext_E_stable, t)
    #visualize_IF_curve(I_ext_E_stable, binned_rate_E, t)
    #visualize_F_rho_curve(binned_rate_E, rho,I_ext_E_stable,t, resets)
    show()


if __name__ == "__main__":
    main()
