from brian2 import *
import csv
import pylab

def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    f1 = figure(figsize=(10, 4))
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
    #show()

class Brian_Simulator:
    def __init__(self, simulation_length, N_E, N_I,sample, params):
        self.simulation_length = simulation_length
        self.N_E = N_E
        self.N_I = N_I
        self.sample = sample
        self.params = params

    def run(self):
        set_device('cpp_standalone')
        prefs.devices.cpp_standalone.openmp_threads = 8
        start_scope()
        #control parameters
        debug = True
        #simulation_length = 10000
        observe_window = 100
        #N = 1000
        E_record_id = range(self.sample)
        I_record_id = range(self.sample)

        F = 1000*Hz

        #Variables used in brian code
        cpre_0 = self.params['cpre_0']
        cpost_0 = self.params['cpost_0']
        rho_0 = self.params['rho_0']
        c = self.params['c']
        dummy =self.params['dummy']
        Ipre =self.params['Ipre']
        Ipost = self.params['Ipost']
        w0 = self.params['w0']
        #LIF specific constants
        tau_lif = self.params['tau_lif']*ms
        V_init = self.params['V_init']
        V_rest = self.params['V_rest']
        V_reset = self.params['V_reset']
        V_threshold = self.params['V_threshold']
        CM = self.params['CM']
        RM = self.params['RM']
        sigma = self.params['sigma']
        refrac = self.params['refrac']
        #Synapse model specific constants
        rho_init = self.params['rho_init']
        ca_initial = self.params['ca_initial']
        ca_delay = self.params['ca_delay']
        Cpre = self.params['Cpre']
        Cpost = self.params['Cpost']
        tau_ca = self.params['tau_ca']
        theta_D = self.params['theta_D']
        theta_P = self.params['theta_P']
        gamma_D = self.params['gamma_D']
        gamma_P = self.params['gamma_P']
        taurho = self.params['taurho']*ms
        taurho_fast = self.params['taurho_fast']*ms # dummy
        taupre = self.params['taupre']*ms
        taupost =self.params['taupost']*ms
        tau_ca = self.params['tau_ca']*ms
        rho_star = self.params['rho_star']
        D = self.params['D']

        lif_eqs = '''
        dv/dt = (- (v+70) + I_ext) / tau_lif + sigma*xi*tau_lif**-0.5 : 1
        I_ext : 1
        '''
        #P = PoissonGroup(N, rates=F)
        # = NeuronGroup(2, eqs, threshold='v>vt', reset='v = vr')
        #S = Synapses(G, G, pre='v+=1*mV', connect='i==0 and j==1')


        G_E = NeuronGroup(self.N_E, lif_eqs, threshold='v>V_threshold', reset='v = V_reset')
        G_E.v = V_init

        G_I = NeuronGroup(self.N_I, lif_eqs, threshold='v>V_threshold', reset='v = V_reset')
        G_I.v = V_init





        synaptic_model = '''
                    w : 1 
                    dcpre/dt = -cpre / taupre : 1
                    dcpost/dt = -cpost / taupost : 1
                    c = cpre + cpost : 1
                    dummy = (c>theta_D) : 1
                    drho/dt = (-rho*(1-rho)*(0.5-rho) + gamma_P*(1-rho)*(c>theta_P) - gamma_D*rho*(c>theta_D)) / taurho : 1
                    '''
        pre_model = '''
                    v_post += rho
                    cpre += Cpre
                    
                    '''
        post_model = '''
                    cpost += Cpost
                    
                    '''




        S_EE = Synapses(G_E, G_E, model = synaptic_model, pre = pre_model, post = post_model)
        S_EE.connect('i!=j', p=0.1)

        #tmp = ((np.arange(len(S))+1) * 4).tolist()
        #S.delay = tmp*ms
        #S.delay = [4, 40, 400, 4000]*ms
        S_EE.w = w0
        S_EE.cpre = cpre_0
        S_EE.cpost= cpost_0
        S_EE.rho = rho_0
        G_E.I_ext = [25] * self.N_E
        G_E.I_ext[0] = 50

        #statemon = StateMonitor(G, 'v', record = True)
        #spikemon = SpikeMonitor(G)

        # Unrecorded simulation
        #statemon_S = StateMonitor(S, ['rho'], record = [0,1], dt=0.1*ms)
        statemon_S = StateMonitor(S_EE, ['rho'], record = E_record_id, dt=0.1*ms)
        run((self.simulation_length-observe_window)*ms, report='stdout', report_period=1*second)
        # Recorded simulation
        run(observe_window*ms, report='stdout', report_period=1*second)

        device.build(directory='output', compile=True, run=True, debug=False)
        # In[4]:

        if debug:
            #p1 = figure(1, figsize=(20,20))
            #visualise_plasticity(statemon_S.t, statemon_S.rho, sample=100)
            visualise_connectivity(S_EE)
            #show()

            f2 = figure(figsize=(10,4))
            plot(statemon_S.t, transpose(statemon_S.rho[E_record_id]))
            legend(E_record_id)
            xlabel('time/s')
            ylabel('synaptic efficacy \rho')
            show()
        #Analysis
        else:
            print statemon_S.rho

        np_rho = np.array(statemon_S.rho)
        stable_rho = np.mean(np_rho, axis=1)
        print stable_rho.shape

        f2 = open('results/stable_rho.csv', 'w')
        writer = csv.writer(f2)
        writer.writerow(stable_rho)
        f2.close()

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

    sim = Brian_Simulator(simulation_length=10000, N_E=80,N_I=20,sample=10, params=params)
    sim.run()
if __name__ == "__main__":
    main()
