from brian2 import *
import csv
import pylab
from subprocess import call
import scipy.io as sio
from scipy.stats import lognorm
#import numpy.matlib
from collections import OrderedDict
import networkx as nx
import itertools
import json
from networkx.readwrite import json_graph

#get_ipython().magic(u'matplotlib inline')

def unpack_EI_connectivity(S_EE, S_IE, S_EI, S_II, N_E, N_I, statemon_EE_rho):
    #print S.i
    #print S.j
    #print S.rho
    print S_IE.i, S_IE.j, statemon_EE_rho.shape
    T = statemon_EE_rho.shape[1]
    S_IE_j_shift = np.asarray(S_IE.j) + N_E
    S_EI_i_shift = np.asarray(S_EI.i) + N_E
    S_II_i_shift = np.asarray(S_II.i) + N_E
    S_II_j_shift = np.asarray(S_II.j) + N_E

    S_i = np.concatenate((S_EE.i, S_EI_i_shift, S_IE.i, S_II_i_shift))
    S_j = np.concatenate((S_EE.j, S_EI.j, S_IE_j_shift, S_II_j_shift))

    print S_EI.w
    S_w = np.concatenate((np.multiply(S_EE.rho,S_EE.w), S_EI.w, S_IE.w, S_II.w))
    print S_i, S_j, S_w


   
    thresh = 0.5 * S_EE.w[0]
    print thresh

    ### All
    #connectivity = zip(S_i, S_j, S_w)
    #trimmed_connectivity = []
    #for conn in connectivity:
    #    print conn
    #    if conn[2] > thresh:
    #        trimmed_connectivity.append(conn)


    ## EE only
    EE_connectivity = zip(S_EE.i, S_EE.j, S_EE.rho)
    thresh = 0.5
    trimmed_connectivity = []
    for conn in EE_connectivity:
        print conn
        if conn[2] > thresh:
            trimmed_connectivity.append(conn)

    G = nx.Graph()
    G.add_weighted_edges_from(trimmed_connectivity)
    figure(figsize=(40,20))
    d = json_graph.node_link_data(G)
    json.dump(d, open('d3js/connectivity.json','w'))
    

    axis('off')
    pos = nx.random_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=8)
    
    elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] > thresh]
    esmall=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] <=thresh]
    pos = nx.random_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=8)
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=0.2)
    #nx.draw_random(G, with_labels=False, arrows=False,width=0.2, node_size=8)
    axis('off')
    draw()
   



def unpack_connectivity(S):
    #print S.i
    #print S.j
    #print S.rho
    connectivity = zip(S.i, S.j, S.rho)
    #print connectivity
    #G = nx.from_edgelist(connectivity)
    #pos = nx.get_node_attributes(G, 'pos')
    G = nx.Graph()
    G.add_weighted_edges_from(connectivity)
    figure(figsize=(40,20))
    thresh = 0.6
    elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] > thresh]
    esmall=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] <=thresh]
    pos = nx.random_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=8)
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=0.2)


    #nx.draw_random(G, with_labels=False, arrows=False,width=0.2, node_size=8)
    axis('off')
    draw()





def unpack_raw_data(filepath):
    raw_data = sio.loadmat(filepath)
    index_exc = raw_data['index_exc'] - 1
    index_inh = raw_data['index_inh'] - 1
    index_ExcNeuron = [17,21,16,42,70,33,48,4,14,2,3,43,19,10]
    index_InhNeuron = [5,15,20,22,26,40,41,49,52]
    R_aver_t_fam = raw_data['R_aver_t_fam']
    R_aver_t_nov = raw_data['R_aver_t_nov']

    R_fam_exc = np.squeeze(R_aver_t_fam[:,index_ExcNeuron])
    R_fam_inh = np.squeeze(R_aver_t_fam[:,index_InhNeuron])
    R_nov_exc = np.squeeze(R_aver_t_nov[:,index_ExcNeuron])
    R_nov_inh = np.squeeze(R_aver_t_nov[:,index_InhNeuron])
    '''
    Norm_R_F_E = zeros(R_fam_exc.shape)
    Norm_R_F_I = zeros(R_fam_inh.shape)
    Norm_R_N_E = zeros(R_nov_exc.shape)
    Norm_R_N_I = zeros(R_nov_inh.shape)

    print R_fam_exc.shape
    print R_nov_inh.shape
    for i in arange(R_fam_exc.shape[0]):
        Norm_R_F_E[i,:] = (R_fam_exc[i,:] - np.mean(R_fam_exc, axis=0)) / np.std(R_fam_exc, axis=0)
        Norm_R_F_I[i,:] = (R_fam_inh[i,:] - np.mean(R_fam_inh, axis=0)) / np.std(R_fam_inh, axis=0)
        Norm_R_N_E[i,:] = (R_nov_exc[i,:] - np.mean(R_nov_exc, axis=0)) / np.std(R_nov_exc, axis=0)
        Norm_R_N_I[i,:] = (R_nov_inh[i,:] - np.mean(R_nov_inh, axis=0)) / np.std(R_nov_inh, axis=0)


    '''
    # use the mean of nov as center
    Norm_R_F_E = (R_fam_exc - np.mean(R_nov_exc, axis=0)) / np.std(R_fam_exc, axis=0)
    Norm_R_N_E = (R_nov_exc - np.mean(R_nov_exc, axis=0)) / np.std(R_nov_exc, axis=0)

    Norm_R_F_I = (R_fam_inh - np.mean(R_nov_inh, axis=0)) / np.std(R_fam_inh, axis=0)
    Norm_R_N_I = (R_nov_inh - np.mean(R_nov_inh, axis=0)) / np.std(R_nov_inh, axis=0)
    #return (Norm_R_F_E, Norm_R_F_I, Norm_R_N_E, Norm_R_N_I)
    return (R_fam_exc, R_fam_inh, R_nov_exc, R_nov_inh)



def visualize_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    figure(figsize=(10, 4))
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
def visualize_all(I=None, F=None, rho=None, t=None, resets=None, legends=None, input_flag=None):
    fig = figure(figsize=(20, 10))
    if input_flag=='7_phase_with_bias':
        plot_num = 3
        ax4 = None
    else:
        plot_num = 4
        ax4 = fig.add_subplot(plot_num, 1, 4)

    ax1 = fig.add_subplot(plot_num, 1, 1)
    visualize_tI_curve(I, t, sub=True,ax=ax1)

    ax2 = fig.add_subplot(plot_num, 1, 2)
    visualize_IF_curve(I, F, t, legends, sub=True, ax=ax2)

    ax3 = fig.add_subplot(plot_num, 1, 3)
    visualize_F_rho_curve(F, rho, I, t, resets, legends, sub=True, ax_a=ax3, ax_b=ax4)

    draw()
    savefig('results/all_1.png')


def visualize_tI_curve(I=None, t=None, sub=False, ax=None):
    if not sub:
        figure(figsize=(10,8))
        ax = plt.gca()
    title('I ext')
    ax.plot(t, I[:,0])

def visualize_IF_curve(I=None, F=None, t=None, legends=None, sub=False, ax=None):
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
    for i in arange(sample):
        #avg_F[:,i] = np.mean(F[:,i].reshape(-1, interp), axis=1)
        #plot(I[:,0], avg_F[:,i])
        #label = 'sigma = '+str(5*(i+1))
        label = legends[i]
        ax.plot(t, avg_F[:,i], label=label)
    ax.legend()
    #savefig('results/e.png')
    draw()
def visualize_F_rho_curve(F=None, rho=None, I=None, t=None, resets=None,legends=None, sub=False, ax_a=None, ax_b=None):
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
    for i in arange(sample):
        ax_a.plot(t, avg_rho[:,i])
        #plot(avg_F[:,i], avg_rho[:,i])


    #if resets > 1:
    if ax_b != None:
        avg_avg_rho = np.mean(avg_rho.reshape((resets, -1, sample)), axis=1)
        #I_downsample = I[0:
        if not sub:
            figure(figsize=(10,8))
            ax_b = plt.gca()
        ax_b.set_title('average rho curve')
        for i in arange(sample):
            #label = 'sigma = '+str(5*(i+1))
            label = legends[i]
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

def visualize_I_ext(I=None):
    #print I.shape
    width = np.floor(np.sqrt(I.shape[0]))
    I_trim = I[0:width*width]
    #print width
    figure()
    imshow(I_trim.reshape((-1,width)))
    draw()
def add_bias(I=None, sigmas=None):
    #print I.shape, sigmas.shape
    for nid in arange(I.shape[1]):
        I[:,nid] = I[:,nid] + sigmas[nid]
    return I

def add_bias_phasewise(I=None, baseline_I_ext=None, I_ext=None, sigmas=None):
    #print I.shape, sigmas.shape
    #print I.shape, sigmas.shape

    for tid in arange(I.shape[0]):
        #I[:,nid] = I[:,nid] * sigmas[I[0,nid]+1, nid] + I_ext
        if I[tid,0] == 0:
            I[tid,:] = baseline_I_ext
        else:
            #I[tid,:] = I[tid,:] * sigmas[I[tid,0]-1,:] + I_ext
            I[tid,:] = sigmas[I[tid,0]-1,:] + I_ext

    return I

def add_bias_phasewise_old(I=None, I_ext=None, sigmas=None):
    #print I.shape, sigmas.shape
    for nid in arange(I.shape[1]):
        I[:,nid] = I[:,nid] * sigmas[nid] + I_ext
    return I



def analyse_spikes(key=None, spikes=None):
    N = len(spikes['t'])
    #print N
    #isi = []
    firing_rate = np.zeros((N, 1))
    for nid in arange(N):
        isi = np.diff(np.sort(spikes['t'][nid]))
        #print isi
        try:
            firing_rate[nid] = 1.0 / np.mean(isi)
        except:
            continue
            #firing_rate[nid] = 0
    mean_r = np.mean(firing_rate)
    std_r = np.std(firing_rate)
    print "mean: ", mean_r
    print "std: ", std_r



    figure(figsize=(20,10))
    title(str(key))
    subplot(211)
    plot(firing_rate)
    subplot(212)
    bins = np.linspace(0, 80)
    hist(firing_rate, bins=bins, histtype='step')
    draw()
    return firing_rate

def lognormal_fit(x_fit = None, data=None, vis=False):
    scatter, loc, mean = lognorm.fit(data)
    print scatter, log, mean
    #x_fit = np.logspace(0,2,num=25)
    pdf_fit = lognorm.pdf(x_fit, scatter, loc, mean)
    if vis:
        figure()
        hist(data, bins=x_fit, normed=True)
        plot(x_fit, pdf_fit)
        xscale('log')
        draw()
        return
    else:
        return pdf_fit

def normalize_distribution(spikes=None, nov_mean=None):
    spikes = (spikes - nov_mean)/ np.std(spikes, axis=0)
    #spikes = (spikes - np.mean(spikes,axis=0))/ np.std(spikes, axis=0)
    return spikes

def analyse_all_parameter_sets(t=None, I=None,spikes=None, R_fam=None,R_nov=None, params=None):
    fig = figure(figsize=(20,10))
    sample_size = len(spikes)
    grid_width = np.ceil(np.sqrt(sample_size))
    grid_height = np.ceil(sample_size/grid_width)

    i = 1

    f_rec = open('results/stat.csv','w')
    writer = csv.writer(f_rec)
    writer.writerow(['','1st familiar fit error','2nd familiar fit error', 'novel fit error'])
    for key in spikes:
        ax = fig.add_subplot(grid_height, grid_width, i)
        print "Key:",key
        i = i + 1
        analyse_spikes_phasewise(t, I,key, spikes[key],R_fam,R_nov,params=params,sub=True, ax=ax, writer=writer)
    savefig('results/all_2')
    f_rec.close()

def analyse_spikes_phasewise(t=None, I=None, key=None, spikes=None, R_fam=None,R_nov=None, params=None, sub=False,ax=None, writer=None):
    N = len(spikes['t'])
   
    baseline = I[0,0]
    
    #index = np.where(I[:,0] - baseline != 0)
    phase_split = np.where(diff(I[:,0] - baseline) != 0)
    phase_split = np.concatenate(([0],np.squeeze(phase_split)) )
    #print new_ind
    phase_num = len(phase_split)
    #starts = new_ind[0::2]
    #ends = new_ind[1::2]

    #print starts, ends

    spike_rates = np.zeros((N, phase_num))
    mean_rates = np.zeros(phase_num)
    std_rates = np.zeros(phase_num)


    R_fam = R_fam.reshape((-1,1))
    mean_fam = np.mean(R_fam)
    std_fam = np.std(R_fam)

    R_nov = R_nov.reshape((-1,1))
    mean_nov = np.mean(R_nov)
    std_nov = np.std(R_nov)
    #lognormal_fit(scaled_real_data)

    for nid in arange(N):
        all_spike_time = np.asarray(np.sort(spikes['t'][nid]))
        for pid in arange(phase_num):
            start_t = double(phase_split[pid]) / 1000
            if pid == phase_num-1:
                end_t = len(t)
            else:
                end_t = double(phase_split[pid+1]) / 1000
            phase_spike_time = all_spike_time[np.logical_and(np.less_equal(all_spike_time, end_t), np.greater(all_spike_time, start_t))]
            #print phase_spike_time
            if len(phase_spike_time) == 1:
                firing_rate = 1
            elif len(phase_spike_time) == 0:
                firing_rate = 0
            else:
                phase_isi = np.diff(phase_spike_time)
                try:
                    firing_rate = 1.0 / np.mean(phase_isi)
                except:
                    continue
            spike_rates[nid, pid] = firing_rate
    spike_rates[np.isnan(spike_rates)] = 0
    print spike_rates.shape

    # find novel mean
    #nov_mean = mean(spike_rates[:,5])
    #spike_rates = normalize_distribution(spike_rates, nov_mean)

    print spike_rates.shape

   


    mean_rates = np.mean(spike_rates, axis=0)
    std_rates = np.std(spike_rates, axis=0)
    #mean_shift = mean_rates[3] - mean_rates[1]

    #print mean_rates, std_rates

    if sub==False:
        figure(figsize=(20,10))
        ax = plt.gca()
    title(str(key))
    log_x = False
    if log_x:
        bins = np.logspace(-1, 2, num=25)
    else:
        bins = np.linspace(0,100, num=51)
        #bins = np.linspace(-2,8, num=51)



    pdf_fit_error = np.zeros(3)
    hist_fit_error = np.zeros(3)
    #writer.writerow(spikemon_G_E.i)
    #f3.close()
    sample = arange(phase_num)
    
    print R_fam, R_nov
    pdf_fam = lognormal_fit(bins, R_fam)
    pdf_nov = lognormal_fit(bins, R_nov)


    '''
    h = ax.plot(bins, pdf_fam)
    ax.hist(R_fam, bins=bins,normed=True, histtype='step', color=h[0].get_color(),label='familiar data mean: %.2f, std: %.2f' % (mean_fam, std_fam))
    h = ax.plot(bins, pdf_nov)
    ax.hist(R_nov, bins=bins,normed=True, histtype='step', color=h[0].get_color(), label='novel data mean: %.2f, std: %.2f' % (mean_nov, std_nov))
    '''

    hf = ax.hist(R_fam, bins=bins,normed=True, histtype='step',label='familiar data mean: %.2f, std: %.2f' % (mean_fam, std_fam))

    hn = ax.hist(R_nov, bins=bins,normed=True, histtype='step',label='novel data mean: %.2f, std: %.2f' % (mean_nov, std_nov))





    for pid in sample[1::2]:
        #print spike_rates[:,pid].shape
        pdf = lognormal_fit(bins, spike_rates[:,pid])
        #h = ax.plot(bins, pdf)
        #ax.hist(spike_rates[:,pid], bins=bins,normed=True, histtype='step',color=h[0].get_color(), label='phase %d mean: %.2f, std: %.2f' % (pid, mean_rates[pid], std_rates[pid]))

        hs = ax.hist(spike_rates[:,pid], bins=bins,normed=True, histtype='step', label='phase %d mean: %.2f, std: %.2f' % (pid, mean_rates[pid], std_rates[pid]))
        if pid != 5:
            pdf_fit_error[(pid-1)/2] = np.linalg.norm(pdf-pdf_fam)
            hist_fit_error[(pid-1)/2] = np.linalg.norm(hs[0]-hf[0])
        else:
            pdf_fit_error[(pid-1)/2] = np.linalg.norm(pdf-pdf_nov)
            hist_fit_error[(pid-1)/2] = np.linalg.norm(hs[0]-hn[0])
    #print pdf_fit_error
    #print hist_fit_error
    writer.writerow([key]+hist_fit_error.tolist())
            




    '''
    for pid in sample[1::2]:
        print spike_rates[:,pid].shape
        pdf = lognormal_fit(spike_rates[:,pid])
        h = ax.plot(bins, pdf)
        ax.hist(spike_rates[:,pid], bins=bins,normed=True, histtype='step', color=h[0].get_color(), label='phase %d mean: %.2f, std: %.2f' % (pid, mean_rates[pid], std_rates[pid]))



    #print scaled_real_data.shape
    pdf = lognormal_fit(real_data)
    h = ax.plot(bins, pdf)
    ax.hist(real_data, bins=bins,normed=True, histtype='step', color=h[0].get_color(), label='real data mean: %.2f, std: %.2f' % (mean_real, std_real))

    #text(60, 8, 'mean_firing_rate_shift'+str(mean_post-mean_pre))
    xscale('log')
    '''

    #ax.text(80, 8, 'Mean_I_ext: %.2f, Shared_sigma: %.2f, Familiar_individual_sigma: %.2f, Novel_individual_sigma: %.2f' % (params['mean_I_ext'],params['sigma'], params['familiar_individual_sigma'],params['familiar_individual_sigma']))
    #ax.plot([], [], color='w',label='Mean_I_ext: %.2f, Shared_sigma: %.2f, Familiar_individual_sigma: %.2f, Novel_individual_sigma: %.2f' % (params['mean_I_ext'],params['sigma'], params['familiar_individual_sigma'],params['familiar_individual_sigma']))

    ax.legend(loc=2, fontsize=8)
    draw()

    return

def build_spike_dict(param_diffs):
    # currently only one parameter can be different from the baseline, later should include combinations
    #spike_dict = {}
    spike_dict = OrderedDict()
    # first add a dummy one, 
    spike_dict['Baseline'] = None
    for key, val in param_diffs.iteritems():
        if val != 0:
            print key, val
            for v in val:
                print v
                if v != 0:
                    spike_dict[(key, v)] = None
    return spike_dict
def build_multivar_spike_dict(param_diffs):
    spike_dict = OrderedDict()
    temp_dict = OrderedDict()
    temp_lists = []
    # first add a dummy one, 
    #spike_dict['Baseline'] = None
    for key, val in param_diffs.iteritems():
        if val != 0:
            #print key, val
            temp_list = []
            for v in val:
                #if v != 0:
                temp_list.append((key,v))
                    #spike_dict[(key, v)] = None
            temp_lists.append(temp_list)
    #print temp_lists
    for i in itertools.product(*temp_lists):
        #print i
        spike_dict[i] = None
    #spike_dict['Baseline'] = None
    return spike_dict
    '''
    keys = []
    dimension = 0
    dims = []
    for key, val in param_diffs.iteritems():
        if val != 0:
            #param_trial = len(val)-1
            #if param_sets == None:
                #param_sets =  
            print key,val
            keys.append(key)
            dims.append(len(val)-1)
            dimension = dimension + 1
            #temp_dict[key] = val
            new_val = []
            for v in val:
                if v != 0:
                    new_val.append(v)
                    #spike_dict[(key, v)] = None
            temp_dict[key] = new_val
    

    print keys,dimension, dims, temp_dict
    param_meshes = np.array((dimension,1))
    param_meshes = np.meshgrid(temp_dict.values())

    #for i in arange(dimension):
    #    mesh = np.zeros(dims)
    nd = 
    print param_meshes[0]




    multivar_spike_dict = OrderedDict()
    tuples = []

    for key, val in spike_dict.iteritems():
        print key[0],key[1], val

    return spike_dict
    '''
            





class Brian_Simulator:
    def __init__(self, simulation_length, N_E, N_I,sample, params, debug):
        self.simulation_length = simulation_length
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

        I_ext_E= add_bias_phasewise(I_ext_E, baseline_I_ext_E, mean_I_ext_E, individual_sigmas_E)
        I_ext_I= add_bias_phasewise(I_ext_I, baseline_I_ext_I, mean_I_ext_I, individual_sigmas_I)
        return (I_ext_E, I_ext_I)



    def run(self, param_diff, mode='cython',I_ext_E=None, I_ext_I=None, resets=1, cpp_directory='output_0'):
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
        I_ext_E, I_ext_I = self.build_7_phase_input(baseline_I_ext_E, baseline_I_ext_I, mean_I_ext_E, mean_I_ext_I, familiar_individual_sigma)
        
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

        S_EE = Synapses(G_E, G_E, model = synaptic_model_plastic, pre = pre_model_E_plastic, post = post_model_E_plastic)
        S_IE = Synapses(G_E, G_I, model = synaptic_model_static, pre = pre_model_E_static, post = None)
        S_EI = Synapses(G_I, G_E, model = synaptic_model_static, pre = pre_model_I_static, post = None)
        S_II = Synapses(G_I, G_I, model = synaptic_model_static, pre = pre_model_I_static, post = None)
        
        
        prob = 0.05
        S_EE.connect('i!=j', p=prob)
        S_IE.connect(True, p=prob)
        S_EI.connect(True, p=prob)
        S_II.connect('i!=j', p=prob)
        

        #tmp = ((np.arange(len(S))+1) * 4).tolist()
        #S.delay = tmp*ms
        #S.delay = [4, 40, 400, 4000]*ms
        
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
            unpack_EI_connectivity(S_EE, S_IE, S_EI, S_II, self.N_E, self.N_I, statemon_S_EE.rho)
            

            figure(figsize=std_size)
            plot(statemon_S_EE.t, transpose(statemon_S_EE.rho[E_record_id]))
            xlabel('time/s')
            ylabel('synaptic efficacy \rho')
            savefig('results/a.png')
            
            figure(figsize=stretch_size)
            subplot(211)
            title('excitatory')
            plot(spikemon_G_E.t, spikemon_G_E.i, '.k', markersize=1)
            xlim([0, self.simulation_length/1000])
            subplot(212)
            title('inhibitory')
            plot(spikemon_G_I.t, spikemon_G_I.i, '.k', markersize=1)
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

        return (binned_rate_E, binned_rate_I, np_rho, spikemon_G_E.all_values())
        #p = pyplot.plot(stable_rho)
        #savefig('results/rho.png')



def main():
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
        'sigma':0,
        'familiar_individual_sigma':0}
        


    # Control variables
    simulation_length = 8000

    stair_length = 500
    resets = 1

    N_E = 400
    N_I = 100
    sample = 10
    debug = True

    ref_data = sio.loadmat('data/nov_stim_rates.mat')
    #raw_data = sio.loadmat('data/Data_Sheinberg_Neuron2012_FiringRates.mat')
    (R_fam_E, R_fam_I, R_nov_E, R_nov_I) = unpack_raw_data('data/Data_Sheinberg_Neuron2012_FiringRates.mat')

    real_data = ref_data['rnov']
    #lognormal_fit(real_data)

    #12 with all excitatory, 15 with E:I=4:1 

    # input pattern candidates

    #input_flag = 'stair'
    #input_flag = 'stable'
    #input_flag = 'stable_with_bias'
    #input_flag = '4_phase'
    #input_flag = '4_phase_with_bias'
    input_flag = '7_phase_with_bias'
    

    baseline_I_ext_E = params['baseline_I_ext_E']
    baseline_I_ext_I = params['baseline_I_ext_I']
    mean_I_ext_E = params['mean_I_ext_E']
    mean_I_ext_I = params['mean_I_ext_I']
    familiar_individual_sigma =params['familiar_individual_sigma']

    individual_sigmas_E_familiar = np.random.normal(0,familiar_individual_sigma,N_E)
    individual_sigmas_I_familiar = np.random.normal(0,familiar_individual_sigma,N_I)

    individual_sigmas_E_novel = np.random.normal(0,familiar_individual_sigma,N_E)
    individual_sigmas_I_novel = np.random.normal(0,familiar_individual_sigma,N_I)

    individual_sigmas_E = np.vstack((individual_sigmas_E_familiar, individual_sigmas_E_novel))
    individual_sigmas_I = np.vstack((individual_sigmas_I_familiar, individual_sigmas_I_novel))


    if input_flag == 'stair':
        I_ext_E = build_increasing_input(0, 40, stair_length, simulation_length, N_E)
        I_ext_I = build_increasing_input(0, 160, stair_length, simulation_length, N_I)
        resets = simulation_length / stair_length

    elif input_flag == 'stable':
        I_ext_E= build_input([mean_I_ext_E], [0,1], simulation_length, N_E)
        I_ext_I= build_input([mean_I_ext_I], [0,1], simulation_length, N_I)

    elif input_flag == 'stable_with_bias':
        I_ext_E= add_bias(I_ext_E_stable, individual_sigmas_E_familiar)
        I_ext_I= add_bias(I_ext_I_stable, individual_sigmas_I_familiar)

    elif input_flag == '4_phase':
        I_ext_E= build_input([0,1,0,1], [0, 0.25,0.5,0.75, 1], simulation_length, N_E)
        I_ext_I= build_input([0,1,0,1], [0, 0.25,0.5,0.75, 1], simulation_length, N_I)


    elif input_flag == '4_phase_with_bias':
        I_ext_E= build_input([0,1,0,1], [0, 0.25,0.5,0.75, 1], simulation_length, N_E)
        I_ext_I= build_input([0,1,0,1], [0, 0.25,0.5,0.75, 1], simulation_length, N_I)

        I_ext_E= add_bias_phasewise_old(I_ext_E, mean_I_ext_E, individual_sigmas_E_familiar)
        I_ext_I= add_bias_phasewise_old(I_ext_I, mean_I_ext_I, individual_sigmas_I_familiar)

    elif input_flag == '7_phase_with_bias':
        #I_ext_E= build_input([0,1,0,1,0,2,0], [0, 0.15,0.3,0.45,0.6,0.75, 0.9, 1], simulation_length, N_E)
        #I_ext_I= build_input([0,1,0,1,0,2,0], [0, 0.15,0.3,0.45,0.6,0.75, 0.9, 1], simulation_length, N_I)
        I_ext_E= build_input([0,1,0,1,0,2,0], [0, 0.4,0.5,0.6,0.7,0.8, 0.9, 1], simulation_length, N_E)
        individual_sigmas_I = 0*individual_sigmas_I
        I_ext_I= build_input([0,1,0,1,0,2,0], [0, 0.4,0.5,0.6,0.7,0.8, 0.9, 1], simulation_length, N_I)
        #I_ext_I= build_input([0,1,0,1,0,2,0], [0, 0.4,0.5,0.6,0.7,0.8, 0.9, 1], simulation_length, N_I)

        I_ext_E= add_bias_phasewise(I_ext_E, baseline_I_ext_E, mean_I_ext_E, individual_sigmas_E)
        I_ext_I= add_bias_phasewise(I_ext_I, baseline_I_ext_I, mean_I_ext_I, individual_sigmas_I)
        #I_ext_I= build_input([baseline_I_ext_I], [0,1], simulation_length, N_I)




    # result variables

    #spike_dict = build_spike_dict(param_diffs) # store spike trains for each parameter set

    spike_dict = build_multivar_spike_dict(param_diffs) # store spike trains for each parameter set

    #print spike_dict
    param_trial_num = len(spike_dict)

    binned_rate_E = np.zeros((simulation_length * 10, param_trial_num))
    binned_rate_I = np.zeros((simulation_length * 10, param_trial_num))
    #rho = np.zeros((sample, simulation_length, param_trial_num))
    rho = np.zeros((N_E, simulation_length, param_trial_num))
    mean_rate_shift =np.zeros((param_trial_num,1))
    #print spike_dict

    if param_trial_num == 1:
        mode = 'cpp_standalone'
    else:
        mode = 'cython'


    t = arange(simulation_length)
    
    sim = Brian_Simulator(simulation_length=simulation_length, N_E=N_E,N_I=N_I,sample=sample,
             params=params, debug=debug)
    #for i in arange(param_trial_num):
    for i, key in enumerate(spike_dict):
        cpp_directory = 'output_'+str(i)

        #print key
        (binned_rate_E[:,i], binned_rate_I[:,i], rho[:,:,i], spike_dict[key]) = sim.run(key, mode=mode, I_ext_E=I_ext_E, I_ext_I=I_ext_I, resets=resets, cpp_directory=cpp_directory)
        #call(['rm','-r',cpp_directory])


    #print spikes['t'][0]
    # spike analysis
    #print len(spike_dict)
    
    if input_flag == '4_phase_with_bias':
        analyse_all_parameter_sets(t, I_ext_E, spike_dict,real_data) 
        #for key in spike_dict:
        #    print "Key:",key
        #    analyse_spikes_phasewise(t, I_ext_E,key, spike_dict[key], real_data)
    elif input_flag == '7_phase_with_bias':
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
    #visualize_tI_curve(I_ext_E_stable, t)
    #visualize_IF_curve(I_ext_E_stable, binned_rate_E, t)
    #visualize_F_rho_curve(binned_rate_E, rho,I_ext_E_stable,t, resets)
    show()


if __name__ == "__main__":
    main()
