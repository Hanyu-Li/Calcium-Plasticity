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

def unpack_EI_connectivity(S_EE, S_IE, S_EI, S_II, N_E, N_I, statemon_EE_rho):
    #print S.i
    #print S.j
    #print S.rho
    #print S_IE.i, S_IE.j, statemon_EE_rho.shape
    T = statemon_EE_rho.shape[1]
    S_IE_j_shift = np.asarray(S_IE.j) + N_E
    S_EI_i_shift = np.asarray(S_EI.i) + N_E
    S_II_i_shift = np.asarray(S_II.i) + N_E
    S_II_j_shift = np.asarray(S_II.j) + N_E

    S_i = np.concatenate((S_EE.i, S_EI_i_shift, S_IE.i, S_II_i_shift))
    S_j = np.concatenate((S_EE.j, S_EI.j, S_IE_j_shift, S_II_j_shift))

    #print S_EI.w
    S_w = np.concatenate((np.multiply(S_EE.rho,S_EE.w), S_EI.w, S_IE.w, S_II.w))
    #print S_i, S_j, S_w


   
    thresh = 0.5 * S_EE.w[0]
    #print thresh

    ## All
    connectivity = zip(S_i, S_j, S_w)
    trimmed_connectivity = []
    for conn in connectivity:
        #print conn
        if conn[2] > thresh:
            trimmed_connectivity.append(conn)


    G = nx.Graph()
    for i in S_EE.i:
        G.add_node(i, group='E')
    for i in range(N_E, N_E+N_I):
        G.add_node(i, group='I')
    G.add_weighted_edges_from(connectivity)
    figure(figsize=(40,20))
    d = json_graph.node_link_data(G)
    json.dump(d, open('d3js/connectivity.json','w'))


    ## EE only
    #EE_connectivity = zip(S_EE.i, S_EE.j, S_EE.rho)
    #thresh = 0.5
    #trimmed_connectivity = []
    #for conn in EE_connectivity:
    #    #print conn
    #    if conn[2] > thresh:
    #        trimmed_connectivity.append(conn)

    #G = nx.Graph()

    #for i in S_EE.i:
    #    G.add_node(i, group='E')
    #G.add_weighted_edges_from(trimmed_connectivity)
    #figure(figsize=(40,20))
    #d = json_graph.node_link_data(G)
    #json.dump(d, open('d3js/connectivity.json','w'))
    

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
    return (Norm_R_F_E, Norm_R_F_I, Norm_R_N_E, Norm_R_N_I)
    #return (R_fam_exc, R_fam_inh, R_nov_exc, R_nov_inh)



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
def visualize_all(I=None, F=None, rho=None, t=None, resets=None,legends=None, input_flag=None):
    fig = figure(figsize=(20, 10))
    if input_flag=='_fam_fam_nov_':
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
        #ax.plot(I, avg_F[:,i],'.', label=label)
        ax.plot(t, avg_F[:,i], label=label)
    ax.legend(loc=2)
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
            print avg_avg_rho[:,i]
            ax_b.plot(avg_avg_rho[:,i], label=label)
        ax_b.legend()
    #savefig('results/f.png')
    draw()
        


def build_input(amplitude, cumulative_share, length, N):
    I = np.zeros((length, N), dtype=float64)
    splits = np.floor(np.multiply(cumulative_share, length))
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
            I[tid,:] = sigmas[np.floor(I[tid,0])-1,:] + I_ext

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
    suptitle("Firing Rate Distribution With"+str(key))
    subplot(211)
    plot(firing_rate)
    subplot(212)
    bins = np.linspace(0, 80)
    hist(firing_rate, bins=bins, histtype='step', label=key)
    legend(loc=2)
    draw()
    return firing_rate

def lognormal_fit(x_fit = None, data=None, vis=False):
    scatter, loc, mean = lognorm.fit(data)
    #print scatter, log, mean
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
        #print "Key:",key
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
    #print "phase split:", phase_split
    #print phase_num
    phase_length = np.diff(np.concatenate((phase_split, [t.shape[0]])))
    phase_length = np.double(phase_length) / 1000
    print phase_length
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
                firing_rate = 1/phase_length[pid]
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

    #print spike_rates.shape

   


    #mean_rates = np.mean(spike_rates, axis=0)
    #std_rates = np.std(spike_rates, axis=0)

    #mean_shift = mean_rates[3] - mean_rates[1]

    #print mean_rates, std_rates

    if sub==False:
        figure(figsize=(20,10))
        ax = plt.gca()
    title(str(key))
    log_x = False
    if log_x:
        bins = np.logspace(-1, 2, num=51)
        xscale('log')
    else:
        #bins = np.linspace(0,100, num=51)
        bins = np.linspace(-2,8, num=51)



    pdf_fit_error = np.zeros(3)
    hist_fit_error = np.zeros(3)
    #writer.writerow(spikemon_G_E.i)
    #f3.close()
    sample = arange(phase_num)
    
    #print R_fam, R_nov
    #print R_fam
    
    pdf_fam = lognormal_fit(bins, R_fam)
    pdf_nov = lognormal_fit(bins, R_nov)


    title("Firing rate distribution")
    h = ax.plot(bins, pdf_fam)
    hf = ax.hist(R_fam, bins=bins,normed=True, histtype='step', color=h[0].get_color(),label='familiar data mean: %.2f, std: %.2f' % (mean_fam, std_fam))
    h = ax.plot(bins, pdf_nov)
    hn = ax.hist(R_nov, bins=bins,normed=True, histtype='step', color=h[0].get_color(), label='novel data mean: %.2f, std: %.2f' % (mean_nov, std_nov))

    '''
    hf = ax.hist(R_fam, bins=bins,normed=True, histtype='step',label='familiar data mean: %.2f, std: %.2f' % (mean_fam, std_fam))

    hn = ax.hist(R_nov, bins=bins,normed=True, histtype='step',color='r',label='novel data mean: %.2f, std: %.2f' % (mean_nov, std_nov))
    '''





    #for pid in sample[1::2]:
    for pid in [5,3,1]:
        #print spike_rates[:,pid].shape
        # perform trimming
        #trimmed_rate = trim_NaN(spike_rates[:,pid])

        trimmed_rate = spike_rates[:,pid]
        trimmed_rate = trimmed_rate[np.greater(trimmed_rate,0)]

        #print trimmed_rate.shape
        if pid == 5:
            nov_mean = mean(trimmed_rate)
        trimmed_rate = normalize_distribution(trimmed_rate, nov_mean)
        trimmed_mean = mean(trimmed_rate)
        trimmed_std = std(trimmed_rate)
        pdf = lognormal_fit(bins, trimmed_rate)
        h = ax.plot(bins, pdf)
        #ax.hist(spike_rates[:,pid], bins=bins,normed=True, histtype='step',color=h[0].get_color(), label='phase %d mean: %.2f, std: %.2f' % (pid, mean_rates[pid], std_rates[pid]))

        if pid == 5:
            label = "During Novel Simulation"
        elif pid == 3:
            label = "During Familiar Simulation"
        elif pid == 1:
            label = "During Training with Familiar Simulation"
        hs = ax.hist(trimmed_rate, bins=bins,normed=True, histtype='step',color=h[0].get_color(), label='%s mean: %.2f, std: %.2f' % (label, trimmed_mean, trimmed_std))
        #hs = ax.hist(trimmed_rate, bins=bins,normed=True, histtype='step', label='phase %d mean: %.2f, std: %.2f' % (pid, trimmed_mean, trimmed_std))
        if pid != 5:
            pdf_fit_error[(pid-1)/2] = np.linalg.norm(pdf-pdf_fam)
            hist_fit_error[(pid-1)/2] = np.linalg.norm(hs[0]-hn[0])
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

    ax.legend(loc=2, fontsize=16)
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

def build_real_value_spike_dict(params, param_diffs):
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
                temp_list.append((key,v+params[key]))
                    #spike_dict[(key, v)] = None
            temp_lists.append(temp_list)
    #print temp_lists
    for i in itertools.product(*temp_lists):
        #print i
        spike_dict[i] = None
    #spike_dict['Baseline'] = None
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
            
