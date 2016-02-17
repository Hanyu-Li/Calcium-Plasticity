
# coding: utf-8

# In[1]:

from core import Brian_Simulator
from brian2 import *
import csv
import pylab
#get_ipython().magic(u'matplotlib inline')


# In[ ]:


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

sim = Brian_Simulator(simulation_length=10000, N_E=1000,N_I=20, params=params)
sim.run()


# In[ ]:

reader = csv.reader(open('results/stable_rho.csv','r'))
stable_rho = reader.next()
#print stable_rho
plot(stable_rho)
show()

xlabel('Excitatory Neuron ID')
ylabel('Stable rho')

