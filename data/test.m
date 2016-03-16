load('Data_Sheinberg_Neuron2012_FiringRates');
index_ExcNeuron = [17 21 16 42 70 33 48 4 14 2 3 43 19 10];
R_E = R_aver_t_fam(:,index_ExcNeuron);
mu = mean(flat_R_E);
st = std(flat_R_E);
norm_R_E = (R_E - mu) / st;
flat_R_E = reshape(norm_R_E, [],1);

hist(flat_R_E,50);
norm_R_E - NormalizedExcR_Fam

