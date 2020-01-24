import os, sys
import random
import ntpath
from collections import defaultdict
import ntpath
import pandas as pd
import numpy as np
import scipy.stats
from pandas import Series
# TORCH: "GPU Tensors"
import torch
from torch.distributions import constraints, transform_to
from torch.optim import Adam, LBFGS
#import tensorflow as tf
# PYRO
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.autoguide import AutoDelta, AutoDiagonalNormal, AutoLowRankMultivariateNormal
from pyro.optim import PyroOptim
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, Trace_ELBO, TraceGraph_ELBO,EmpiricalMarginal
#Biopython Superimposer
from Bio.SVDSuperimposer import SVDSuperimposer
# Matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import tqdm
# Early STOPPING
from ignite.handlers import EarlyStopping
from ignite.engine import Engine, Events
# NUTS sampler
from pyro.infer.abstract_infer import TracePredictive
from pyro.infer.mcmc import MCMC, NUTS
mpl.use('agg') #TkAgg
tqdm.monitor_interval = 0
# Posterior probabilities
PyroOptim.state_dict = lambda self: self.get_state()
#LOGGING (stats for posterior)
import logging
# Use GPU?

use_cuda = torch.cuda.is_available()
cuda = torch.device('cuda')
cuda0 = torch.device('cuda:0')
cuda2 = torch.device('cuda:2')
if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor') #PyTorch tensors have a dimension limit of 25 integers in CUDA and 64 in CPU
    device = torch.device(cuda2)
    print("Using the GPU",torch.cuda.get_device_name(0))
else:
    print("Using CPU detected")
torch.multiprocessing.set_sharing_strategy('file_system')  # Needed to avoid runtime errors

class SVIEngine(Engine):
    def __init__(self, *args, step_args=None, **kwargs):
        self.svi = SVI(*args, **kwargs)
        self._step_args = step_args or {}
        super(SVIEngine, self).__init__(self._update)
    def _update(self, engine, batch):
        return -engine.svi.step(batch, **self._step_args)
class Loader():
    def Average_Structure(self,tuple_struct):
        average = sum(list(tuple_struct)) / len(
            tuple_struct)  # sum element-wise the list of tensors containing the coordinates #tf.add_n
        return average
    def Max_variance(self,structure):
        '''Calculates the maximum distance to the origin of the structure, this value will define the variance of the prior's distribution'''
        centered = self.Center_torch(structure)
        mul = centered@torch.t(structure)
        max_var = torch.sqrt(torch.max(torch.diag(mul)))
        return max_var
    def Center_numpy(self,Array):
        '''Centering to the origin the data'''
        mean = np.mean(Array, axis=0)
        centered_array = Array - mean
        return centered_array
    def Center_torch(self,Array):
        '''Centering to the origin the data'''
        mean = torch.mean(Array, dim=0)
        centered_array = Array - mean
        return centered_array
    def RMSD_numpy(self,X1, X2):
        import torch.nn.functional as F
        return F.pairwise_distance(torch.from_numpy(X1), torch.from_numpy(X2))
    def PairwiseDistances(self,X1,X2):
        '''Computes pairwise distances among coordinates'''
        import torch.nn.functional as F
        return F.pairwise_distance(X1,X2)
    def RMSD_biopython(self,x, y):
        sup = SVDSuperimposer()
        sup.set(x, y)
        sup.run()
        rot, tran = sup.get_rotran()
        return rot
class Model():
    def sample_R(self,ri_vec):
        """Inputs a sample of unit quaternion and transforms it into a rotation matrix"""
        theta1 = 2 * np.pi * ri_vec[1]
        theta2 = 2 * np.pi * ri_vec[2]

        r1 = torch.sqrt(1 - ri_vec[0])
        r2 = torch.sqrt(ri_vec[0])

        qw = r2 * torch.cos(theta2)
        qx = r1 * torch.sin(theta1)
        qy = r1 * torch.cos(theta1)
        qz = r2 * torch.sin(theta2)

        R = torch.eye(3, 3) # device =cuda

        # Row one
        R[0, 0] = qw ** 2 + qx ** 2 - qy ** 2 - qz ** 2
        R[0, 1] = 2 * (qx * qy - qw * qz)
        R[0, 2] = 2 * (qx * qz + qw * qy)

        # Row two
        R[1, 0] = 2 * (qx * qy + qw * qz)
        R[1, 1] = qw ** 2 - qx ** 2 + qy ** 2 - qz ** 2
        R[1, 2] = 2 * (qy * qz - qw * qx)

        # Row three
        R[2, 0] = 2 * (qx * qz - qw * qy)
        R[2, 1] = 2 * (qy * qz + qw * qx)
        R[2, 2] = qw ** 2 - qx ** 2 - qy ** 2 + qz ** 2
        return R
    def model(self,data):
        max_var,data1, data2 = data
        ### 1. prior over mean M
        M = pyro.sample("M", dist.StudentT(1,0, 3).expand_by([data1.size(0),data1.size(1)]).to_event(2))
        ### 2. Prior over variances for the normal distribution
        U = pyro.sample("U", dist.HalfNormal(1).expand_by([data1.size(0)]).to_event(1))
        U =  U.reshape(data1.size(0),1).repeat(1,3).view(-1)
        ## 3. prior over translations T_i: Sample translations for each of the x,y,z coordinates
        T2 = pyro.sample("T2", dist.Normal(0, 1).expand_by([3]).to_event(1))
        ## 4. prior over rotations R_i
        ri_vec = pyro.sample("ri_vec",dist.Uniform(0, 1).expand_by([3]).to_event(1))
        R = self.sample_R(ri_vec)
        M_T1 = M
        M_R2_T2 = M @ R + T2
        # 5. Likelihood
        with pyro.plate("plate_univariate", data1.size(0)*data1.size(1),dim=-1):
            pyro.sample("X1", dist.StudentT(1,M_T1.view(-1), U),obs=data1.view(-1))
            pyro.sample("X2", dist.StudentT(1,M_R2_T2.view(-1), U), obs=data2.view(-1))
    def _get_initial_trace(self,data_obs, average):
        '''Initialize MCMC and NUTS with the VI MAP estimate'''
        if use_cuda:
            data_obs = [data.cuda() for data in data_obs]
            average = average.cuda()
        else:
            pass
        # GUIDE
        global_guide = AutoDelta(self.model)
        # OPTIMIZER
        optim = pyro.optim.AdagradRMSProp(dict())
        elbo = Trace_ELBO()
        # STOCHASTIC VARIATIONAL INFERENCE
        svi_engine = SVIEngine(self.model, global_guide, optim, loss=elbo)
        pbar = tqdm.tqdm()
        loss_list = []
        # INITIALIZING PRIORS for mean structure and the unit quaternion
        pyro.param("auto_ri_vec", torch.Tensor([0.9, 0.1, 0.9]),constraint=constraints.unit_interval)
        pyro.param("auto_M",average)
        @svi_engine.on(Events.EPOCH_COMPLETED)
        def update_progress(svi_engine):
            pbar.update(1)
            loss_list.append(-svi_engine.state.output)
            pbar.set_description( "[epoch {}] avg train loss: {}".format(svi_engine.state.epoch, svi_engine.state.output))
        # HANDLER
        handler = EarlyStopping(patience=25, score_function=lambda eng: eng.state.output, trainer=svi_engine)
        # SVI
        svi_engine.add_event_handler(Events.EPOCH_COMPLETED, handler)
        svi_engine.run([data_obs], max_epochs=15000)
        return svi_engine.svi.exec_traces
    def Run(self,data_obs, average,name1):
        if use_cuda:
            data_obs = [data.cuda() for data in data_obs]
            average = average.cuda()
        else:
            pass
        ### MCMC and NUTS
        # I had to fix a problem at /home/lys/anaconda3/lib/python3.5/site-packages/pyro/util.py by initializing the seed to rng_seed = random.randint(0,2**32-1)
        nuts_kernel = NUTS(self.model,max_tree_depth=5)
        # INITIALIZING PRIOR Changing in the first iteration the value for the prior in order to constrain the prior over the rotations
        nuts_kernel.initial_trace = self._get_initial_trace(data_obs, average)
        samples = 1250
        warmup =  250
        chains = 1 #Also, I cannot use more than 1 chain, because I get the error "RuntimeError: CUDA error: initialization error" at mcmc.py, I suspect it has to do with torch.multiprocessing.set_sharing_strategy('file_system')
        mcmc = MCMC(nuts_kernel, num_samples=samples, warmup_steps=warmup, num_chains=chains).run(data_obs)
        max_var, data1, data2 = data_obs

        # EXTRACT PARAMETERS
        #Rotation matrix stats
        ri_vec_marginal_1 = mcmc.marginal(sites=["ri_vec"])
        ri_vec_marginal_1 = torch.cat(list(ri_vec_marginal_1.support(flatten=True).values()), dim=-1).cpu().numpy() #Samples
        params = ['ri_vec[0]','ri_vec[1]','ri_vec[2]']
        df = pd.DataFrame(ri_vec_marginal_1, columns= params).transpose()
        df_summary = df.apply(pd.Series.describe, axis=1)[["mean", "std", "25%", "50%", "75%"]]
        df_summary.to_csv("ri_vec_stats_{}.txt".format(name1),sep='\t')
        # Rotation matrix output
        ri_vec_marginal = mcmc.marginal(["ri_vec"]).empirical["ri_vec"]
        ri_vec_mean = ri_vec_marginal.mean
        ri_vec_variance = ri_vec_marginal.variance
        R = self.sample_R(ri_vec_mean)

        # Mean structure stats
        M_marginal_1 = mcmc.marginal(sites=["M"])
        M_marginal_1 = torch.cat(list(M_marginal_1.support(flatten=True).values()), dim=-1).cpu().numpy()
        params = ['M[{}]'.format(i) for i in range(0,len(data1))]
        label_one = np.array(params)
        label_two = np.array(['x', 'y', 'z'])
        cols = pd.MultiIndex.from_product([label_one, label_two])
        df = pd.DataFrame(M_marginal_1.T.reshape(samples, -1), columns=cols).transpose()
        df_summary = df.apply(pd.Series.describe, axis=1)[["mean", "std", "25%", "50%", "75%"]]
        df_summary.to_csv("M_stats_{}.txt".format(name1),sep='\t')

        # Mean structure M output
        M_vec_marginal = mcmc.marginal(["M"]).empirical["M"]
        M_vec_mean = M_vec_marginal.mean
        M_vec_variance= M_vec_marginal.variance
        M = M_vec_mean
        #M = Center_torch(M_vec_mean.detach())

        # Translation stats
        T_marginal_1 = mcmc.marginal(sites=["T2"])
        T_marginal_1 = torch.cat(list(T_marginal_1.support(flatten=True).values()), dim=-1).cpu().numpy()
        params = ['T[0]','T[1]','T[2]']
        df = pd.DataFrame(T_marginal_1, columns= params).transpose()
        df_summary = df.apply(pd.Series.describe, axis=1)[["mean", "std", "25%", "50%", "75%"]]
        df_summary.to_csv("T_stats_{}.txt".format(name1),sep='\t')
        # Translation T output
        T2_vec_marginal = mcmc.marginal(["T2"]).empirical["T2"]
        T2_vec_mean = T2_vec_marginal.mean
        T2_vec_variance = T2_vec_marginal.variance

        #Observed
        X1 = data1.detach().cpu().numpy() #- T1_vec_mean.cpu().numpy()  # X1 -T1
        X2 = np.dot(data2.detach().cpu().numpy() - T2_vec_mean.cpu().numpy(), np.transpose(R.cpu()))  # (X2-T2)R-1


        import matplotlib
        matplotlib.rcParams['legend.fontsize'] = 10

        #################PLOTS################################################
        fig = plt.figure(figsize=(18, 16), dpi=80)
        ax = fig.add_subplot(111, projection='3d')

        # blue graph
        x = X1[:, 0]
        y = X1[:, 1]
        z = X1[:, 2]

        ax.plot(x, y, z, c='b', label='X1', linewidth=3.0)

        # red graph
        x2 = X2[:, 0]
        y2 = X2[:, 1]
        z2 = X2[:, 2]

        ax.plot(x2, y2, z2, c='r', label='X2', linewidth=3.0)

        ###green graph
        x3 = M.cpu().numpy()[:, 0]
        y3 = M.cpu().numpy()[:, 1]
        z3 = M.cpu().numpy()[:, 2]
        ax.plot(x3, y3, z3, c='g', label='M', linewidth=3.0)
        ax.legend()

        plt.title("Initialized MCMC and NUTS model")
        plt.savefig("Bayesian_Result_Samples_{}_{}_chains_{}".format(name1,samples + warmup,chains))

        plt.clf()
        plt.plot(Loader.PairwiseDistances(data1.cpu(),data2.cpu()).numpy(), linewidth = 8.0)
        plt.plot(Loader.PairwiseDistances(torch.from_numpy(X1),torch.from_numpy(X2)).numpy(), linewidth=8.0)
        plt.ylabel('Pairwise distances',fontsize='46')
        plt.xlabel('Amino acid position',fontsize='46')
        plt.title('{}'.format(name1.upper()),fontsize ='46')
        plt.gca().legend(('RMSD', 'Theseus-PP'),fontsize='40')
        plt.savefig(r"Distance_Differences_{}_Bayesian_{}".format(name1,chains))
        plt.close()

        return  T2_vec_mean.cpu().numpy(), R.cpu(), M.cpu().numpy(), X1, X2, ri_vec_marginal_1,M_marginal_1,T_marginal_1 #Values for the mean structure



#CALLING THE FUNCTIONS
Loader = Loader()
Model = Model()
samples =3
#Dataframes
data1 = torch.Tensor([[-25.5741, -20.0990, -12.2635],
        [-27.3981, -16.9150, -11.1845],
        [-26.0351, -14.8840,  -8.2475],
        [-27.7071, -12.0190,  -6.3435],
        [-25.3701,  -9.9950,  -4.1205],
        [-26.1141,  -6.5100,  -2.7135],
        [-25.8601,  -3.3260,  -4.8175],
        [-23.1851,  -0.8920,  -3.5765],
        [-19.6731,  -2.1700,  -2.8335],
        [-20.4441,  -5.2620,  -4.9435],
        [-19.9741,  -3.2250,  -8.1355],
        [-17.4981,  -0.9230,  -6.3325],
        [-18.9441,   1.5130,  -3.7675],
        [-16.0231,   2.3120,  -1.4315],
        [-15.1381,   5.8500,  -0.3475],
        [-11.7441,   6.0280,   1.5325],
        [-11.7831,   5.7160,   5.2995],
        [-10.4681,   9.2910,   5.5775],
        [ -7.5391,   7.8190,   7.4965],
        [ -6.4841,   6.9920,   3.9335],
        [ -6.5741,  10.6170,   3.1165],
        [ -3.8771,  10.4470,   5.8385],
        [ -2.2381,   7.3010,   4.5395],
        [ -1.0621,   9.0730,   1.4055],
        [ -0.5581,  12.1340,   3.6185],
        [  2.0649,  10.1120,   5.4645],
        [  4.5759,   7.6370,   4.0035],
        [  6.0179,  10.7960,   2.3605],
        [  9.5829,  11.8140,   3.3305],
        [ 10.9439,  14.0420,   6.1655],
        [  7.3559,  15.1630,   6.8365],
        [  7.8169,  13.0660,  10.0025],
        [  7.2589,  15.7070,  12.7295],
        [ 10.0779,  13.9220,  14.3255],
        [ 10.5379,  10.4630,  12.8165],
        [ 11.5229,   9.0270,  16.2225],
        [  7.7969,   8.8340,  17.0025],
        [  6.5979,   9.0580,  13.4815],
        [  8.2229,   5.6740,  13.1785],
        [  5.4289,   4.5460,  15.4435],
        [  2.8209,   6.8870,  13.9365],
        [  1.5629,   4.6780,  11.1035],
        [  3.4289,   5.7040,   7.9225],
        [  3.1859,   2.4770,   5.8845],
        [ -0.0741,   0.5800,   5.5405],
        [ -2.4381,  -1.7250,   3.7485],
        [ -6.1601,  -1.0540,   3.6055],
        [ -9.3271,  -2.7290,   2.2835],
        [-12.9451,  -1.4730,   2.7505],
        [-15.1131,  -1.3120,   5.9165],
        [-17.3571,  -3.2210,   8.3275],
        [-19.9721,  -3.5060,   5.5755],
        [-17.0531,  -2.9980,   3.1415],
        [-16.6521,   0.4110,   1.5735],
        [-14.0701,   3.1760,   2.3985],
        [-10.5361,   2.1130,   1.6755],
        [ -9.9911,   0.6520,   5.1645],
        [ -7.8331,  -1.1530,   7.6145],
        [ -9.6951,  -4.4890,   7.5015],
        [ -7.3241,  -5.8790,  10.0205],
        [ -3.8941,  -5.1800,  11.5155],
        [ -1.6531,  -4.7530,   8.4595],
        [  0.4229,  -1.5630,   8.2875],
        [  4.1379,  -0.9960,   8.5335],
        [  5.1799,   2.3520,   9.8905],
        [  8.8469,   2.8980,   9.0315],
        [  9.7199,   2.5550,  12.7505],
        [  9.1189,  -1.2030,  12.8085],
        [ 11.5479,  -1.3360,   9.8655],
        [ 14.4459,  -2.7220,  11.9255],
        [ 12.4349,  -4.8970,  14.3405],
        [ 10.3569,  -7.6000,  12.7075],
        [ 11.5969,  -5.8580,   9.5455],
        [ 14.7789,  -4.0310,   8.4725],
        [ 13.9809,  -1.3070,   5.8785],
        [ 12.9399,  -3.4610,   2.8565],
        [  9.1769,  -4.0090,   2.5325],
        [  8.3839,  -7.6820,   1.6515],
        [  7.6499,  -8.0510,   5.3505],
        [  4.7899,  -5.6560,   4.7335],
        [  3.7689,  -7.9080,   1.8275],
        [  3.6259, -10.5430,   4.5165],
        [  1.2339,  -8.2430,   6.3905],
        [ -1.0831,  -8.6500,   3.4505],
        [ -1.6341, -12.3370,   2.7035],
        [ -4.1781, -12.8090,  -0.1065],
        [ -7.6491, -13.3450,   1.3865],
        [ -8.2571,  -9.7660,   2.6585],
        [ -6.8901,  -6.2780,   1.6565],
        [ -5.7561,  -4.5110,  -1.5445],
        [ -4.7131,  -0.7860,  -1.2905],
        [ -1.5261,  -0.0820,   0.6735],
        [  0.8199,   2.8720,   1.0785],
        [  4.0849,   1.8740,   2.6825],
        [  6.6859,   4.5440,   3.4675],
        [  9.2689,   3.4860,   0.8575],
        [ 12.7399,   4.2630,   2.2295],
        [ 14.9579,   3.6680,  -0.8015],
        [ 17.8599,   5.5100,  -2.4065],
        [ 17.8009,   6.1960,  -6.0975],
        [ 20.0069,   7.9450,  -8.6645],
        [ 18.4989,  10.7720,  -6.5925],
        [ 19.5069,   8.9300,  -3.3585],
        [ 17.6949,   7.9420,  -0.1585],
        [ 14.7309,  10.2570,  -0.3845],
        [ 11.8519,   8.0020,   0.6565],
        [  8.9419,   7.7400,  -1.7925],
        [  6.0549,   5.4980,  -0.6885],
        [  4.0869,   3.5550,  -3.2515],
        [  0.4709,   2.6280,  -3.8915],
        [ -0.1911,  -1.0910,  -3.4905],
        [ -3.2291,  -2.6240,  -5.2175],
        [ -2.5191,  -5.8360,  -3.3395],
        [ -4.9391,  -7.8870,  -5.4815],
        [ -4.4371, -11.5590,  -4.7365],
        [ -5.9451, -13.9810,  -7.2405],
        [ -7.8041, -15.4740,  -4.2435],
        [ -9.0581, -12.3820,  -2.3405],
        [-12.7871, -11.9580,  -2.0915],
        [-14.6061, -10.5220,  -5.1255],
        [-15.5721,  -7.3950,  -3.1565],
        [-11.8231,  -6.6380,  -3.1005],
        [-11.4681,  -6.9220,  -6.8855],
        [-14.6821,  -5.6400,  -8.4595],
        [-14.5241,  -2.8880,  -5.9435],
        [-10.9831,  -1.7350,  -6.1665],
        [-11.7061,  -1.5040,  -9.8425],
        [-13.4641,   1.7550,  -8.9755],
        [-11.8541,   2.1940,  -5.5445],
        [ -8.7581,   3.4200,  -7.3215],
        [-10.3621,   6.7330,  -7.9115],
        [-11.7021,   7.1350,  -4.4205],
        [ -8.3501,   6.3710,  -2.8225],
        [ -6.9671,   8.3780,  -5.7225],
        [ -9.0381,  11.4060,  -4.6645],
        [ -6.9951,  11.1450,  -1.4625],
        [ -3.8421,   9.7260,  -3.0795],
        [ -3.2421,  13.1340,  -4.6115],
        [  0.3349,  11.8600,  -4.9785],
        [  0.6089,   8.0670,  -4.8635],
        [  4.2649,   8.1370,  -5.8285],
        [  3.8399,   4.6340,  -7.1895],
        [  0.5989,   3.0420,  -8.3095],
        [  1.8199,  -0.5520,  -8.0505],
        [ -0.5801,  -3.5230,  -8.2745],
        [  0.3329,  -7.0510,  -7.0975],
        [ -1.6921, -10.1980,  -7.7945],
        [  0.9699, -12.1250,  -6.0545],
        [  2.9789, -10.9200,  -3.1115],
        [  5.8739, -10.9700,  -5.6385],
        [  4.5939,  -8.0660,  -7.7795],
        [  5.3239,  -6.1690,  -4.5745],
        [  8.7569,  -7.7480,  -4.5675],
        [ 11.1939,  -4.9830,  -3.6485],
        [ 13.3499,  -6.1800,  -6.5515],
        [ 10.2989,  -5.4540,  -8.6605],
        [  9.7119,  -2.3200,  -6.7725],
        [ 13.3509,  -1.9660,  -7.6775],
        [ 12.0499,  -2.0210, -11.2135],
        [  9.0959,   0.2730, -10.7425],
        [ 11.8269,   2.4730,  -9.2495],
        [ 14.8679,   0.3490, -10.1635],
        [ 13.5079,  -0.1140, -13.6425],
        [ 14.4969,   3.4560, -14.4755],
        [ 16.9579,   5.3010, -12.2165],
        [ 17.2869,   4.6290,  -8.4925],
        [ 20.3169,   3.2500,  -6.5695],
        [ 18.9109,   0.9940,  -3.8275],
        [ 16.1689,  -1.6230,  -3.9655],
        [ 17.6249,  -2.1110,  -7.4025],
        [ 21.1649,  -1.1840,  -6.2835],
        [ 22.3599,  -1.4410,  -2.6085],
        [ 19.7779,  -3.6650,  -0.8735],
        [ 18.5399,  -5.2350,  -4.1375]])
data2 = torch.Tensor([[-3.7667e+01,  5.5708e-02, -1.8390e+01],
        [-3.7796e+01, -1.8971e+00, -1.5108e+01],
        [-3.4426e+01, -1.4353e+00, -1.3374e+01],
        [-3.2713e+01, -1.2440e+00, -9.9734e+00],
        [-3.0059e+01,  1.4666e+00, -9.6119e+00],
        [-2.8001e+01,  1.0846e-01, -6.6896e+00],
        [-2.4242e+01, -3.6792e-01, -6.1745e+00],
        [-2.2667e+01, -3.4342e+00, -4.5052e+00],
        [-1.8941e+01, -4.2862e+00, -4.4476e+00],
        [-1.9265e+01, -7.7003e+00, -6.1284e+00],
        [-1.7500e+01, -6.0622e+00, -9.0810e+00],
        [-1.5847e+01, -3.2125e+00, -7.1375e+00],
        [-1.8185e+01, -1.2270e+00, -4.8721e+00],
        [-1.5737e+01, -3.6709e-01, -2.0488e+00],
        [-1.6295e+01,  2.8815e+00, -1.6363e-01],
        [-1.2795e+01,  3.4590e+00,  1.1698e+00],
        [-1.2291e+01,  4.1199e+00,  4.8081e+00],
        [-1.1692e+01,  7.7470e+00,  5.6982e+00],
        [-8.5330e+00,  6.4371e+00,  7.3114e+00],
        [-7.4450e+00,  5.9770e+00,  3.6824e+00],
        [-7.3097e+00,  9.7666e+00,  3.3199e+00],
        [-5.2975e+00,  9.7079e+00,  6.5765e+00],
        [-2.9400e+00,  6.8947e+00,  5.5941e+00],
        [-1.4930e+00,  8.5992e+00,  2.5310e+00],
        [-1.6981e+00,  1.1674e+01,  4.7726e+00],
        [ 7.0756e-01,  9.8795e+00,  7.1319e+00],
        [ 3.1625e+00,  9.4263e+00,  4.2785e+00],
        [ 3.9532e+00,  1.2939e+01,  2.9453e+00],
        [ 1.4040e+00,  1.5166e+01,  4.7110e+00],
        [ 4.0397e+00,  1.7733e+01,  5.5667e+00],
        [ 7.2994e+00,  1.6029e+01,  4.5231e+00],
        [ 7.4900e+00,  1.4331e+01,  7.9557e+00],
        [ 8.1005e+00,  1.7473e+01,  1.0131e+01],
        [ 1.1231e+01,  1.5752e+01,  1.1103e+01],
        [ 1.1098e+01,  1.2222e+01,  9.7857e+00],
        [ 1.3335e+01,  1.1006e+01,  1.2663e+01],
        [ 1.0189e+01,  1.0847e+01,  1.4806e+01],
        [ 7.7464e+00,  1.0602e+01,  1.1968e+01],
        [ 8.7209e+00,  6.9049e+00,  1.1706e+01],
        [ 6.6835e+00,  6.5643e+00,  1.4879e+01],
        [ 3.8472e+00,  8.6436e+00,  1.3445e+01],
        [ 2.3857e+00,  5.6345e+00,  1.1591e+01],
        [ 2.3375e+00,  5.5856e+00,  7.7884e+00],
        [ 3.1072e+00,  2.0765e+00,  6.5887e+00],
        [-1.5865e-01,  3.4659e-01,  6.4756e+00],
        [-2.3930e+00, -2.1574e+00,  4.7938e+00],
        [-6.1388e+00, -1.6923e+00,  4.6123e+00],
        [-9.0631e+00, -3.9540e+00,  3.6226e+00],
        [-1.2761e+01, -3.2794e+00,  2.8400e+00],
        [-1.4743e+01, -4.5406e+00,  5.8583e+00],
        [-1.6645e+01, -7.5683e+00,  7.1868e+00],
        [-1.8599e+01, -8.1611e+00,  3.9659e+00],
        [-1.5550e+01, -6.5079e+00,  2.3628e+00],
        [-1.6180e+01, -3.0656e+00,  9.1459e-01],
        [-1.4126e+01,  1.7063e-02,  2.0463e+00],
        [-1.0347e+01,  4.2111e-01,  1.8255e+00],
        [-1.0241e+01, -8.5528e-02,  5.5961e+00],
        [-7.1299e+00,  2.3059e-01,  7.8083e+00],
        [-5.0026e+00, -3.0198e+00,  7.8709e+00],
        [-6.8585e+00, -5.9479e+00,  9.4513e+00],
        [-4.2941e+00, -5.3906e+00,  1.2183e+01],
        [-1.1763e+00, -4.8647e+00,  9.9736e+00],
        [ 8.6310e-01, -1.7304e+00,  9.0509e+00],
        [ 4.4705e+00, -8.0493e-01,  9.6432e+00],
        [ 5.8171e+00,  2.4226e+00,  1.1171e+01],
        [ 8.7855e+00,  3.1666e+00,  8.9012e+00],
        [ 1.0820e+01,  3.5939e+00,  1.2130e+01],
        [ 1.0246e+01, -8.8331e-02,  1.2937e+01],
        [ 1.2325e+01, -5.2095e-01,  9.7922e+00],
        [ 1.5735e+01, -1.2238e+00,  1.1310e+01],
        [ 1.4644e+01, -2.9416e+00,  1.4510e+01],
        [ 1.2770e+01, -6.0454e+00,  1.3377e+01],
        [ 1.2874e+01, -4.7924e+00,  9.7943e+00],
        [ 1.5822e+01, -3.3515e+00,  7.7911e+00],
        [ 1.4465e+01, -1.0378e+00,  5.0279e+00],
        [ 1.3680e+01, -3.6449e+00,  2.2947e+00],
        [ 9.9406e+00, -4.5172e+00,  2.2090e+00],
        [ 9.5731e+00, -8.3384e+00,  2.4861e+00],
        [ 8.8429e+00, -7.8264e+00,  6.1895e+00],
        [ 5.7631e+00, -5.6913e+00,  5.5773e+00],
        [ 4.7654e+00, -8.4718e+00,  3.1622e+00],
        [ 4.9058e+00, -1.0694e+01,  6.2193e+00],
        [ 2.0979e+00, -8.4363e+00,  7.4521e+00],
        [ 5.4881e-01, -9.2215e+00,  4.1044e+00],
        [-7.9133e-01, -1.2777e+01,  4.5100e+00],
        [-2.1124e+00, -1.4799e+01,  1.5783e+00],
        [-5.7688e+00, -1.5093e+01,  2.6474e+00],
        [-6.5158e+00, -1.1384e+01,  3.1526e+00],
        [-7.6360e+00, -8.8076e+00,  4.9082e-01],
        [-6.1005e+00, -5.5512e+00,  1.7211e+00],
        [-4.4305e+00, -2.6141e+00, -8.3951e-02],
        [-1.4890e+00, -8.9470e-01,  1.7072e+00],
        [ 6.5703e-01,  2.2364e+00,  1.7743e+00],
        [ 4.0708e+00,  1.4981e+00,  3.1417e+00],
        [ 6.3965e+00,  4.4964e+00,  3.3152e+00],
        [ 9.3649e+00,  3.0514e+00,  1.4068e+00],
        [ 1.2903e+01,  3.9959e+00,  2.3934e+00],
        [ 1.5012e+01,  4.0010e+00, -7.6855e-01],
        [ 1.7093e+01,  6.4813e+00, -2.7672e+00],
        [ 1.6860e+01,  6.7094e+00, -6.5478e+00],
        [ 1.8441e+01,  8.7735e+00, -9.3399e+00],
        [ 1.6478e+01,  1.1277e+01, -7.3236e+00],
        [ 1.8603e+01,  1.0358e+01, -4.3089e+00],
        [ 1.7802e+01,  8.9948e+00, -8.4623e-01],
        [ 1.4045e+01,  9.3341e+00, -6.8345e-01],
        [ 1.1752e+01,  7.3556e+00,  1.5611e+00],
        [ 8.6478e+00,  7.2151e+00, -5.9994e-01],
        [ 5.6772e+00,  4.9649e+00,  2.2082e-01],
        [ 4.0700e+00,  2.9106e+00, -2.4876e+00],
        [ 4.6885e-01,  1.9136e+00, -2.8620e+00],
        [ 1.0034e-01, -1.8380e+00, -2.2303e+00],
        [-2.8967e+00, -3.8074e+00, -3.3828e+00],
        [-1.8608e+00, -6.7016e+00, -1.2613e+00],
        [-4.2541e+00, -9.1792e+00, -2.9105e+00],
        [-3.1465e+00, -1.2811e+01, -2.6667e+00],
        [-4.1430e+00, -1.5311e+01, -5.3609e+00],
        [-5.8165e+00, -1.7114e+01, -2.4499e+00],
        [-7.5048e+00, -1.3971e+01, -1.1387e+00],
        [-1.1181e+01, -1.3627e+01, -1.8806e+00],
        [-1.2285e+01, -1.2524e+01, -5.3611e+00],
        [-1.3486e+01, -9.3916e+00, -3.5419e+00],
        [-9.8778e+00, -8.2060e+00, -3.4371e+00],
        [-9.2396e+00, -8.8153e+00, -7.1549e+00],
        [-1.2534e+01, -7.7498e+00, -8.7803e+00],
        [-1.3099e+01, -5.0235e+00, -6.2983e+00],
        [-9.6994e+00, -3.5562e+00, -5.7720e+00],
        [-9.6287e+00, -3.4394e+00, -9.5232e+00],
        [-1.1911e+01, -4.0471e-01, -9.2136e+00],
        [-1.1039e+01,  3.1594e-01, -5.5594e+00],
        [-7.8530e+00,  1.8875e+00, -6.8414e+00],
        [-9.7300e+00,  4.9142e+00, -7.9297e+00],
        [-1.1715e+01,  5.3270e+00, -4.7592e+00],
        [-8.6223e+00,  5.1373e+00, -2.5782e+00],
        [-7.0012e+00,  7.2420e+00, -5.2813e+00],
        [-9.3327e+00,  1.0146e+01, -4.4527e+00],
        [-7.5716e+00,  1.0082e+01, -1.0842e+00],
        [-4.1812e+00,  8.8744e+00, -2.3806e+00],
        [-3.6666e+00,  1.2267e+01, -3.9815e+00],
        [-1.4082e-02,  1.1159e+01, -4.0774e+00],
        [ 2.0017e-01,  7.3519e+00, -3.9269e+00],
        [ 3.8078e+00,  7.4229e+00, -5.0519e+00],
        [ 3.3117e+00,  3.9223e+00, -6.3552e+00],
        [ 1.9425e-01,  2.0287e+00, -7.3102e+00],
        [ 1.6307e+00, -1.4503e+00, -6.8255e+00],
        [-6.4297e-01, -4.5105e+00, -6.6728e+00],
        [ 7.5864e-01, -7.8279e+00, -5.4364e+00],
        [-9.2582e-01, -1.1129e+01, -6.2041e+00],
        [ 1.7941e+00, -1.2828e+01, -4.2631e+00],
        [ 4.1304e+00, -1.1401e+01, -1.6988e+00],
        [ 6.6473e+00, -1.1498e+01, -4.6187e+00],
        [ 5.0123e+00, -8.6510e+00, -6.5802e+00],
        [ 6.1120e+00, -6.6786e+00, -3.5163e+00],
        [ 9.5556e+00, -8.1083e+00, -4.1800e+00],
        [ 1.1919e+01, -5.2798e+00, -3.5449e+00],
        [ 1.3617e+01, -6.0776e+00, -6.8487e+00],
        [ 1.0213e+01, -5.6377e+00, -8.4238e+00],
        [ 9.7394e+00, -2.6235e+00, -6.3382e+00],
        [ 1.3197e+01, -1.9677e+00, -7.7195e+00],
        [ 1.1449e+01, -2.1738e+00, -1.1047e+01],
        [ 8.4544e+00, -6.5794e-02, -1.0209e+01],
        [ 1.1171e+01,  2.3286e+00, -9.0127e+00],
        [ 1.4243e+01,  4.2315e-01, -1.0233e+01],
        [ 1.2496e+01, -7.3520e-02, -1.3538e+01],
        [ 1.2503e+01,  3.6668e+00, -1.4197e+01],
        [ 1.5140e+01,  5.7839e+00, -1.2433e+01],
        [ 1.6269e+01,  5.0735e+00, -8.8733e+00],
        [ 1.9678e+01,  4.0976e+00, -7.3916e+00],
        [ 1.8747e+01,  1.8398e+00, -4.4834e+00],
        [ 1.6008e+01, -7.6466e-01, -4.4968e+00],
        [ 1.7389e+01, -1.4468e+00, -7.9421e+00],
        [ 2.0938e+01, -5.7324e-01, -6.8117e+00],
        [ 2.2108e+01, -4.1629e-01, -3.1356e+00],
        [ 1.9727e+01, -2.8834e+00, -1.4537e+00],
        [ 1.8751e+01, -4.4652e+00, -4.7833e+00]])
data_obs = data1, data2
#Average between structures
average = Loader.Average_Structure(data_obs)
max_var = Loader.Max_variance(data_obs[0])
data_obs = max_var, data1, data2
#Run model
T2, R, M, X1, X2, ri_vec_samples,M_samples,T_samples = Model.Run(data_obs, average,"TEST")





















