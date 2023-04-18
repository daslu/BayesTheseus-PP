#%%

import os, sys
import random as rnd
import time
import ntpath
from collections import defaultdict
import ntpath
import pandas as pd  # Windows users: I copy-paste the pandas,dateutil and pyltz folders from anaconda 2!! into the site-packages folder of pymol(only for temporary use, other wise it gets confused with the paths of the packages)
import numpy as np
from numpyro.compat.infer import Trace_ELBO
from pandas import Series
# Biopython
from Bio import SeqRecord, Alphabet, SeqIO
from Bio.Seq import Seq
import Bio.PDB as PDB
from Bio.Seq import MutableSeq
from Bio.PDB.Polypeptide import is_aa
from Bio.SVDSuperimposer import SVDSuperimposer
#Pymol
import pymol
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
import scipy.stats
# TORCH:
import torch
from torch.distributions import constraints, transform_to
from torch.optim import Adam, LBFGS
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import tqdm
from ignite.handlers import EarlyStopping
from ignite.engine import Engine, Events
import torch.multiprocessing as mp
import jax.numpy as np_jax
import jax.random as random
from jax.scipy.special import logsumexp
import numpyro
import numpyro.distributions as dist
from numpyro.examples.datasets import BASEBALL, load_dataset
from numpyro.infer import MCMC, NUTS, Predictive, log_likelihood, SVI, init_to_median
from numpyro.infer.util import initialize_model, init_to_median
from numpyro.optim import _NumpyroOptim
#mp.set_start_method('spawn') ###For increasing number of chains
tqdm.monitor_interval = 0
#XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/lib/cuda"
#numpyro.set_platform("gpu") #slowwwww
tqdm.monitor_interval = 0
mpl.use('agg') #TkAgg
numpyro.set_host_device_count(2)
torch.multiprocessing.set_sharing_strategy('file_system')  # Needed to avoid runtime errors
class SVIEngine(Engine):
    def __init__(self, *args, step_args=None, **kwargs):
        self.svi = SVI(*args, **kwargs)
        self._step_args = step_args or {}
        super(SVIEngine, self).__init__(self._update)
    def _update(self, engine, batch):
        return -engine.svi.step(batch, **self._step_args)
_NumpyroOptim.state_dict = lambda self: self.get_state()
def Max_variance(structure):
    '''Calculates the maximum distance to the origin of the structure, this value will define the variance of the prior's distribution'''
    centered = Center_Jax(structure)
    mul = centered@np_jax.transpose(structure)
    max_var = np.sqrt(np_jax.max(np_jax.diag(mul)))
    return max_var
def Extract_coordinates_from_PDB(PDB_file,type):
    ''' Returns both the alpha carbon coordinates contained in the PDB file and the residues coordinates for the desired chains'''
    from Bio.PDB.PDBParser import PDBParser
    from Bio.PDB import MMCIFParser
    Name = ntpath.basename(PDB_file).split('.')[0]

    try:
        parser = PDB.PDBParser()
        structure = parser.get_structure('%s' % (Name), PDB_file)
    except:
        parser = MMCIFParser()
        structure = parser.get_structure('%s' % (Name), PDB_file)

    ############## Iterating over residues to extract all of them even if there is more than 1 chain
    if type=='models':
        CoordinatesPerModel = []
        for model in structure:
            model_coord =[]
            for chain in model:
                for residue in chain:
                    if is_aa(residue.get_resname(), standard=True):
                            model_coord.append(residue['CA'].get_coord())
            CoordinatesPerModel.append(model_coord)

        return CoordinatesPerModel
    elif type=='chains':
        CoordinatesPerChain=[]
        for model in structure:
            for chain in model:
                chain_coord = []
                for residue in chain:
                    if is_aa(residue.get_resname(), standard=True):
                        chain_coord.append(residue['CA'].get_coord())
                CoordinatesPerChain.append(chain_coord)
        return CoordinatesPerChain

    elif type =='all':
        alpha_carbon_coordinates = []
        for chain in structure.get_chains():
            for residue in chain:
                if is_aa(residue.get_resname(), standard=True):
                    alpha_carbon_coordinates.append(residue['CA'].get_coord())
        return alpha_carbon_coordinates
def Average_Structure(tuple_struct):
    average = sum(list(tuple_struct)) / len(
        tuple_struct)  # sum element-wise the list of tensors containing the coordinates #tf.add_n
    return average
def Center_numpy(Array):
    '''Centering to the origin the data'''
    mean = np.mean(Array, axis=0)
    centered_array = Array - mean
    return centered_array
def Center_Jax(Array):
    '''Centering to the origin the data'''
    mean = np_jax.mean(Array, axis=0)
    centered_array = Array - mean
    return centered_array
def sample_R(ri_vec):
    """Inputs a sample of unit quaternion and transforms it into a rotation matrix"""
    # argument i guarantees that the symbolic variable name will be identical everytime this method is called
    # repeating a symbolic variable name in a model will throw an error
    # the first argument states that i will be the name of the rotation made
    theta1 = 2 * np_jax.pi * ri_vec[1]
    theta2 = 2 * np_jax.pi * ri_vec[2]

    r1 = np_jax.sqrt(1 - ri_vec[0])
    r2 = np_jax.sqrt(ri_vec[0])

    qw = r2 * np_jax.cos(theta2)
    qx = r1 * np_jax.sin(theta1)
    qy = r1 * np_jax.cos(theta1)
    qz = r2 * np_jax.sin(theta2)

    # filling the rotation matrix
    # Evangelos A. Coutsias, et al "Using quaternions to calculate RMSD" In: Journal of Computational Chemistry 25.15 (2004)
    # Row one
    R_00 = qw ** 2 + qx ** 2 - qy ** 2 - qz ** 2
    R_01 = 2 * (qx * qy - qw * qz)
    R_02 = 2 * (qx * qz + qw * qy)

    # Row two
    R_10 = 2 * (qx * qy + qw * qz)
    R_11 = qw ** 2 - qx ** 2 + qy ** 2 - qz ** 2
    R_12 = 2 * (qy * qz - qw * qx)

    # Row three
    R_20 = 2 * (qx * qz - qw * qy)
    R_21 = 2 * (qy * qz + qw * qx)
    R_22 = qw ** 2 - qx ** 2 - qy ** 2 + qz ** 2

    R = np_jax.array([[R_00,R_01,R_02],[R_10,R_11,R_12],[R_20,R_21,R_22]]) #google coloured brackets


    return R
def RMSD_numpy(X1, X2):
    "Calculates average distance"
    return np.linalg.norm(X1-X2)
def RMSD(X1, X2):
    "Calculates pairwise distances per aa position"
    import torch.nn.functional as F
    return F.pairwise_distance(X1, X2)
def RMSD_biopython(x, y):
    "Kalbsh algorithm : Performs SVD to find rotation and traslation"
    sup = SVDSuperimposer()
    sup.set(x, y)
    sup.run()
    rot, tran = sup.get_rotran()
    return rot
def Read_Data(prot1,prot2,type='models',models =(0,1),RMSD=True):
    '''Reads different types of proteins and extracts the alpha carbons from the models, chains or all . The model,
    chain or aminoacid range numbers are indicated by the tuple models'''

    if type == 'models':
        X1_coordinates = Extract_coordinates_from_PDB('{}'.format(prot1),type)[models[0]]
        X2_coordinates = Extract_coordinates_from_PDB('{}'.format(prot2),type)[models[1]]
    elif type == 'chains':
        X1_coordinates = Extract_coordinates_from_PDB('{}'.format(prot1),type)[models[0]][1:141]
        X2_coordinates = Extract_coordinates_from_PDB('{}'.format(prot2),type)[models[1]][0:140]

    elif type == 'all':
        X1_coordinates = Extract_coordinates_from_PDB('{}'.format(prot1),type)[models[0]:models[1]]
        X2_coordinates = Extract_coordinates_from_PDB('{}'.format(prot2),type)[models[0]:models[1]]

    #Apply RMSD to the protein that needs to be superimposed
    X1_Obs_Stacked = Center_numpy(np.vstack(X1_coordinates))
    X2_Obs_Stacked = Center_numpy(np.vstack(X2_coordinates))
    if RMSD:
        X2_Obs_Stacked = np.dot(X2_Obs_Stacked,RMSD_biopython(X1_Obs_Stacked,X2_Obs_Stacked))
        X1_Obs_Stacked = X1_Obs_Stacked
    else:
        X1_Obs_Stacked = X1_Obs_Stacked
        X2_Obs_Stacked = X2_Obs_Stacked

    data_obs = (X1_Obs_Stacked,X2_Obs_Stacked)

    # ###PLOT INPUT DATA################
    x = Center_numpy(np.vstack(X1_coordinates))[:, 0]
    y=Center_numpy(np.vstack(X1_coordinates))[:, 1]
    z=Center_numpy(np.vstack(X1_coordinates))[:, 2]
    fig = plt.figure(figsize=(18, 16), dpi=80)
    ax = fig.add_subplot(111, projection='3d')
    plt.plot(x, y, z)
    ax.plot(x, y,z ,c='b', label='data1',linewidth=3.0)
    #orange graph
    x2 = Center_numpy(np.vstack(X2_coordinates))[:, 0]
    y2=Center_numpy(np.vstack(X2_coordinates))[:, 1]
    z2=Center_numpy(np.vstack(X2_coordinates))[:, 2]
    ax.plot(x2, y2,z2, c='r', label='data2',linewidth=3.0)
    ax.legend()
    plt.savefig(r"Initial.png")
    plt.clf() #Clear the plot, otherwise it will give an error when plotting the loss
    plt.close()

    return data_obs
def expand_by(x,sample_shape):
    x_shape = x.shape
    sample_shape = tuple(sample_shape)
    for i in range(len(sample_shape)):
        x = np.expand_dims(x,-1)
    return np.broadcast_to(x,x_shape+sample_shape)
def model(data):
    max_var,data1, data2 = data
    ### 1. prior over mean M

    #M = numpyro.sample("M", dist.StudentT(1, 0, 3), sample_shape=(data1.shape[0], data1.shape[1])) #Also works, change to this
    mu_M = np.array(0)
    mu_M =expand_by(mu_M,[data1.shape[0],data1.shape[1]])
    sd_M = np.array(1)
    sd_M = expand_by(sd_M, [data1.shape[0], data1.shape[1]])
    df_M = np.array(0)
    df_M = expand_by(df_M, [data1.shape[0], data1.shape[1]])
    M = numpyro.sample("M", dist.Cauchy(mu_M,sd_M).to_event(2)) #Student T with df = 1 is Cauchy

    # # ### 2. Prior over variances for the normal distribution
    mu_U = np.array(1)
    mu_U = expand_by(mu_U,[data1.shape[0]])
    U = numpyro.sample("U", dist.HalfNormal(mu_U).to_event(1))
    U = U.reshape(data1.shape[0], 1) #.repeat(1, 3).view(-1)
    U = U.repeat(3,axis=-1)
    U = U.reshape(-1)
    # # ## 3. prior over translations T_i: Sample translations for each of the x,y,z coordinates
    mu_T2= np.array(0)
    mu_T2 = expand_by(mu_T2,[3])
    sd_T2 =np.array(1)
    sd_T2 = expand_by(sd_T2,[3])
    T2 = numpyro.sample("T2", dist.Normal(mu_T2, sd_T2).to_event(1))
    # # ## 4. prior over rotations R_i
    mu_ri_vec = expand_by(np.array(0),[3])
    sd_ri_vec = expand_by(np.array(1),[3])
    ri_vec = numpyro.sample("ri_vec",dist.Uniform(mu_ri_vec, sd_ri_vec))
    R =sample_R(ri_vec)
    #M_T1 = M
    M_R2_T2 = M@R  + T2
    # # # 5. Sampling from several Univariate Distributions (approximating the posterior distribution ): The observations are conditionally independant given the U, which is sampled outside the loop
    with numpyro.plate("plate_univariate", data1.shape[0]*data1.shape[1],dim=-1):
          numpyro.sample("X1", dist.StudentT(1,M.reshape(-1), U),obs=data1.reshape(-1))
          numpyro.sample("X2", dist.StudentT(1,M_R2_T2.reshape(-1), U), obs=data2.reshape(-1))
def Run(data_obs, average,name1):

    ###POSTERIOR PROBABILITY calculations: MCMC and NUTS
    # I had to fix a problem at /home/lys/anaconda3/lib/python3.5/site-packages/pyro/util.py by initializing the seed to rng_seed = random.randint(0,2**32-1)
    #map_points = _get_initial_trace(data_obs,average)
    # MCMC initialization
    chains=1
    warmup= 500
    samples = 1000
    rng_key = random.PRNGKey(4)

    # Running MCMC using NUTS as selected kernel
    nuts_kernel = NUTS(model, max_tree_depth=10, init_strategy=init_to_median(15),target_accept_prob=0.8)#,step_size=3)
    #nuts_kernel.initial_trace = _get_initial_trace(data_obs, average)
    mcmc = MCMC(nuts_kernel, warmup, samples, chains,chain_method='parallel')
    mcmc.run(rng_key,data_obs)
    max_var, data1, data2 = data_obs
    #######SAMPLES FROM THE POSTERIOR
    mcmc_samples = mcmc.get_samples()
    #######PARAMETERS:
    #ROTATION
    ri_vec_post_samples = mcmc_samples["ri_vec"]
    ri_vec_mean = ri_vec_post_samples.mean(axis=0)
    ri_vec_variance = ri_vec_post_samples.var(axis=0)
    R = sample_R(ri_vec_mean)
    # MEAN STRUCTURE
    M_post_samples = mcmc_samples["M"]
    M = M_post_samples.mean(axis=0)
    M_variance = M_post_samples.var(axis=0)
    #TRANSLATION
    T2_post_samples = mcmc_samples["T2"]
    T2_mean = T2_post_samples.mean(axis=0)
    T2_variance = T2_post_samples.var(axis=0)

    #Observed
    X1 = data1 #- T1_vec_mean.cpu().numpy()  # X1 -T1
    X2 = np.dot(data2 - T2_mean, np.transpose(R))  # (X2-T2)R-1

    import matplotlib
    matplotlib.rcParams['legend.fontsize'] = 15

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
    x3 = M[:, 0]
    y3 = M[:, 1]
    z3 = M[:, 2]
    #ax.plot(x3, y3, z3, c='g', label='M', linewidth=3.0)
    ax.legend()

    plt.title("Initialized MCMC and NUTS model")
    plt.savefig("{}_PLOTS_and_FILES/Bayesian_Result_Samples_{}_{}".format(name1,name1,samples + warmup))

    plt.clf()
    plt.figure(figsize=(5, 5))
    plt.plot(RMSD(torch.from_numpy(data1),torch.from_numpy(data2)), linewidth = 2.0)
    plt.plot(RMSD(torch.from_numpy(X1),torch.from_numpy(X2)), linewidth=2.0)
    plt.ylabel('Pairwise distances',fontsize='10')
    plt.xlabel('Amino acid position',fontsize='10')
    plt.ylim(0,40)
    plt.tick_params(labelsize=10)
    plt.title('{}'.format(name1.upper()),fontsize ='10')
    plt.gca().legend(('Kabsch', 'Theseus-PP MAP'),fontsize='10')
    plt.savefig("{}_PLOTS_and_FILES/Distance_Differences_Average_Bayesian_{}".format(name1,name1),dpi=600)
    plt.close()

    return T2_mean, R, M, X1, X2, ri_vec_post_samples,M_post_samples,T2_post_samples #Values for the mean structure
def Write_PDB(initialPDB, Rotation, Translation):
    ''' Transform the atom coordinates from the original PDB file and overwrite it '''
    from Bio.PDB.PDBParser import PDBParser
    from Bio.PDB import MMCIFParser, PDBIO
    Name = ntpath.basename(initialPDB).split('.')[0]

    try:
        parser = PDB.PDBParser()
        structure = parser.get_structure('%s' % (Name), initialPDB)
    except:
        parser = MMCIFParser()
        structure = parser.get_structure('%s' % (Name), initialPDB)
    for atom in structure.get_atoms():
        atom.transform(Rotation, -Translation)
    io = PDBIO()
    io.set_structure(structure)
    io.save("{}_PLOTS_and_FILES/Transformed_{}".format(initialPDB,ntpath.basename(initialPDB)))
def write_ATOM_line(structure, file_name):
    import os
    """Transform coordinates to PDB file: Add intermediate coordinates to be able to visualize Mean structure in PyMOL"""
    expanded_structure = np.ones(shape=(2 * len(structure) - 1, 3))  # The expanded structure contains extra rows between the alpha carbons
    averagearray = np.zeros(shape=(len(structure) - 1, 3))  # should be of size len(structure) -1
    for index, row in enumerate(structure):
        if index != len(structure) and index != len(structure) - 1:
            averagearray[int(index)] = (structure[int(index)] + structure[int(index) + 1]) / 2
        else:
            pass
    # split the expanded structure in sets , where each set will be structure[0] + number*average
    # The even rows of the 'expanded structure' are simply the rows of the original structure
    expanded_structure[0::2] = structure
    expanded_structure[1::2] = averagearray
    structure = expanded_structure
    aa_name = "ALA"
    aa_type = "CA"

    if os.path.isfile(file_name):
        os.remove(file_name)
        for i in range(len(structure)):
            with open(file_name, 'a') as f:
                f.write(
                    "ATOM{:7d} {}   {} A{:4d}{:12.3f}{:8.3f}{:8.3f}  0.00  0.00    X    \n".format(i, aa_type, aa_name,
                                                                                                   i, structure[i, 0],
                                                                                                   structure[i, 1],
                                                                                                   structure[i, 2]))
    else:
        for i in range(len(structure)):
            with open(file_name, 'a') as f:
                f.write(
                    "ATOM{:7d} {}   {} A{:4d}{:12.3f}{:8.3f}{:8.3f}  0.00  0.00    X    \n".format(i, aa_type, aa_name,
                                                                                                   i, structure[i, 0],
                                                                                                   structure[i, 1],
                                                                                                   structure[i, 2]))
def Pymol(*args):
    '''Visualization program'''
    #LAUNCH PYMOL
    launch=False
    if launch:
        pymol.pymol_argv = ['pymol'] + sys.argv[1:]
        pymol.finish_launching(['pymol','-q'])

    pymol.cmd.set("max_threads", 4) #Deactivate to call pymol GUI
    def Colour_Backbone(selection,color,color_digit):
        #pymol.cmd.select("alphas", "name ca") #apparently nothing is ca
        #pymol.cmd.select("sidechains", "! alphas") #select the opposite from ca, which should be the side chains, not working :(
        pymol.cmd.show("sticks", selection)
        pymol.cmd.set_color(color,color_digit)
        pymol.cmd.color(color,selection)

    # Load Structures and apply the function
    #colornames=['red','green','blue','orange','purple','yellow','black','aquamarine']
    #Palette of colours
    pal1 = sns.color_palette("OrRd",1)
    pal2 = sns.color_palette("PuBuGn_d",100) #RGB numbers for the palette colours
    colornames1 = ["red_{}".format(i) for i in range(0, len(pal1))]#So that X1 is red
    colornames2 = ["blue_{}".format(i) for i in range(0,len(pal2))]
    colornames = colornames1 + colornames2
    pal = pal1 + pal2

    snames=[]
    for file,color,color_digit in zip(args,colornames,pal):
        sname = ntpath.basename(file)
        snames.append(sname)
        pymol.cmd.load(file, sname) #discrete 1 will create different sets of atoms for each model
        pymol.cmd.bg_color("white")
        pymol.cmd.extend("Colour_Backbone", Colour_Backbone)
        Colour_Backbone(sname,color,color_digit)
    pymol.cmd.png("{}_PLOTS_and_FILES/Superposition_Bayesian_Pymol_{}".format(snames[0].split('_')[1],snames[0].split('_')[1]))
def Pymol_Samples(data1,data2,name1,R_samples,T_samples,samples):
    '''Create the PDB files to be sent to plot to PyMOL. Make plots showing pairwise distances'''
    #Process the dataframes
    indexes = rnd.sample(range(0, samples), samples) #not warm up samples
    X1 = data1
    T_average = T_samples.mean(axis=0)
    R_average = sample_R(R_samples.mean(axis=0))
    X2_average = np.dot(data2 - T_average, np.transpose(R_average))
    plt.clf()
    fig1, ax1 = plt.subplots(figsize=(5,5))
    fig2, ax2 = plt.subplots(figsize=(5,5))
    ax1.plot(RMSD(torch.from_numpy(data1),torch.from_numpy(data2)), linewidth=2.0, color="blue", label= "Kalbsh SVD")
    ax2.hist(RMSD_numpy(data1,data2),alpha=1,facecolor="blue",label="Kalbsh SVD")
    for i in range(0,samples):
        Rotation = sample_R(R_samples[i,:]) #torch
        Translation = T_samples[i,:] #numpy
        X2 = np.dot(data2 - Translation, np.transpose(Rotation))
        write_ATOM_line(X2, os.path.join("{}_PLOTS_and_FILES".format(name1),'Result_{}_X2_{}.pdb'.format(name1,i)))
        ax1.plot(RMSD(torch.from_numpy(X1),torch.from_numpy(X2)), linewidth=0.5,color = plt.cm.autumn(i))
        ax2.hist(RMSD_numpy(X1,X2),alpha=1,facecolor= plt.cm.autumn(i), label="Theseus-PP NUTS samples")

    ax1.plot(RMSD(torch.from_numpy(X1), torch.from_numpy(X2_average)), linewidth=0.5, color="green",label="Theseus-PP NUTS average")
    ax1.set_ylabel('Pairwise distances', fontsize='12')
    ax1.set_xlabel('Amino acid position', fontsize='12')
    ax1.set_title('{}'.format(name1.upper()), fontsize='12')
    ax1.set_ylim(0,40)
    green_patch = mpatches.Patch(color='green', label='Theseus-PP NUTS average')
    orange_patch = mpatches.Patch(color='orange', label='Theseus-PP NUTS samples')
    blue_patch = mpatches.Patch(color='blue', label='Kabsch SVD')
    ax1.legend(handles = [blue_patch,orange_patch,green_patch], fontsize=10)
    #ax1.legend(('Kabsch SVD', "Theseus-PP NUTS samples",'Theseus-PP NUTS average'), fontsize='10',prop={'size': 10})
    fig1.savefig("{}_PLOTS_and_FILES/Distance_Differences_Bayesian_{}.png".format(name1,name1),dpi=600)
    ax2.set_ylabel("probability")
    ax2.set_title('{} Probability histogram'.format(name1.upper()), fontsize='12')
    fig2.savefig("{}_PLOTS_and_FILES/Probability_Histogram_{}.png".format(name1, name1))
    plt.close()
    names = [os.path.join("{}_PLOTS_and_FILES/".format(name1),'Result_{}_X2_{}.pdb'.format(name1,i)) for i in indexes] #exchange indexes with range(0,samples)
    names = [os.path.join("{}_PLOTS_and_FILES/".format(name1),'Result_{}_X1.pdb'.format(name1))] + names
    Pymol(*names)
def Folders(folder_name):
    """ Folder for all the generated images It will updated everytime!!! Save the previous folder before running again. Creates folder in current directory"""
    import os
    import shutil
    basepath = os.getcwd()
    if not basepath:
        newpath = folder_name
    else:
        newpath = basepath + "/%s" % folder_name

    if not os.path.exists(newpath):
        try:
            original_umask = os.umask(0)
            os.makedirs(newpath, 0o777)
        finally:
            os.umask(original_umask)
    else:
        shutil.rmtree(newpath)  # removes all the subdirectories!
        os.makedirs(newpath,0o777)

#if __name__ == "__main__":
name1 = '2lmp' #2nl7:139 #2do0=114
name2 ='2lmp'
models = (0,3)
samples =100
print(name1 + "\t" + str(models))
Folders("{}_PLOTS_and_FILES/".format(name1))
data_obs = Read_Data('../PDB_files/{}.pdb'.format(name1), '../PDB_files/{}.pdb'.format(name2),type='models',models =models,RMSD=True)
max_var = Max_variance(data_obs[0])
average = Average_Structure(data_obs)
data1, data2 = data_obs
print(len(data1))
write_ATOM_line(data1, os.path.join('{}_PLOTS_and_FILES/'.format(name1),'RMSD_{}_data1.pdb'.format(name1)))
write_ATOM_line(data2, os.path.join('{}_PLOTS_and_FILES/'.format(name1),'RMSD_{}_data2.pdb'.format(name1)))

#Pymol('{}_PLOTS_and_FILES//RMSD_{}_data1.pdb'.format(name1,name1), '{}_PLOTS_and_FILES//RMSD_{}_data2.pdb'.format(name2,name2))
#exit()
data_obs = max_var, data1, data2
start  = time.time()
T2, R, M, X1, X2, ri_vec_samples,M_samples,T_samples = Run(data_obs, average,name1)
stop = time.time()


print("Time:", stop-start)

write_ATOM_line(M, os.path.join("{}_PLOTS_and_FILES/".format(name1),'M_{}.pdb'.format(name1)))
write_ATOM_line(X1, os.path.join("{}_PLOTS_and_FILES/".format(name1),'Result_{}_X1.pdb'.format(name1)))
write_ATOM_line(X2, os.path.join("{}_PLOTS_and_FILES/".format(name1),'Result_{}_X2.pdb'.format(name2)))
#Write_PDB(r"../PDB_files/{}.pdb".format(name1), np.transpose(R), T1)
#Write_PDB(r"../PDB_files/{}.pdb".format(name2), np.transpose(R), T2)
#Pymol("Result_{}_X1.pdb".format(name1), "Result_{}_X2.pdb".format(name2))
Pymol_Samples(data1,data2,name1,ri_vec_samples,T_samples,samples)




















