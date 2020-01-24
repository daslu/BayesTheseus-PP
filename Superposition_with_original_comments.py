import os,sys
import ntpath
from collections import defaultdict
import ntpath
import pandas as pd #Windows users: I copy-paste the pandas,dateutil and pyltz folders from anaconda 2!! into the site-packages folder of pymol(only for temporary use, other wise it gets confused with the paths of the packages)
import numpy as np
from pandas import Series
#Biopython
from Bio import SeqRecord,Alphabet,SeqIO
from Bio.Seq import Seq
import Bio.PDB as PDB
from Bio.Seq import MutableSeq
from Bio.PDB.Polypeptide import is_aa
from Bio.SVDSuperimposer import SVDSuperimposer
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats
#TORCH: "Tensors"
import torch
from torch.distributions import constraints, transform_to
#PYRO
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.autoguide import AutoDelta, AutoDiagonalNormal
from torch.optim import Adam, LBFGS
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate,Trace_ELBO, TraceGraph_ELBO
#from pyro.infer.static_svi import StaticSVI
#from static_svi import StaticSVI
import matplotlib.pyplot as plt
import tqdm
#Early STOPPING
from ignite.handlers import EarlyStopping
from ignite.engine import Engine,Events
from pyro.infer import SVI
from pyro.optim import PyroOptim

class SVIEngine(Engine):
    def __init__(self, *args, step_args=None, **kwargs):
        self.svi = SVI(*args, **kwargs)
        self._step_args = step_args or {}
        super(SVIEngine, self).__init__(self._update)

    def _update(self, engine, batch):
        return -engine.svi.step(batch, **self._step_args)
PyroOptim.state_dict = lambda self: self.get_state()
class GaussianRandomWalk(dist.TorchDistribution):
    has_rsample = True
    arg_constraints = {'scale': constraints.positive}
    support = constraints.real

    def __init__(self, scale, num_steps=1):
        self.scale = scale
        batch_shape, event_shape = scale.shape, torch.Size([num_steps])
        super(GaussianRandomWalk, self).__init__(batch_shape, event_shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = sample_shape + self.batch_shape + self.event_shape
        walks = self.scale.new_empty(shape).normal_()
        return walks.cumsum(-1) * self.scale.unsqueeze(-1)

    def log_prob(self, x):
        init_prob = dist.Normal(self.scale.new_tensor(0.), self.scale).log_prob(x[..., 0])
        step_probs = dist.Normal(x[..., :-1], self.scale).log_prob(x[..., 1:])
        return init_prob + step_probs.sum(-1)
def Extract_coordinates_from_PDB(PDB_file):
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
    alpha_carbon_coordinates = []

    for chain in structure.get_chains():
        for residue in chain:
            if is_aa(residue.get_resname(), standard=True):
                #try:
                    alpha_carbon_coordinates.append(residue['CA'].get_coord() )
                #except:
                    #pass
    return alpha_carbon_coordinates
def Center_numpy(Array):
    '''Centering to the origin the data'''
    mean = np.mean(Array,axis=0)
    centered_array = Array-mean
    return centered_array
def Center_torch(Array):
    '''Centering to the origin the data'''
    mean = torch.mean(Array, dim=0)
    centered_array = Array - mean
    return centered_array
def kronecker(t1, t2):
    """
    Computes the Kronecker product between two tensors.
    See https://en.wikipedia.org/wiki/Kronecker_product
    """
    t1_height, t1_width = t1.size()
    t2_height, t2_width = t2.size()
    out_height = t1_height * t2_height
    out_width = t1_width * t2_width

    tiled_t2 = t2.repeat(t1_height, t1_width)
    expanded_t1 = (
        t1.unsqueeze(2)
          .unsqueeze(3)
          .repeat(1, t2_height, t2_width, 1)
          .view(out_height, out_width)
    )

    return expanded_t1 * tiled_t2
def sample_R(ri_vec):
    """Inputs a sample of unit quaternion and transforms it into a rotation matrix"""
    # argument i guarantees that the symbolic variable name will be identical everytime this method is called
    # repeating a symbolic variable name in a model will throw an error
    # the first argument states that i will be the name of the rotation made
    theta1 = 2 * np.pi * ri_vec[1]
    theta2 = 2 * np.pi * ri_vec[2]

    r1 = torch.sqrt(1 - ri_vec[0])
    r2 = torch.sqrt(ri_vec[0])

    qw = r2 * torch.cos(theta2)
    qx = r1 * torch.sin(theta1)
    qy = r1 * torch.cos(theta1)
    qz = r2 * torch.sin(theta2)

    R= torch.eye(3,3)
    # filling the rotation matrix
    # Evangelos A. Coutsias, et al "Using quaternions to calculate RMSD" In: Journal of Computational Chemistry 25.15 (2004)

    # Row one
    R[0, 0]= qw**2 + qx**2 - qy**2 - qz**2
    R[0, 1]= 2*(qx*qy - qw*qz)
    R[0, 2]= 2*(qx*qz + qw*qy)

    # Row two
    R[1, 0]= 2*(qx*qy + qw*qz)
    R[1, 1]= qw**2 - qx**2 + qy**2 - qz**2
    R[1, 2]= 2*(qy*qz - qw*qx)

    # Row three
    R[2, 0]= 2*(qx*qz - qw*qy)
    R[2, 1]= 2*(qy*qz + qw*qx)
    R[2, 2]= qw**2 - qx**2 - qy**2 + qz**2
    return R
def RMSD_numpy(X1,X2):
    import torch.nn.functional as F
    return F.pairwise_distance(torch.from_numpy(X1),torch.from_numpy(X2))
def RMSD(X1,X2):
    import torch.nn.functional as F
    return F.pairwise_distance(X1,X2)
def RMSD_biopython(x,y):
    sup = SVDSuperimposer()
    sup.set(x, y)
    sup.run()
    rot, tran = sup.get_rotran()
    return rot
def Read_Data(prot1,prot2):
    #DATASETS#######################
    #Create the observed variables: Center, stack and convert to a tensor
    X1_coordinates = Extract_coordinates_from_PDB('../PDB_files/{}'.format(prot1)) #71,3 #List of tensors
    X2_coordinates = Extract_coordinates_from_PDB('../PDB_files/{}'.format(prot2)) #71,3
    #Stack and Convert numpy to pytorch tensor : torch.from_numpy()
    # X1_Obs_Stacked = torch.from_numpy(Center_numpy(np.vstack(X1_coordinates)))
    # X2_Obs_Stacked = torch.from_numpy(Center_numpy(np.vstack(X2_coordinates)))

    #Apply RMSD to the protein that needs to be superimposed
    X1_Obs_Stacked = Center_numpy(np.vstack(X1_coordinates))
    X2_Obs_Stacked = Center_numpy(np.vstack(X2_coordinates))
    X2_Obs_Stacked = torch.from_numpy(np.dot(X2_Obs_Stacked,RMSD_biopython(X1_Obs_Stacked,X2_Obs_Stacked)))
    X1_Obs_Stacked = torch.from_numpy(X1_Obs_Stacked)
    data_obs = (X1_Obs_Stacked,X2_Obs_Stacked)

    ###PLOT INPUT DATA################
    x = Center_numpy(np.vstack(X1_coordinates))[:, 0]
    y=Center_numpy(np.vstack(X1_coordinates))[:, 1]
    z=Center_numpy(np.vstack(X1_coordinates))[:, 2]
    fig = plt.figure(figsize=(18, 16), dpi=80)
    ax = fig.add_subplot(111, projection='3d')
    plt.plot(x, y, z)
    ax.plot(x, y,z ,c='b', label='y1',linewidth=7.0)
    #orange graph
    x2 = Center_numpy(np.vstack(X2_coordinates))[:, 0]
    y2=Center_numpy(np.vstack(X2_coordinates))[:, 1]
    z2=Center_numpy(np.vstack(X2_coordinates))[:, 2]
    ax.plot(x2, y2,z2, c='r', label='y2')
    #plt.show()
    #plt.savefig(r"C:\Users\Lys Sanz Moreta\Dropbox\PhD\Superpositioning\Initial.png")
    plt.savefig(r"Initial.png")
    plt.clf() #Clear the plot, otherwise it will give an error when plotting the loss

    # rmsd = RMSD_numpy(Center_numpy(np.vstack(X1_coordinates)),Center_numpy(np.vstack(X2_coordinates)))
    # plt.plot(rmsd.numpy())
    # plt.show()
    return data_obs
def model(data):
    data1, data2 = data

    ### 1. prior over mean M
    # Sample the x, y and z coordinates as randomly as possible
    M1 = pyro.sample("M1", GaussianRandomWalk(torch.tensor(3.), torch.tensor(len(data1))))
    M2 = pyro.sample("M2", GaussianRandomWalk(torch.tensor(3.), torch.tensor(len(data1))))
    M3 = pyro.sample("M3", GaussianRandomWalk(torch.tensor(3.), torch.tensor(len(data1))))
    M = torch.stack([M1, M2, M3], dim=1)
    M = Center_torch(M)#centering within the model breaks the structure
    ## 2. prior over translations T_i: Sample translations for each of the x,y,z coordinates
    T1 = pyro.sample("T1", dist.Normal(0, 1).expand_by([3]))
    T2 = pyro.sample("T2", dist.Normal(0, 1).expand_by([3]))
    ## 3. prior over rotations R_i
    #element1 = pyro.sample("element1",dist.Beta(18, 2).expand_by([1]).independent(1))  # Beta distribution with mean 0.9
    #element2 = pyro.sample("element2",dist.Beta(2, 18).expand_by([1]).independent(1))  # Beta distribution with mean 0.1
    #element3 = pyro.sample("element3",dist.Beta(18, 28).expand_by([1]).independent(1))  # Beta distribution with mean 0.9
    #ri_vec = torch.cat((element1,element2, element3), 0)
    ri_vec = pyro.sample("ri_vec",dist.Uniform(0, 1).expand_by([3]))  # Uniform distribution

    #element1 = pyro.sample("element1",dist.Uniform(0, 1).expand_by([1]).independent(1))  # First element from uniform distribution
    #element2 = pyro.sample("element2",dist.Uniform(0, 1).expand_by([1]).independent(1))  # Second element from uniform distribution
    #element3 = pyro.sample("element3",dist.Uniform(0, 1).expand_by([1]).independent(1))  # Third element from uniform distribution

    R = sample_R(ri_vec)
    ### 4. putting the model together
    # prior over the first structure
    M_T1 = M + T1 #Thanks to broadcasting, we don't need to have nX3 and nx3 to sum , we can do nx3 + 3,
    #M_T1 = M
    M_T1 = M_T1.view((np.shape(M_T1)[0] * np.shape(M_T1)[1]))  # Reshape to flatten the matrix to len(data1)*3
    #M_R2_T2 = M@R
    M_R2_T2 = M@R + T2
    M_R2_T2 = M_R2_T2.view((np.shape(M_R2_T2)[0] * np.shape(M_R2_T2)[1]))

    ### 5. prior over gaussian noise E_i: Several alternatives to sample parameters for multivariate distribution
    # a) Row covariance matrix (U): Sample from halfnormal distribution, covariance between atoms
    U = torch.diag(pyro.sample("U", dist.HalfNormal(0.01).expand_by([len(data1)])))  # Concentration=Alpha=1 #Rate=Beta=1
    #U = torch.diag(pyro.sample("U", dist.Delta(torch.tensor(0.1)).expand_by([len(data1)]).independent(1)))
    # b) Column covariance matrix with shape 3x3: Covariance between coordinates x,y and z
    V = torch.eye(3)
    # c) Kronecker product
    #k = kronecker(U, V)  # 162,162
    k = kronecker(U + 1e-7, V)  # Adding jitter noise to avoid underflow in matrixnormal
    # Reshaping the observed data to match it to the modelled one
    data1 = data1.view(3 * np.shape(data1)[0], )
    data2 = data2.view(3 * np.shape(data2)[0], )
    # X1
    M_E1_T1 = pyro.sample("M_E1_T1", dist.MultivariateNormal(M_T1, k), obs=data1)
    M_E2_T2 = pyro.sample("M_E2_T2", dist.MultivariateNormal(M_R2_T2, k), obs=data2)
def Run(data_obs):
    #GUIDE
    global_guide = AutoDelta(model)
    #global_guide = AutoDiagonalNormal(model)
    #OPTIMIZER
    #optim = torch.optim.Adagrad()
    #optim = LBFGS({})
    optim =pyro.optim.AdagradRMSProp(dict())
    #optim = pyro.optim.Adam({'lr': 1, 'betas': [0.8, 0.99]}) #1 for delta
    #elbo = TraceGraph_ELBO() #use if there are for loops in the model
    elbo= Trace_ELBO()
    #STOCHASTIC VARIATIONAL INFERENCE
    #pyro.param("auto_ri_vec", torch.Tensor([0.9,0.1,0.9]),constraint = constraints.unit_interval) #constraint = constraints.unit_interval#not bad but it doesn't stop
    svi_engine = SVIEngine(model,global_guide,optim,loss=elbo)
    pbar = tqdm.tqdm()
    loss_list = []

    #INITIALIZING PRIOR Changing in the first iteration the value for the prior in order to constrain the prior over the rotations
    pyro.param("auto_ri_vec", torch.Tensor([0.9,0.1,0.9]),constraint = constraints.unit_interval) #constraint = constraints.simplex doesn't work exactly

    @svi_engine.on(Events.EPOCH_COMPLETED)
    def update_progress(svi_engine):
        pbar.update(1)
        loss_list.append(-svi_engine.state.output)
        pbar.set_description("[epoch {}] avg train loss: {}".format(svi_engine.state.epoch,svi_engine.state.output))
    #HANDLER
    handler = EarlyStopping(patience=15, score_function=lambda eng: eng.state.output, trainer=svi_engine)
    #SVI
    svi_engine.add_event_handler(Events.EPOCH_COMPLETED, handler)
    svi_engine.run([data_obs],max_epochs=5000) #max_epochs=3
    #PLOTTING ELBO
    #plt.figure(1)
    plt.plot(loss_list)
    plt.savefig(r"ELBO_Loss.png")
    #exit()

    #### PARAMETERS ################
    data1,data2 = data_obs
    map_estimates = global_guide(data_obs)

    M1 = map_estimates["M1"].detach()
    M2 = map_estimates["M2"].detach()
    M3 = map_estimates["M3"].detach()
    M = torch.stack([M1,M2,M3], dim=1)
    M = Center_numpy(M.numpy())
    #ri_vec= torch.cat((map_estimates["element1"].detach(),map_estimates["element2"].detach(),map_estimates["element3"].detach()),0)
    ri_vec = map_estimates["ri_vec"].detach()
    R = sample_R(ri_vec)

    T1 = map_estimates["T1"].detach()
    T2 = map_estimates["T2"].detach()

    #X1 = Center_numpy(data1.detach().numpy() - T1.numpy()) #X1 -T1
    X1 = data1.detach().numpy() - T1.numpy() #X1 -T1
    #X2 = Center_numpy(np.dot(data2.detach().numpy() - T2.numpy(), np.linalg.inv(R))) #(X2-T2)R-1
    X2 = np.dot(data2.detach().numpy() - T2.numpy(), np.transpose(R)) #(X2-T2)R-1

    #################PLOTS################################################
    fig = plt.figure(figsize=(18, 16), dpi=80)
    ax = fig.add_subplot(111, projection='3d')

    #blue graph
    x = X1[:,0]
    y = X1[:,1]
    z = X1[:,2]

    ax.plot(x, y,z ,c='b', label='y1',linewidth=3.0)

    #red graph
    x2 = X2[:,0]
    y2 = X2[:,1]
    z2 = X2[:,2]

    ax.plot(x2, y2,z2, c='r', label='y2',linewidth=3.0)

    ###green graph
    x3=M[:,0]
    y3=M[:,1]
    z3=M[:,2]
    ax.plot(x3, y3,z3, c='g', label='y3',linewidth=3.0)

    #plt.title("Error : {}".format(loss_list[-1]) + " " "Iterations: {} ".format(iterations))
    plt.savefig(r"EarlyStopping")
    #plt.savefig(r"{}_Iterations".format(iterations))
    #plt.show()
    plt.clf()
    plt.plot(RMSD(data1,data2).numpy())
    plt.plot(RMSD(torch.from_numpy(X1),torch.from_numpy(X2)).numpy())
    plt.savefig(r"RMSD")
    #plt.show()
    return T1.numpy(),T2.numpy(),R.numpy(),M
def Write_PDB(initialPDB,Rotation,Translation):


    ''' Transform the atom coordinates from the original PDB file and rewrite it '''
    from Bio.PDB.PDBParser import PDBParser
    from Bio.PDB import MMCIFParser,PDBIO
    Name = ntpath.basename(initialPDB).split('.')[0]

    try:
        parser = PDB.PDBParser()
        structure = parser.get_structure('%s' % (Name), initialPDB)
    except:
        parser = MMCIFParser()
        structure = parser.get_structure('%s' % (Name), initialPDB)
    #Transformation
    #for atom, i in zip(structure.get_atoms(),range(len(structure.get_atoms()))):
    #[x for b in a for x in b]
    #size = int(len([_ for chain in structure.get_chains() for _ in chain.get_residues() if PDB.is_aa(_)]))
    #Translation = np.zeros(3) # 0 values for the translations
    for atom in structure.get_atoms():
        atom.transform(Rotation, -Translation)
    io = PDBIO()
    io.set_structure(structure)
    io.save("Transformed_{}".format(ntpath.basename(initialPDB)))
def write_ATOM_line(structure, file_name):
    import os
    """Transform coordinates to PDB file: Add intermediate coordinates to be able to visualize Mean structure in PyMOL"""
    expanded_structure = np.ones(shape=(2*len(structure)-1,3)) #The expanded structure contains extra rows between the alpha carbons
    averagearray = np.zeros(shape=(len(structure)-1,3)) #should be of size len(structure) -1
    for index,row in enumerate(structure):
        #print(index,row)
        if index != len(structure) and index != len(structure)-1:
            averagearray[int(index)] = (structure[int(index)]+structure[int(index)+1])/2
        else:
            pass
    #split the expanded structure in sets , where each set will be structure[0] + number*average
    #The even rows of the 'expanded structure' are simply the rows of the original structure
    expanded_structure[0::2] = structure
    expanded_structure[1::2] = averagearray
    structure = expanded_structure
    aa_name = "ALA"
    aa_type="CA"
    if os.path.isfile(file_name):
        os.remove(file_name)
        for i in range(len(structure)):
            with open(file_name,'a') as f:
                f.write("ATOM{:7d} {}   {} A{:4d}{:12.3f}{:8.3f}{:8.3f}  0.00  0.00    X    \n".format(i, aa_type, aa_name, i, structure[i, 0], structure[i, 1], structure[i, 2]))

    else:
        for i in range(len(structure)):
            with open(file_name,'a') as f:
                f.write("ATOM{:7d} {}   {} A{:4d}{:12.3f}{:8.3f}{:8.3f}  0.00  0.00    X    \n".format(i, aa_type, aa_name, i, structure[i, 0], structure[i, 1], structure[i, 2]))
def Pymol(*args):
    '''Visualization program'''
    import pymol
    #LAUNCH PYMOL
    pymol.pymol_argv = ['pymol'] + sys.argv[1:]
    pymol.finish_launching()

    def Colour_Backbone(selection,color):
        #pymol.cmd.select("alphas", "name ca") #apparently nothing is ca
        #pymol.cmd.select("sidechains", "! alphas") #select the opposite from ca, which should be the side chains, not working
        pymol.cmd.show("sticks", selection)
        pymol.cmd.color(color,selection)

    # Load Structures and apply the function
    colornames=['red','green','blue','orange']
    for file,color in zip(args,colornames):
        sname = ntpath.basename(file)
        pymol.cmd.load(file, sname)
        pymol.cmd.extend("Colour_Backbone", Colour_Backbone)
        Colour_Backbone(sname,color)



if __name__ == "__main__":

    data_obs = Read_Data('1adz0T.pdb','1adz1T.pdb')
    T1,T2,R,M = Run(data_obs)
    write_ATOM_line(M, 'M.pdb')
    Write_PDB(r"../PDB_files/1adz0T.pdb", np.transpose(R), T1)
    Write_PDB(r"../PDB_files/1adz1T.pdb", np.transpose(R), T2)
    Pymol("../PDB_files/1adz0T.pdb", "Transformed_1adz1T.pdb", "../PDB_files/1adz1T.pdb","M.pdb")













