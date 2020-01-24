import os,sys
import ntpath

from collections import defaultdict
import ntpath
import pandas as pd #Windows users: I copy-paste the pandas,dateutil and pyltz folders from anaconda 2!! into the site-packages folder of pymol(only for temporary use, other wise it gets confused with the paths of the packages)
import numpy as np
#Biopython
from Bio import SeqRecord,Alphabet,SeqIO
from Bio.Seq import Seq
import Bio.PDB as PDB
from Bio.SVDSuperimposer import SVDSuperimposer
from Bio.Seq import MutableSeq
from Bio.PDB.Polypeptide import is_aa
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
from pyro.optim import AdagradRMSProp
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate,Trace_ELBO, TraceGraph_ELBO
#from pyro.infer.static_svi import StaticSVI
#from static_svi import StaticSVI
import matplotlib.pyplot as plt
import tqdm
#torch.manual_seed(999)
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
                    # try:
                    alpha_carbon_coordinates.append(residue['CA'].get_coord())
                # except:
                # pass
        return alpha_carbon_coordinates
def Max_variance(structure):
    '''Calculates the maximum distance to the origin of the structure, this value will define the variance of the prior's distribution'''
    centered = Center_torch(structure)
    mul = centered@torch.t(structure)
    max_var = torch.sqrt(torch.max(torch.diag(mul)))
    return max_var
def Average_Structure(tuple_struct):
    average = sum(list(tuple_struct))/len(tuple_struct) #sum element-wise the list of tensors containing the coordinates #tf.add_n
    return average
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
def RMSD_biopython(x,y):
    sup = SVDSuperimposer()
    sup.set(x, y)
    sup.run()
    rot, tran = sup.get_rotran()
    return rot
def RMSD_numpy(X1,X2):
    import torch.nn.functional as F
    return F.pairwise_distance(torch.from_numpy(X1),torch.from_numpy(X2))
def RMSD(X1,X2):
    import torch.nn.functional as F
    return F.pairwise_distance(X1,X2)
def Read_Data(prot1,prot2,type='models',models =(0,1),RMSD=True):
    '''Reads different types of proteins and extracts the alpha carbons from the models, chains or all . The model,
    chain or aminoacid range numbers are indicated by the tuple models'''

    if type == 'models':
        X1_coordinates = Extract_coordinates_from_PDB('{}'.format(prot1),type)[models[0]]
        X2_coordinates = Extract_coordinates_from_PDB('{}'.format(prot2),type)[models[1]]
    elif type == 'chains':
        X1_coordinates = Extract_coordinates_from_PDB('{}'.format(prot1),type)[models[0]][0:141]
        X2_coordinates = Extract_coordinates_from_PDB('{}'.format(prot2),type)[models[1]][0:141]

    elif type == 'all':
        X1_coordinates = Extract_coordinates_from_PDB('{}'.format(prot1),type)[models[0]:models[1]]
        X2_coordinates = Extract_coordinates_from_PDB('{}'.format(prot2),type)[models[0]:models[1]]

    #Apply RMSD to the protein that needs to be superimposed
    X1_Obs_Stacked = Center_numpy(np.vstack(X1_coordinates))
    X2_Obs_Stacked = Center_numpy(np.vstack(X2_coordinates))
    if RMSD:
        X2_Obs_Stacked = torch.from_numpy(np.dot(X2_Obs_Stacked,RMSD_biopython(X1_Obs_Stacked,X2_Obs_Stacked)))
        X1_Obs_Stacked = torch.from_numpy(X1_Obs_Stacked)
    else:
        X1_Obs_Stacked = torch.from_numpy(X1_Obs_Stacked)
        X2_Obs_Stacked = torch.from_numpy(X2_Obs_Stacked)
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

    # rmsd = RMSD_numpy(Center_numpy(np.vstack(X1_coordinates)),Center_numpy(np.vstack(X2_coordinates)))
    # plt.plot(rmsd.numpy())
    # plt.show()
    return data_obs
def model(data):

    max_var,data1, data2 = data
    ### 1. prior over mean M
    M = pyro.sample("M", dist.Normal(0, 3).expand_by([data1.size(0),data1.size(1)]).to_event(2))
    M = Center_torch(M)
    ### 2. Prior over variances for the normal distribution
    U = pyro.sample("U", dist.HalfNormal(0.01).expand_by([data1.size(0)]).to_event(1))
    U = U.reshape(data1.size(0),1).repeat(1,3).view(-1)  #Triplicate the rows for the subsequent mean calculation
    ## 3. prior over translations T_i: Sample translations for each of the x,y,z coordinates
    T1 = pyro.sample("T1", dist.Normal(0, 1).expand_by([3]).to_event(1))
    T2 = pyro.sample("T2", dist.Normal(0, 1).expand_by([3]).to_event(1))
    ## 4. prior over rotations R_i
    ri_vec = pyro.sample("ri_vec",dist.Uniform(0, 1).expand_by([3]).to_event(1))  # Uniform distribution
    M_T1 = M + T1
    R = sample_R(ri_vec)
    M_R2_T2 = M @ R + T2
    # 5. Sampling from several Univariate Distributions (approximating the posterior distribution ): The observations are conditionally independant given the U, which is sampled outside the loop
    #UNIVARIATE NORMALS
    with pyro.plate("plate_univariate", data1.size(0)*data1.size(1),dim=-1):
        pyro.sample("X1", dist.Normal(M_T1.view(-1), U),obs=data1.view(-1))
        pyro.sample("X2", dist.Normal(M_R2_T2.view(-1), U), obs=data2.view(-1))
def Run(data_obs,iterations,average):
    #GUIDE
    global_guide = AutoDelta(model)
    optim = pyro.optim.AdagradRMSProp({'eta': 0.25, 'delta': 1e-16, 't': 5e-1})

    #optim = pyro.optim.Adam({'lr': 1, 'betas': [0.8, 0.99]}) #1 for delta
    #elbo = TraceGraph_ELBO() #use if there are for loops in the model
    elbo= Trace_ELBO()
    #STOCHASTIC VARIATIONAL INFERENCE
    svi = SVI(model, global_guide, optim, loss=elbo)
    #svi = StaticSVI(model, global_guide, optim, loss=elbo)
    steps = tqdm.tqdm(range(iterations))
    loss_list = []
    #Changing in the first iteration the value for the prior in order to constrain the prior over the rotations
    svi.step(data_obs)
    store = pyro.get_param_store()
    #store.get_param("auto_ri_vec")
    store.replace_param("auto_ri_vec",torch.Tensor([0.9,0.1,0.9]),store.get_param("auto_ri_vec"))
    store.replace_param("auto_M",average,store.get_param("auto_M"))


    for i in steps:
        svi.step(data_obs) #step returns a noisy estimate of the loss (i.e. minus the ELBO).
        # this estimate is not normalized in any way, so e.g. it scales with the size of the mini-batch
        loss = svi.loss(model, global_guide, data_obs)
        loss_list.append(loss)
        steps.set_description("Error: {}".format(loss))

    #PLOTTING ELBO
    plt.figure(1)
    plt.plot(loss_list)
    #plt.savefig(r"C:\Users\Lys Sanz Moreta\Dropbox\PhD\Superpositioning\ELBO_Loss.png")
    plt.savefig(r"ELBO_Loss.png")
    #plt.show()


    #### PARAMETERS ################
    max_var,data1,data2 = data_obs
    map_estimates = global_guide(data_obs)

    M = map_estimates["M"].detach()
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



    plt.title("Error : {}".format(loss_list[-1]) + " " "Iterations: {} ".format(iterations))
    plt.savefig(r"{}_Iterations".format(iterations))
    #plt.show()
    plt.clf()
    plt.plot(RMSD(data1,data2).numpy())
    plt.plot(RMSD(torch.from_numpy(X1),torch.from_numpy(X2)).numpy())
    plt.savefig(r"RMSD")
    #plt.show()
    return T1.numpy(),T2.numpy(),R.numpy(),M,X1,X2
def Write_PDB(initialPDB,Rotation,Translation,N):
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
        atom.transform(Rotation, Translation)
    io = PDBIO()
    io.set_structure(structure)
    io.save("{}_{}".format(N,ntpath.basename(initialPDB)))
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
    colornames=['red','green','blue','orange','purple','yellow','black','aquamarine']
    for file,color in zip(args,colornames):
        sname = ntpath.basename(file)
        pymol.cmd.load(file, sname)
        pymol.cmd.bg_color("white")
        pymol.cmd.extend("Colour_Backbone", Colour_Backbone)
        Colour_Backbone(sname,color)
















