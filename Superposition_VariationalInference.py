import os,sys
import time
import random
import ntpath
from collections import defaultdict
import ntpath
import pandas as pd
import numpy as np
import math
import timeit
#Biopython
import pymol
from Bio import SeqRecord,Alphabet,SeqIO
from Bio.Seq import Seq
import Bio.PDB as PDB
from Bio.Seq import MutableSeq
from Bio.PDB.Polypeptide import is_aa
from Bio.SVDSuperimposer import SVDSuperimposer
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats
import seaborn as sns
#PYRO
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.autoguide import AutoDelta, AutoDiagonalNormal, AutoLowRankMultivariateNormal, AutoMultivariateNormal, \
    AutoGuide, init_to_median, AutoGuideList
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate,Trace_ELBO, TraceGraph_ELBO, JitTrace_ELBO
import matplotlib as mpl #If errors when running Pymol: conda install pyqt=5.6 downgrade
import matplotlib.pyplot as plt
import tqdm
#Early STOPPING
from ignite.handlers import EarlyStopping
from ignite.engine import Engine,Events
from pyro.infer import SVI
from pyro.optim import PyroOptim
#TORCH: "Tensors"
import torch
import math
from torch.distributions import constraints, transform_to
from torch.optim import Adam, LBFGS
#Pre-set ups: Necessary for some torch errors
torch.backends.cudnn.deterministic = True
PyroOptim.state_dict = lambda self: self.get_state()
mpl.use('agg') #Change matplotlib graphics backend to avoid conflict with Pymol
tqdm.monitor_interval = 0
pyro.enable_validation(True)

#torch.manual_seed(5179758458530145964)
class SVIEngine(Engine):
    def __init__(self, *args, step_args=None, **kwargs):
        self.svi = SVI(*args, **kwargs)
        self._step_args = step_args or {}
        super(SVIEngine, self).__init__(self._update)

    def _update(self, engine, batch):
        return -engine.svi.step(batch, **self._step_args)
class DataManagement():
    def Extract_coordinates_from_PDB(self,PDB_file,type):
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
    def Average_Structure(self,tuple_struct):
        average = sum(list(tuple_struct))/len(tuple_struct)
        return average
    def Max_variance(self,structure):
        '''Calculates the maximum distance to the origin of the structure, this value will define the variance of the prior's distribution'''
        centered = self.Center_torch(structure)
        mul = centered@torch.t(structure)
        max_var = torch.sqrt(torch.max(torch.diag(mul)))
        return max_var
    def Center_numpy(self,Array):
        '''Centering to the origin the data'''
        mean = np.mean(Array,axis=0)
        centered_array = Array-mean
        return centered_array
    def Center_torch(self,Array):
        '''Centering to the origin the data'''
        mean = torch.mean(Array, dim=0)
        centered_array = Array - mean
        return centered_array
    def PairwiseDistances(self,X1,X2):
        '''Computes pairwise distances among coordinates'''
        import torch.nn.functional as F
        return F.pairwise_distance(X1,X2)
    def RMSD_biopython(self,x,y):
        """Kalbsch algorithm"""
        sup = SVDSuperimposer()
        sup.set(x, y)
        sup.run()
        rot, tran = sup.get_rotran()
        return rot
    def Read_Data(self,prot1,prot2,type='models',models =(0,1),RMSD=True):
        '''Reads different types of proteins and extracts the alpha carbons from the models, chains or all . The model,
        chain or aminoacid range numbers are indicated by the tuple models'''

        if type == 'models':
            X1_coordinates = self.Extract_coordinates_from_PDB('{}'.format(prot1),type)[models[0]]
            X2_coordinates = self.Extract_coordinates_from_PDB('{}'.format(prot2),type)[models[1]]
        elif type == 'chains':
            X1_coordinates = self.Extract_coordinates_from_PDB('{}'.format(prot1),type)[models[0]][0:141]
            X2_coordinates = self.Extract_coordinates_from_PDB('{}'.format(prot2),type)[models[1]][0:141]

        elif type == 'all':
            X1_coordinates = self.Extract_coordinates_from_PDB('{}'.format(prot1),type)[models[0]:models[1]]
            X2_coordinates = self.Extract_coordinates_from_PDB('{}'.format(prot2),type)[models[0]:models[1]]
        #Apply RMSD to the protein that needs to be superimposed
        X1_Obs_Stacked = self.Center_numpy(np.vstack(X1_coordinates))
        X2_Obs_Stacked = self.Center_numpy(np.vstack(X2_coordinates))
        if RMSD:
            X2_Obs_Stacked = torch.from_numpy(np.dot(X2_Obs_Stacked,self.RMSD_biopython(X1_Obs_Stacked,X2_Obs_Stacked)))
            X1_Obs_Stacked = torch.from_numpy(X1_Obs_Stacked)
        else:
            X1_Obs_Stacked = torch.from_numpy(X1_Obs_Stacked)
            X2_Obs_Stacked = torch.from_numpy(X2_Obs_Stacked)

        data_obs = (X1_Obs_Stacked,X2_Obs_Stacked)

        # ###PLOT INPUT DATA################
        x= self.Center_numpy(np.vstack(X1_coordinates))[:, 0]
        y= self.Center_numpy(np.vstack(X1_coordinates))[:, 1]
        z= self.Center_numpy(np.vstack(X1_coordinates))[:, 2]
        fig = plt.figure(figsize=(18, 16), dpi=80)
        ax = fig.add_subplot(111, projection='3d')
        plt.plot(x, y, z)
        ax.plot(x, y,z ,c='b', label='data1',linewidth=3.0)
        #orange graph
        x2=self.Center_numpy(np.vstack(X2_coordinates))[:, 0]
        y2=self.Center_numpy(np.vstack(X2_coordinates))[:, 1]
        z2=self.Center_numpy(np.vstack(X2_coordinates))[:, 2]
        ax.plot(x2, y2,z2, c='r', label='data2',linewidth=3.0)
        ax.legend()
        plt.savefig(r"Initial.png")
        plt.clf() #Clear the plot
        plt.close()

        return data_obs
    def Folders(self,folder_name):
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
            os.makedirs(newpath, 0o777)
    def Write_PDB(self,initialPDB,Rotation,Translation,N):
        ''' Transform by rotating and translating the atom coordinates from the original PDB file and rewrite it '''
        from Bio.PDB.PDBParser import PDBParser
        from Bio.PDB import MMCIFParser,PDBIO
        Name = ntpath.basename(initialPDB).split('.')[0]

        try:
            parser = PDB.PDBParser()
            structure = parser.get_structure('%s' % (Name), initialPDB)
        except:
            parser = MMCIFParser()
            structure = parser.get_structure('%s' % (Name), initialPDB)

        for atom in structure.get_atoms():
            atom.transform(Rotation, Translation)
        io = PDBIO()
        io.set_structure(structure)
        io.save("{}_{}".format(N,ntpath.basename(initialPDB)))
    def write_ATOM_line(self,structure, file_name):
        """Write a completely new PDB file with the C_alpha trace coordinates extracted from the model. It adds an intermediate coordinate between the C_alpha atoms in order to be able
        to visualize a conected line in Pymol"""
        #Create an expanded array: Contains an extra row between each C_alpha atom, for the intermediate coordinate
        expanded_structure = np.ones(shape=(2*len(structure)-1,3))
        #Calculate the average coordinate between each C alpha atom
        averagearray = np.zeros(shape=(len(structure)-1,3))
        for index,row in enumerate(structure):
            if index != len(structure) and index != len(structure)-1:
                averagearray[int(index)] = (structure[int(index)]+structure[int(index)+1])/2
            else:
                pass
        #The even rows of the 'expanded structure' are simply the rows of the original structure
        #The odd rows are the intermediate coordinate
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

    def Pymol(self,*args):
        '''Visualization program'''
        # LAUNCH PYMOL
        launch = True
        if launch:
            pymol.pymol_argv = ['pymol'] + sys.argv[1:]
            pymol.finish_launching(['pymol'])

        def Colour_Backbone(selection, color, color_digit):
            # pymol.cmd.select("alphas", "name ca") #apparently nothing is ca
            # pymol.cmd.select("sidechains", "! alphas") #select the opposite from ca, which should be the side chains, not working
            pymol.cmd.show("sticks", selection)
            pymol.cmd.set_color(color, color_digit)
            pymol.cmd.color(color, selection)

        # Load Structures and apply the function
        # colornames=['red','green','blue','orange','purple','yellow','black','aquamarine']
        # Palette of colours
        pal = sns.color_palette("GnBu_d", 10)  # RGB numbers for the palette colours
        colornames = ["blue_{}".format(i) for i in range(0, len(pal))]
        snames = []
        for file, color, color_digit in zip(args, colornames, pal):
            sname = ntpath.basename(file)
            snames.append(sname)
            pymol.cmd.load(file, sname)  # discrete 1 will create different sets of atoms for each model
            pymol.cmd.bg_color("white")
            pymol.cmd.extend("Colour_Backbone", Colour_Backbone)
            Colour_Backbone(sname, color, color_digit)
        pymol.cmd.png("Superposition_Bayesian_Pymol_{}".format(snames[0].split('_')[2]))
    def Pymol_Samples(self,data1, data2, name1, R_samples, T_samples, samples):
        # Process the dataframes
        R_samples = torch.from_numpy(R_samples)
        #indexes = random.sample(range(0, 9), samples)
        #for i in indexes:
        for i in range(0,samples):
            Rotation = SuperpositionModel.sample_R(R_samples[i, :])  # torch
            Translation = T_samples[i, :]  # numpy
            X2 = np.dot(data2.numpy() - Translation, np.transpose(Rotation.numpy()))
            self.write_ATOM_line(X2, os.path.join("{}_PDB_files".format(name1), 'Result_AutoDiagonalNormal_{}_X2_{}.pdb'.format(name1, i)))
        names = [os.path.join("{}_PDB_files".format(name1), 'Result_AutoDiagonalNormal_{}_X2_{}.pdb'.format(name1, i)) for i in range(0, samples)]
        self.Pymol(*names)
class SuperpositionModel():
    def sample_R(self,ri_vec):
        """Inputs a sample of unit quaternion and transforms it into a rotation matrix"""
        theta1 = 2 * math.pi * ri_vec[1]
        theta2 = 2 * math.pi * ri_vec[2]

        r1 = torch.sqrt(1 - ri_vec[0])
        r2 = torch.sqrt(ri_vec[0])

        qw = r2 * torch.cos(theta2)
        qx = r1 * torch.sin(theta1)
        qy = r1 * torch.cos(theta1)
        qz = r2 * torch.sin(theta2)

        R= torch.eye(3,3)
        # Filling the rotation matrix
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
    def model(self,data):
        max_var,data1, data2 = data
        ### 1. prior over mean M
        M = pyro.sample("M", dist.StudentT(1,0, 3).expand_by([data1.size(0),data1.size(1)]).to_event(2))
        ### 2. Prior over variances for the normal distribution
        U = pyro.sample("U", dist.HalfNormal(1).expand_by([data1.size(0)]).to_event(1))
        U =  U.reshape(data1.size(0),1).repeat(1,3).view(-1)  #Triplicate the rows for the subsequent mean calculation
        ## 3. prior over translations T_i: Sample translations for each of the x,y,z coordinates
        T2 = pyro.sample("T2", dist.Normal(0, 1).expand_by([3]).to_event(1))
        ## 4. prior over rotations R_i
        ri_vec = pyro.sample("ri_vec",dist.Uniform(0, 1).expand_by([3]).to_event(1))  # Uniform distribution
        R = self.sample_R(ri_vec)
        M_T1 = M
        M_R2_T2 = M @ R + T2
        # 5. Likelihood
        with pyro.plate("plate_univariate", data1.size(0)*data1.size(1),dim=-1):
            pyro.sample("X1", dist.StudentT(1,M_T1.view(-1), U),obs=data1.view(-1))
            pyro.sample("X2", dist.StudentT(1,M_R2_T2.view(-1), U), obs=data2.view(-1))
    def Run(self,data_obs,average,name,samples):
        #INITIALIZING PRIOR :
        def init_prior(site):
            if site["name"] == "ri_vec":
                return torch.tensor([0.9,0.1,0.9])
            elif site["name"] == "M":
                return average
            else:
                return init_to_median(site)
        #GUIDE: https://github.com/pyro-ppl/pyro/tree/672c662b4bd9105fdd50c66dcfc83dc8f7d4ac7d/pyro/contrib/autoguide
        global_guide = AutoGuideList(self.model)
        global_guide.add(AutoDiagonalNormal(poutine.block(self.model, hide=["M"],expose=["ri_vec","T2","U"]),init_loc_fn=init_prior))
        #AutoDiagonalNormal might not be able to capture the variance around the more variable areas of the protein
        global_guide.add(AutoDelta(poutine.block(self.model, expose=["M"],hide=["ri_vec","T2","U"]),init_loc_fn=init_prior))
        global_guide = AutoLowRankMultivariateNormal(self.model,rank=8) #Rank expresses the level of noise over the variance of the parameters
        #OPTIMIZER
        optim =pyro.optim.AdagradRMSProp(dict())
        #ELBO
        elbo = JitTrace_ELBO()
        #STOCHASTIC VARIATIONAL INFERENCE
        svi_engine = SVIEngine(self.model,global_guide,optim,loss=elbo)
        #Initialize variance to improve learning: Another parameter to play with, it gives a dimension error at the moment and it did not used to, so I am not sure what is going on
        # pyro.clear_param_store()
        # latent_dim = data_obs[1].size(0) + 6 # plus 6 is just to arithmetically match the number of latent dimensions that pyro uses to do its calculations,(rule of 3)
        # pyro.param("auto_loc", torch.randn(latent_dim))
        # pyro.param("auto_cov_factor", torch.randn(latent_dim, 8)*1e-4)
        # pyro.param("auto_cov_diag", torch.ones(latent_dim)*1e-2,constraint = constraints.positive)

        pbar = tqdm.tqdm()
        loss_list = []
        #ERROR LOSS
        @svi_engine.on(Events.EPOCH_COMPLETED)
        def update_progress(svi_engine):
            loss_list.append(-svi_engine.state.output)

        # Register hooks to monitor gradient norms.
        svi_engine.run([data_obs],max_epochs=1)
        gradient_norms = defaultdict(list)
        for name_i, value in pyro.get_param_store().named_parameters():
            value.register_hook(lambda g, name_i=name_i: gradient_norms[name_i].append(g.norm().item()))
        #Earlystop
        handler = EarlyStopping(patience=25, score_function=lambda engine: engine.state.output, trainer=svi_engine)
        #SVI
        svi_engine.add_event_handler(Events.EPOCH_COMPLETED, handler)
        start = time.time()
        svi_engine.run([data_obs],max_epochs=15000)
        svi_engine.svi.run(data_obs)
        stop=time.time()
        duration = stop-start
        #PLOTTING ELBO
        plt.plot(loss_list)
        plt.savefig(r"ELBO_Loss.png")
        plt.close()
        ###PLOTTING THE GRADIENT
        plt.figure(figsize=(10, 4), dpi=100).set_facecolor('white')
        for name_i, grad_norms in gradient_norms.items():
            plt.plot(grad_norms, label=name_i)
        plt.xlabel('iters')
        plt.ylabel('gradient norm')
        plt.yscale('log')
        plt.legend(loc='best')
        plt.title('Gradient norms during SVI')
        plt.savefig("Gradients")
        plt.close()
        #### PARAMETERS ################
        max_var,data1,data2 = data_obs
        map_estimates = global_guide(data_obs)
        #Mean structure MAP
        M = map_estimates["M"].detach().numpy()
        #Rotations MAP
        ri_vec = map_estimates["ri_vec"].detach()
        R = self.sample_R(ri_vec)
        #Translation MAP
        T2 = map_estimates["T2"].detach()
        #Prediction
        X1 = data1.detach().numpy()  #X1
        X2 = np.dot(data2.detach().numpy() - T2.numpy(), np.transpose(R)) #(X2-T2)R-1
        ####PARAMETERS STATS ####################3
        # Rotation matrix stats
        ri_vec_marginal_1 = svi_engine.svi.marginal(sites=["ri_vec"])
        ri_vec_marginal_1 = torch.cat(list(ri_vec_marginal_1.support(flatten=True).values()),dim=-1).detach().numpy()  # Where the samples are stored
        params = ['ri_vec[0]', 'ri_vec[1]', 'ri_vec[2]']
        df = pd.DataFrame(ri_vec_marginal_1, columns=params).transpose()
        df_summary = df.apply(pd.Series.describe, axis=1)[["mean", "std", "25%", "50%", "75%"]]
        df_summary.to_csv("ri_vec_stats_{}.txt".format(name), sep='\t')
        # Rotation matrix output
        ri_vec_mean = svi_engine.svi.marginal(["ri_vec"]).empirical["ri_vec"].mean
        ri_vec_variance = svi_engine.svi.marginal(["ri_vec"]).empirical["ri_vec"].variance
        R = self.sample_R(ri_vec_mean)

        # # Mean structure stats
        M_marginal_1 = svi_engine.svi.marginal(sites=["M"])
        M_marginal_1 = torch.cat(list(M_marginal_1.support(flatten=True).values()), dim=-1).detach().numpy()
        params = ['M[{}]'.format(i) for i in range(0, len(data1))]
        label_one = np.array(params)
        label_two = np.array(['x', 'y', 'z'])
        cols = pd.MultiIndex.from_product([label_one, label_two])
        df = pd.DataFrame(M_marginal_1.T.reshape(10, -1), columns=cols).transpose() #AutoDiagonal only generates 10 samples
        df_summary = df.apply(pd.Series.describe, axis=1)[["mean", "std", "25%", "50%", "75%"]]
        df_summary.to_csv("M_stats_{}.txt".format(name), sep='\t')

        # Mean structure M output
        M_vec_marginal = svi_engine.svi.marginal(["M"]).empirical["M"]
        M_vec_mean = M_vec_marginal.mean
        M_vec_variance = M_vec_marginal.variance
        M = M_vec_mean.detach().numpy()

        # Translation stats
        T_marginal_1 = svi_engine.svi.marginal(sites=["T2"])
        T_marginal_1 = torch.cat(list(T_marginal_1.support(flatten=True).values()), dim=-1).detach().numpy()
        params = ['T[0]', 'T[1]', 'T[2]']
        df = pd.DataFrame(T_marginal_1, columns=params).transpose()
        df_summary = df.apply(pd.Series.describe, axis=1)[["mean", "std", "25%", "50%", "75%"]]
        df_summary.to_csv("T_stats_{}.txt".format(name), sep='\t')
        # Translation T output
        T2_vec_mean = svi_engine.svi.marginal(["T2"]).empirical["T2"].mean
        T2_vec_variance = svi_engine.svi.marginal(["T2"]).empirical["T2"].variance


        #################PLOTS################################################
        fig = plt.figure(figsize=(18, 16), dpi=80)
        ax = fig.add_subplot(111, projection='3d')

        #blue graph
        x = X1[:,0]
        y = X1[:,1]
        z = X1[:,2]

        ax.plot(x, y,z ,c='b', label='X1',linewidth=3.0)

        #red graph
        x2 = X2[:,0]
        y2 = X2[:,1]
        z2 = X2[:,2]

        ax.plot(x2, y2,z2, c='r', label='X2',linewidth=3.0)

        ###green graph
        x3=M[:,0]
        y3=M[:,1]
        z3=M[:,2]
        ax.plot(x3, y3,z3, c='g', label='M',linewidth=3.0)
        ax.legend()

        plt.savefig(r"Plating_{}".format(name))
        distances = DataManagement.PairwiseDistances(torch.from_numpy(X1),torch.from_numpy(X2)).numpy()
        plt.clf()
        plt.plot(DataManagement.PairwiseDistances(data1,data2).numpy(), linewidth = 8.0)
        plt.plot(DataManagement.PairwiseDistances(torch.from_numpy(X1),torch.from_numpy(X2)).numpy(), linewidth=8.0)
        plt.ylabel('Pairwise distances',fontsize='46')
        plt.xlabel('Amino acid position',fontsize='46')
        plt.yticks(fontsize=30)
        plt.xticks(fontsize=30)
        plt.title('{}'.format(name.upper()),fontsize ='46')
        plt.gca().legend(('RMSD', 'Theseus-PP'),fontsize='40')
        plt.savefig(r"Distance_Differences_{}".format(name))
        plt.close()

        return T2.detach().numpy(),R.detach().numpy(),M, X1,X2,distances,(svi_engine.state.epoch,svi_engine.state.output),duration,ri_vec_marginal_1,M_marginal_1,T_marginal_1

DataManagement = DataManagement()
SuperpositionModel = SuperpositionModel()



















