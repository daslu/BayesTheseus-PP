import numpy as np
import torch
    
def kronecker(first_matrix, second_matrix):
    n,m = np.shape(first_matrix) # the dimensions of the first matrix
    p,q = np.shape(second_matrix) # the dimensions of the second matrix
    result_matrix = np.zeros((n*p,m*q)) # a starting matrix in which we will put the correct elements
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for l in range(q):
                    result_matrix[i*p+k,j*q+l] = first_matrix[i,j]*second_matrix[k,l]
    return torch.tensor(result_matrix)
def kronecker(matrix1, matrix2):
    '''Performs the Kronecker product between 2 matrices'''
    final_list = []
    sub_list = []

    count = len(matrix2)

    for row_1 in matrix1:
        counter = 0
        check = 0
        while check < count:
            for element_1 in row_1:
                for element_2 in matrix2[counter]: #matrix2[counter] == row_2
                    sub_list.append(element_1 * element_2)
            counter += 1
            final_list.append(sub_list)
            sub_list = []
            check += 1

    return torch.Tensor(final_list)

def write_ATOM_line(structure, file_name):
    import os
    """Transform coordinates to PDB file"""
    _ATOM_FORMAT_STRING = "%s%5i %-4s%c%3s %c%4i%c   %8.3f%8.3f%8.3f%6.2f%6.2f %4s%2s%2s\n"
    # with open(file_name, 'a') as f:
    #     f.write("ATOM{:7d} {}   {} A{:4d}{:12.3f}{:8.3f}{:8.3f}  0.00  0.00    X    \n".format(i, name,aa_type, i, x, y, z))
    # import re
    # open the xml file for reading:
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


def avg(myArray, N=2):
    cum = np.cumsum(myArray, 0)
    result = cum[N - 1::N] / float(N)
    result[1:] = result[1:] - result[:-1]

    remainder = myArray.shape[0] % N
    if remainder != 0:
        if remainder < myArray.shape[0]:
            lastAvg = (cum[-1] - cum[-1 - remainder]) / float(remainder)
        else:
            lastAvg = cum[-1] / float(remainder)
        result = np.vstack([result, lastAvg])

    return result




def model(data1,data2):
    import pyro
    #means_vector=torch.rand((data1.size(0),)) #Random numbers from normal distribution?
    variance_vector=torch.empty(data1.size(0), ).uniform_(0, 3)

    with pyro.plate("plate_normal", data1.size(-1) ): #-1 ?
        Triads = torch.Tensor()
        for i in range(data1.size(0)): #create len(data1)*3 values for the diagonal
            triad = pyro.sample('Triad_{}'.format(i),dist.HalfNormal(variance_vector[i]).expand_by([3]))
            Triads=torch.cat((Triads,triad),0)
    #Substitute the diagonal of a zero array for the results in the sampling
    zeros = torch.zeros(data1.size(0)*data1.size(1),data1.size(0)*data1.size(1))
    mask = torch.diag(torch.ones_like(Triads))
    out = mask * torch.diag(Triads) + (1. - mask) * zeros



#How to triplicate torch tensors rows
  #M_T1 = M_T1.repeat(1,3).view(-1, 3) #Triplicate the rows for the subsequent mean calculation
    #mean_vector1 = torch.mean(M_T1, dim=1)
    #print(mean_vector1)
    #exit()
    #M_R2_T2 = M_R2_T2.repeat(1,3).view(-1, 3) #Tri
    #mean_vector2 = torch.mean(M_R2_T2, dim=1)


#Model with MixtureofDiagNormals

    #MIXTUREOFDIAGNORMALS
    #U = torch.stack((U,U,U),dim=1)

    #:param torch.Tensor locs: N x D mean matrix
    #:param torch.Tensor coord_scale: N x D scale matrix
    #:param torch.Tensor component_logits: N-dimensional vector of softmax logits, weights for the NN
    #with pyro.plate('plate_mixture',data1.size(0),dim=-1):
        #M_E1_T1 = pyro.sample("M_E1_T1",dist.MixtureOfDiagNormals(M_T1,U,torch.ones(data1.size(0))),obs =data1)
        #M_E2_T2 = pyro.sample("M_E2_T2",dist.MixtureOfDiagNormals(M_R2_T2,U,torch.ones(data2.size(0))),obs =data2)


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




#LBFGS and StaticSVI

    #https://github.com/fehiepsi/pyro/blob/c2fa8927234e3582b575a386ec19c0f6ff905218/tests/infer/test_static_svi.py
    # OPTIMIZER: https://github.com/pyro-ppl/pyro/blob/6dd78c2f0b077f3c6ba58dc7ea096ea28294ed99/examples/mixed_hmm/experiment.py
    ##################################
    #Check rethinking.py for alternative implementation of params

    # loss_fn = TraceEnum_ELBO(max_plate_nesting=2).differentiable_loss
    # with pyro.poutine.trace(param_only=True) as param_capture:
    #      loss_fn(model, global_guide,data_obs)
    # params = [site["value"].unconstrained() for site in param_capture.trace.nodes.values()]
    # optim = torch.optim.LBFGS(params)
    #svi_engine = SVIEngine(model, global_guide, optim, loss=Trace_ELBO())



def Calculate_Rotation():
    a = sample_R(torch.Tensor([0.5, np.pi / 4, np.pi / 3]))
    b = np.transpose(a)
    c = a @ b
    print(np.linalg.det(c), c)



def Alternative_Guide():
    # GUIDE
    global_guide = AutoMultivariateNormal(model)
    # OPTIMIZER
    optim = pyro.optim.AdagradRMSProp(dict())
    # optim = pyro.optim.Adam({'lr': 1, 'betas': [0.8, 0.99]}) #1 for delta
    # elbo = TraceGraph_ELBO() #use if there are for loops in the

    elbo = Trace_ELBO()
    # STOCHASTIC VARIATIONAL INFERENCE
    # pyro.param("auto_ri_vec", torch.Tensor([0.9,0.1,0.9]),constraint = constraints.unit_interval) #constraint = constraints.unit_interval#not bad but it doesn't stop
    svi_engine = SVIEngine(model, global_guide, optim, loss=elbo)
    pbar = tqdm.tqdm()
    loss_list = []
    # INITIALIZING PRIOR Changing in the first iteration the value for the prior in order to constrain the prior over the rotations
    pyro.param("ri_vec_loc", torch.Tensor([0.9, 0.1, 0.9]),
               constraint=constraints.unit_interval)  # constraint = constraints.simplex doesn't work exactly
    # Initialize the Mean Structure (each coordinate separately): NO CONSTRAINTS!!!
    pyro.param("M_loc", average[:, 0])




def RMSD_numpy(X1,X2):
    import torch.nn.functional as F
    return F.pairwise_distance(torch.from_numpy(X1),torch.from_numpy(X2))

# INITIALIZING PRIOR : Constraint the unit vector for the rotations and the mean structure
# pyro.param("auto_ri_vec", torch.Tensor([0.9,0.1,0.9]),constraint = constraints.unit_interval)
# pyro.param("auto_M",average)


















