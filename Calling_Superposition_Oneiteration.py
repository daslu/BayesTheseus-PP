
import numpy as np
earlystop=True
Cuda = False
name1 = '1adz0T'
name2 ='1adz1T'
iterations = 30000
# '2ain': Models 0-5
# '2lkq' : Models 0-3,0-4
# '2mx2' : All models very close to each other, raises a lot of errors
# '1pwj' : All models pretty close to each other--> checked until 0-7
# '1bz1' : Is too 'challenging' to superimpose (models 0-1, 0-3)--Try not-earlystop ---> not doing much
# '2ys9' : Models 0-3 (faulty), Models 0-2 (working?)
# '2lkl' : Models 0-3
# '2cpd' : Models 0-2, models 0-3
# '1ak7': Models 0-1
# '2lkl' : Models 0-1, 0-8
# '2yuq' : Models 0-2
# '1adz0T' and '1adz1T'
#  '1ahl': Models 0-2
# '2khi' : Models 0-2

if earlystop:
    from Superposition_EarlyStop import *
    import torch
    #from Superposition_StaticSVI_EarlyStop import *
    # data_obs = Read_Data('../PDB_files/{}.pdb'.format(name1), '../PDB_files/{}.pdb'.format(name2),type='models',models =(0,3),RMSD=False)
    # max_var = Max_variance(data_obs[0])  # calculate the max pairwise distance to origin of the structure to set as value for max variance in the prior for mean struct
    # average = Average_Structure(data_obs)
    # data1, data2 = data_obs
    # write_ATOM_line(data1, 'data1clean.pdb')
    # write_ATOM_line(data2, 'data2clean.pdb')
    # Pymol('data1clean.pdb', 'data2clean.pdb')

    data_obs = Read_Data('../PDB_files/{}.pdb'.format(name1), '../PDB_files/{}.pdb'.format(name2),type='models',models =(0,3),RMSD=True)
    #Pymol('../PDB_files/{}.pdb'.format(name1), '../PDB_files/{}.pdb'.format(name2))
    max_var = Max_variance(data_obs[0])  # calculate the max pairwise distance to origin of the structure to set as value for max variance in the prior for mean struct
    average = Average_Structure(data_obs)
    data1, data2 = data_obs
    write_ATOM_line(data1, 'RMSD_2ys9_data1.pdb')
    write_ATOM_line(data2, 'RMSD_2ys9_data2.pdb')
    #Pymol('RMSD_2ys9_data1.pdb', 'RMSD_2ys9_data2.pdb')
    data_obs = max_var, data1, data2
    T1, T2, R, M, X1, X2,distances = Run(data_obs, average, name1)
    write_ATOM_line(M, 'M.pdb')
    write_ATOM_line(X1, 'Result_2ys9_X1.pdb')
    write_ATOM_line(X2, 'Result_2ys9_X2.pdb')
    Write_PDB(r"../PDB_files/{}.pdb".format(name1), np.transpose(R), -T1,'Transformed')
    Write_PDB(r"../PDB_files/{}.pdb".format(name2), np.transpose(R), -T2,'Transformed')
    #Pymol("Transformed_{}.pdb".format(name2))
    Pymol("Result_2ys9_X1.pdb", "Result_2ys9_X2.pdb")
    #Pymol("Result_2ys9_X1.pdb","Result_2ys9_X2.pdb","M.pdb",r"../PDB_files/{}.pdb".format(name1), "Transformed_{}.pdb".format(name2), "../PDB_files/{}.pdb".format(name2))

else:
    #from Superposition_NotEarlyStop import *
    from Superposition_StaticSVI_NotEarlyStop import *
    import numpy as np
    data_obs = Read_Data('../PDB_files/{}.pdb'.format(name1),'../PDB_files/{}.pdb'.format(name2),type='all',models =(0,100),RMSD=True)
    max_var = Max_variance(data_obs[0]) #calculate the max pairwise distance to origin of the structure to set as value for max variance in the prior for mean struct
    average = Average_Structure(data_obs)
    data1,data2=data_obs
    write_ATOM_line(data1,'RMSD_2ys9_data1.pdb')
    write_ATOM_line(data2,'RMSD_2ys9_data2.pdb')
    Pymol('RMSD_2ys9_data1.pdb','RMSD_2ys9_data2.pdb')
    data_obs = max_var,data1,data2
    T1,T2,R,M,X1,X2 = Run(data_obs,iterations,average)
    write_ATOM_line(M, 'M.pdb')
    write_ATOM_line(X1,'Result_2ys9_X1.pdb')
    write_ATOM_line(X2,'Result_2ys9_X2.pdb')
    Write_PDB(r"../PDB_files/{}.pdb".format(name1), np.transpose(R), -T1,'Transformed')
    Write_PDB(r"../PDB_files/{}.pdb".format(name2), np.transpose(R), -T2,'Transformed')
    #Pymol(r"../PDB_files/{}.pdb".format(name1), "Transformed_{}.pdb".format(name2), "../PDB_files/{}.pdb".format(name2),"M.pdb","Result_2ys9_X1.pdb","Result_2ys9_X2.pdb")
    Pymol("Result_2ys9_X1.pdb", "Result_2ys9_X2.pdb", "M.pdb")


exit()

if Cuda:
    from Superposition_EarlyStop import *
    import numpy as np
    ###With EarlyStopping
    data_obs = Read_Data('../PDB_files/{}.pdb'.format(name1), '../PDB_files/{}.pdb'.format(name2),type='models',models =(0,3),RMSD=True)
    max_var = Max_variance(data_obs[0])  # calculate the max pairwise distance to origin of the structure to set as value for max variance in the prior for mean struct
    average = Average_Structure(data_obs)
    data1, data2 = data_obs
    write_ATOM_line(data1, 'RMSD_2ys9_data1.pdb')
    write_ATOM_line(data2, 'RMSD_2ys9_data2.pdb')
    Pymol('RMSD_2ys9_data1.pdb', 'RMSD_2ys9_data2.pdb')
    data_obs = max_var, data1, data2
    T1, T2, R, M, X1, X2 = Run(data_obs, average, name1)
    write_ATOM_line(M, 'M.pdb')
    write_ATOM_line(X1, 'Result_2ys9_X1.pdb')
    write_ATOM_line(X2, 'Result_2ys9_X2.pdb')
    Write_PDB(r"../PDB_files/{}.pdb".format(name1), np.transpose(R), -T1,'Transformed')
    Write_PDB(r"../PDB_files/{}.pdb".format(name2), np.transpose(R), -T2,'Transformed')
    Pymol("Result_2ys9_X1.pdb", "Result_2ys9_X2.pdb", "M.pdb")
    #Pymol("Result_2ys9_X1.pdb","Result_2ys9_X2.pdb","M.pdb",r"../PDB_files/{}.pdb".format(name1), "Transformed_{}.pdb".format(name2), "../PDB_files/{}.pdb".format(name2))














