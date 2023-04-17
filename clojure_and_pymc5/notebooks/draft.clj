
(defn read-data [prot1 prot2
                 {:keys [data-type moedls rmsd?]
                  :or {data-type :models
                       :models [0 1]
                       :rmsd true}}]
  )

(let [name1 "1d3z"
      name2 "1ubq"
      models [0 3]
      samples 100]
  )

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
d








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
