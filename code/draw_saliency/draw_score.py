import pickle
import numpy as np
import matplotlib.pyplot as plt

exp_name = '/exp'
root_file = '/root/path'
save_name = '/attn_score.pkl'
file_path = root_file + exp_name + save_name
file_path = save_name

with open(file_path, 'rb') as f:
    loaded_array = pickle.load(f)

loaded_array = loaded_array.mean(0).squeeze()
data = loaded_array/loaded_array.sum(axis=1,keepdims=True)

plt.rcParams.update({'font.size': 18})

# Set the line width
plt.rcParams.update({'lines.linewidth': 2})

# Create the plot
plt.plot(data[:,0], label=r'$S_v$',color='r')
plt.plot(data[:,1], label=r'$S_t$',color='g')
plt.plot(data[:,2], label=r'$S_o$',color='b')

# Set the title and labels
plt.xlabel('Layer')
plt.ylabel(r'$S$')

# Show the legend and plot
plt.grid()
plt.legend()
plt.xlim(0,31)
plt.ylim(0,0.8)
plt.tight_layout()
plt.savefig('temp.pdf')

