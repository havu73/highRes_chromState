import pandas as pd 
import numpy as np
import seaborn as sns 
import os 
import sys

beta_fn = sys.argv[1]
plot_fn = sys.argv[2]
beta_fn
beta_df = pd.read_csv(beta_fn, header = None, index_col = None, sep = '\t')# just a matrix right now. implicit: rows: states that are from ref_epig, columns: states that are in sample of interest. row sum should be 1
print(beta_df)
S25_state_annot_fn = '/gstore/project/gepiviz_data/vuh6/roadmap/state_annot.txt'
state_annot_df = pd.read_csv(S25_state_annot_fn, header = 0, index_col = None, sep = '\t') # columns: state   menumonics      description     color_name      rgb
beta_df.columns = state_annot_df.menumonics 
ax = sns.heatmap(beta_df, vmin = 0, cmap = sns.color_palette('Blues', n_colors = 30), annot = False, fmt='.2f')
ax.set(xlabel = 'States in ref_epig', ylabel = 'States in sample of interest')
plt.savefig(output_fn_prefix + '.png')
