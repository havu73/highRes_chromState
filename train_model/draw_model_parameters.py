import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os 
import sys
import helper
import pandas as pd 
import numpy as np
CELL_GROUP_COLOR_CODE = {'Neurosph': '#FFD924', 'HSC & B-cell': '#678C69', 'Mesench': '#B65C73', 'Brain': '#C5912B', 'Adipose': '#AF5B39', 'Thymus': '#DAB92E', 'Sm. Muscle': '#F182BC', 'IMR90': '#E41A1C', 'Myosat': '#E67326', 'iPSC': '#69608A', 'Muscle': '#C2655D', 'Digestive': '#C58DAA', 'ESC': '#924965', 'Epithelial': '#FF9D0C', 'Heart': '#D56F80', 'ENCODE2012': '#000000', 'ES-deriv': '#4178AE', 'Other': '#999999', 'Blood & T-cell': '#55A354', 'NA' : 'black'}
fn = sys.argv[1]
save_fn = sys.argv[2]
df = pd.read_csv(fn, header = None, sep = '\t', index_col = None, squeeze = True)
df.columns = ['EID', 'pi']

processed_raw_metadata_fn = '../../data/hg19/processed_raw_metadata_roadmap_epig_summary.csv'
annot_df = pd.read_csv(processed_raw_metadata_fn, header = 0, index_col = None, sep = ',')
annot_df = annot_df.rename(columns = {'Epigenome ID (EID)': 'EID'})
df = df.merge(annot_df, left_on = 'EID', right_on = 'EID', how = 'left')
df = df.sort_values('GROUP')
fig, axes = plt.subplots(ncols = 1, nrows = 1, figsize = (13,7))
ax = sns.pointplot(data = df, x = 'EID', y = 'pi',linestyles=' ', hue = 'GROUP', palette = CELL_GROUP_COLOR_CODE)
xlabels = ax.get_xticklabels() 
ax.set_xticklabels(xlabels, rotation = 270, size = 5)
plt.legend(bbox_to_anchor=(1.15, 0.3), loc='center right', borderaxespad=0)
fig.tight_layout()
fig.savefig(save_fn)
print ('Done')