import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os 
hg19_chromSize_fn = '../../../data/hg19.chrom-sizes'
CHROMOSOME_LIST = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X'] # there was actually no data of chromatin state for chromosome Y
# CHROMOSOME_LIST = ['1']
ct_input_folder = '../../../data/hg19/K562'
ct_code = 'E123'
all_model_folder = '../../../model_analysis/K562_hg19'
model_name_list = ['all_mark_model', 'three_mark_model']
num_state_dict = {'all_mark_model': 5, 'three_mark_model': 25}
NUM_STATE = 5
NUM_BP_PER_WINDOW = 1000000
NUM_BP_PER_BIN = 200
NUM_BIN_PER_WINDOW = int(NUM_BP_PER_WINDOW / NUM_BP_PER_BIN)
# for enrichment analysis
COORD_DIR_FOR_ENRICHMENT = '../../../data/hg19/genome_context_from_ha/for_enrichment'
COORD_DIR_FOR_NEIGHBORHOOD = '../../../program_source/ChromHMM/ChromHMM/ANCHORFILES/hg19'

rule all:
	input:
		expand(os.path.join(all_model_folder, 'compare_models', 'confusionM_stateOverlap.{ext}'), ext = ['txt.gz', 'png']),

# first, we want to compare how the two models' state tend to overlap with each other
rule calculate_state_overlap_two_models: # create the confusion-matrix-type result to see how the states in big and small models tend to overlap
	input:
		os.path.join(all_model_folder, 'all_mark_model', 'genome_25_segments_clean.bed.gz'),
		os.path.join(all_model_folder, 'three_mark_model', 'genome_5_segments_clean.bed.gz'),
	output:
		expand(os.path.join(all_model_folder, 'compare_models', 'confusionM_stateOverlap.{ext}'), ext = ['txt.gz', 'png']),
	params:
		output_no_tail = os.path.join(all_model_folder, 'compare_models', 'confusionM_stateOverlap')
	shell:
		"""
		python get_confusion_matrix_two_models.py {input[0]} {input[1]} {params.output_no_tail}
		"""