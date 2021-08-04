import pandas as pd 
import numpy as np 
import seaborn as sns
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')

hg19_chrom_length_fn = '../../model_analysis/hg19_chrom_length_for_segmentation.bed' # chrom, start, end, start is always 0
model_analysis_dir = '../../model_analysis'
all_model_dir = '../../model_analysis/K562_hg19'
COORD_CRISPR_MPRA_DIR = '../../data/hg19/K562/COORD_CRISPR_MPRA_DATA/for_enrichment'
COORD_CRISPR_MPRA_COLUMN_MATCH_FN = os.path.join(COORD_CRISPR_MPRA_DIR, 'match_foreground_background')
context_list = ['Fulco19_significant_DE_G_ST6a', 'gasperini19_altScale_664_enhancerGenePairs_ST2B', 'gasperini19_pilot_145_EnhancerGenePairs_ST1B', 'klann21_S1_discovery_K562_sgrna', 'klann21_S2_discovery_K562_bin2', 'klann21_S3_discovery_K562_bin3', 'klann21_S5_discovery_K562_dhs', 'klann21_S6_validation_K562_sgrna', 'klann21_S7_validation_K562_bin2', 'klann21_S8_validation_K562_bin3']
ROADMAP_dir = '../../data/hg19/K562/roadmap_published_chromHMM/'
simple_rule_annot_dir = '../../model_analysis/K562_hg19/simple_rules_from_natalie'
def colored_excel_fn_list():
	results = []
	# expand(os.path.join(all_model_dir, '{num_mark_model}', 'state_{num_state}', 'full_model', 'segment_based_on_entropy', 'overlap_crispr_mpra', '{fore_or_back}', 'gw_overlap.xlsx'), num_mark_model = ['all_mark_model'], num_state = [25], fore_or_back = ['foreground', 'background']),
	# expand(os.path.join(all_model_dir, '{num_mark_model}', 'state_{num_state}', 'full_model', 'segment_based_on_entropy', 'overlap_crispr_mpra', '{fore_or_back}', 'gw_overlap.xlsx'), num_mark_model = ['three_mark_model'], num_state = [8], fore_or_back = ['foreground', 'background']),
	# expand(os.path.join(all_model_dir, '{num_mark_model}', 'state_{num_state}', 'full_model', 'segment_based_on_entropy', 'overlap_crispr_mpra', '{fore_or_back}', 'gw_overlap_against_bg.xlsx'), num_mark_model = ['all_mark_model'], num_state = [25], fore_or_back = ['fg_against_bg']),
	# expand(os.path.join(all_model_dir, '{num_mark_model}', 'state_{num_state}', 'full_model', 'segment_based_on_entropy', 'overlap_crispr_mpra', '{fore_or_back}', 'gw_overlap_against_bg.xlsx'), num_mark_model = ['three_mark_model'], num_state = [8], fore_or_back = ['fg_against_bg']),
	# expand(os.path.join(all_model_dir, '{num_mark_model}', 'state_{num_state}', 'full_model', 'segment_based_on_entropy', '{compare_to_median}', 'genome_segments_sorted.bed.gz'), num_mark_model = ['all_mark_model'], num_state = [25], compare_to_median = ['higher', 'leq']),
	return results

rule all:
	input:
		expand(os.path.join(all_model_dir, '{num_mark_model}', 'state_{num_state}', 'full_model', 'POSTERIOR', 'entropy', 'plots', 'entropy_density_gw.png'), num_mark_model = ['all_mark_model'], num_state = range(25,26)),
		expand(os.path.join(all_model_dir, '{num_mark_model}', 'state_{num_state}', 'full_model', 'POSTERIOR', 'entropy', 'plots', 'entropy_density_gw.png'), num_mark_model = ['three_mark_model'], num_state = range(8,9)),
		# os.path.join(ROADMAP_dir, '25_state','POSTERIOR', 'entropy', 'plots', 'entropy_density_gw.png'),
		# expand(os.path.join(all_model_dir, '{num_mark_model}', 'state_{num_state}', 'full_model', 'segment_based_on_entropy', 'genome_segments_sorted.bed.gz'), num_mark_model = ['three_mark_model'], num_state = range(8,9)),
		# expand(os.path.join(all_model_dir, '{num_mark_model}', 'state_{num_state}', 'full_model', 'segment_based_on_entropy', 'genome_segments_sorted.bed.gz'), num_mark_model = ['all_mark_model'], num_state = range(25,26)),
		# expand(os.path.join(all_model_dir, '{num_mark_model}', 'state_{num_state}', 'full_model', 'segment_based_on_entropy', 'overlap_crispr_mpra','fg_against_bg', '{TT}_overlap_against_bg.txt'), num_mark_model = ['all_mark_model'], num_state = range(25,26), TT = ['train', 'test']),
		# expand(os.path.join(all_model_dir, '{num_mark_model}', 'state_{num_state}', 'full_model', 'segment_based_on_entropy', 'overlap_crispr_mpra','fg_against_bg', '{TT}_overlap_against_bg.txt'), num_mark_model = ['three_mark_model'], num_state = range(8,9), TT = ['train', 'test']),
		# expand(os.path.join(all_model_dir, 'compare_models_recover_crispr_mpra', 'entropy_stratified', 'fg_against_bg', 'auc_{context}.txt'), context = context_list),
		# expand(os.path.join(all_model_dir, 'compare_models_recover_crispr_mpra', 'fg_against_bg', 'auc_{context}.txt'), context = context_list),
		expand(os.path.join(all_model_dir, '{num_mark_model}', 'state_{num_state}', 'full_model', 'segment_based_on_entropy', '{compare_to_median}', 'genome_segments_sorted.bed.gz'), num_mark_model = ['three_mark_model'], num_state = [8], compare_to_median = ['higher', 'leq']),
		expand(os.path.join(all_model_dir, '{num_mark_model}', 'state_{num_state}', 'full_model', 'segment_based_on_entropy', '{compare_to_median}', 'genome_segments_sorted.bed.gz'), num_mark_model = ['all_mark_model'], num_state = [25], compare_to_median = ['higher', 'leq']),




def get_corresponding_binarized_dir(wildcards):
	all_data_dir = '../../data/hg19/K562/'
	if wildcards.num_mark_model == "three_mark_model":
		result = os.path.join(all_data_dir, 'three_mark_model_training')
	elif wildcards.num_mark_model == 'all_mark_model':
		result = os.path.join(all_data_dir, 'big_model_training', 'full')
	else: 
		result = ''
	return result 

rule get_posterior_chromHMM: # this function is applicable to cases where we have trained the model ourselves (not the roadmap published ones), and we want to get the posterior probabilities of state assignment for these models
	input:
		os.path.join(all_model_dir, '{num_mark_model}', 'state_{num_state}', 'full_model', 'model_{num_state}.txt'), 
	output:
		expand(os.path.join(all_model_dir, '{{num_mark_model}}', 'state_{{num_state}}', 'full_model', 'POSTERIOR', 'genome_{{num_state}}_chr{chrom}_posterior.txt.gz'), chrom = ['1','X']),
	params:	
		out_dir = os.path.join(all_model_dir, '{num_mark_model}', 'state_{num_state}', 'full_model'),
		binarized_dir = lambda w: get_corresponding_binarized_dir(w),
	shell:	
		"""
		java -jar ../../program_source/ChromHMM/ChromHMM/ChromHMM.jar MakeSegmentation -gzip -splitrows -printposterior {input[0]} {params.binarized_dir} {params.out_dir}
		"""

rule calculate_entropy_posterior:
	input:
		expand(os.path.join(all_model_dir, '{{num_mark_model}}', 'state_{{num_state}}', 'full_model', 'POSTERIOR', 'genome_{{num_state}}_chr{chrom}_posterior.txt.gz'), chrom = ['1','X']), # from get_posterior_chromHMM
	output:
		expand(os.path.join(all_model_dir, '{{num_mark_model}}', 'state_{{num_state}}', 'full_model', 'POSTERIOR', 'entropy', 'entropy_chr{chrom}.txt.gz'), chrom =  ['1', 'X'])
	params:
		input_dir = os.path.join(all_model_dir, '{num_mark_model}', 'state_{num_state}', 'full_model', 'POSTERIOR'),
		output_dir = os.path.join(all_model_dir, '{num_mark_model}', 'state_{num_state}', 'full_model', 'POSTERIOR', 'entropy')
	shell:
		"""
		python ./scripts/calculate_entropy_posterior_chromHMM.py {params.input_dir} {params.output_dir}
		"""

rule calculate_entropy_posterior_ROADMAP:
	input:
		expand(os.path.join(ROADMAP_dir, '25_state', 'POSTERIOR', 'E123_25_imputed12marks_chr{chrom}_posterior.txt.gz'), chrom = ['1','X']), # from get_posterior_chromHMM
	output:
		expand(os.path.join(ROADMAP_dir, '25_state', 'POSTERIOR', 'entropy', 'entropy_chr{chrom}.txt.gz'), chrom =  ['1', 'X'])
	params:
		input_dir = os.path.join(ROADMAP_dir, '25_state', 'POSTERIOR'),
		output_dir = os.path.join(ROADMAP_dir, '25_state', 'POSTERIOR', 'entropy')
	shell:
		"""
		python ./scripts/calculate_entropy_posterior_chromHMM.py {params.input_dir} {params.output_dir}
		"""

rule plot_histogram_entropy: # given the entropy from any model, we will plot the histogram
	input:
		expand(os.path.join('{{model_folder}}', 'POSTERIOR', 'entropy', 'entropy_chr{chrom}.txt.gz'), chrom =  ['1', 'X'])
	output:
		os.path.join('{model_folder}', 'POSTERIOR', 'entropy', 'plots', 'entropy_density_gw.png')
	run:
		output_dir = os.path.join(wildcards.model_folder, 'POSTERIOR', 'entropy', 'plots')
		CHROMOSOME_LIST = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X'] # there was actually no data of chromatin state for chromosome Y
		entropy_folder = os.path.join(wildcards.model_folder, 'POSTERIOR', 'entropy')
		entropy_fn_list = glob.glob(entropy_folder +  '/entropy_chr*.txt.gz')
		entropy_df_list = list(map(lambda x: pd.read_csv(x, header = None, index_col = None, sep = '\t'), entropy_fn_list[:1]))
		entropy_df_gw = pd.concat(entropy_df_list, ignore_index = True)
		entropy_df_gw.columns = ['entropy', 'state']
		grouped_df = entropy_df_gw.groupby('state')
		plot_nrow = (np.floor(np.sqrt(float(wildcards.num_state)))).astype(int)
		plot_ncol = int(np.ceil(float(wildcards.num_state) / plot_nrow))
		fig, axes = plt.subplots(ncols = plot_ncol, nrows = plot_nrow, figsize = (9,9))
		plot_index = 0
		for state, state_df in grouped_df:
			ax = (axes.flat)[plot_index] 
			sns.distplot(state_df['entropy'], kde= True, ax = ax).set_title(state)
			plot_index += 1
			# save_fn = os.path.join(output_dir, 'entropy_density_{state}.png'.format(state = state))
			# plt.tight_layout()
			# plt.savefig(save_fn)
		save_fn = os.path.join(output_dir, 'entropy_density_all_states.png')
		fig.tight_layout()
		fig.savefig(save_fn)
		# done with individual state, now we will look at the genome-wide entropy
		fig, ax = plt.subplots()
		sns.distplot(entropy_df_gw['entropy'], kde = True, ax = ax).set_title('all_states')
		save_fn = os.path.join(output_dir, 'entropy_density_gw.png')
		fig.savefig(save_fn)

rule get_segmentation_based_on_entropy:
# this rule will change from the traditional segment of N states into 2N states, where each of the N states will now get stratified into 2 states: those that have have high entropy and lower entropy. State numberings are such that from state N --> state 2N-1 if low entropy, and state 2N if high entropy.
	input:
		expand(os.path.join('{{model_folder}}', 'POSTERIOR', 'entropy', 'entropy_chr{chrom}.txt.gz'), chrom =  ['1', 'X'])
	output:
		os.path.join('{model_folder}', 'segment_based_on_entropy', 'genome_segments_sorted.bed.gz'),
		expand(os.path.join('{{model_folder}}', 'segment_based_on_entropy', '{compare_to_median}', 'genome_segments_sorted.bed.gz'), compare_to_median = ['higher', 'leq'])
	params:
		entropy_folder = os.path.join('{model_folder}', 'POSTERIOR', 'entropy'),
		output_dir = os.path.join('{model_folder}', 'segment_based_on_entropy'),
	shell:
		"""
		python ./scripts/change_segmentation_based_on_entropy.py {params.entropy_folder} {params.output_dir} 
		"""

rule get_segment_file_train_test:
	input:
		os.path.join('{model_folder}', 'genome_segments_sorted.bed.gz'), # this is from the model training and from rule sort_segment_file
		os.path.join(model_analysis_dir, 'train_segments_sorted.bed.gz'), # from rule get_train_test_bed_fn
		os.path.join(model_analysis_dir, 'test_segments_sorted.bed.gz') # from rule get_train_test_bed_fn
	output:
		expand(os.path.join('{{model_folder}}', 'train_and_test_segments_sorted', '{TT}_segments.bed.gz'), TT = ['train', 'test'])
	params:
		out_dir = os.path.join('{model_folder}', 'train_and_test_segments_sorted')
	shell:	
		"""	
		bedtools map -a {input[1]} -b {input[0]} -c 4 -o collapse > {params.out_dir}/train_segments.bed
		gzip {params.out_dir}/train_segments.bed
		bedtools map -a {input[2]} -b {input[0]} -c 4 -o collapse > {params.out_dir}/test_segments.bed
		gzip {params.out_dir}/test_segments.bed
		"""

rule overlap_with_genome_contexts: # for both the model trained by me and roadmap published 25-state model
	input:
		expand(os.path.join('{{model_folder}}', 'train_and_test_segments_sorted', '{TT}_segments.bed.gz'), TT = ['train', 'test']),
	output:
		expand(os.path.join('{{model_folder}}', 'overlap_crispr_mpra', '{{fore_or_back}}', '{TT}_overlap.txt'), TT = ['train', 'test'])
	params:
		train_out_prefix = os.path.join('{model_folder}', 'overlap_crispr_mpra', '{fore_or_back}', 'train_overlap'),
		test_out_prefix = os.path.join('{model_folder}', 'overlap_crispr_mpra', '{fore_or_back}', 'test_overlap')
	shell:
		"""
		java -jar ../../program_source/ChromHMM/ChromHMM/ChromHMM.jar OverlapEnrichment -noimage {input[0]} {COORD_CRISPR_MPRA_DIR}/{wildcards.fore_or_back} {params.train_out_prefix}
		java -jar ../../program_source/ChromHMM/ChromHMM/ChromHMM.jar OverlapEnrichment -noimage {input[1]} {COORD_CRISPR_MPRA_DIR}/{wildcards.fore_or_back} {params.test_out_prefix}
		"""

rule overlap_against_background:
	input:
		expand(os.path.join('{{model_folder}}', 'overlap_crispr_mpra', '{fore_or_back}', '{{TT}}_overlap.txt'), fore_or_back = ['foreground', 'background']) # from overlap_with_genome_contexts , TT: train, test, gw
	output:
		os.path.join('{model_folder}', 'overlap_crispr_mpra','fg_against_bg', '{TT}_overlap_against_bg.txt'), # TT: train or test or gw
	params:
		foreground_fn = os.path.join('{model_folder}', 'overlap_crispr_mpra', 'foreground', '{TT}_overlap.txt'),
		background_fn = os.path.join('{model_folder}', 'overlap_crispr_mpra', 'background', '{TT}_overlap.txt'),
	shell:
		"""
		python ./scripts/calculate_fold_enrichment_against_background.py {params.background_fn} {params.foreground_fn} {COORD_CRISPR_MPRA_COLUMN_MATCH_FN} {output[0]}
		"""

rule overlap_with_genome_contexts_gw: # for both the model trained by me and roadmap published 25-state model
	input:
		os.path.join('{model_folder}', 'genome_segments_sorted.bed.gz'),
	output:
		os.path.join('{model_folder}', 'overlap_crispr_mpra', '{fore_or_back}', 'gw_overlap.txt')
	params:
		output_prefix = os.path.join('{model_folder}', 'overlap_crispr_mpra', '{fore_or_back}', 'gw_overlap'),
	shell:
		"""
		java -jar ../../program_source/ChromHMM/ChromHMM/ChromHMM.jar OverlapEnrichment -noimage {input[0]} {COORD_CRISPR_MPRA_DIR}/{wildcards.fore_or_back} {params.output_prefix}
		"""

rule get_colored_excel_enrichment_gw:
	input:
		os.path.join('{model_folder}', 'overlap_crispr_mpra','{fore_or_back}', '{TT}_{overlap_type}.txt'), # TT: train or test or gw, fore_or_back can be foreground, background, or fg_against_bg, overlap_type can be overlap or overlap_against_bg. Overlap_type: overlap or overlap_against_bp
	output:	
		os.path.join('{model_folder}', 'overlap_crispr_mpra','{fore_or_back}', '{TT}_{overlap_type}.xlsx'), # TT: train or test or gw, fore_or_back can be foreground, background, or fg_against_bg, overlap_type can be overlap or overlap_against_bg. Overlap_type: overlap or overlap_against_bp
	shell:
		"""
		python ../utils/create_excel_overlap_enrichment.py {input[0]} {output[0]} ''
		"""

rule calculate_train_test_ROC_precall_raw: 
	input: 
		expand(os.path.join(all_model_dir, 'all_mark_model', 'state_{num_state}', 'full_model', 'segment_based_on_entropy', 'overlap_crispr_mpra', 'foreground', '{TT}_overlap.txt'), num_state = range(2, 26), TT = ['train', 'test']),
		expand(os.path.join(all_model_dir, 'three_mark_model', 'state_{num_state}', 'full_model', 'segment_based_on_entropy', 'overlap_crispr_mpra', 'foreground', '{TT}_overlap.txt'), num_state = range(2,9), TT = ['train', 'test']),
		expand(os.path.join(ROADMAP_dir, '25_state', 'overlap_crispr_mpra', 'foreground', '{TT}_overlap.txt'), TT = ['train', 'test']),
		expand(os.path.join(simple_rule_annot_dir, 'overlap_crispr_mpra', 'foreground', '{TT}_overlap.txt'), TT = ['train', 'test'])
	output:	
		expand(os.path.join(all_model_dir, 'compare_models_recover_crispr_mpra',  'entropy_stratified', 'foreground_only', 'auc_{context}.txt'), context = context_list)
	params:
		out_dir = os.path.join(all_model_dir, 'compare_models_recover_crispr_mpra', 'entropy_stratified', 'foreground_only')
	shell:	
		"""
		model_type="all_mark_model three_mark_model"
		all_mark_max_nState=25
		three_mark_max_nState=8
		overlap_folder_list=""
		model_name_list=""
		num_model=0
		for ns in `seq 2 ${{all_mark_max_nState}}`
		do
			this_overlap_folder={all_model_dir}/all_mark_model/state_${{ns}}/full_model/overlap_crispr_mpra/foreground
			overlap_folder_list="$overlap_folder_list $this_overlap_folder"
			this_model_name="M13_S${{ns}}"
			model_name_list="$model_name_list $this_model_name"
			num_model=$(( $num_model + 1 ))
		done
		for ns in `seq 2 ${{three_mark_max_nState}}`
		do
			this_overlap_folder={all_model_dir}/three_mark_model/state_${{ns}}/full_model/overlap_crispr_mpra/foreground
			overlap_folder_list="$overlap_folder_list $this_overlap_folder"
			this_model_name="M3_S${{ns}}"
			model_name_list="$model_name_list $this_model_name"
			num_model=$(( $num_model + 1 ))
		done
		num_model=$(( $num_model + 2 ))
		roadmap_overlap_folder={ROADMAP_dir}/25_state/overlap_crispr_mpra/foreground/
		simple_rules_overlap_folder={simple_rule_annot_dir}/overlap_crispr_mpra/foreground/
		overlap_folder_list="$overlap_folder_list $roadmap_overlap_folder $simple_rules_overlap_folder"
		model_name_list="$model_name_list roadmap_S25 simpleRules_S3" 
		python ./scripts/calculate_roc_train_test_enrichment.py $num_model {params.out_dir} ${{overlap_folder_list}} ${{model_name_list}}
		"""

rule calculate_train_test_ROC_precall_against_bg: 
	input: 
		expand(os.path.join(all_model_dir, 'all_mark_model', 'state_{num_state}', 'full_model', 'segment_based_on_entropy', 'overlap_crispr_mpra', 'fg_against_bg', '{TT}_overlap_against_bg.txt'), num_state = range(25, 26), TT = ['train', 'test']),
		expand(os.path.join(all_model_dir, 'three_mark_model', 'state_{num_state}', 'full_model', 'segment_based_on_entropy', 'overlap_crispr_mpra', 'fg_against_bg', '{TT}_overlap_against_bg.txt'), num_state = range(8,9), TT = ['train', 'test']),
		expand(os.path.join(ROADMAP_dir, '25_state', 'overlap_crispr_mpra', 'segment_based_on_entropy', 'fg_against_bg', '{TT}_overlap_against_bg.txt'), TT = ['train', 'test']),
		expand(os.path.join(simple_rule_annot_dir, 'overlap_crispr_mpra', 'fg_against_bg', '{TT}_overlap_against_bg.txt'), TT = ['train', 'test']),
	output:	
		expand(os.path.join(all_model_dir, 'compare_models_recover_crispr_mpra', 'entropy_stratified', 'fg_against_bg', 'auc_{context}.txt'), context = context_list)
	params:
		out_dir = os.path.join(all_model_dir, 'compare_models_recover_crispr_mpra', 'entropy_stratified', 'fg_against_bg')
	shell:	
		"""
		model_type="all_mark_model three_mark_model"
		all_mark_max_nState=25
		three_mark_max_nState=8
		overlap_folder_list=""
		model_name_list=""
		num_model=0
		for ns in `seq 25 ${{all_mark_max_nState}}`
		do
			this_overlap_folder={all_model_dir}/all_mark_model/state_${{ns}}/full_model/segment_based_on_entropy/overlap_crispr_mpra/fg_against_bg
			overlap_folder_list="$overlap_folder_list $this_overlap_folder"
			this_model_name="M13_S${{ns}}"
			model_name_list="$model_name_list $this_model_name"
			num_model=$(( $num_model + 1 ))
		done
		for ns in `seq 8 ${{three_mark_max_nState}}`
		do
			this_overlap_folder={all_model_dir}/three_mark_model/state_${{ns}}/full_model/segment_based_on_entropy/overlap_crispr_mpra/fg_against_bg
			overlap_folder_list="$overlap_folder_list $this_overlap_folder"
			this_model_name="M3_S${{ns}}"
			model_name_list="$model_name_list $this_model_name"
			num_model=$(( $num_model + 1 ))
		done
		num_model=$(( $num_model + 2 ))
		roadmap_overlap_folder={ROADMAP_dir}/25_state/overlap_crispr_mpra/fg_against_bg
		simple_rules_overlap_folder={simple_rule_annot_dir}/overlap_crispr_mpra/fg_against_bg
		overlap_folder_list="$overlap_folder_list $roadmap_overlap_folder $simple_rules_overlap_folder"
		model_name_list="$model_name_list roadmap_S25 simpleRules_S3" 
		python ./scripts/calculate_roc_enrichment_against_bg_train_test.py $num_model {params.out_dir} ${{overlap_folder_list}} ${{model_name_list}}
		"""
