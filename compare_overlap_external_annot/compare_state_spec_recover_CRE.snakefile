import os 
import numpy as np
COORD_CRISPR_MPRA_DIR = '../../data/hg19/K562/COORD_CRISPR_MPRA_DATA/for_enrichment'
COORD_CRISPR_MPRA_COLUMN_MATCH_FN = os.path.join(COORD_CRISPR_MPRA_DIR, 'match_foreground_background')
simple_rule_annot_dir = '../../model_analysis/K562_hg19/simple_rules_from_natalie'
context_list = ['Fulco19_significant_DE_G_ST6a', 'gasperini19_altScale_664_enhancerGenePairs_ST2B', 'gasperini19_pilot_145_EnhancerGenePairs_ST1B', 'klann21_S1_discovery_K562_sgrna', 'klann21_S2_discovery_K562_bin2', 'klann21_S3_discovery_K562_bin3', 'klann21_S5_discovery_K562_dhs', 'klann21_S6_validation_K562_sgrna', 'klann21_S7_validation_K562_bin2', 'klann21_S8_validation_K562_bin3']
model_analysis_dir = '../../model_analysis'
big_model_posterior_dir = '/gstore/home/vuh6/model_analysis/K562_hg19/all_mark_model/state_25/full_model/POSTERIOR'
three_mark_model_posterior_dir = '/gstore/home/vuh6/model_analysis/K562_hg19/three_mark_model/state_8/full_model/POSTERIOR'
ROADMAP_posterior_dir = '/gstore/project/gepiviz_data/vuh6/roadmap/posterior/E123'
all_model_dir = '/gstore/home/vuh6/model_analysis/K562_hg19/'
rule all:
	input:
		expand(os.path.join(big_model_posterior_dir, 'state_spec_segmentation', 'E{state}_train_test', 'overlap_crispr_mpra','fg_against_bg', '{TT}_overlap_against_bg.txt'), state = np.arange(25)+1, TT = ['train', 'test']),
		expand(os.path.join(three_mark_model_posterior_dir, 'state_spec_segmentation', 'E{state}_train_test', 'overlap_crispr_mpra','fg_against_bg', '{TT}_overlap_against_bg.txt'), state = np.arange(8)+1, TT = ['train', 'test']),
		expand(os.path.join(ROADMAP_posterior_dir, 'state_spec_segmentation', 'E{state}_train_test', 'overlap_crispr_mpra','fg_against_bg', '{TT}_overlap_against_bg.txt'), state = np.arange(25)+1, TT = ['train', 'test']),
		expand(os.path.join(all_model_dir, 'compare_models_recover_crispr_mpra', 'state_spec_segmentation', 'fg_against_bg', 'auc_{context}.txt'), context = context_list)

# rule get_per_state_segmentation_based_on_posterior:
# 	input:
# 		expand(os.path.join('{{model_folder}}', 'genome_{{num_state}}_chr{chrom}_posterior.txt.gz'), chrom = ['1', 'X']),
# 	output:
# 		expand(os.path.join('{{model_folder}}', 'state_spec_segmentation', 'genome_E{state}_posterior_segment.txt.gz'), state = [1,3]),
# 	params: 
# 		quantile_fn = os.path.join('{model_folder}', 'quantile_posterior_chr1.txt'),
# 		output_folder = os.path.join('{model_folder}', 'state_spec_segmentation')
# 	shell: 
# 		"""
# 		python ./scripts/get_state_spec_segmentation_based_on_quantile.py {wildcards.model_folder} {params.quantile_fn} {params.output_folder} {wildcards.num_state}
# 		"""

rule sort_segment_file: # sort segmentation of state-specific segmentation
	input:
		os.path.join('{model_folder}', 'state_spec_segmentation', 'genome_E{state}_posterior_segment.txt.gz'), # this is from the model training, and from the rule soft_link_segments
	output:
		os.path.join('{model_folder}', 'state_spec_segmentation', 'genome_E{state}_posterior_sorted.txt.gz')
	params:
		output_no_gz = os.path.join('{model_folder}', 'state_spec_segmentation', 'genome_E{state}_posterior_sorted.txt'),
	shell:
		"""
		zcat {input[0]} | sort -k1,1 -k2,2n > {params.output_no_gz}
		gzip {params.output_no_gz}
		"""

rule get_segment_file_train_test:
	input:
		os.path.join('{model_folder}', 'state_spec_segmentation', 'genome_E{state}_posterior_sorted.txt.gz'), # this is from get_per_state_segmentation_based_on_posterior and from rule sort_segment_file
		os.path.join(model_analysis_dir, 'train_segments_sorted.bed.gz'), # from rule get_train_test_bed_fn
		os.path.join(model_analysis_dir, 'test_segments_sorted.bed.gz') # from rule get_train_test_bed_fn
	output:
		expand(os.path.join('{{model_folder}}', 'state_spec_segmentation', 'E{{state}}_train_test', '{TT}_segments.bed.gz'), TT = ['train', 'test'])
	params:
		out_dir = os.path.join('{model_folder}', 'state_spec_segmentation', 'E{state}_train_test')
	shell:	
		"""	
		bedtools map -a {input[1]} -b {input[0]} -c 4 -o collapse > {params.out_dir}/train_segments.bed
		gzip {params.out_dir}/train_segments.bed
		bedtools map -a {input[2]} -b {input[0]} -c 4 -o collapse > {params.out_dir}/test_segments.bed
		gzip {params.out_dir}/test_segments.bed
		"""

rule overlap_with_genome_contexts_train_test: # for both the model trained by me and roadmap published 25-state model
	input:
		expand(os.path.join('{{state_folder}}', '{TT}_segments.bed.gz'), TT = ['train', 'test']),
	output:
		expand(os.path.join('{{state_folder}}', 'overlap_crispr_mpra', '{{fore_or_back}}', '{TT}_overlap.txt'), TT = ['train', 'test'])
	params:
		train_out_prefix = os.path.join('{state_folder}', 'overlap_crispr_mpra', '{fore_or_back}', 'train_overlap'),
		test_out_prefix = os.path.join('{state_folder}', 'overlap_crispr_mpra', '{fore_or_back}', 'test_overlap')
	shell:
		"""
		java -jar ../../program_source/ChromHMM/ChromHMM/ChromHMM.jar OverlapEnrichment -labels -noimage {input[0]} {COORD_CRISPR_MPRA_DIR}/{wildcards.fore_or_back} {params.train_out_prefix}
		java -jar ../../program_source/ChromHMM/ChromHMM/ChromHMM.jar OverlapEnrichment -labels -noimage {input[1]} {COORD_CRISPR_MPRA_DIR}/{wildcards.fore_or_back} {params.test_out_prefix}
		"""

rule overlap_against_background:
### the TT wildcards here can be: train, test or gw
	input:
		expand(os.path.join('{{state_folder}}', 'overlap_crispr_mpra', '{fore_or_back}', '{{TT}}_overlap.txt'), fore_or_back = ['foreground', 'background']) # from overlap_with_genome_contexts_train_test
	output:
		os.path.join('{state_folder}', 'overlap_crispr_mpra','fg_against_bg', '{TT}_overlap_against_bg.txt'), # TT: train or test or gw
	params:
		foreground_fn = os.path.join('{state_folder}', 'overlap_crispr_mpra', 'foreground', '{TT}_overlap.txt'),
		background_fn = os.path.join('{state_folder}', 'overlap_crispr_mpra', 'background', '{TT}_overlap.txt'),
	shell:
		"""
		python ./scripts/calculate_fold_enrichment_against_background.py {params.background_fn} {params.foreground_fn} {COORD_CRISPR_MPRA_COLUMN_MATCH_FN} {output[0]}
		"""

rule calculate_train_test_ROC_precall_against_bg: 
	input: 
		expand(os.path.join(big_model_posterior_dir, 'state_spec_segmentation', 'E{state}_train_test', 'overlap_crispr_mpra','fg_against_bg', '{TT}_overlap_against_bg.txt'), state = np.arange(25)+1, TT = ['train', 'test']),
		expand(os.path.join(three_mark_model_posterior_dir, 'state_spec_segmentation', 'E{state}_train_test', 'overlap_crispr_mpra','fg_against_bg', '{TT}_overlap_against_bg.txt'), state = np.arange(8)+1, TT = ['train', 'test']),
		expand(os.path.join(ROADMAP_posterior_dir, 'state_spec_segmentation', 'E{state}_train_test', 'overlap_crispr_mpra','fg_against_bg', '{TT}_overlap_against_bg.txt'), state = np.arange(25)+1, TT = ['train', 'test']),
		expand(os.path.join(simple_rule_annot_dir, 'overlap_crispr_mpra', 'fg_against_bg', '{TT}_overlap_against_bg.txt'), TT = ['train', 'test']), # calculated before in other pipelines
	output:	
		expand(os.path.join(all_model_dir, 'compare_models_recover_crispr_mpra', 'state_spec_segmentation', 'fg_against_bg', 'auc_{context}.txt'), context = context_list)
	params:
		out_dir = os.path.join(all_model_dir, 'compare_models_recover_crispr_mpra', 'state_spec_segmentation', 'fg_against_bg')
	shell:	
		"""
		model_type="all_mark_model three_mark_model"
		all_mark_nState=25
		three_mark_nState=8
		overlap_folder_list=""
		model_name_list=""
		num_model=0
		for ns in `seq 1 ${{all_mark_nState}}`
		do
			this_overlap_folder={big_model_posterior_dir}/state_spec_segmentation/E${{ns}}_train_test/overlap_crispr_mpra/fg_against_bg
			overlap_folder_list="$overlap_folder_list $this_overlap_folder"
			this_model_name="M13_S25_posterior_E${{ns}}"
			model_name_list="$model_name_list $this_model_name"
			num_model=$(( $num_model + 1 ))
		done
		for ns in `seq 1 ${{three_mark_nState}}`
		do
			this_overlap_folder={three_mark_model_posterior_dir}/state_spec_segmentation/E${{ns}}_train_test/overlap_crispr_mpra/fg_against_bg
			overlap_folder_list="$overlap_folder_list $this_overlap_folder"
			this_model_name="M3_S8_posterior_E${{ns}}"
			model_name_list="$model_name_list $this_model_name"
			num_model=$(( $num_model + 1 ))
		done
		for ns in `seq 1 ${{all_mark_nState}}`
		do
			this_overlap_folder={ROADMAP_posterior_dir}/state_spec_segmentation/E${{ns}}_train_test/overlap_crispr_mpra/fg_against_bg
			overlap_folder_list="$overlap_folder_list $this_overlap_folder"
			this_model_name="roadmap25_posterior_E${{ns}}"
			model_name_list="$model_name_list $this_model_name"
			num_model=$(( $num_model + 1 ))
		done
		num_model=$(( $num_model + 1 ))
		simple_rules_overlap_folder={simple_rule_annot_dir}/overlap_crispr_mpra/fg_against_bg
		overlap_folder_list="$overlap_folder_list $simple_rules_overlap_folder"
		model_name_list="$model_name_list simpleRules_S3" 
		python ./scripts/calculate_roc_enrichment_against_bg_train_test_state_spec.py $num_model {params.out_dir} ${{overlap_folder_list}} ${{model_name_list}}
		"""


