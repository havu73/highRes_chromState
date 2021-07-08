hg19_chrom_length_fn = '../../model_analysis/hg19_chrom_length_for_segmentation.bed' # chrom, start, end, start is always 0
model_analysis_dir = '../../model_analysis'
all_model_dir = '../../model_analysis/K562_hg19'
COORD_CRISPR_MPRA_DIR = '../../data/hg19/K562/COORD_CRISPR_MPRA_DATA/for_enrichment'
context_list = ['Fulco19_significant_DE_G_ST6a', 'gasperini19_altScale_664_enhancerGenePairs_ST2B', 'gasperini19_pilot_145_EnhancerGenePairs_ST1B', 'klann21_S1_discovery_K562_sgrna', 'klann21_S2_discovery_K562_bin2', 'klann21_S3_discovery_K562_bin3', 'klann21_S5_discovery_K562_dhs', 'klann21_S6_validation_K562_sgrna', 'klann21_S7_validation_K562_bin2', 'klann21_S8_validation_K562_bin3']
ROADMAP_dir = '../../data/hg19/K562/roadmap_published_chromHMM/'

rule all:
	input:
		# os.path.join(model_analysis_dir, 'train_segments.bed.gz'),
		# os.path.join(model_analysis_dir, 'test_segments.bed.gz'),
		# expand(os.path.join(all_model_dir, '{num_mark_model}', 'state_{num_state}', 'full_model', 'overlap_crispr_mpra', '{TT}_overlap.txt'), num_mark_model = ['all_mark_model'], num_state = range(2, 26), TT = ['train', 'test']),
		expand(os.path.join(all_model_dir, 'compare_models_recover_crispr_mpra', 'auc_{context}.txt'), context = context_list),

rule get_train_test_bed_fn:
	input:
		hg19_chrom_length_fn,
	output:
		os.path.join(model_analysis_dir, 'train_segments_sorted.bed.gz'),
		os.path.join(model_analysis_dir, 'test_segments_sorted.bed.gz')
	params:
		out_dir = model_analysis_dir,
		test_fn_no_gz = os.path.join(model_analysis_dir, 'test_segments_sorted.bed'),
		train_fn_no_gz = os.path.join(model_analysis_dir, 'train_segments_sorted.bed'),
	shell:
		"""
		python ./scripts/get_sample_genome_bed.py {input} 0.5 {params.out_dir}
		zcat {params.out_dir}/train_segments.bed.gz | sort -k1,1 -k2,2n > {params.train_fn_no_gz}
		gzip {params.train_fn_no_gz}
		zcat {params.out_dir}/test_segments.bed.gz | sort -k1,1 -k2,2n > {params.test_fn_no_gz}
		gzip {params.test_fn_no_gz}
		"""

rule sort_segment_file:
	input:
		os.path.join(all_model_dir, '{num_mark_model}', 'state_{num_state}', 'full_model', 'genome_{num_state}_segments.bed.gz'), # this is from the model training
	output:
		os.path.join(all_model_dir, '{num_mark_model}', 'state_{num_state}', 'full_model', 'genome_{num_state}_segments_sorted.bed.gz')
	params:
		output_no_gz = os.path.join(all_model_dir, '{num_mark_model}', 'state_{num_state}', 'full_model', 'genome_{num_state}_segments_sorted.bed'),
	shell:
		"""
		zcat {input[0]} | sort -k1,1 -k2,2n > {params.output_no_gz}
		gzip {params.output_no_gz}
		"""

rule ROADMAP_sort_segment_file:
	input: # exactly the same as sort_segment_file, just for ROADMAP 25-state segmentation 
		os.path.join(ROADMAP_dir, '25_state' 'E123_15_coreMarks_segments.bed.gz')
	output:
		os.path.join(ROADMAP_dir, '25_state' 'E123_25_imputed12marks_segments.bed.gz')
	params:
		output_no_gz = os.path.join(ROADMAP_dir, '25_state' 'E123_15_coreMarks_segments_sorted.bed'),
	shell:
		"""
		zcat {input[0]} | sort -k1,1 -k2,2n > {params.output_no_gz}
		gzip {params.output_no_gz}
		"""

rule get_segment_file_train_test:
	input:
		os.path.join(all_model_dir, '{num_mark_model}', 'state_{num_state}', 'full_model', 'genome_{num_state}_segments_sorted.bed.gz'), # this is from the model training
		os.path.join(model_analysis_dir, 'train_segments_sorted.bed.gz'), # from rule get_train_test_bed_fn
		os.path.join(model_analysis_dir, 'test_segments_sorted.bed.gz') # from rule get_train_test_bed_fn
	output:
		expand(os.path.join(all_model_dir, '{{num_mark_model}}', 'state_{{num_state}}', 'full_model', 'train_and_test_segments_sorted', '{TT}_segments.bed.gz'), TT = ['train', 'test'])
	params:
		out_dir = os.path.join(all_model_dir, '{num_mark_model}', 'state_{num_state}', 'full_model', 'train_and_test_segments_sorted')
	shell:	
		"""	
		bedtools map -a {input[1]} -b {input[0]} -c 4 -o collapse > {params.out_dir}/train_segments.bed
		gzip {params.out_dir}/train_segments.bed
		bedtools map -a {input[2]} -b {input[0]} -c 4 -o collapse > {params.out_dir}/test_segments.bed
		gzip {params.out_dir}/test_segments.bed
		"""

rule ROADMAP_get_segment_file_train_test:
	input:
		os.path.join(ROADMAP_dir, '25_state' 'E123_25_imputed12marks_segments.bed.gz'), # from ROADMAP_sort_segment_file
		os.path.join(model_analysis_dir, 'train_segments_sorted.bed.gz'), # from rule get_train_test_bed_fn
		os.path.join(model_analysis_dir, 'test_segments_sorted.bed.gz'), # from rule get_train_test_bed_fn
	output:
		expand(os.path.join(ROADMAP_dir, '25_state' 'train_and_test_segments_sorted', '{TT}_segments.bed.gz'), TT = ['train', 'test'])
	params:
		out_dir = os.path.join(ROADMAP_dir, '25_state' 'train_and_test_segments_sorted')
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
		expand(os.path.join('{{model_folder}}', 'overlap_crispr_mpra', '{TT}_overlap.txt'), TT = ['train', 'test'])
	params:
		train_out_prefix = os.path.join('{model_folder}', 'overlap_crispr_mpra', 'train_overlap'),
		test_out_prefix = os.path.join('{model_folder}', 'overlap_crispr_mpra', 'test_overlap')
	shell:
		"""
		java -jar ../../program_source/ChromHMM/ChromHMM/ChromHMM.jar OverlapEnrichment -noimage {input[0]} {COORD_CRISPR_MPRA_DIR} {params.train_out_prefix}
		java -jar ../../program_source/ChromHMM/ChromHMM/ChromHMM.jar OverlapEnrichment -noimage {input[1]} {COORD_CRISPR_MPRA_DIR} {params.test_out_prefix}
		"""

rule calculate_train_test_ROC_precall: 
	input: 
		expand(os.path.join(all_model_dir, 'all_mark_model', 'state_{num_state}', 'full_model', 'overlap_crispr_mpra', '{TT}_overlap.txt'), num_state = range(2, 26), TT = ['train', 'test']),
		expand(os.path.join(all_model_dir, 'three_mark_model', 'state_{num_state}', 'full_model', 'overlap_crispr_mpra', '{TT}_overlap.txt'), num_state = range(2,9), TT = ['train', 'test']),
		expand(os.path.join(ROADMAP_dir, '25_state' 'overlap_crispr_mpra', '{TT}_overlap.txt'), TT = ['train', 'test']),
	output:	
		expand(os.path.join(all_model_dir, 'compare_models_recover_crispr_mpra', 'auc_{context}.txt'), context = context_list)
	params:
		out_dir = os.path.join(all_model_dir, 'compare_models_recover_crispr_mpra')
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
			this_overlap_folder={all_model_dir}/all_mark_model/state_${{ns}}/full_model/overlap_crispr_mpra/
			overlap_folder_list="$overlap_folder_list $this_overlap_folder"
			this_model_name="M13_S${{ns}}"
			model_name_list="$model_name_list $this_model_name"
			num_model=$(( $num_model + 1 ))
		done
		for ns in `seq 2 ${{three_mark_max_nState}}`
		do
			this_overlap_folder={all_model_dir}/three_mark_model/state_${{ns}}/full_model/overlap_crispr_mpra/
			overlap_folder_list="$overlap_folder_list $this_overlap_folder"
			this_model_name="M3_S${{ns}}"
			model_name_list="$model_name_list $this_model_name"
			num_model=$(( $num_model + 1 ))
		done
		num_model=$(( $num_model + 1 ))
		roadmap_overlap_folder={ROADMAP_dir}/25_state/overlap_crispr_mpra/
		overlap_folder_list="$overlap_folder_list $roadmap_overlap_folder"
		model_name_list="$model_name_list roadmap_S25" 
		python ./scripts/calculate_roc_train_test_enrichment.py $num_model {params.out_dir} ${{overlap_folder_list}} ${{model_name_list}}
		"""

# rule visualize_auc_results:
# 	input:
# 		expand(os.path.join(all_model_dir, 'compare_models_recover_crispr_mpra', 'auc_{context}.txt'), context = context_list) # from calculate_train_test_overlap
# 	output:
		