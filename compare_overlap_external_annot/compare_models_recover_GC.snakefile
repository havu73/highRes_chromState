hg19_chrom_length_fn = '/gstore/home/vuh6/model_analysis/hg19_chrom_length_for_segmentation.bed' # chrom, start, end, start is always 0
model_analysis_dir = '/gstore/home/vuh6/model_analysis'
all_model_dir = '/gstore/home/vuh6/model_analysis/K562_hg19'
COORD_CRISPR_MPRA_DIR = '/gstore/home/vuh6/data/hg19/K562/COORD_CRISPR_MPRA_DATA/for_enrichment'
COORD_CRISPR_MPRA_COLUMN_MATCH_FN = os.path.join(COORD_CRISPR_MPRA_DIR, 'match_foreground_background')
context_list = ['Fulco19_significant_DE_G_ST6a', 'gasperini19_altScale_664_enhancerGenePairs_ST2B', 'gasperini19_pilot_145_EnhancerGenePairs_ST1B', 'klann21_S1_discovery_K562_sgrna', 'klann21_S2_discovery_K562_bin2', 'klann21_S3_discovery_K562_bin3', 'klann21_S5_discovery_K562_dhs', 'klann21_S6_validation_K562_sgrna', 'klann21_S7_validation_K562_bin2', 'klann21_S8_validation_K562_bin3']
ROADMAP_dir = '/gstore/home/vuh6/data/hg19/K562/roadmap_published_chromHMM/'
simple_rule_annot_dir = '/gstore/home/vuh6/model_analysis/K562_hg19/simple_rules_from_natalie'

rule all:
	input:
		# os.path.join(model_analysis_dir, 'train_segments.bed.gz'),
		# os.path.join(model_analysis_dir, 'test_segments.bed.gz'),
		# expand(os.path.join(all_model_dir, '{num_mark_model}', 'state_{num_state}', 'full_model', 'overlap_crispr_mpra','fg_against_bg','{TT}_overlap_against_bg.txt'), num_mark_model = ['three_mark_model'], num_state = range(2, 9), TT = ['train', 'test']),
		# expand(os.path.join(all_model_dir, '{num_mark_model}', 'state_{num_state}', 'full_model', 'overlap_crispr_mpra', 'fg_against_bg','{TT}_overlap_against_bg.txt'), num_mark_model = ['all_mark_model'], num_state = range(2, 26), TT = ['train', 'test']),
		expand(os.path.join(all_model_dir, 'compare_models_recover_crispr_mpra', 'fg_against_bg', 'auc_{context}.txt'), context = context_list),

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

rule soft_link_segments: #we need to soft link so that all the segmentation files inside different models' folder will have the same name. This will be helpful beacue it makes the code more managable 
	input: 
		expand(os.path.join(all_model_dir, '{num_mark_model}', 'state_{num_state}', 'full_model', 'genome_{num_state}_segments.bed.gz'), num_mark_model = ['all_mark_model'], num_state = range(2,26)),
		expand(os.path.join(all_model_dir, '{num_mark_model}', 'state_{num_state}', 'full_model', 'genome_{num_state}_segments.bed.gz'), num_mark_model = ['three_mark_model'], num_state = range(3,9)),
		os.path.join(ROADMAP_dir, '25_state', 'E123_25_imputed12marks_segments.bed.gz'), # roadmap published model
		os.path.join(simple_rule_annot_dir, 'genome_simple_3_segments.bed.gz'), # simple rules from Natalie
	output:
		expand(os.path.join(all_model_dir, '{num_mark_model}', 'state_{num_state}', 'full_model', 'genome_segments.bed.gz'), num_mark_model = ['all_mark_model'], num_state = range(2,26)),
		expand(os.path.join(all_model_dir, '{num_mark_model}', 'state_{num_state}', 'full_model', 'genome_segments.bed.gz'), num_mark_model = ['three_mark_model'], num_state = range(3,9)),
		os.path.join(ROADMAP_dir, '25_state', 'genome_segments.bed.gz'), # roadmap published model
		os.path.join(simple_rule_annot_dir, 'genome_segments.bed.gz'), # simple rules from Natalie
	shell:
		"""
		for f in {input}
		do
			model_folder=$(dirname $f) # function dirname in bash shows the directory of a file
			ln -s $f ${{model_folder}}/genome_segments.bed.gz
		done
		"""


rule sort_segment_file:
	input:
		os.path.join('{model_folder}', 'genome_segments.bed.gz'), # this is from the model training, and from the rule soft_link_segments
	output:
		os.path.join('{model_folder}', 'genome_segments_sorted.bed.gz')
	params:
		output_no_gz = os.path.join('{model_folder}','genome_segments_sorted.bed'),
	shell:
		"""
		zcat {input[0]} | sort -k1,1 -k2,2n > {params.output_no_gz}
		gzip {params.output_no_gz}
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
		java -jar /gstore/home/vuh6/program_source/ChromHMM/ChromHMM/ChromHMM.jar OverlapEnrichment -noimage {input[0]} {COORD_CRISPR_MPRA_DIR}/{wildcards.fore_or_back} {params.train_out_prefix}
		java -jar /gstore/home/vuh6/program_source/ChromHMM/ChromHMM/ChromHMM.jar OverlapEnrichment -noimage {input[1]} {COORD_CRISPR_MPRA_DIR}/{wildcards.fore_or_back} {params.test_out_prefix}
		"""

rule overlap_against_background:
	input:
		expand(os.path.join('{{model_folder}}', 'overlap_crispr_mpra', '{fore_or_back}', '{{TT}}_overlap.txt'), fore_or_back = ['foreground', 'background']) # from overlap_with_genome_contexts
	output:
		os.path.join('{model_folder}', 'overlap_crispr_mpra','fg_against_bg', '{TT}_overlap_against_bg.txt'), # TT: train or test
	params:
		foreground_fn = os.path.join('{model_folder}', 'overlap_crispr_mpra', 'foreground', '{TT}_overlap.txt'),
		background_fn = os.path.join('{model_folder}', 'overlap_crispr_mpra', 'background', '{TT}_overlap.txt'),
	shell:
		"""
		python ./scripts/calculate_fold_enrichment_against_background.py {params.background_fn} {params.foreground_fn} {COORD_CRISPR_MPRA_COLUMN_MATCH_FN} {output[0]}
		"""

rule calculate_train_test_ROC_precall_raw: 
	input: 
		expand(os.path.join(all_model_dir, 'all_mark_model', 'state_{num_state}', 'full_model', 'overlap_crispr_mpra', 'foreground', '{TT}_overlap.txt'), num_state = range(2, 26), TT = ['train', 'test']),
		expand(os.path.join(all_model_dir, 'three_mark_model', 'state_{num_state}', 'full_model', 'overlap_crispr_mpra', 'foreground', '{TT}_overlap.txt'), num_state = range(2,9), TT = ['train', 'test']),
		expand(os.path.join(ROADMAP_dir, '25_state', 'overlap_crispr_mpra', 'foreground', '{TT}_overlap.txt'), TT = ['train', 'test']),
		expand(os.path.join(simple_rule_annot_dir, 'overlap_crispr_mpra', 'foreground', '{TT}_overlap.txt'), TT = ['train', 'test'])
	output:	
		expand(os.path.join(all_model_dir, 'compare_models_recover_crispr_mpra', 'foreground_only', 'auc_{context}.txt'), context = context_list)
	params:
		out_dir = os.path.join(all_model_dir, 'compare_models_recover_crispr_mpra', 'foreground_only')
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
		expand(os.path.join(all_model_dir, 'all_mark_model', 'state_{num_state}', 'full_model', 'overlap_crispr_mpra', 'fg_against_bg', '{TT}_overlap_against_bg.txt'), num_state = range(2, 26), TT = ['train', 'test']),
		expand(os.path.join(all_model_dir, 'three_mark_model', 'state_{num_state}', 'full_model', 'overlap_crispr_mpra', 'fg_against_bg', '{TT}_overlap_against_bg.txt'), num_state = range(2,9), TT = ['train', 'test']),
		expand(os.path.join(ROADMAP_dir, '25_state', 'overlap_crispr_mpra', 'fg_against_bg', '{TT}_overlap_against_bg.txt'), TT = ['train', 'test']),
		expand(os.path.join(simple_rule_annot_dir, 'overlap_crispr_mpra', 'fg_against_bg', '{TT}_overlap_against_bg.txt'), TT = ['train', 'test']),
	output:	
		expand(os.path.join(all_model_dir, 'compare_models_recover_crispr_mpra', 'fg_against_bg', 'auc_{context}.txt'), context = context_list)
	params:
		out_dir = os.path.join(all_model_dir, 'compare_models_recover_crispr_mpra', 'fg_against_bg')
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
			this_overlap_folder={all_model_dir}/all_mark_model/state_${{ns}}/full_model/overlap_crispr_mpra/fg_against_bg
			overlap_folder_list="$overlap_folder_list $this_overlap_folder"
			this_model_name="M13_S${{ns}}"
			model_name_list="$model_name_list $this_model_name"
			num_model=$(( $num_model + 1 ))
		done
		for ns in `seq 2 ${{three_mark_max_nState}}`
		do
			this_overlap_folder={all_model_dir}/three_mark_model/state_${{ns}}/full_model/overlap_crispr_mpra/fg_against_bg
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
