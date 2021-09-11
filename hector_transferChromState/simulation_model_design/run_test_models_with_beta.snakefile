import numpy as np
import os 
state_list = np.arange(3, 15, 5) # 3 --> 18
mark_list = [3, 30]
reference_list = [3,10,20] # 3 --> 63
num_groups = 3 # # groups of references
state_vary_rate = 0.01 # fraction of genome where there are differences in states btw 2 refs of the same group
bins_list = [10000]
hidden_ref_list = range(10, 31, 10)
hidden_comb_list = range(10, 31, 10)

genData_beta_outDir='./experiments/experiments/strict_genData_realScen/counts/with_beta/'
real_data_outdir = '/gstore/home/vuh6/source/hector_transferChromState/simulation_model_design/experiments/real_data'
rule all:
	input:
		# expand(os.path.join(genData_beta_outDir, 'b{bins}_r{reference}_m{signal}_s{state}', 'report_ratio_CR.txt'), bins = bins_list, reference = reference_list, signal = mark_list, state = state_list),
		expand(os.path.join(genData_beta_outDir, 'b{batch}_e{epoch}_s{hidden_sig}_r{hidden_ref}_c{hidden_comb}_h{hidden}', 'report_ratio_CR.txt'), bins = bins_list, epoch = [4000], batch = [200], hidden_sig=[1],  hidden_ref = hidden_ref_list, hidden_comb = hidden_comb_list, hidden = [32]),
		expand(os.path.join(genData_beta_outDir, 'b{batch}_e{epoch}_s{hidden_sig}_r{hidden_ref}_c{hidden_comb}_h{hidden}', 'report_ratio_CR.txt'), bins = bins_list, epoch = [4000], batch = [200], hidden_sig=[1],  hidden_ref = 25, hidden_comb = 15, hidden = range(10,31,10))

rule test_one_genData_param_set: 
# run test model on simulated data generated from generate_toy_data.py
	input:
	output:
		os.path.join(genData_beta_outDir, 'b{bins}_r{reference}_m{signal}_s{state}', 'report_ratio_CR.txt')
	params:
		output_folder=os.path.join(genData_beta_outDir, 'b{bins}_r{reference}_m{signal}_s{state}')
	shell:
		"""
		python scripts/test_model.py  --num_bins {wildcards.bins} --num_references {wildcards.reference} --num_signals {wildcards.signal} --num_states {wildcards.state} --output_folder {params.output_folder}
		"""

rule test_one_genData_realSce_counts_param_set:
	input:
	output:
		os.path.join(genData_beta_outDir, 'b{batch}_e{epoch}_s{hidden_sig}_r{hidden_ref}_c{hidden_comb}_h{hidden}', 'report_ratio_CR.txt')
	params:
		output_folder = os.path.join(genData_beta_outDir, 'b{batch}_e{epoch}_s{hidden_sig}_r{hidden_ref}_c{hidden_comb}_h{hidden}')
	shell:
		"""
		python scripts/refState_counts/test_realScen_withBeta.py --output_folder {params.output_folder} --num_epochs {wildcards.epoch} --batch_size {wildcards.batch} --num_hidden_sig={wildcards.hidden_sig} --num_hidden_ref={wildcards.hidden_ref} --num_hidden_comb={wildcards.hidden_comb} --num_hidden={wildcards.hidden} 
		"""

rule test_one_realData_param_set:
	input:
		'/gstore/home/vuh6/pyro_model/old_generative/E123_hg19/train_data/maxBin_5000_chr22/observed_mark_signals_for_training.bed.gz', # mark_fn
		'/gstore/home/vuh6/pyro_model/old_generative/train_segmentation_data/maxBin_5000_chr22/ref_epig_segment_for_training.bed.gz', # ref_fn
		'/gstore/project/gepiviz_data/vuh6/roadmap/emission/emissions_25_imputed12marks.txt', # emission_fn
	output:
		os.path.join(real_data_outdir, 'SigRef_pos.txt.gz')
	params:
		output_folder = real_data_outdir
	shell:
		"""
		python scripts/test_models_real_data.py --mark_fn={input[0]} --ref_fn={input[1]} --emission_fn={input[2]} --output_folder={params.output_folder} --emission_scale=200 --num_states=25 --batch_size=300 --num_epochs=1500 --num_hidden=32 --dropout=0.2
		"""

