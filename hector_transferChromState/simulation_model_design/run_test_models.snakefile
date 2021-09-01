import numpy as np
import os 
state_list = np.arange(3, 15, 5) # 3 --> 18
mark_list = [3, 30]
reference_list = [3,10,20] # 3 --> 63
num_groups = 3 # # groups of references
state_vary_rate = 0.01 # fraction of genome where there are differences in states btw 2 refs of the same group
bins_list = [10000]

output_folder='./experiments/strict_genData_circularState'

rule all:
	input:
		expand(os.path.join(output_folder, 'b{bins}_r{reference}_m{signal}_s{state}', 'report_ratio_CR.txt'), bins = bins_list, reference = reference_list, signal = mark_list, state = state_list)

rule test_one_genData_param_set: 
# run test model on simulated data generated from generate_toy_data.py
	input:
	output:
		os.path.join(output_folder, 'b{bins}_r{reference}_m{signal}_s{state}', 'report_ratio_CR.txt')
	params:
		output_folder=os.path.join(output_folder, 'b{bins}_r{reference}_m{signal}_s{state}')
	shell:
		"""
		python scripts/test_models.py --num_bins {wildcards.bins} --num_references {wildcards.reference} --num_signals {wildcards.signal} --num_states {wildcards.state} --output_folder {params.output_folder}
		"""

