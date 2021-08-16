NUM_TRAIN_ITERATIONS_list = [10,100,200,500,1000,4000]
NUM_BINS_SAMPLE_PER_ITER_list = [100, 1000]
out_dir = ' ../../pyro_model/simulation_study/tiny_toy'
rule all:
	input:
		expand(os.path.join(out_dir, 'with_beta_guide', 'n{NUM_TRAIN_ITERATIONS}_b{NUM_BINS_SAMPLE_PER_ITER}.txt'), NUM_TRAIN_ITERATIONS = NUM_TRAIN_ITERATIONS_list, NUM_BINS_SAMPLE_PER_ITER = NUM_BINS_SAMPLE_PER_ITER_list),
		expand(os.path.join(out_dir, 'no_beta_guide', 'n{NUM_TRAIN_ITERATIONS}_b{NUM_BINS_SAMPLE_PER_ITER}.txt'), NUM_TRAIN_ITERATIONS = NUM_TRAIN_ITERATIONS_list, NUM_BINS_SAMPLE_PER_ITER = NUM_BINS_SAMPLE_PER_ITER_list)

rule tiny_example_with_beta_guide:
	input:
	output:
		os.path.join(out_dir, 'with_beta_guide', 'n{NUM_TRAIN_ITERATIONS}_b{NUM_BINS_SAMPLE_PER_ITER}.txt')
	shell:
		'''
			python tiny_toy_example_with_betaGuide.py -n {wildcards.NUM_TRAIN_ITERATIONS} -o {wildcards.NUM_BINS_SAMPLE_PER_ITER} {output[0]}
		'''

rule tiny_example_no_beta_guide:
	input:
	output:
		os.path.join(out_dir, 'no_beta_guide', 'n{NUM_TRAIN_ITERATIONS}_b{NUM_BINS_SAMPLE_PER_ITER}.txt')
	shell:
		'''
			python tiny_toy_example_no_betaGuide.py -n {wildcards.NUM_TRAIN_ITERATIONS} -o {wildcards.NUM_BINS_SAMPLE_PER_ITER} {output[0]}
		'''