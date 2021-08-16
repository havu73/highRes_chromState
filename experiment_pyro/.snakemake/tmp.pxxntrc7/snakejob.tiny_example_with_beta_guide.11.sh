#!/bin/sh
# properties = {"type": "single", "rule": "tiny_example_with_beta_guide", "local": false, "input": [], "output": [" ../../pyro_model/simulation_study/tiny_toy/with_beta_guide/n4000_b100.txt"], "wildcards": {"NUM_TRAIN_ITERATIONS": "4000", "NUM_BINS_SAMPLE_PER_ITER": "100"}, "params": {}, "log": [], "threads": 1, "resources": {}, "jobid": 11, "cluster": {}}
 cd /gstore/home/vuh6/source/experiment_pyro && \
/gstore/apps/Anaconda3/5.0.1/bin/python \
-m snakemake ' ../../pyro_model/simulation_study/tiny_toy/with_beta_guide/n4000_b100.txt' --snakefile /gstore/home/vuh6/source/experiment_pyro/evaluate_beta_pi_tiny_example.snakefile \
--force -j --keep-target-files --keep-remote \
--wait-for-files /gstore/home/vuh6/source/experiment_pyro/.snakemake/tmp.pxxntrc7 --latency-wait 5 \
 --attempt 1 --force-use-threads \
--wrapper-prefix https://github.com/snakemake/snakemake-wrappers/raw/ \
   --allowed-rules tiny_example_with_beta_guide --nocolor --notemp --no-hooks --nolock \
--mode 2  && touch /gstore/home/vuh6/source/experiment_pyro/.snakemake/tmp.pxxntrc7/11.jobfinished || (touch /gstore/home/vuh6/source/experiment_pyro/.snakemake/tmp.pxxntrc7/11.jobfailed; exit 1)

