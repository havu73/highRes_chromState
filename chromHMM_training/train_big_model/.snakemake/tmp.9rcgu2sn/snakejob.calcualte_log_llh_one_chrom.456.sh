#!/bin/sh
# properties = {"type": "single", "rule": "calcualte_log_llh_one_chrom", "local": false, "input": ["../../../model_analysis/K562_hg19/all_mark_model/state_115/half_for_model_evaluation/model_115.txt", "../../../data/hg19/K562/big_model_training/test/genome_chr2.0_binary.txt.gz", "../../../data/hg19/K562/big_model_training/test/genome_chr2.1_binary.txt.gz", "../../../data/hg19/K562/big_model_training/test/genome_chr2.2_binary.txt.gz", "../../../data/hg19/K562/big_model_training/test/genome_chr4.0_binary.txt.gz", "../../../data/hg19/K562/big_model_training/test/genome_chr4.1_binary.txt.gz", "../../../data/hg19/K562/big_model_training/test/genome_chr4.2_binary.txt.gz", "../../../data/hg19/K562/big_model_training/test/genome_chr6.0_binary.txt.gz", "../../../data/hg19/K562/big_model_training/test/genome_chr6.1_binary.txt.gz", "../../../data/hg19/K562/big_model_training/test/genome_chr6.2_binary.txt.gz", "../../../data/hg19/K562/big_model_training/test/genome_chr8.0_binary.txt.gz", "../../../data/hg19/K562/big_model_training/test/genome_chr8.1_binary.txt.gz", "../../../data/hg19/K562/big_model_training/test/genome_chr8.2_binary.txt.gz", "../../../data/hg19/K562/big_model_training/test/genome_chr10.0_binary.txt.gz", "../../../data/hg19/K562/big_model_training/test/genome_chr10.1_binary.txt.gz", "../../../data/hg19/K562/big_model_training/test/genome_chr10.2_binary.txt.gz", "../../../data/hg19/K562/big_model_training/test/genome_chr12.0_binary.txt.gz", "../../../data/hg19/K562/big_model_training/test/genome_chr12.1_binary.txt.gz", "../../../data/hg19/K562/big_model_training/test/genome_chr12.2_binary.txt.gz", "../../../data/hg19/K562/big_model_training/test/genome_chr14.0_binary.txt.gz", "../../../data/hg19/K562/big_model_training/test/genome_chr14.1_binary.txt.gz", "../../../data/hg19/K562/big_model_training/test/genome_chr14.2_binary.txt.gz", "../../../data/hg19/K562/big_model_training/test/genome_chr16.0_binary.txt.gz", "../../../data/hg19/K562/big_model_training/test/genome_chr16.1_binary.txt.gz", "../../../data/hg19/K562/big_model_training/test/genome_chr16.2_binary.txt.gz", "../../../data/hg19/K562/big_model_training/test/genome_chr18.0_binary.txt.gz", "../../../data/hg19/K562/big_model_training/test/genome_chr18.1_binary.txt.gz", "../../../data/hg19/K562/big_model_training/test/genome_chr18.2_binary.txt.gz", "../../../data/hg19/K562/big_model_training/test/genome_chr20.0_binary.txt.gz", "../../../data/hg19/K562/big_model_training/test/genome_chr20.1_binary.txt.gz", "../../../data/hg19/K562/big_model_training/test/genome_chr20.2_binary.txt.gz", "../../../data/hg19/K562/big_model_training/test/genome_chr22.0_binary.txt.gz", "../../../data/hg19/K562/big_model_training/test/genome_chr22.1_binary.txt.gz", "../../../data/hg19/K562/big_model_training/test/genome_chr22.2_binary.txt.gz"], "output": ["../../../model_analysis/K562_hg19/all_mark_model/state_115/half_for_model_evaluation/log_llh/log_llh_chr10.txt"], "wildcards": {"num_state": "115", "chrom": "10"}, "params": {"binarized_for_test_dir": "../../../data/hg19/K562/big_model_training/test"}, "log": [], "threads": 1, "resources": {}, "jobid": 456, "cluster": {}}
 cd /gstore/home/vuh6/source/chromHMM_training/train_big_model && \
/gstore/apps/Anaconda3/5.0.1/bin/python \
-m snakemake ../../../model_analysis/K562_hg19/all_mark_model/state_115/half_for_model_evaluation/log_llh/log_llh_chr10.txt --snakefile /gstore/home/vuh6/source/chromHMM_training/train_big_model/evaluate_multiple_models.snakefile \
--force -j --keep-target-files --keep-remote \
--wait-for-files /gstore/home/vuh6/source/chromHMM_training/train_big_model/.snakemake/tmp.9rcgu2sn ../../../model_analysis/K562_hg19/all_mark_model/state_115/half_for_model_evaluation/model_115.txt ../../../data/hg19/K562/big_model_training/test/genome_chr2.0_binary.txt.gz ../../../data/hg19/K562/big_model_training/test/genome_chr2.1_binary.txt.gz ../../../data/hg19/K562/big_model_training/test/genome_chr2.2_binary.txt.gz ../../../data/hg19/K562/big_model_training/test/genome_chr4.0_binary.txt.gz ../../../data/hg19/K562/big_model_training/test/genome_chr4.1_binary.txt.gz ../../../data/hg19/K562/big_model_training/test/genome_chr4.2_binary.txt.gz ../../../data/hg19/K562/big_model_training/test/genome_chr6.0_binary.txt.gz ../../../data/hg19/K562/big_model_training/test/genome_chr6.1_binary.txt.gz ../../../data/hg19/K562/big_model_training/test/genome_chr6.2_binary.txt.gz ../../../data/hg19/K562/big_model_training/test/genome_chr8.0_binary.txt.gz ../../../data/hg19/K562/big_model_training/test/genome_chr8.1_binary.txt.gz ../../../data/hg19/K562/big_model_training/test/genome_chr8.2_binary.txt.gz ../../../data/hg19/K562/big_model_training/test/genome_chr10.0_binary.txt.gz ../../../data/hg19/K562/big_model_training/test/genome_chr10.1_binary.txt.gz ../../../data/hg19/K562/big_model_training/test/genome_chr10.2_binary.txt.gz ../../../data/hg19/K562/big_model_training/test/genome_chr12.0_binary.txt.gz ../../../data/hg19/K562/big_model_training/test/genome_chr12.1_binary.txt.gz ../../../data/hg19/K562/big_model_training/test/genome_chr12.2_binary.txt.gz ../../../data/hg19/K562/big_model_training/test/genome_chr14.0_binary.txt.gz ../../../data/hg19/K562/big_model_training/test/genome_chr14.1_binary.txt.gz ../../../data/hg19/K562/big_model_training/test/genome_chr14.2_binary.txt.gz ../../../data/hg19/K562/big_model_training/test/genome_chr16.0_binary.txt.gz ../../../data/hg19/K562/big_model_training/test/genome_chr16.1_binary.txt.gz ../../../data/hg19/K562/big_model_training/test/genome_chr16.2_binary.txt.gz ../../../data/hg19/K562/big_model_training/test/genome_chr18.0_binary.txt.gz ../../../data/hg19/K562/big_model_training/test/genome_chr18.1_binary.txt.gz ../../../data/hg19/K562/big_model_training/test/genome_chr18.2_binary.txt.gz ../../../data/hg19/K562/big_model_training/test/genome_chr20.0_binary.txt.gz ../../../data/hg19/K562/big_model_training/test/genome_chr20.1_binary.txt.gz ../../../data/hg19/K562/big_model_training/test/genome_chr20.2_binary.txt.gz ../../../data/hg19/K562/big_model_training/test/genome_chr22.0_binary.txt.gz ../../../data/hg19/K562/big_model_training/test/genome_chr22.1_binary.txt.gz ../../../data/hg19/K562/big_model_training/test/genome_chr22.2_binary.txt.gz --latency-wait 5 \
 --attempt 1 --force-use-threads \
--wrapper-prefix https://github.com/snakemake/snakemake-wrappers/raw/ \
   --allowed-rules calcualte_log_llh_one_chrom --nocolor --notemp --no-hooks --nolock \
--mode 2  && touch /gstore/home/vuh6/source/chromHMM_training/train_big_model/.snakemake/tmp.9rcgu2sn/456.jobfinished || (touch /gstore/home/vuh6/source/chromHMM_training/train_big_model/.snakemake/tmp.9rcgu2sn/456.jobfailed; exit 1)

