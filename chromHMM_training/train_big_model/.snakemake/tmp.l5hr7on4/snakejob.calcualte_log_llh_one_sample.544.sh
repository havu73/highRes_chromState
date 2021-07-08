#!/bin/sh
# properties = {"type": "single", "rule": "calcualte_log_llh_one_sample", "local": false, "input": ["../../../model_analysis/K562_hg19/all_mark_model/state_20/model_20.txt", "../../../data/hg19/K562/big_model_training/genome_chr1.0_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr1.1_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr1.2_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr2.0_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr2.1_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr2.2_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr3.0_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr3.1_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr3.2_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr4.0_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr4.1_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr4.2_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr5.0_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr5.1_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr5.2_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr6.0_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr6.1_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr6.2_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr7.0_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr7.1_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr7.2_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr8.0_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr8.1_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr8.2_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr9.0_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr9.1_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr9.2_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr10.0_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr10.1_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr10.2_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr11.0_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr11.1_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr11.2_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr12.0_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr12.1_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr12.2_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr13.0_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr13.1_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr13.2_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr14.0_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr14.1_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr14.2_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr15.0_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr15.1_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr15.2_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr16.0_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr16.1_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr16.2_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr17.0_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr17.1_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr17.2_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr18.0_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr18.1_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr18.2_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr19.0_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr19.1_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr19.2_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr20.0_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr20.1_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr20.2_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr21.0_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr21.1_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr21.2_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr22.0_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr22.1_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chr22.2_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chrX.0_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chrX.1_binary.txt.gz", "../../../data/hg19/K562/big_model_training/genome_chrX.2_binary.txt.gz", "../../../model_analysis/K562_hg19/all_mark_model/log_llh_select_regions/seed_257.txt"], "output": ["../../../model_analysis/K562_hg19/all_mark_model/state_20/log_llh/log_llh_seed_257.txt"], "wildcards": {"num_state": "20", "seed": "257"}, "params": {"sample_region_fn": "../../../model_analysis/K562_hg19/all_mark_model/log_llh_select_regions/seed_257.txt"}, "log": [], "threads": 1, "resources": {}, "jobid": 544, "cluster": {}}
 cd /gstore/home/vuh6/source/chromHMM_training/train_big_model && \
/gstore/apps/Anaconda3/5.0.1/bin/python \
-m snakemake ../../../model_analysis/K562_hg19/all_mark_model/state_20/log_llh/log_llh_seed_257.txt --snakefile /gstore/home/vuh6/source/chromHMM_training/train_big_model/evaluate_multiple_models.snakefile \
--force -j --keep-target-files --keep-remote \
--wait-for-files /gstore/home/vuh6/source/chromHMM_training/train_big_model/.snakemake/tmp.l5hr7on4 ../../../model_analysis/K562_hg19/all_mark_model/state_20/model_20.txt ../../../data/hg19/K562/big_model_training/genome_chr1.0_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr1.1_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr1.2_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr2.0_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr2.1_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr2.2_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr3.0_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr3.1_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr3.2_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr4.0_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr4.1_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr4.2_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr5.0_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr5.1_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr5.2_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr6.0_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr6.1_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr6.2_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr7.0_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr7.1_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr7.2_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr8.0_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr8.1_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr8.2_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr9.0_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr9.1_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr9.2_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr10.0_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr10.1_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr10.2_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr11.0_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr11.1_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr11.2_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr12.0_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr12.1_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr12.2_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr13.0_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr13.1_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr13.2_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr14.0_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr14.1_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr14.2_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr15.0_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr15.1_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr15.2_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr16.0_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr16.1_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr16.2_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr17.0_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr17.1_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr17.2_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr18.0_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr18.1_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr18.2_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr19.0_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr19.1_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr19.2_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr20.0_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr20.1_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr20.2_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr21.0_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr21.1_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr21.2_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr22.0_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr22.1_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chr22.2_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chrX.0_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chrX.1_binary.txt.gz ../../../data/hg19/K562/big_model_training/genome_chrX.2_binary.txt.gz ../../../model_analysis/K562_hg19/all_mark_model/log_llh_select_regions/seed_257.txt --latency-wait 5 \
 --attempt 1 --force-use-threads \
--wrapper-prefix https://github.com/snakemake/snakemake-wrappers/raw/ \
   --allowed-rules calcualte_log_llh_one_sample --nocolor --notemp --no-hooks --nolock \
--mode 2  && touch /gstore/home/vuh6/source/chromHMM_training/train_big_model/.snakemake/tmp.l5hr7on4/544.jobfinished || (touch /gstore/home/vuh6/source/chromHMM_training/train_big_model/.snakemake/tmp.l5hr7on4/544.jobfailed; exit 1)

