#!/bin/sh
# properties = {"type": "single", "rule": "binarize_multiple_bigwigAverageOverBed_output", "local": false, "input": ["../../../data/hg19/K562/roadmap_pval_signals/DNase_chr5.tab", "../../../data/hg19/K562/roadmap_pval_signals/H3K27ac_chr5.tab", "../../../data/hg19/K562/roadmap_pval_signals/H3K4me3_chr5.tab"], "output": ["../../../data/hg19/K562/three_mark_model_training/genome_chr5.0_binary.txt", "../../../data/hg19/K562/three_mark_model_training/genome_chr5.1_binary.txt", "../../../data/hg19/K562/three_mark_model_training/genome_chr5.2_binary.txt"], "wildcards": {"chrom": "5"}, "params": {}, "log": [], "threads": 1, "resources": {}, "jobid": 28, "cluster": {}}
 cd /gstore/home/vuh6/source/chromHMM_training/train_big_model && \
/gstore/apps/Anaconda3/5.0.1/bin/python \
-m snakemake ../../../data/hg19/K562/three_mark_model_training/genome_chr5.0_binary.txt --snakefile /gstore/home/vuh6/source/chromHMM_training/train_big_model/learn_three_mark_model.snakefile \
--force -j --keep-target-files --keep-remote \
--wait-for-files /gstore/home/vuh6/source/chromHMM_training/train_big_model/.snakemake/tmp.cfjlz8tz ../../../data/hg19/K562/roadmap_pval_signals/DNase_chr5.tab ../../../data/hg19/K562/roadmap_pval_signals/H3K27ac_chr5.tab ../../../data/hg19/K562/roadmap_pval_signals/H3K4me3_chr5.tab --latency-wait 5 \
 --attempt 1 --force-use-threads \
--wrapper-prefix https://github.com/snakemake/snakemake-wrappers/raw/ \
   --allowed-rules binarize_multiple_bigwigAverageOverBed_output --nocolor --notemp --no-hooks --nolock \
--mode 2  && touch /gstore/home/vuh6/source/chromHMM_training/train_big_model/.snakemake/tmp.cfjlz8tz/28.jobfinished || (touch /gstore/home/vuh6/source/chromHMM_training/train_big_model/.snakemake/tmp.cfjlz8tz/28.jobfailed; exit 1)

