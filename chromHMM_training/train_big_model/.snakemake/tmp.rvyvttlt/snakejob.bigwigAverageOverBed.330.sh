#!/bin/sh
# properties = {"type": "single", "rule": "bigwigAverageOverBed", "local": false, "input": ["../../../data/hg19/K562/basic_bed/chr22.bed.gz", "../../../data/hg19/K562/roadmap_pval_signals/E123-H3K9me1.pval.signal.bigwig"], "output": ["../../../data/hg19/K562/roadmap_pval_signals/H3K9me1_chr22.tab"], "wildcards": {"mark": "H3K9me1", "chrom": "22"}, "params": {}, "log": [], "threads": 1, "resources": {}, "jobid": 330, "cluster": {}}
 cd /gstore/home/vuh6/source/chromHMM_training/train_big_model && \
/gstore/apps/Anaconda3/5.0.1/bin/python \
-m snakemake ../../../data/hg19/K562/roadmap_pval_signals/H3K9me1_chr22.tab --snakefile /gstore/home/vuh6/source/chromHMM_training/train_big_model/learn_model.snakefile \
--force -j --keep-target-files --keep-remote \
--wait-for-files /gstore/home/vuh6/source/chromHMM_training/train_big_model/.snakemake/tmp.rvyvttlt ../../../data/hg19/K562/basic_bed/chr22.bed.gz ../../../data/hg19/K562/roadmap_pval_signals/E123-H3K9me1.pval.signal.bigwig --latency-wait 5 \
 --attempt 1 --force-use-threads \
--wrapper-prefix https://github.com/snakemake/snakemake-wrappers/raw/ \
   --allowed-rules bigwigAverageOverBed --nocolor --notemp --no-hooks --nolock \
--mode 2  && touch /gstore/home/vuh6/source/chromHMM_training/train_big_model/.snakemake/tmp.rvyvttlt/330.jobfinished || (touch /gstore/home/vuh6/source/chromHMM_training/train_big_model/.snakemake/tmp.rvyvttlt/330.jobfailed; exit 1)

