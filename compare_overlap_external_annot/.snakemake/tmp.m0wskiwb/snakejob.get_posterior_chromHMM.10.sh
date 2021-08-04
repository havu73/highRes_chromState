#!/bin/sh
# properties = {"type": "single", "rule": "get_posterior_chromHMM", "local": false, "input": ["../../model_analysis/K562_hg19/all_mark_model/state_11/full_model/model_11.txt"], "output": ["../../model_analysis/K562_hg19/all_mark_model/state_11/full_model/POSTERIOR/genome_11_chr1_posterior.txt.gz", "../../model_analysis/K562_hg19/all_mark_model/state_11/full_model/POSTERIOR/genome_11_chrX_posterior.txt.gz"], "wildcards": {"num_mark_model": "all_mark_model", "num_state": "11"}, "params": {"out_dir": "../../model_analysis/K562_hg19/all_mark_model/state_11/full_model"}, "log": [], "threads": 1, "resources": {}, "jobid": 10, "cluster": {}}
 cd /gstore/home/vuh6/source/compare_overlap_external_annot && \
/gstore/apps/Anaconda3/5.0.1/bin/python \
-m snakemake ../../model_analysis/K562_hg19/all_mark_model/state_11/full_model/POSTERIOR/genome_11_chr1_posterior.txt.gz --snakefile /gstore/home/vuh6/source/compare_overlap_external_annot/compare_models_recover_GC.snakefile \
--force -j --keep-target-files --keep-remote \
--wait-for-files /gstore/home/vuh6/source/compare_overlap_external_annot/.snakemake/tmp.m0wskiwb ../../model_analysis/K562_hg19/all_mark_model/state_11/full_model/model_11.txt --latency-wait 5 \
 --attempt 1 --force-use-threads \
--wrapper-prefix https://github.com/snakemake/snakemake-wrappers/raw/ \
   --allowed-rules get_posterior_chromHMM --nocolor --notemp --no-hooks --nolock \
--mode 2  && touch /gstore/home/vuh6/source/compare_overlap_external_annot/.snakemake/tmp.m0wskiwb/10.jobfinished || (touch /gstore/home/vuh6/source/compare_overlap_external_annot/.snakemake/tmp.m0wskiwb/10.jobfailed; exit 1)

