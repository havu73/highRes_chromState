fulco19_dir = '../../data/hg19/K562/fulco_2019/'
klann21_dir = '../../data/hg19/K562/klann_2021/'
gasperini19_dir = '../../data/hg19/K562/gasperini_2019'
rule all:
	input:
		os.path.join(fulco19_dir, 'Fulco19_significant_DE_G_ST6a.txt'), # fulco_2019
		expand(os.path.join(klann21_dir, 'for_enrichment', '{fn}.bed'), fn = ['klann21_S1_discovery_K562_sgrna', 'klann21_S2_discovery_K562_bin2', 'klann21_S3_discovery_K562_bin3', 'klann21_S5_discovery_K562_dhs', 'klann21_S6_validation_K562_sgrna', 'klann21_S7_validation_K562_bin2', 'klann21_S8_validation_K562_bin3']), # klann_2021
		expand(os.path.join(gasperini19_dir, 'for_enrichment', '{fn}'), fn = ['gasperini19_altScale_664_enhancerGenePairs_ST2B.bed', 'gasperini19_pilot_145_EnhancerGenePairs_ST1B.bed']) # gasperini_2019

rule prepare_sign_DEG_fulco19: # get significant sites of distal elements and genes from fulco et al 2019
	input:
		os.path.join(fulco19_dir, 'ST6a_expTested_DE-G.csv')
	output:	
		os.path.join(fulco19_dir, 'Fulco19_significant_DE_G_ST6a.txt')
	shell:
		"""
		cat {input[0]} | awk -F',' -v significant='TRUE' 'BEGIN{{OFS="\t"}}{{if (NR>1 && $10==significant) print $1,$2,$3}}' > {output[0]} # there are only 202 out of ~5K such elements
		"""

rule prepare_sign_DE_cell_fitness_klann_2021:
	input:
		expand(os.path.join(klann21_dir, 'raw_data', '{fn}'), fn = ['supplementary_table_1_discovery_screen_k562_sgrna_deseq2_results_hg19.csv.gz', \
			'supplementary_table_2_discovery_screen_k562_bin2_deseq2_results_hg19.csv.gz', \
			'supplementary_table_3_discovery_screen_k562_bin3_deseq2_results_hg19.csv.gz', \
			'supplementary_table_5_DHS_summary_results.txt.gz', \
			'supplementary_table_6_validation_screen_k562_sgrna_deseq2_results_hg19.csv.gz', \
			'supplementary_table_7_validation_screen_k562_bin2_deseq2_results_hg19.csv.gz', \
			'supplementary_table_8_validation_screen_k562_bin3_deseq2_results_hg19.csv.gz']) #, \'supplementary_table_9_validation_screen_k562_dhs_deseq2_results_hg19.csv.gz'])
	output:
		expand(os.path.join(klann21_dir, 'for_enrichment', '{fn}.bed'), fn = ['klann21_S1_discovery_K562_sgrna', 'klann21_S2_discovery_K562_bin2', 'klann21_S3_discovery_K562_bin3', 'klann21_S5_discovery_K562_dhs', 'klann21_S6_validation_K562_sgrna', 'klann21_S7_validation_K562_bin2', 'klann21_S8_validation_K562_bin3'])
	shell:
		"""
		zcat {input[0]} | awk -F',' 'BEGIN{{OFS="\t"}}{{if ($13 <= 0.1) print $1,$2,$3,$5}}' > {output[0]} # chrom, start, end, gRNAid. Filter adj pvalue <= 0.1 # S1
		zcat {input[1]} | awk -F',' 'BEGIN{{OFS="\t"}}{{if ($12 <= 0.1) print $1,$2,$3,$5,$6}}' > {output[1]} # chrom, start, end, binID, DHS identifier. Filter adj pvalue <= 0.1 # S2
		zcat {input[2]} | awk -F',' 'BEGIN{{OFS="\t"}}{{if ($12 <= 0.1) print $1,$2,$3,$5,$6}}' > {output[2]} # chrom, start, end, binID, DHS identifier. Filter adj pvalue <= 0.1 #S3
		zcat {input[3]} | awk -F'\t' 'BEGIN{{OFS="\t"}}{{if ($14 == 1.0) print $1,$2,$3,$4,$19}}' > {output[3]} # chrom, start, end, name of DHS, summary_direction_discovery_K562. IF $14==1, meaning the DHS is significant at FDR 0.1, then select them. $19: summary_direction_discovery_K562: depleted, enriched, non-significant, both #S5
		zcat {input[4]} | awk -F',' 'BEGIN{{OFS="\t"}}{{if ($13 <= 0.1) print $1,$2,$3,$5}}' > {output[4]} # chrom, start, end, gRNAid. Filter adj pvalue <= 0.1 #S6
		zcat {input[5]} | awk -F',' 'BEGIN{{OFS="\t"}}{{if ($12 <= 0.1) print $1,$2,$3,$5,$6}}' > {output[5]} # chrom, start, end, binID, DHS identifier. Filter adj pvalue <= 0.1 #S7
		zcat {input[6]} | awk -F',' 'BEGIN{{OFS="\t"}}{{if ($12 <= 0.1) print $1,$2,$3,$5,$6}}' > {output[6]} # chrom, start, end, binID, DHS identifier. Filter adj pvalue <= 0.1 #S8
		"""


rule prepare_candidate_DEG_gasperini19:
	input:	
		expand(os.path.join(gasperini19_dir, 'raw_data', '{fn}'), fn = ['gasperini_2019_altScale_664_enhancerGenePairs_ST2B.csv', 'gasperini_2019_pilot_145_EnhancerGenePairs_ST1B.csv'])
	output:
		expand(os.path.join(gasperini19_dir, 'for_enrichment', '{fn}'), fn = ['gasperini19_altScale_664_enhancerGenePairs_ST2B.bed', 'gasperini19_pilot_145_EnhancerGenePairs_ST1B.bed'])
	shell:	
		"""
		cat {input[0]} | awk -F',' 'BEGIN{{OFS="\t"}}{{print $9,$10,$11,$1,$2,$3}}' > {output[0]} # chrom, start, end, target_site, ENSG, target_gene_name
		cat {input[1]} | awk -F',' 'BEGIN{{OFS="\t"}}{{print $6,$7,$8,$1,$2,$3}}' > {output[1]} # chrom, start, end, target_side, ENSG, target_gene_name
		"""