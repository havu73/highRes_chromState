fulco19_dir = '../../data/hg19/K562/fulco_2019/'
klann21_dir = '../../data/hg19/K562/klann_2021/'
gasperini19_dir = '../../data/hg19/K562/gasperini_2019'
COORD_CRISPR_MPRA_DIR = '../../data/hg19/K562/COORD_CRISPR_MPRA_DATA/for_enrichment'

rule all:
	input:
		os.path.join(COORD_CRISPR_MPRA_DIR, 'match_foreground_background')

rule prepare_sign_DEG_fulco19: # get significant sites of distal elements and genes from fulco et al 2019
	input:
		os.path.join(fulco19_dir, 'fulco19_ST6a_expTested_DE-G.csv'),
		os.path.join(fulco19_dir, 'fulco19_ST3a_CRISPRi_flowFish_EGpairs.csv')
	output:	
		os.path.join(fulco19_dir, 'for_enrichment','significant', 'Fulco19_significant_DE_G_ST6a.txt'),
		os.path.join(fulco19_dir, 'for_enrichment', 'significant', 'Fulco19_significant_CRIPSRi_flowFish_EGpairs.txt'),
		os.path.join(fulco19_dir, 'for_enrichment', 'background', 'Fulco19_DE_G_ST6a.txt'),
		os.path.join(fulco19_dir, 'for_enrichment', 'background', 'Fulco19_CRIPSRi_flowFish_EGpairs.txt'),
		os.path.join(fulco19_dir, 'for_enrichment', 'match_foreground_background.txt')
	shell:
		"""
		cat {input[0]} | awk -F',' -v significant='TRUE' 'BEGIN{{OFS="\t"}}{{if (NR>1 && $10==significant) print $1,$2,$3}}' > {output[0]} # there are only 202 out of ~5K such elements
		cat {input[1]} | awk -F',' -v significant='TRUE' 'BEGIN{{OFS="\t"}}{{if (NR>1 && $8==significant) print $1,$2,$3}}' > {output[1]} # filter the significant result from their CRIPSR-FlowFish experiments
		cat {input[0]} | awk -F',' 'BEGIN{{OFS="\t"}}{{if (NF>1) print $1,$2,$3}}' > {output[2]} # list of the places that they tested
		cat {input[1]} | awk -F',' 'BEGIN{{OFS="\t"}}{{if (NF>1) print $1,$2,$3}}' > {output[3]} # list of the places that they tested in the CRISPR-FlowFish expeirments
		echo -e "Fulco19_significant_DE_G_ST6a.txt\tFulco19_DE_G_ST6a.txt" > {output[4]} 
		echo -e "Fulco19_significant_CRIPSRi_flowFish_EGpairs.txt\tFulco19_CRIPSRi_flowFish_EGpairs.txt" >> {output[4]}
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
		expand(os.path.join(klann21_dir, 'for_enrichment', 'significant', '{fn}.bed'), fn = ['klann21_S1_discovery_K562_sgrna', 'klann21_S2_discovery_K562_bin2', 'klann21_S3_discovery_K562_bin3', 'klann21_S5_discovery_K562_dhs', 'klann21_S6_validation_K562_sgrna', 'klann21_S7_validation_K562_bin2', 'klann21_S8_validation_K562_bin3'])
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

rule prepare_background_DE_cell_fitness_klann_2021:
	input:
		expand(os.path.join(klann21_dir, 'raw_data', '{fn}'), fn = ['supplementary_table_1_discovery_screen_k562_sgrna_deseq2_results_hg19.csv.gz', \
			'supplementary_table_2_discovery_screen_k562_bin2_deseq2_results_hg19.csv.gz', \
			'supplementary_table_3_discovery_screen_k562_bin3_deseq2_results_hg19.csv.gz', \
			'supplementary_table_5_DHS_summary_results.txt.gz', \
			'supplementary_table_6_validation_screen_k562_sgrna_deseq2_results_hg19.csv.gz', \
			'supplementary_table_7_validation_screen_k562_bin2_deseq2_results_hg19.csv.gz', \
			'supplementary_table_8_validation_screen_k562_bin3_deseq2_results_hg19.csv.gz']) #, \'supplementary_table_9_validation_screen_k562_dhs_deseq2_results_hg19.csv.gz'])
	output:
		expand(os.path.join(klann21_dir, 'for_enrichment', 'background', '{fn}.bed'), fn = ['klann21_S1_discovery_K562_sgrna', 'klann21_S2_discovery_K562_bin2', 'klann21_S3_discovery_K562_bin3', 'klann21_S5_discovery_K562_dhs', 'klann21_S6_validation_K562_sgrna', 'klann21_S7_validation_K562_bin2', 'klann21_S8_validation_K562_bin3'])
	shell:
		"""
		zcat {input[0]} | awk -F',' 'BEGIN{{OFS="\t"}}{{print $1,$2,$3,$5}}' > {output[0]} # chrom, start, end, gRNAid.  # S1
		zcat {input[1]} | awk -F',' 'BEGIN{{OFS="\t"}}{{ print $1,$2,$3,$5,$6}}' > {output[1]} # chrom, start, end, binID, DHS identifier. # S2
		zcat {input[2]} | awk -F',' 'BEGIN{{OFS="\t"}}{{print $1,$2,$3,$5,$6}}' > {output[2]} # chrom, start, end, binID, DHS identifier. #S3
		zcat {input[3]} | awk -F'\t' 'BEGIN{{OFS="\t"}}{{print $1,$2,$3,$4,$19}}' > {output[3]} # chrom, start, end, name of DHS, summary_direction_discovery_K562. $19: summary_direction_discovery_K562: depleted, enriched, non-significant, both #S5
		zcat {input[4]} | awk -F',' 'BEGIN{{OFS="\t"}}{{print $1,$2,$3,$5}}' > {output[4]} # chrom, start, end, gRNAid.  #S6
		zcat {input[5]} | awk -F',' 'BEGIN{{OFS="\t"}}{{ print $1,$2,$3,$5,$6}}' > {output[5]} # chrom, start, end, binID, DHS identifier. #S7
		zcat {input[6]} | awk -F',' 'BEGIN{{OFS="\t"}}{{print $1,$2,$3,$5,$6}}' > {output[6]} # chrom, start, end, binID, DHS identifier. #S8
		"""

rule prepare_candidate_DEG_gasperini19:
	input:	
		expand(os.path.join(gasperini19_dir, 'raw_data', '{fn}'), fn = ['gasperini_2019_altScale_664_enhancerGenePairs_ST2B.csv', 'gasperini_2019_pilot_145_EnhancerGenePairs_ST1B.csv'])
	output:
		expand(os.path.join(gasperini19_dir, 'for_enrichment', 'significant','{fn}'), fn = ['gasperini19_altScale_664_enhancerGenePairs_ST2B.bed', 'gasperini19_pilot_145_EnhancerGenePairs_ST1B.bed'])
	shell:	
		"""
		cat {input[0]} | awk -F',' 'BEGIN{{OFS="\t"}}{{print $9,$10,$11,$1,$2,$3}}' > {output[0]} # chrom, start, end, target_site, ENSG, target_gene_name
		cat {input[1]} | awk -F',' 'BEGIN{{OFS="\t"}}{{print $6,$7,$8,$1,$2,$3}}' > {output[1]} # chrom, start, end, target_side, ENSG, target_gene_name
		"""

rule prepare_background_DEG_gasperini19:
	input:
		expand(os.path.join(gasperini19_dir, 'raw_data', '{fn}'), fn = ['gasperini_2019_AtScale_gRNALibrary.csv', 'gasperini_2019_Pilot_gRNALibrary.csv'])
	output:
		expand(os.path.join(gasperini19_dir, 'for_enrichment', 'background','{fn}'), fn = ['gasperini_2019_AtScale_gRNALibrary.bed', 'gasperini_2019_Pilot_gRNALibrary.bed'])
	shell:
		"""
		cat {input[0]} | awk -F',' 'BEGIN{{OFS="\t"}}{{if (NR>1 && $3!="") print $3,$4,$5}}' > {output[0]}
		cat {input[1]} | awk -F',' 'BEGIN{{OFS="\t"}}{{if (NR>1 && $3!="") print $3,$4,$5}}' > {output[1]}
		"""

rule prepare_match_column_name_file:
	input:
		expand(os.path.join(fulco19_dir, 'for_enrichment','significant', '{fn}.txt'), fn = ['Fulco19_significant_DE_G_ST6a', 'Fulco19_significant_CRIPSRi_flowFish_EGpairs']), # fulco_2019, significant distal elements
		expand(os.path.join(fulco19_dir, 'for_enrichment', 'background', '{fn}.txt'), fn = ['Fulco19_DE_G_ST6a', 'Fulco19_CRIPSRi_flowFish_EGpairs']), # fulco_2019 all the elements that the chose to experimentally test
		expand(os.path.join(klann21_dir, 'for_enrichment', 'significant','{fn}.bed'), fn = ['klann21_S1_discovery_K562_sgrna', 'klann21_S2_discovery_K562_bin2', 'klann21_S3_discovery_K562_bin3', 'klann21_S5_discovery_K562_dhs', 'klann21_S6_validation_K562_sgrna', 'klann21_S7_validation_K562_bin2', 'klann21_S8_validation_K562_bin3']), # klann_2021
		expand(os.path.join(klann21_dir, 'for_enrichment', 'background','{fn}.bed'), fn = ['klann21_S1_discovery_K562_sgrna', 'klann21_S2_discovery_K562_bin2', 'klann21_S3_discovery_K562_bin3', 'klann21_S5_discovery_K562_dhs', 'klann21_S6_validation_K562_sgrna', 'klann21_S7_validation_K562_bin2', 'klann21_S8_validation_K562_bin3']), # klann_2021
		expand(os.path.join(gasperini19_dir, 'for_enrichment', 'significant','{fn}'), fn = ['gasperini19_altScale_664_enhancerGenePairs_ST2B.bed', 'gasperini19_pilot_145_EnhancerGenePairs_ST1B.bed']), # gasperini_2019
		expand(os.path.join(gasperini19_dir, 'for_enrichment', 'background', '{fn}'), fn = ['gasperini_2019_AtScale_gRNALibrary.bed', 'gasperini_2019_Pilot_gRNALibrary.bed'])
	output:
		os.path.join(COORD_CRISPR_MPRA_DIR, 'match_foreground_background')
	shell:
		"""	
		echo -e "Fulco19_significant_DE_G_ST6a\tFulco19_DE_G_ST6a" > {output[0]} 
		echo -e "Fulco19_significant_CRIPSRi_flowFish_EGpairs\tFulco19_CRIPSRi_flowFish_EGpairs" >> {output[0]}
		echo -e "klann21_S1_discovery_K562_sgrna\tklann21_S1_discovery_K562_sgrna" >> {output[0]}
		echo -e "klann21_S2_discovery_K562_bin2\tklann21_S2_discovery_K562_bin2" >> {output[0]}
		echo -e "klann21_S3_discovery_K562_bin3\tklann21_S3_discovery_K562_bin3" >> {output[0]}
		echo -e "klann21_S5_discovery_K562_dhs\tklann21_S5_discovery_K562_dhs" >> {output[0]}
		echo -e "klann21_S6_validation_K562_sgrna\tklann21_S6_validation_K562_sgrna" >> {output[0]}
		echo -e "klann21_S7_validation_K562_bin2\tklann21_S7_validation_K562_bin2" >> {output[0]}
		echo -e "klann21_S8_validation_K562_bin3\tklann21_S8_validation_K562_bin3" >> {output[0]}
		echo -e "gasperini19_altScale_664_enhancerGenePairs_ST2B\tgasperini_2019_AtScale_gRNALibrary" >> {output[0]}
		echo -e "gasperini19_pilot_145_EnhancerGenePairs_ST1B\tgasperini_2019_Pilot_gRNALibrary" >> {output[0]}
		"""