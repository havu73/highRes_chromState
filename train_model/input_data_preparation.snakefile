import os 
import pandas as pd 
import numpy as np 
raw_metadata_fn = '../../data/hg19/roadmap_metadata_consolidated_epig_summary.csv'
processed_raw_metadata_fn = '../../data/hg19/processed_raw_metadata_roadmap_epig_summary.csv'
roadmap_S25_segment_folder = '../../data/hg19/roadmap/S25_segment/'
def get_ct_list(processed_raw_metadata_fn):
	df = pd.read_csv(processed_raw_metadata_fn, header = 0, index_col = None, sep = '\t')
	return list(df.EID)

rule all:
	input:
		processed_raw_metadata_fn,
		expand(os.path.join(roadmap_S25_segment_folder, '{ct}', '{ct}_25_imputed12marks_segments.bed.gz'), ct = get_ct_list(processed_raw_metadata_fn)),

rule process_raw_metadata: # get clean metadata
	input:
		raw_metadata_fn,
	output:
		processed_raw_metadata_fn
	shell:
		"""
		cat {input} | awk -F',' '{{print $2,$4,$5,$6,$17}}' | awk  'BEGIN{{OFS="\t"}}{{if (NR == 1 || NR > 6) print $1,$2,$3,$4,$5}}' > {output}
		"""

rule download_roadmap_S25_segment:
	input:
		processed_raw_metadata_fn
	output:
		os.path.join(roadmap_S25_segment_folder, '{ct}', '{ct}_25_imputed12marks_segments.bed.gz')
	params:
		ct_out_folder=os.path.join(roadmap_S25_segment_folder, '{ct}')
	shell:
		"""
		link_prefix='https://egg2.wustl.edu/roadmap/data/byFileType/chromhmmSegmentations/ChmmModels/imputed12marks/jointModel/final/'
		link="${{link_prefix}}/{wildcards.ct}_25_imputed12marks_segments.bed.gz"
		echo $link
		wget $link -P {params.ct_out_folder}
		"""


