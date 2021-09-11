This directory contains files for Sharpr-MPRA regulatory activity scores from the scale-up K562 experiments.

The files are:
basepredictions_K562_ScaleUpDesign1and2_combinedP.txt
basepredictions_K562_ScaleUpDesign1and2_minP.txt
basepredictions_K562_ScaleUpDesign1and2_SV40P.txt

The file name indicates if the scores are based on minimal promoter (minP) data only,
SV40 promoter (SV40P) only, or combining the minP and SV40P data (combinedP).

Files contain the concatenation of scores from the two designs.
Each row corresponds to one 295-bp region profiled.
The first column contains the ID for the region. The remaining
columns contain the score for each base in the 295-bp region in order.

IDs are of the form CELLTYPE_STATE_REGIONIDINSTATE_CHR_CENTER where
-CELLTYPE and STATE are the cell type (H1hesc, Hepg2, K562, or Huvec) and chromatin state ID (1..25) respectively 
based on which the regulatory region was selected
-REGIONIDINSTATE is an integer ID for the region which is 
unique among region selections from the same CELLTYPE and STATE combination 
-CHR and CENTER are the chromosome and coordinate of the center base (hg19) respectively of the regulatory region
