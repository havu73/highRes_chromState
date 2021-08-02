
### Classifying regions into promoter and enhancer states for the ROADMAP data ###
### Binarized files were provided by Ha Vu ###

# load data into GRanges

input.dir <- '/gstore/scratch/u/foxn3/roadmap_data_from_ha/';
output.dir <- '/gstore/scratch/u/foxn3/';
files <- list.files(input.dir);
gr.list <- list()
for(i in files) {
  binarized.data <- read.table(paste0(input.dir,i));
  gr <- GRanges(
    seqnames = binarized.data[,1],
    ranges = IRanges(
      start = binarized.data[,2],
      end = binarized.data[,3]
    ),
    strand = rep('*',nrow(binarized.data)),
    target = binarized.data[,4]
  );
  target <- sub('_.*','',i);
  if(!target %in% names(gr.list)) {
    gr.list[[target]] <- gr;
  } else {
    gr.list[[target]] <- c(gr.list[[target]],gr);
  }
}

# check that the ranges are still the same between histone marks
# then estimate promoters and enhancers using DNase, H3K27ac, H3K4me3 in a simple rule

if(all(gr.list[['H3K4me3']] == gr.list[['H3K27ac']]) & all(gr.list[['H3K4me3']] == gr.list[['DNase']])) {
  
  # Combine the calls into the metadata for a single GRanges
  gr <- gr.list[['H3K4me3']];
  colnames(mcols(gr)) <- 'H3K4me3';
  gr$H3K27ac <- gr.list[['H3K27ac']]$target;
  gr$DNase <- gr.list[['DNase']]$target;
  
  # Define promoters and enhancers
  gr$promoter <- as.numeric(gr$DNase & gr$H3K27ac & gr$H3K4me3);
  gr$enhancer <- as.numeric(gr$DNase & gr$H3K27ac & !gr$H3K4me3);
  gr$score <- 2*gr$promoter + gr$enhancer;
  gr$name <- c('inactive','enhancer','promoter')[gr$score+1];
  
  # Check the results
  print(gr[gr$promoter | gr$enhancer]);
  
  # save to file
  write.table(
    as.data.frame(gr)[,-c(4,5)],
    file=paste0(paste0(output.dir,Sys.Date(),'_simple_promoter_enhancer_classification.txt')),
    quote=FALSE,
    col.names=TRUE,
    row.names=FALSE,
    sep='\t'
  );
  
  library(rtracklayer);
  # rtracklayer seems write the bigbed file with the start one less than here so add one to counter that
  start(gr) <- start(gr) +1;
  # save as a bed file
  export.bed(
    gr[,c(7,6)],
    paste0(output.dir,Sys.Date(),'_simple_promoter_enhancer_classification.bed')
  );
}
