library(ggplot2)
library(dplyr)
library(tidyr)
library(ggpubr)
##########################################################################
############ ALL THE FUNCTIONS ###########################################

get_enrichment_context_name <- function(roc_fn, tail_to_cut) {
	# tail_to_cut should be : '_prec_recall.csv' or '_roc.csv'
	roc_fn <- unlist(strsplit(roc_fn, "/")) %>% last() # last file name in a string of full fill path, each component separated by '/'
	gc_name <- unlist(strsplit(roc_fn, tail_to_cut)) %>% first()
	return(gc_name) # get rid of the '.bed.gz_roc.csv' tail
}

get_roc_df <- function(roc_fn){
	df <- read.table(roc_fn, sep = ',', header = FALSE, fill = TRUE)
	rownames(df) <- df$V1
	df <- df %>% select(c(-'V1', -'V2')) # first column is already assigned the rownames of the df
	df <- as.data.frame(t(df))
	return(df)
}

draw_precall_three_models <- function(precall_fn){
	df <- get_roc_df (precall_fn)
	df <- df %>% select(c('M3_S8_recall', 'M3_S8_precision', 'M13_S25_recall', 'M13_S25_precision', 'roadmap_S25_recall', 'roadmap_S25_precision', 'simpleRules_S3_precision', 'simpleRules_S3_recall'))
	context_name <- get_enrichment_context_name(precall_fn, '_prec_recall.csv')
	p <- ggplot() +
  geom_point(data = df, aes(x = M3_S8_recall, y = M3_S8_precision, color = 'M3_S8', alpha = 0.5, size = 5)) +
  geom_line(data = df, aes(x = M3_S8_recall, y = M3_S8_precision, color = 'M3_S8', alpha = 0.5)) +
  theme_bw()+ 
  geom_point(data = df, aes(x = M13_S25_recall, y = M13_S25_precision, color = 'M13_S25', alpha = 0.5, size = 5)) + 
  geom_line(data = df, aes(x = M13_S25_recall, y = M13_S25_precision, color = 'M3_S25', alpha = 0.5)) +
  geom_point(data = df, aes(x = roadmap_S25_recall, y = roadmap_S25_precision, color = 'roadmap_S25', alpha = 0.5, size = 5)) + 
  geom_line(data = df, aes(x = roadmap_S25_recall, y = roadmap_S25_precision, color = 'roadmap_S25', alpha = 0.5)) + 
  geom_point(data = df, aes(x = simpleRules_S3_recall, y = simpleRules_S3_precision, color = 'simpleRules_S3', alpha = 0.5, size = 5)) +
  geom_line(data = df, aes(x = simpleRules_S3_recall, y = simpleRules_S3_precision, color = 'simpleRules_S3', alpha = 0.5)) +
  scale_color_manual(values = c('M3_S8' = 'blue', 'M13_S25' = 'red', 'roadmap_S25' = 'green', 'simpleRules_S3' = 'black')) +
  theme(legend.position = 'bottom') +
  ggtitle(context_name)
  # ggsave(save_fn) 
  return (p)
}

draw_roc_three_models <- function(roc_fn){
	df <- get_roc_df(roc_fn)
	df <- df %>% select(c('M3_S8_false_pos', 'M3_S8_true_pos', 'M13_S25_false_pos', 'M13_S25_true_pos', 'roadmap_S25_false_pos', 'roadmap_S25_true_pos', 'simpleRules_S3_false_pos', 'simpleRules_S3_true_pos'))
	context_name <- get_enrichment_context_name(roc_fn, '_roc.csv')
	p <- ggplot() +
  geom_point(data = df, aes(x = M3_S8_false_pos, y = M3_S8_true_pos, color = 'M3_S8', alpha = 0.5, size = 5)) +
  theme_bw()+ 
  geom_point(data = df, aes(x = M13_S25_false_pos, y = M13_S25_true_pos, color = 'M13_S25', alpha = 0.5, size = 5)) + 
  geom_point(data = df, aes(x = roadmap_S25_false_pos, y = roadmap_S25_true_pos, color = 'roadmap_S25', alpha = 0.5, size = 5)) + 
  geom_point(data = df, aes(x = simpleRules_S3_false_pos, y = simpleRules_S3_true_pos, color = 'simpleRules_S3', alpha = 0.5, size = 5)) +
  scale_color_manual(values = c('M3_S8' = 'blue', 'M13_S25' = 'red', 'roadmap_S25' = 'green', 'simpleRules_S3' = 'black')) +
  theme(legend.position = 'bottom') +
  ggtitle(context_name)
  # ggsave(save_fn) 
  return (p)
}

arrange_10_plots <- function(plot_list, save_fn){
	ggarrange(plot_list[[1]], plot_list[[2]], plot_list[[3]], plot_list[[4]], plot_list[[5]], plot_list[[6]], plot_list[[7]], plot_list[[8]], plot_list[[9]], plot_list[[10]], ncol = 2, nrow = 5)
	ggsave(save_fn, width = 15, height = 30)
}

get_all_precall_plots <- function(compare_folder, save_fn){
	precall_fn_list <- Sys.glob(paste0(compare_folder, '/*_prec_recall.csv'))
	precall_plot_list <- lapply(precall_fn_list, draw_precall_three_models)
	arrange_10_plots(precall_plot_list, save_fn)
}

get_all_roc_plots <- function(compare_folder, save_fn){
	roc_fn_list <- Sys.glob(paste0(compare_folder, '/*_roc.csv'))
	roc_plot_list <- lapply(roc_fn_list, draw_roc_three_models)
	arrange_10_plots(roc_plot_list, save_fn)
}
#!/usr/bin/env Rscript
################ GETTING THE COMMAND LINE ARGUMENTS #####################
args = commandArgs(trailingOnly=TRUE)
print(args)
print(length(args))
if (length(args) != 2)
{	
	stop("wrong command line argument format", call.=FALSE)
}
compare_folder <- args[1] # roc_fn is the output of /u/home/h/havu73/project-ernst/source/model_evaluation/create_roc_curve_overlap_enrichment.py
output_folder <- args[2] # the output of /u/home/h/havu73/project-ernst/source/model_evaluation/create_roc_curve_overlap_enrichment.py
dir.create(output_folder, recursive = TRUE)

################ CALLING FUNCTIONS FOR THIS PROGRAM ######################
precall_save_fn <- file.path(output_folder, 'precall_all_contexts.png')
roc_save_fn <- file.path(output_folder, 'roc_all_contexts.png')
get_all_precall_plots(compare_folder, precall_save_fn)
get_all_roc_plots(compare_folder, roc_save_fn)
##########################################################################