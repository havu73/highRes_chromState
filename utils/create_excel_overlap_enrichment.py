import seaborn as sns
import pandas as pd 
import numpy as np 
import os
import sys
import helper

print ("This file will take as input the chromHMM OverlapEnrichment output file, and create a replicate of such file in Excel format. The slick  thing about this is that the excel file will be colored based on the levels of enrichment in each column. The color betweeen columns are not relevant. However, the colors of cells within one column shows the enrichment levels comparable among states.")
print ("")
print ("")


def get_enrichment_df(enrichment_fn):
    enrichment_df = pd.read_csv(enrichment_fn, sep = "\t", header = 0)
    enrichment_df = enrichment_df.rename(columns = {u'state (Emission order)': 'state', u'Genome %' : 'percent_in_genome'})
    (num_state, num_enr_cont) = (enrichment_df.shape[0] - 1, enrichment_df.shape[1] - 1)  
    percent_genome_of_cont = enrichment_df.iloc[num_state, 2:]
    enrichment_df = enrichment_df.loc[:(num_state - 1)] # rid of the last row because that is the row about percentage each genomic context occupies
    enrichment_df['state'] = (enrichment_df['state']).astype(str).astype(int)    
    return enrichment_df, num_state, num_enr_cont, percent_genome_of_cont

def get_rid_of_stupid_file_tail(context_name):
    # first, get rid of stupid_file_tail
    if context_name.endswith('.bed.gz'):
        context_name = context_name[:-7]
    else:
        context_name = context_name
    # fix if the context name is from variant prioritization analysis
    if 'non_coding' in context_name or 'whole_genome' in context_name: # that means context_name can be FATHMM_non_coding_top_0.01, so we want FATHMM instead
        enr_cont_list = context_name.split('_')
        enr_cont_list = enr_cont_list[:-4] # get rid of the last 4 items because 
        context_name = '_'.join(enr_cont_list)
    return(context_name)

def add_comments_on_states(enrichment_df, num_state, num_enr_cont, context_prefix):
    enr_cont_list = enrichment_df.columns[2:]
    NUM_TOP_STATE_TO_COMMENT = 1 #7
    output_comment = pd.Series([''] * enrichment_df.shape[0]) # we choose the number of rows in this enrichment_df as the size of output_comment, instead of num_state because there's one last line that's the base line and we want that line to be an empty string too. 
    # output_comment will be later used as a column in the enrichment_df to details what enrichment context each state is associated with
    for enr_cont in enr_cont_list:
        top_states = enrichment_df[enr_cont][:num_state].nlargest(NUM_TOP_STATE_TO_COMMENT)
        top_states = np.around(top_states, decimals = 1) # round the values of enrichment 
        comment_this_cont = list(map(lambda x: 'rank' + str(x + 1) + '_'+ context_prefix + '_' + enr_cont + ':' + str(top_states[top_states.index[x]])+ "; ", range(NUM_TOP_STATE_TO_COMMENT))) # the string that specify the content of the comment: rank<rank>_<enrichment_context>: <enrichment_values>
        for i in range(NUM_TOP_STATE_TO_COMMENT):
            top_state_index = top_states.index[i] # the index of the state that has rank i 
            output_comment[top_state_index] = output_comment[top_state_index] + comment_this_cont[i] # concatenate comment string 
    return output_comment

def color_state_annotation(row_data, index_to_color):
    results = [""] * len(row_data) # current paint all the cells in the rows with nothing, no format yet---> all white for now
    state_annot_color = row_data['color']
    if not pd.isna(row_data['color']):
        results[index_to_color] = 'background-color: %s' % state_annot_color # the third cell from the left is the state annotation cells
    return results

def combine_enrichment_df_with_state_annot_df(enrichment_df, enr_cont_list):
    state_annot_df = get_state_annot_df()
    enrichment_df['state'] = (enrichment_df['state']).astype(str).astype(int)
    enrichment_df = enrichment_df.merge(state_annot_df, how = 'left', left_on = 'state', right_on = 'state', suffixes = ('_x', '_y'))
    enrichment_df = enrichment_df.sort_values(by = 'state_order_by_group')
    enrichment_df = enrichment_df.reset_index()  
    enrichment_df = enrichment_df.drop(columns = ['index'])
    enrichment_df['state'] = enrichment_df['state'].astype(str).astype(int)
    enrichment_df['percent_in_genome'] = enrichment_df['percent_in_genome'].astype(str).astype(float)
    for enr_cont in enr_cont_list:
        enrichment_df[enr_cont] = enrichment_df[enr_cont].astype(str).astype(float)
    enrichment_df['comment'] = enrichment_df['comment'].astype(str)
    enrichment_df['max_fold_context'] = enrichment_df['max_fold_context'].astype(str)
    enrichment_df['max_enrich'] = enrichment_df['max_enrich'].astype(float)
    enrichment_df['mneumonics'] = enrichment_df['mneumonics'].astype(str)
    enrichment_df['color'] = enrichment_df['color'].astype(str)
    enrichment_df['state_order_by_group'] = enrichment_df['state_order_by_group'].astype(int)
    return enrichment_df

def get_enrichment_colored_df(enrichment_fn, save_fn, context_prefix):
    cm = sns.light_palette("red", as_cmap=True)
    enrichment_df = pd.read_csv(enrichment_fn, sep = "\t", header = 0)
    enrichment_df = enrichment_df.rename(columns = {u'state (Emission order)': 'state', u'Genome %' : 'percent_in_genome'})
    NUM_NON_CONTEXT_COLUMNS = 2
    if ('percent_in_genome' not in enrichment_df.columns):
        NUM_NON_CONTEXT_COLUMNS = 1
    (num_state, num_enr_cont) = (enrichment_df.shape[0] - 1, enrichment_df.shape[1] - NUM_NON_CONTEXT_COLUMNS)
    enr_cont_list = enrichment_df.columns[NUM_NON_CONTEXT_COLUMNS:]
    enr_cont_list = list(map(get_rid_of_stupid_file_tail, enr_cont_list)) # fix the name of the enrichment context. If it contains the tail '.bed.gz' then we get rid of it
    enrichment_df.columns = list(enrichment_df.columns[:NUM_NON_CONTEXT_COLUMNS]) + list(enr_cont_list) 
    percent_genome_of_cont = enrichment_df.iloc[num_state, NUM_NON_CONTEXT_COLUMNS:]
    enrichment_df = enrichment_df.loc[:(num_state-1)] # rid of the last row because that is the row about percentage each genomic context occupies    
    # now change the column names of the enrichment contexts. If the contexts' names contain '.bed.gz' then get rid of it.
    no_state_df = enrichment_df[enrichment_df.columns[NUM_NON_CONTEXT_COLUMNS:]] # only get the data of enrichment contexts for now, don't consider state and percent_in_genome
    # now call functions to add comments to the enrichment patterns of states
    # add comments about states, calculate the maximum fold enrichment for each state and what context is associated with such fold enrichment
    enrichment_df['comment'] = add_comments_on_states(enrichment_df, num_state, num_enr_cont, context_prefix)
    enrichment_df['max_fold_context'] = no_state_df.apply(lambda x: x.idxmax(), axis = 1) # name of the context that are most enriched in this state
    enrichment_df['max_enrich'] = no_state_df.apply(lambda x: x.max(), axis = 1) # the value of the max fold enrichment in this state
    # now add back data associated with the percentage in genome of each of the
    # enrichment_df = combine_enrichment_df_with_state_annot_df(enrichment_df, enr_cont_list) #NOTE THIS LINE IS COMMENTED OUT BECAUSE WE DO NOT WANT TO 
    num_remaining_columns = len(enrichment_df.columns) - NUM_NON_CONTEXT_COLUMNS - len(enr_cont_list)
    enrichment_df.loc[num_state] = [''] * NUM_NON_CONTEXT_COLUMNS + list(percent_genome_of_cont) + [''] * num_remaining_columns  
    # mneumonics_index_in_row = enrichment_df.columns.get_loc('mneumonics') # column index, zero-based of the menumonics entries, which we will use to paint the right column later
    print(num_state)
    print(enrichment_df.shape)
    print(enrichment_df.tail())
    colored_df = enrichment_df.style.background_gradient(subset = pd.IndexSlice[:(num_state-1), [enr_cont_list[0]]], cmap = cm)
    for enr_cont in enr_cont_list[1:]:
        colored_df = colored_df.background_gradient(subset = pd.IndexSlice[:(num_state-1), [enr_cont]], cmap = cm)
    # colored_df = colored_df.apply(lambda x: color_state_annotation(x, mneumonics_index_in_row), axis = 1)
    colored_df.to_excel(save_fn, engine = 'openpyxl')
    return colored_df


def color_genome_context(val):
    # this function is not really valuable anymore
    GENOME_CONTEXT_COLOR = {"CpGIsland" : '#73C6B6' ,"RefSeqExon" : '#D2B4DE', "RefSeqGene" : '#F1948A' ,"RefSeqTES" : '#F7DC6F', "RefSeqTSS" : '#2980B9', "RefSeqTSS2kb" : '#27AE60', "laminB1lads" : '#D7DBDD', 'clean_assembly_gap_hg19': '#8E44AD', 'znf_genes_hg19' : '#7B241C'}
    color = GENOME_CONTEXT_COLOR[val]
    return 'background-color: %s' % color

def get_colored_df_for_gc_znf_assembly_gap_enrichment(enrichment_fn, save_fn):
    cm = sns.light_palette("red", as_cmap=True)
    enrichment_df = pd.read_csv(enrichment_fn, sep = "\t", header = 0)
    try:
        enrichment_df = enrichment_df.rename(columns = {u'state (Emission order)': 'state', u'Genome %' : 'percent_in_genome'})
    except:
        print ("Could not convert columns: 'state (Emission order)' to 'state' and 'Genome %' to 'percent_in_genome'")
    (num_state, num_enr_cont) = (enrichment_df.shape[0] - 1, enrichment_df.shape[1] - 1)

    enr_cont_list = enrichment_df.columns[2:]
    enr_cont_list = map(get_rid_of_stupid_file_tail, enr_cont_list) # fix the name of the enrichment context. If it contains the tail '.bed.gz' then we get rid of it
    enrichment_df.columns = list(enrichment_df.columns[:2]) + list(enr_cont_list) # now change the column names of the enrichment contexts. If the contexts' names contain '.bed.gz' then get rid of it.
    no_state_df = enrichment_df[enrichment_df.columns[2:]] # only get the data of enrichment contexts for now, don't consider state and percent_in_genome

    enrichment_df['max_fold_context'] = no_state_df.apply(lambda x: x.idxmax(), axis = 1) # name of the context that are most enriched in this state
    enrichment_df['max_enrich'] = no_state_df.apply(lambda x: x.max(), axis = 1) # the value of the max fold enrichment in this state
    # now join the data that we usually have wtih data of consHMM state annotation
    # color the max enrichment consHMM state for each full stack state with the color of the consHMM state
    colored_df = enrichment_df.style.applymap(color_genome_context, subset = pd.IndexSlice[:, ['max_fold_context']])
    # now color the enrichment values
    colored_df = colored_df.background_gradient(subset = pd.IndexSlice[:(num_state - 1), [enr_cont_list[0]]], cmap = cm)
    for enr_cont in enr_cont_list[1:]:
        colored_df = colored_df.background_gradient(subset = pd.IndexSlice[:(num_state - 1), [enr_cont]], cmap = cm)
    colored_df.to_excel(save_fn, engine = 'openpyxl')
    return colored_df


def main():
    if len(sys.argv) != 4:
    	usage()
    input_fn = sys.argv[1]
    helper.check_file_exist(input_fn)
    output_fn = sys.argv[2]
    helper.create_folder_for_file(output_fn)
    context_prefix = sys.argv[3]
    print ("Done getting command line")

    get_enrichment_colored_df(input_fn, output_fn, context_prefix)


def usage():
    print ("python create_excel_painted_overlap_enrichment_output.py")
    print ("input_fn: the output of chromHMM OverlapEnrichment")
    print ("output_fn: the name of the excel file that you want to create") 
    print ("context_prefix: the prefix to all the enrichment contexts in this input_fn. For example, if we do enrichments with gnomad variants of varying maf, the column names in input_fn are 'maf_0_0.1<etc>', we can specify context_prefix to gnomad. If we dont need such prefix, specify to empty string. This is useful when we try to write comment on states that are most enriched in each genomic context.")
    exit(1)

main()
