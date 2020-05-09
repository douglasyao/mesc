'''
Compute expression scores for gene sets. Expression scores can be meta-analyzed over multiple tissues/conditions or not.
'''

from __future__ import division
from collections import OrderedDict, Iterable, defaultdict
import numpy as np
import pandas as pd
import os
import argparse
import mesc.ldscore as ld
import mesc.parse as ps
import sys
import copy
import subprocess

N_CHR=2
dirname = os.path.dirname(__file__)
pd.set_option('display.max_rows',10)

class Suppressor(object):
    '''
    Suppresses output from subprocess
    '''
    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = self

    def __exit__(self, type, value, traceback):
        sys.stdout = self.stdout
        if type is not None:
            pass

    def write(self, x): pass

def read_file_line(fname):
    '''
    Read file with one item per line into list
    '''
    lines = []
    with open(fname, 'rb') as f:
        for l in f:
            l = l.strip()
            lines.append(l)
    return lines

def create_gset_expscore(args):
    '''
    Create gene set expression scores
    '''
    input_prefix = '{}.{}'.format(args.input_prefix, args.chr)
    gsets = read_gene_sets(args.gene_sets)

    print('Reading eQTL weights')
    h2cis = pd.DataFrame()
    # read in all chromosome, since partitioning should be by gene across genome (makes a difference for small gene sets)
    for i in range(1, N_CHR+1):
        temp_h2cis = pd.read_csv('{}.{}.hsq'.format(args.input_prefix, i), sep='\t')
        h2cis = h2cis.append(temp_h2cis)
    h2cis.dropna(inplace=True)

    lasso = pd.read_csv(input_prefix + '.lasso', sep='\t')

    keep_snps = pd.read_csv(args.keep, header=None)
    geno_fname = args.bfile
    bim = pd.read_csv(geno_fname + '.bim', sep='\t', header=None) # load bim
    bim = bim.loc[(bim[0] == args.chr).values & bim[1].isin(keep_snps[0]).values, 0:3]
    bim.columns = ['CHR', 'SNP', 'CM', 'BP']

    # keep genes with positive h2cis and converged LASSO
    snp_indices = dict(zip(bim['SNP'].tolist(), range(len(bim))))  # SNP indices for fast merging
    filtered_h2cis = h2cis[h2cis['h2cis'] > 0]  # filter out genes w/h2cis < 0
    filtered_h2cis = filtered_h2cis[~np.isnan(filtered_h2cis['h2cis'])]
    if args.genes:
        keep_genes = read_file_line(args.genes)
        filtered_h2cis = filtered_h2cis[filtered_h2cis['Gene'].isin(keep_genes)]
    # retain genes across all chromosome for binning
    filtered_gene_indices = dict(zip(filtered_h2cis['Gene'].tolist(), range(len(filtered_h2cis))))

    # get gset names
    gset_names = ['Cis_herit_bin_{}'.format(x) for x in range(1,args.num_background_bins+1)]
    for k in gsets.keys():
        gset_names.extend(['{}_Cis_herit_bin_{}'.format(k, x) for x in range(1,args.num_gene_bins+1)])

    # create dict indicating gene membership in each gene set
    ave_h2cis = []  # compute average cis-heritability of genes in bin
    gene_gset_dict = defaultdict(list)
    # background gene set
    gene_bins = pd.qcut(filtered_h2cis['h2cis'], args.num_background_bins, labels=range(args.num_background_bins)).astype(int).tolist()
    temp_combined_herit = pd.DataFrame(np.c_[filtered_h2cis[['Gene', 'Chrom','h2cis']], gene_bins])
    temp_combined_herit[1] = temp_combined_herit[1].astype(int)
    temp_combined_herit[2] = temp_combined_herit[2].astype(float)
    temp_combined_herit[3] = temp_combined_herit[3].astype(int)
    temp_combined_herit = temp_combined_herit[temp_combined_herit[1] == args.chr]
    temp_h2cis = temp_combined_herit[[2,3]].groupby([3]).mean()
    temp_h2cis = temp_h2cis[2].values
    ave_h2cis.extend(temp_h2cis)
    for i, gene in enumerate(filtered_h2cis['Gene']):
        gene_gset_dict[gene].append('Cis_herit_bin_{}'.format(gene_bins[i]+1))

    # remaining gene sets
    for k, v in gsets.items():
        temp_genes = [x for x in v if x in filtered_h2cis['Gene'].tolist()]
        temp_herit = filtered_h2cis.iloc[[filtered_gene_indices[x] for x in temp_genes], [0,1,2]]
        gene_bins = pd.qcut(temp_herit['h2cis'], args.num_gene_bins, labels=range(args.num_gene_bins)).astype(int).tolist() # bin first, then subset chr
        temp_combined_herit = pd.DataFrame(np.c_[temp_herit, gene_bins])
        temp_combined_herit[1] = temp_combined_herit[1].astype(int)
        temp_combined_herit[2] = temp_combined_herit[2].astype(float)
        temp_combined_herit[3] = temp_combined_herit[3].astype(int)
        temp_combined_herit = temp_combined_herit[temp_combined_herit[1] == args.chr] # subset chr

        # sometimes for small gene sets, bins will contain no genes for individual chromosomes
        bins = temp_combined_herit[3].tolist()
        copy_herit = copy.deepcopy(temp_combined_herit)
        for i in range(args.num_gene_bins):
            if i not in bins:
                copy_herit = copy_herit.append([['GENE',0,0,i]])
        temp_h2cis = copy_herit[[2, 3]].groupby([3]).mean()
        temp_h2cis = temp_h2cis[2].values
        ave_h2cis.extend(temp_h2cis)
        for i, gene in enumerate(temp_combined_herit[0].values):
            gene_gset_dict[gene].append('{}_Cis_herit_bin_{}'.format(k, temp_combined_herit[3].values[i]+1))
    gset_indices = dict(zip(gset_names, range(len(gset_names))))
    filtered_h2cis = filtered_h2cis[filtered_h2cis['Gene'].isin(lasso['GENE'])] # finally retain just genes on input chr

    g_annot = []
    glist = []
    eqtl_annot = np.zeros((len(bim), len(gset_names)))

    # create eQTL annot (for expscore) and gene annot
    print('Combining eQTL weights')
    for i in range(len(filtered_h2cis)):
        gene = filtered_h2cis.iloc[i, 0]
        temp_h2cis = filtered_h2cis.iloc[i, 2]
        temp_lasso = lasso[lasso['GENE'] == gene]
        if len(temp_lasso) == 0:
            continue
        if gene not in gene_gset_dict.keys():
            g_annot.append(np.zeros(len(gset_names)))
        else:
            snp_idx = [snp_indices[x] for x in temp_lasso['SNP'].tolist()]
            temp_lasso_weights = temp_lasso['EFFECT'].values
            emp_herit = np.sum(np.square(temp_lasso_weights))
            if emp_herit <= 0:  # scale eQTL weights to h2cis
                bias = 0
            else:
                bias = np.sqrt(temp_h2cis / emp_herit)
            temp_lasso_weights *= bias
            temp_gset_indices = [gset_indices[x] for x in gene_gset_dict[gene]]
            for gset in temp_gset_indices:
                eqtl_annot[snp_idx, gset] += np.square(temp_lasso_weights)
            g_annot_toadd = np.zeros(len(gset_names))
            g_annot_toadd[temp_gset_indices] = 1
            g_annot.append(g_annot_toadd)
        glist.append(gene)

    g_annot = np.array(g_annot)
    g_annot_final = pd.DataFrame(np.c_[glist, g_annot])
    g_annot_final.columns = ['Gene'] + gset_names
    for i in range(1, g_annot_final.shape[1]):
        g_annot_final.iloc[:,i] = pd.to_numeric(g_annot_final.iloc[:,i]).astype(int)
    g_annot_final.to_csv('{}.{}.gannot.gz'.format(args.out, args.chr), sep='\t', index=False, compression='gzip')

    # output .G and .ave_h2cis files
    G = np.sum(g_annot, axis=0)
    np.savetxt('{}.{}.G'.format(args.out, args.chr), G.reshape((1, len(G))), fmt='%d')
    np.savetxt('{}.{}.ave_h2cis'.format(args.out, args.chr), np.array(ave_h2cis).reshape((1, len(ave_h2cis))),
               fmt="%.5f")

    print('Computing expression scores')
    # load genotypes
    array_indivs = ps.PlinkFAMFile(geno_fname + '.fam')
    array_snps = ps.PlinkBIMFile(geno_fname + '.bim')
    keep_snps_indices = np.where((array_snps.df['CHR'] == args.chr).values & array_snps.df['SNP'].isin(keep_snps[0]).values)[0]

    with Suppressor():
        geno_array = ld.PlinkBEDFile(geno_fname + '.bed', array_indivs.n, array_snps,
                                     keep_snps=keep_snps_indices)

    block_left = ld.getBlockLefts(geno_array.df[:, 2], 1e6)

    # estimate expression scores
    res = geno_array.ldScoreVarBlocks(block_left, c=50, annot=eqtl_annot)
    expscore = pd.DataFrame(np.c_[geno_array.df[:, :3], res])
    expscore.columns = geno_array.colnames[:3] + gset_names

    for name in gset_names:
        expscore[name] = expscore[name].astype(float)

    # output files
    expscore.to_csv('{}.{}.expscore.gz'.format(args.out, args.chr), sep='\t', index=False, compression='gzip',
                    float_format='%.5f')
    print('Done!')

def create_gset_expscore_meta(args):
    '''
    Create gene set expression scores meta-analyzed over several tissues/conditions
    '''
    input_prefixes = read_file_line(args.input_prefix_meta)
    genes = get_gene_list(input_prefixes)
    gsets = read_gene_sets(args.gene_sets)

    # gene indices for fast merging
    gene_indices = dict(zip(genes['Gene'].tolist(), range(len(genes))))
    num = np.zeros(len(genes))
    count = np.zeros(len(genes))
    all_lasso = pd.DataFrame()

    print('Reading eQTL weights')
    # meta-analyze REML h2cis estimates by taking simple average
    # inverse-variance weighing has issues, since REML SE is downwardly biased for small h2 estimates
    for input in input_prefixes:
        cond = os.path.basename(input)
        lasso = pd.read_csv('{}.{}.lasso'.format(input, args.chr), sep='\t')
        lasso['COND'] = cond
        all_lasso = all_lasso.append(lasso)
        all_reml = pd.DataFrame()
        for i in range(1, N_CHR+1):
            reml = pd.read_csv('{}.{}.hsq'.format(input, i), sep='\t')
            reml.dropna(inplace=True)
            all_reml = all_reml.append(reml)
        gene_idx = [gene_indices[x] for x in all_reml['Gene'].tolist()]
        num[gene_idx] += all_reml['h2cis'].values
        count[gene_idx] += 1

    count[count == 0] = np.nan
    meta_h2cis = num / count
    meta_h2cis_out = pd.DataFrame({'Gene': genes['Gene'],
                                   'Chrom': genes['Chrom'],
                                   'h2cis': meta_h2cis}, columns=['Gene', 'Chrom', 'h2cis'])

    keep_snps = pd.read_csv(args.keep, header=None)
    geno_fname = args.bfile
    bim = pd.read_csv(geno_fname + '.bim', sep='\t', header=None) # load bim
    bim = bim.loc[(bim[0] == args.chr).values & bim[1].isin(keep_snps[0]).values, 0:3]
    bim.columns = ['CHR', 'SNP', 'CM', 'BP']

    # keep genes with positive h2cis and converged LASSO
    snp_indices = dict(zip(bim['SNP'].tolist(), range(len(bim))))  # SNP indices for fast merging
    filtered_h2cis = meta_h2cis_out[meta_h2cis_out['h2cis'] > 0]  # filter out genes w/h2cis < 0
    filtered_h2cis = filtered_h2cis[~np.isnan(filtered_h2cis['h2cis'])]
    if args.genes:
        keep_genes = read_file_line(args.genes)
        filtered_h2cis = filtered_h2cis[filtered_h2cis['Gene'].isin(keep_genes)]
    # retain genes across all chromosome for binning
    filtered_gene_indices = dict(zip(filtered_h2cis['Gene'].tolist(), range(len(filtered_h2cis))))

    # get gset names
    gset_names = ['Cis_herit_bin_{}'.format(x) for x in range(1,args.num_background_bins+1)]
    for k in gsets.keys():
        gset_names.extend(['{}_Cis_herit_bin_{}'.format(k, x) for x in range(1,args.num_gene_bins+1)])

    # create dict indicating gene membership in each gene set
    ave_h2cis = []  # compute average cis-heritability of genes in bin
    gene_gset_dict = defaultdict(list)
    # background gene set
    gene_bins = pd.qcut(filtered_h2cis['h2cis'], args.num_background_bins,
                        labels=range(args.num_background_bins)).astype(int).tolist()
    temp_combined_herit = pd.DataFrame(np.c_[filtered_h2cis[['Gene', 'Chrom', 'h2cis']], gene_bins])
    temp_combined_herit[1] = temp_combined_herit[1].astype(int)
    temp_combined_herit[2] = temp_combined_herit[2].astype(float)
    temp_combined_herit[3] = temp_combined_herit[3].astype(int)
    temp_combined_herit = temp_combined_herit[temp_combined_herit[1] == args.chr]
    temp_h2cis = temp_combined_herit[[2, 3]].groupby([3]).mean()
    temp_h2cis = temp_h2cis[2].values
    ave_h2cis.extend(temp_h2cis)
    for i, gene in enumerate(filtered_h2cis['Gene']):
        gene_gset_dict[gene].append('Cis_herit_bin_{}'.format(gene_bins[i] + 1))

    # remaining gene sets
    for k, v in gsets.items():
        temp_genes = [x for x in v if x in filtered_h2cis['Gene'].tolist()]
        temp_herit = filtered_h2cis.iloc[[filtered_gene_indices[x] for x in temp_genes], [0, 1, 2]]
        gene_bins = pd.qcut(temp_herit['h2cis'], args.num_gene_bins, labels=range(args.num_gene_bins)).astype(
            int).tolist()  # bin first, then subset chr
        temp_combined_herit = pd.DataFrame(np.c_[temp_herit, gene_bins])
        temp_combined_herit[1] = temp_combined_herit[1].astype(int)
        temp_combined_herit[2] = temp_combined_herit[2].astype(float)
        temp_combined_herit[3] = temp_combined_herit[3].astype(int)
        temp_combined_herit = temp_combined_herit[temp_combined_herit[1] == args.chr]  # subset chr

        # sometimes for small gene sets, bins will contain no genes for individual chromosomes
        bins = temp_combined_herit[3].tolist()
        copy_herit = copy.deepcopy(temp_combined_herit)
        for i in range(args.num_gene_bins):
            if i not in bins:
                copy_herit = copy_herit.append([['GENE', 0, 0, i]])
        temp_h2cis = copy_herit[[2, 3]].groupby([3]).mean()
        temp_h2cis = temp_h2cis[2].values
        ave_h2cis.extend(temp_h2cis)
        for i, gene in enumerate(temp_combined_herit[0].values):
            gene_gset_dict[gene].append('{}_Cis_herit_bin_{}'.format(k, temp_combined_herit[3].values[i] + 1))
    gset_indices = dict(zip(gset_names, range(len(gset_names))))
    filtered_h2cis = filtered_h2cis[filtered_h2cis['Gene'].isin(all_lasso['GENE'])]  # finally retain just genes on input chr

    g_annot = []
    g_annot_names = []
    eqtl_annot = np.zeros((len(bim), len(gset_names)))

    # create eQTL annot (for expscore) and gene annot
    print('Combining eQTL weights')
    for i in range(len(filtered_h2cis)):
        gene = filtered_h2cis.iloc[i, 0]
        temp_h2cis = filtered_h2cis.iloc[i, 2]
        temp_lasso = all_lasso[all_lasso['GENE'] == gene]
        unique_conds = pd.unique(temp_lasso['COND'])
        if gene not in gene_gset_dict.keys():
            for temp_cond in unique_conds.tolist():
                g_annot.append(np.zeros(len(gset_names)))
                g_annot_names.append('{}_{}'.format(gene, temp_cond))
            continue

        for temp_cond in unique_conds:  # for each condition
            temp_temp_lasso = temp_lasso[temp_lasso['COND'] == temp_cond]
            snp_idx = [snp_indices[x] for x in temp_temp_lasso['SNP'].tolist()]
            temp_lasso_weights = temp_temp_lasso['EFFECT'].values
            emp_herit = np.sum(np.square(temp_lasso_weights))
            if emp_herit <= 0:  # scale eQTL weights to meta-tissue h2cis
                bias = 0
            else:
                bias = np.sqrt(temp_h2cis / emp_herit)
            temp_lasso_weights *= bias
            temp_gset_indices = [gset_indices[x] for x in gene_gset_dict[gene]]
            for gset in temp_gset_indices:
                eqtl_annot[snp_idx, gset] += np.square(temp_lasso_weights)
            g_annot_toadd = np.zeros(len(gset_names))
            g_annot_toadd[temp_gset_indices] = 1
            g_annot.append(g_annot_toadd)
            g_annot_names.append(gene + '_' + temp_cond)

    g_annot = np.array(g_annot)
    g_annot_final = pd.DataFrame(np.c_[g_annot_names, g_annot])
    g_annot_final.columns = ['Gene'] + gset_names
    for i in range(1, g_annot_final.shape[1]):
        g_annot_final.iloc[:,i] = pd.to_numeric(g_annot_final.iloc[:,i]).astype(int)
    g_annot_final.to_csv('{}.{}.gannot.gz'.format(args.out, args.chr), sep='\t', index=False, compression='gzip')

    # output .G and .ave_h2cis files
    G = np.sum(g_annot, axis=0)
    np.savetxt('{}.{}.G'.format(args.out, args.chr), G.reshape((1, len(G))), fmt='%d')
    np.savetxt('{}.{}.ave_h2cis'.format(args.out, args.chr), np.array(ave_h2cis).reshape((1, len(ave_h2cis))),
               fmt="%.5f")

    print('Computing expression scores')
    # load genotypes
    array_indivs = ps.PlinkFAMFile(geno_fname + '.fam')
    array_snps = ps.PlinkBIMFile(geno_fname + '.bim')
    keep_snps_indices = np.where((array_snps.df['CHR'] == args.chr).values & array_snps.df['SNP'].isin(keep_snps[0]).values)[0]

    with Suppressor():
        geno_array = ld.PlinkBEDFile(geno_fname + '.bed', array_indivs.n, array_snps,
                                     keep_snps=keep_snps_indices)

    block_left = ld.getBlockLefts(geno_array.df[:, 2], 1e6)

    # estimate expression scores
    res = geno_array.ldScoreVarBlocks(block_left, c=50, annot=eqtl_annot)
    expscore = pd.DataFrame(np.c_[geno_array.df[:, :3], res])
    expscore.columns = geno_array.colnames[:3] + gset_names

    for name in gset_names:
        expscore[name] = expscore[name].astype(float)

    # output files
    expscore.to_csv('{}.{}.expscore.gz'.format(args.out, args.chr), sep='\t', index=False, compression='gzip',
                    float_format='%.5f')
    print('Done!')

##### CHANGE N_CHR BACK TO 22 BEFORE COMMITTING ######################################################
def create_gset_expscore_meta_batch(args):
    '''
    Create gene set expression scores meta-analyzed over several tissues/conditions. Analyze gene sets in batches.
    '''
    input_prefixes = read_file_line(args.input_prefix_meta)
    genes = get_gene_list(input_prefixes)
    gsets = read_gene_sets(args.gene_sets)

    # gene indices for fast merging
    gene_indices = dict(zip(genes['Gene'].tolist(), range(len(genes))))
    num = np.zeros(len(genes))
    count = np.zeros(len(genes))
    all_lasso = pd.DataFrame()

    print('Reading eQTL weights')
    # meta-analyze REML h2cis estimates by taking simple average
    # inverse-variance weighing has issues, since REML SE is downwardly biased for small h2 estimates
    for input in input_prefixes:
        cond = os.path.basename(input)
        lasso = pd.read_csv('{}.{}.lasso'.format(input, args.chr), sep='\t')
        lasso['COND'] = cond
        all_lasso = all_lasso.append(lasso)
        all_reml = pd.DataFrame()
        for i in range(1, N_CHR + 1):
            reml = pd.read_csv('{}.{}.hsq'.format(input, i), sep='\t')
            reml.dropna(inplace=True)
            all_reml = all_reml.append(reml)
        gene_idx = [gene_indices[x] for x in all_reml['Gene'].tolist()]
        num[gene_idx] += all_reml['h2cis'].values
        count[gene_idx] += 1

    count[count == 0] = np.nan
    meta_h2cis = num / count
    meta_h2cis_out = pd.DataFrame({'Gene': genes['Gene'],
                                   'Chrom': genes['Chrom'],
                                   'h2cis': meta_h2cis}, columns=['Gene', 'Chrom', 'h2cis'])

    keep_snps = pd.read_csv(args.keep, header=None)
    geno_fname = args.bfile
    bim = pd.read_csv(geno_fname + '.bim', sep='\t', header=None)  # load bim
    bim = bim.loc[(bim[0] == args.chr).values & bim[1].isin(keep_snps[0]).values, 0:3]
    bim.columns = ['CHR', 'SNP', 'CM', 'BP']

    # keep genes with positive h2cis and converged LASSO
    snp_indices = dict(zip(bim['SNP'].tolist(), range(len(bim))))  # SNP indices for fast merging
    filtered_h2cis = meta_h2cis_out[meta_h2cis_out['h2cis'] > 0]  # filter out genes w/h2cis < 0
    filtered_h2cis = filtered_h2cis[~np.isnan(filtered_h2cis['h2cis'])]
    if args.genes:
        keep_genes = read_file_line(args.genes)
        filtered_h2cis = filtered_h2cis[filtered_h2cis['Gene'].isin(keep_genes)]
    # retain genes across all chromosome for binning
    filtered_gene_indices = dict(zip(filtered_h2cis['Gene'].tolist(), range(len(filtered_h2cis))))
    chr_filtered_h2cis = filtered_h2cis[filtered_h2cis['Gene'].isin(all_lasso['GENE'])]  # finally retain just genes on input chr

    # load genotypes
    array_indivs = ps.PlinkFAMFile(geno_fname + '.fam')
    array_snps = ps.PlinkBIMFile(geno_fname + '.bim')
    keep_snps_indices = \
        np.where((array_snps.df['CHR'] == args.chr).values & array_snps.df['SNP'].isin(keep_snps[0]).values)[0]

    with Suppressor():
        geno_array = ld.PlinkBEDFile(geno_fname + '.bed', array_indivs.n, array_snps,
                                     keep_snps=keep_snps_indices)

    block_left = ld.getBlockLefts(geno_array.df[:, 2], 1e6)


    print('Computing expression scores for background gene sets')
    # analyze background gset
    gset_names = ['Cis_herit_bin_{}'.format(x) for x in range(1, args.num_background_bins + 1)]
    gset_indices = dict(zip(gset_names, range(len(gset_names))))

    # create dict indicating gene membership in each gene set
    all_ave_h2cis = []  # compute average cis-heritability of genes in bin
    all_G = []
    gene_gset_dict = defaultdict(list)
    # background gene set
    gene_bins = pd.qcut(filtered_h2cis['h2cis'], args.num_background_bins,
                        labels=range(args.num_background_bins)).astype(int).tolist()
    temp_combined_herit = pd.DataFrame(np.c_[filtered_h2cis[['Gene', 'Chrom', 'h2cis']], gene_bins])
    temp_combined_herit[1] = temp_combined_herit[1].astype(int)
    temp_combined_herit[2] = temp_combined_herit[2].astype(float)
    temp_combined_herit[3] = temp_combined_herit[3].astype(int)
    temp_combined_herit = temp_combined_herit[temp_combined_herit[1] == args.chr]
    temp_h2cis = temp_combined_herit[[2, 3]].groupby([3]).mean()
    temp_h2cis = temp_h2cis[2].values
    all_ave_h2cis.extend(temp_h2cis)
    for i, gene in enumerate(filtered_h2cis['Gene']):
        gene_gset_dict[gene].append('Cis_herit_bin_{}'.format(gene_bins[i] + 1))

    # compute expression scores
    G, g_annot_final, expscore = batch_expscore(gene_gset_dict, gset_names, chr_filtered_h2cis, all_lasso, bim, snp_indices, gset_indices,
                                                geno_array, block_left)
    all_G.extend(G)

    g_annot_name = '{}.{}.gannot'.format(args.out, args.chr)
    expscore_name = '{}.{}.expscore'.format(args.out, args.chr)

    g_annot_final.to_csv(g_annot_name, sep='\t', index=False)
    expscore.to_csv(expscore_name, sep='\t', index=False, float_format='%.5f')

    # remaining gene sets
    for i in range(0, len(gsets), args.batch_size):
        print('Computing expression scores for gene sets {} to {} (out of {} total)'.format(i+1, min(i+args.batch_size, len(gsets)), len(gsets)))
        temp_gsets = OrderedDict(gsets.items()[i:(i+args.batch_size)])
        rest_gset_names = []
        rest_gene_gset_dict = defaultdict(list)

        for k in temp_gsets.keys():
            rest_gset_names.extend(['{}_Cis_herit_bin_{}'.format(k, x) for x in range(1, args.num_gene_bins + 1)])

        gset_indices = dict(zip(rest_gset_names, range(len(rest_gset_names))))

        for k, v in temp_gsets.items():
            temp_genes = [x for x in v if x in filtered_h2cis['Gene'].tolist()]
            temp_herit = filtered_h2cis.iloc[[filtered_gene_indices[x] for x in temp_genes], [0, 1, 2]]
            gene_bins = pd.qcut(temp_herit['h2cis'], args.num_gene_bins, labels=range(args.num_gene_bins)).astype(
                int).tolist()  # bin first, then subset chr
            temp_combined_herit = pd.DataFrame(np.c_[temp_herit, gene_bins])
            temp_combined_herit[1] = temp_combined_herit[1].astype(int)
            temp_combined_herit[2] = temp_combined_herit[2].astype(float)
            temp_combined_herit[3] = temp_combined_herit[3].astype(int)
            temp_combined_herit = temp_combined_herit[temp_combined_herit[1] == args.chr]  # subset chr

            # sometimes for small gene sets, bins will contain no genes for individual chromosomes
            bins = temp_combined_herit[3].tolist()
            copy_herit = copy.deepcopy(temp_combined_herit)
            for i in range(args.num_gene_bins):
                if i not in bins:
                    copy_herit = copy_herit.append([['GENE', 0, 0, i]])
            temp_h2cis = copy_herit[[2, 3]].groupby([3]).mean()
            temp_h2cis = temp_h2cis[2].values
            all_ave_h2cis.extend(temp_h2cis)
            for i, gene in enumerate(temp_combined_herit[0].values):
                rest_gene_gset_dict[gene].append('{}_Cis_herit_bin_{}'.format(k, temp_combined_herit[3].values[i] + 1))

        G, g_annot_final, expscore = batch_expscore(rest_gene_gset_dict, rest_gset_names, chr_filtered_h2cis, all_lasso, bim,
                                                    snp_indices, gset_indices, geno_array, block_left)
        all_G.extend(G)
        temp_g_annot_name = '{}.{}.temp.gannot'.format(args.out, args.chr)
        temp_expscore_name = '{}.{}.temp.expscore'.format(args.out, args.chr)

        g_annot_final.iloc[:,1:].to_csv(temp_g_annot_name, sep='\t', index=False)
        expscore.iloc[:,3:].to_csv(temp_expscore_name, sep='\t', index=False, float_format='%.5f')
        subprocess.Popen('paste {} {} > {}.temp'.format(g_annot_name, temp_g_annot_name, temp_g_annot_name), shell=True)
        subprocess.Popen('mv {}.temp {}'.format(temp_g_annot_name, g_annot_name), shell=True)
        subprocess.Popen('rm {}'.format(temp_g_annot_name), shell=True)

        subprocess.Popen('paste {} {} > {}.temp'.format(expscore_name, temp_expscore_name, temp_expscore_name), shell=True)
        subprocess.Popen('mv {}.temp {}'.format(temp_expscore_name, expscore_name), shell=True)
        subprocess.Popen('rm {}'.format(temp_expscore_name), shell=True)

    subprocess.Popen('gzip {}'.format(g_annot_name), shell=True)
    subprocess.Popen('gzip {}'.format(expscore_name), shell=True)

    np.savetxt('{}.{}.G'.format(args.out, args.chr), np.array(all_G).reshape((1, len(all_G))), fmt='%d')
    np.savetxt('{}.{}.ave_h2cis'.format(args.out, args.chr), np.array(all_ave_h2cis).reshape((1, len(all_ave_h2cis))),
               fmt="%.5f")

    print('Done!')


def batch_expscore(gene_gset_dict, gset_names, filtered_h2cis, all_lasso, bim, snp_indices, gset_indices, geno_array, block_left):
    '''
    Only gene_gset_dict and gset_names will change
    '''
    g_annot = []
    g_annot_names = []
    eqtl_annot = np.zeros((len(bim), len(gset_names)))
    for i in range(len(filtered_h2cis)):
        gene = filtered_h2cis.iloc[i, 0]
        temp_h2cis = filtered_h2cis.iloc[i, 2]
        temp_lasso = all_lasso[all_lasso['GENE'] == gene]
        unique_conds = pd.unique(temp_lasso['COND'])
        if gene not in gene_gset_dict.keys():
            for temp_cond in unique_conds.tolist():
                g_annot.append(np.zeros(len(gset_names)))
                g_annot_names.append('{}_{}'.format(gene, temp_cond))
            continue

        for temp_cond in unique_conds:  # for each condition
            temp_temp_lasso = temp_lasso[temp_lasso['COND'] == temp_cond]
            snp_idx = [snp_indices[x] for x in temp_temp_lasso['SNP'].tolist()]
            temp_lasso_weights = temp_temp_lasso['EFFECT'].values
            emp_herit = np.sum(np.square(temp_lasso_weights))
            if emp_herit <= 0:  # scale eQTL weights to meta-tissue h2cis
                bias = 0
            else:
                bias = np.sqrt(temp_h2cis / emp_herit)
            temp_lasso_weights *= bias
            temp_gset_indices = [gset_indices[x] for x in gene_gset_dict[gene]]
            for gset in temp_gset_indices:
                eqtl_annot[snp_idx, gset] += np.square(temp_lasso_weights)
            g_annot_toadd = np.zeros(len(gset_names))
            g_annot_toadd[temp_gset_indices] = 1
            g_annot.append(g_annot_toadd)
            g_annot_names.append(gene + '_' + temp_cond)

    g_annot = np.array(g_annot)
    g_annot_final = pd.DataFrame(np.c_[g_annot_names, g_annot])
    g_annot_final.columns = ['Gene'] + gset_names
    for i in range(1, g_annot_final.shape[1]):
        g_annot_final.iloc[:, i] = pd.to_numeric(g_annot_final.iloc[:, i]).astype(int)
    G = np.sum(g_annot, axis=0)

    # estimate expression scores
    res = geno_array.ldScoreVarBlocks(block_left, c=50, annot=eqtl_annot)
    geno_array._currentSNP = 0
    expscore = pd.DataFrame(np.c_[geno_array.df[:, :3], res])
    expscore.columns = geno_array.colnames[:3] + gset_names

    for name in gset_names:
        expscore[name] = expscore[name].astype(float)

    # output files
    return(G, g_annot_final, expscore)


def read_gene_sets(fname):
    '''
    Read gene sets from file.
    '''
    gsets = OrderedDict()
    with open(fname, 'rb') as f:
        for line in f:
            line = line.strip().split()
            gsets[line[0]] = line[1:]
    return gsets

def get_gene_list(input_prefixes):
    '''
    Get union of all genes found in all input files
    '''
    genes = []
    for file in input_prefixes:
        for i in range(1,N_CHR+1):
            fname = '{}.{}.hsq'.format(file, i)
            with open(fname, 'rb') as f:
                next(f)
                for line in f:
                    genes.append(line.split()[0:2])
    genes = pd.DataFrame(genes)
    genes = genes.drop_duplicates()
    genes.columns = ['Gene', 'Chrom']
    return genes

def flatten(items):
    '''
    Flatten nested list
    '''
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x

parser = argparse.ArgumentParser()
parser.add_argument('--out', default=None, type=str,
                    help='Output filename prefix.')
parser.add_argument('--input-prefix', default=None, type=str,
                    help='File prefix for .lasso and .hsq files.')
parser.add_argument('--input-prefix-meta', default=None, type=str,
                    help='File containing list of file prefixes for .lasso and .hsq files to meta-analyze over.'
                         'One name per line.')
parser.add_argument('--gene-sets', default=None, type=str,
                    help='File containing gene sets. One gene set per line. First column is gene set name, remaining '
                         'columns are gene names.')
parser.add_argument('--bfile', default=None, type=str,
                    help='Genotypes used to compute expression scores, which should be ancestry-matched to GWAS samples.'
                         'We recommend using 1000 Genomes.')
parser.add_argument('--chr', default=None, type=int,
                    help='Chromosome number.')
parser.add_argument('--keep', default=os.path.join(dirname, 'data/hm3_snps.txt'), type=str,
                    help='File with SNPs to include in expression score estimation. '
                         'The file should contain one SNP ID per row.')
parser.add_argument('--num-gene-bins', default=3, type=int,
                    help='Number of bins to split each gene set into. Default 3.')
parser.add_argument('--num-background-bins', default=5, type=int,
                    help='Number of bins to split background set of all genes into. Default 5.')
parser.add_argument('--genes', default=None, type=str,
                    help='File containing set of background genes to retain in analysis. h2med enrichment is computed '
                         'relative to background set of genes. One name per line')
parser.add_argument('--batch-size', default=None, type=int,
                    help='Analyze gene sets in batches of input size x. Useful to save memory if many gene sets are present.')

if __name__ == '__main__':

    args = parser.parse_args()
    if args.out is None:
        raise ValueError('Must specify --out')
    if not (args.input_prefix or args.input_prefix_meta):
        raise ValueError('Must specify --input-prefix or --input-prefix-meta')
    if (args.input_prefix and args.input_prefix_meta):
        raise ValueError('Cannot specify both --input-prefix and --input-prefix-meta')
    if args.bfile is None:
        raise ValueError('Must specify --bfile')
    if args.chr is None:
        raise ValueError('Must specify --chr')
    if args.gene_sets is None:
        raise ValueError('Must specify --gene-sets')
    if args.input_prefix_meta:
        if args.batch_size:
            create_gset_expscore_meta_batch(args)
        else:
            create_gset_expscore_meta(args)
    elif args.input_prefix:
        create_gset_expscore(args)


