#!/usr/bin/env python
'''
Meta-analyze LASSO-predicted eQTL weights across multiple tissues/conditions.
Takes a commma separated list of file prefixes for .lasso and .hsq files and PLINK bed file, outputs .expscore, .G,
and .ave_h2cis files meta-analyzed across all conditions.
'''

from __future__ import division
import numpy as np
import pandas as pd
import os
import argparse
import mesc.ldscore as ld
import mesc.parse as ps
import sys

dirname = os.path.dirname(__file__)

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

def meta_analyze(args):
    '''
    Meta-analyze weights
    '''
    input_prefixes = read_file_line(args.input_prefixes)
    input_prefixes_name = ['{}.{}'.format(x, args.chr) for x in input_prefixes]
    genes = get_gene_list(input_prefixes_name)
    if args.genes:
        keep_genes = read_file_line(args.genes)
        genes = [x for x in genes if x in keep_genes]
    # gene indices for fast merging
    gene_indices = dict(zip(genes, range(len(genes))))
    num = np.zeros(len(genes))
    count = np.zeros(len(genes))
    all_lasso = pd.DataFrame()

    # meta-analyze REML h2cis estimates by taking simple average
    # inverse-variance weighing has issues, since REML SE is downwardly biased for small h2 estimates
    for input in input_prefixes:
        cond = os.path.basename(input)
        lasso = pd.read_csv('{}.{}.lasso'.format(input, args.chr), sep='\t')
        lasso['COND'] = cond
        all_lasso = all_lasso.append(lasso)
        reml = pd.read_csv('{}.{}.hsq'.format(input, args.chr), sep='\t')
        reml.dropna(inplace=True)
        reml = reml[reml['Gene'].isin(genes)]
        gene_idx = [gene_indices[x] for x in reml['Gene'].tolist()]
        num[gene_idx] += reml['h2cis'].values
        count[gene_idx] += 1

    count[count == 0] = np.nan
    meta_h2cis = num / count
    meta_h2cis_out = pd.DataFrame({'Gene': genes,
                                   'Chrom': args.chr,
                                   'h2cis': meta_h2cis})
    meta_h2cis_out.to_csv(args.out + '.hsq', sep='\t', index=False)

    keep_snps = pd.read_csv(args.keep, header=None)
    geno_fname = args.bfile
    bim = pd.read_csv(geno_fname + '.bim', sep='\t', header=None) # load bim
    bim = bim.loc[(bim[0] == args.chr).values & bim[1].isin(keep_snps[0]).values, 0:3]
    bim.columns = ['CHR', 'SNP', 'CM', 'BP']

    # create gene membership dict
    snp_indices = dict(zip(bim['SNP'].tolist(), range(len(bim)))) # SNP indices for fast merging
    filtered_meta_h2cis = meta_h2cis_out[meta_h2cis_out['h2cis'] > 0] # filter out genes w/h2cis < 0
    filtered_meta_h2cis = filtered_meta_h2cis[~np.isnan(filtered_meta_h2cis['h2cis'])]
    g_annot_meta = {} # gene membership in bins
    gene_bins = pd.qcut(filtered_meta_h2cis['h2cis'], args.num_bins, labels=range(args.num_bins)).astype(int)
    filtered_meta_h2cis['Bin'] = gene_bins
    for j in range(len(filtered_meta_h2cis)):
        g_annot_meta[filtered_meta_h2cis.iloc[j,1]] = gene_bins.iloc[j]

    g_annot = []
    g_annot_names = []
    eqtl_annot = np.zeros((len(bim), args.num_bins))

    # create eQTL annot (for expscore) and gene annot
    print('Combining eQTL weights')
    for i in range(len(filtered_meta_h2cis)):
        gene = filtered_meta_h2cis.iloc[i, 1]
        temp_h2cis = filtered_meta_h2cis.iloc[i, 2]
        temp_lasso = all_lasso[all_lasso['GENE'] == gene]
        unique_conds = pd.unique(temp_lasso['COND'])
        for temp_cond in unique_conds: # for each condition
            temp_temp_lasso = temp_lasso[temp_lasso['COND'] == temp_cond]
            snp_idx = [snp_indices[x] for x in temp_temp_lasso['SNP'].tolist()]
            temp_lasso_weights = temp_temp_lasso['EFFECT'].values
            emp_herit = np.sum(np.square(temp_lasso_weights))
            if emp_herit <= 0: # scale eQTL weights to meta-tissue h2cis
                bias = 0
            else:
                bias = np.sqrt(temp_h2cis / emp_herit)
            temp_lasso_weights *= bias
            eqtl_annot[snp_idx, g_annot_meta[gene]] += np.square(temp_lasso_weights)
            g_annot_toadd = np.zeros(args.num_bins)
            g_annot_toadd[g_annot_meta[gene]] = 1
            g_annot.append(g_annot_toadd)
            g_annot_names.append(gene + '_' + temp_cond)

    g_annot = np.array(g_annot).astype(int)
    g_annot_final = pd.DataFrame(np.c_[g_annot_names, g_annot])
    g_bin_names = ['Cis_herit_bin_{}'.format(x) for x in range(1, args.num_bins+1)]
    g_annot_final.columns = ['Gene'] + g_bin_names
    g_annot_final.to_csv('{}.{}.gannot.gz'.format(args.out, args.chr), sep='\t', index=False, compression='gzip')

    G = np.sum(g_annot, axis=0)
    ave_cis_herit = filtered_meta_h2cis.groupby(['Bin']).mean()
    ave_cis_herit = ave_cis_herit['h2cis'].values

    np.savetxt('{}.{}.G'.format(args.out, args.chr), G.reshape((1, len(G))), fmt='%d')
    np.savetxt('{}.{}.ave_h2cis'.format(args.out, args.chr), ave_cis_herit.reshape((1, len(ave_cis_herit))),
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
    expscore = pd.concat([
        pd.DataFrame(geno_array.df[:, :3]),
        pd.DataFrame(res)], axis=1)
    expscore.columns = geno_array.colnames[:3] + g_bin_names

    # output files
    expscore.to_csv('{}.{}.expscore.gz'.format(args.out, args.chr), sep='\t', index=False, compression='gzip',
                    float_format='%.5f')
    print('Done!')


def get_gene_list(input_prefixes):
    '''
    Get union of all genes found in all input files
    '''
    genes = []
    for file in input_prefixes:
        fname = file + '.hsq'
        with open(fname, 'rb') as f:
            next(f)
            for line in f:
                genes.append(line.split()[0])
    genes = list(set(genes))
    return genes

parser = argparse.ArgumentParser()
parser.add_argument('--out', default=None, type=str,
                    help='Output filename prefix.')
parser.add_argument('--input-prefixes', default=None, type=str,
                    help='File containing list of file prefixes for .lasso and .hsq files to meta-analyze over.'
                         'One name per line.')
parser.add_argument('--bfile', default=None, type=str,
                    help='Genotypes used to compute expression scores, which should be ancestry-matched to GWAS samples.'
                         'We recommend using 1000 Genomes.')
parser.add_argument('--chr', default=None, type=int,
                    help='Chromosome number.')
parser.add_argument('--keep', default=os.path.join(dirname, 'data/hm3_snps.txt'), type=str,
                    help='File with SNPs to include in expression score estimation. '
                         'The file should contain one SNP ID per row. Default is Hapmap3 SNPs')
parser.add_argument('--num-bins', default=5, type=int,
                    help='Number of expression cis-heritability bins. Default 5.')
parser.add_argument('--genes', default=None, type=str,
                    help='File containing list of genes to retain in the analysis. One gene name per line.')

if __name__ == '__main__':

    args = parser.parse_args()
    if args.out is None:
        raise ValueError('Must specify --out')
    if args.input_prefixes is None:
        raise ValueError('Must specify --input-prefixes')
    if args.bfile is None:
        raise ValueError('Must specify --bfile')
    if args.chr is None:
        raise ValueError('Must specify --chr')

    meta_analyze(args)