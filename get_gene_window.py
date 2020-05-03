'''
Given list of gene sets, compute SNP annotation corresponding to x kb window around genes in gene set. Compute LD scores from
these annotations.
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

dirname = os.path.dirname(__file__)

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


def create_window_ldsc(args):
    '''
    Create SNP annotation corresponding to x kb window around genes in gene sets. Estimate LD scores using these annotations.
    '''
    gsets = read_gene_sets(args.gene_sets)
    gene_coords = pd.read_csv(args.gene_coords, sep='\t')

    # read SNPs
    keep_snps = pd.read_csv(args.keep, header=None)
    geno_fname = args.bfile
    bim = pd.read_csv(geno_fname + '.bim', sep='\t', header=None)  # load bim
    bim = bim.loc[bim[1].isin(keep_snps[0]).values, 0:3]
    bim.columns = ['CHR', 'SNP', 'CM', 'BP']

    # create gene window annot file
    for gset, genes in gsets.items():
        temp_genes = [x for x in genes if x in gene_coords.iloc[:,2].tolist()]
        toadd_annot = np.zeros(len(bim))
        for gene in temp_genes:
            coord = gene_coords[gene_coords.iloc[:,2] == gene]
            try:
                chr = int(coord.iloc[0,0])
            except:
                continue
            start = int(coord.iloc[0,1]) - 1000 * args.make_kb_window
            end = int(coord.iloc[0,1]) + 1000 * args.make_kb_window
            toadd_annot[(bim['CHR'] == chr).values & (bim['BP'] > start).values & (bim['BP'] < end).values] = 1
        bim[gset] = toadd_annot

    bim.to_csv('{}.annot.gz'.format(args.out), sep='\t', index=False, compression='gzip')
    # estimate LD scores
    array_indivs = ps.PlinkFAMFile(geno_fname + '.fam')
    array_snps = ps.PlinkBIMFile(geno_fname + '.bim')
    keep_snps_indices = np.where(array_snps.df['SNP'].isin(keep_snps[0]).values)[0]

    geno_array = ld.PlinkBEDFile(geno_fname + '.bed', array_indivs.n, array_snps,
                                     keep_snps=keep_snps_indices)

    block_left = ld.getBlockLefts(geno_array.df[:, 2], 1e6)

    # estimate expression scores
    res = geno_array.ldScoreVarBlocks(block_left, c=50, annot=bim.iloc[:, 4:].values)
    expscore = pd.DataFrame(np.c_[geno_array.df[:, :3], res])
    expscore.columns = geno_array.colnames[:3] + bim.columns[4:].tolist()

    expscore.to_csv('{}.l2.ldscore.gz'.format(args.out), sep='\t', index=False, compression='gzip',
                    float_format='%.5f')

parser = argparse.ArgumentParser()
parser.add_argument('--make-kb-window', default=100, type=int,
                    help='Generate SNP annotation corresponding to x kb window around genes in gene set, '
                         'where x is the input value. Default is ')
parser.add_argument('--gene-coords', default=os.path.join(dirname, 'data/gene_coords.txt'), type=str,
                    help='File containing the gene locations of files. '
                         'By default, we supply this file in data/gene_locations.txt')
parser.add_argument('--bfile', default=None, type=str,
                    help='Genotypes that are ancestry-matched to GWAS samples')
parser.add_argument('--gene-sets', default=None, type=str,
                    help='File containing gene sets. One gene set per line. First column is gene set name, remaining '
                         'columns are gene names.')
parser.add_argument('--keep', default=os.path.join(dirname, 'data/hm3_snps.txt'), type=str,
                    help='File with SNPs to include in expression score estimation. '
                         'The file should contain one SNP ID per row.')
parser.add_argument('--out', default=None, type=str,
                    help='Output prefix')

if __name__ == '__main__':

    args = parser.parse_args()
    if not args.bfile:
        raise ValueError('Must specify --bfile')
    if not args.gene_sets:
        raise ValueError('Must specify --gene-sets')
    if not args.out:
        raise ValueError('Must specify --out')

    create_window_ldsc(args)