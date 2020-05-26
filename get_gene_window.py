#!/usr/bin/env python
'''
Given list of gene sets, compute SNP annotation corresponding to x kb window around genes in gene set. Compute LD scores from
these annotations.
'''

from __future__ import division
from collections import OrderedDict
import numpy as np
import pandas as pd
import os
import argparse
import mesc.ldscore as ld
import mesc.parse as ps
import subprocess

dirname = os.path.dirname(__file__)

def read_gene_sets(fname, start=None, end=None):
    '''
    Read gene sets from file.
    '''
    gsets = OrderedDict()
    with open(fname, 'rb') as f:
        for line in f:
            line = line.strip().split()
            gsets[line[0]] = line[1:]
    if start and end:
        gsets = OrderedDict(gsets.items()[start - 1:end])
    elif start:
        gsets = OrderedDict(gsets.items()[start - 1:])
    elif end:
        gsets = OrderedDict(gsets.items()[:end])
    return gsets


def create_window_ldsc(args):
    '''
    Create SNP annotation corresponding to x kb window around genes in gene sets. Estimate LD scores using these annotations.
    '''
    gsets = read_gene_sets(args.gene_sets, args.gset_start, args.gset_end)
    gene_coords = pd.read_csv(args.gene_coords, sep='\t')

    # read SNPs
    geno_fname = args.bfile
    bim = pd.read_csv(geno_fname + '.bim', sep='\t', header=None)  # load bim
    bim = bim.loc[:, 0:3]
    bim.columns = ['CHR', 'SNP', 'CM', 'BP']

    # create gene window annot file
    for gset, genes in gsets.items():
        temp_genes = [x for x in genes if x in gene_coords.iloc[:,2].tolist()]
        toadd_annot = np.zeros(len(bim), dtype=int)
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

    # estimate LD scores
    array_indivs = ps.PlinkFAMFile(geno_fname + '.fam')
    array_snps = ps.PlinkBIMFile(geno_fname + '.bim')

    geno_array = ld.PlinkBEDFile(geno_fname + '.bed', array_indivs.n, array_snps)
    block_left = ld.getBlockLefts(geno_array.df[:, 2], 1e6)

    # compute M
    M_5_50 = np.sum(bim.iloc[np.array(geno_array.freq) > 0.05, 4:].values, axis=0)

    # estimate ld scores
    res = geno_array.ldScoreVarBlocks(block_left, c=50, annot=bim.iloc[:, 4:].values)
    keep_snps = pd.read_csv(args.keep, header=None)
    keep_snps_indices = array_snps.df['SNP'].isin(keep_snps[0]).values
    res = res[keep_snps_indices,:]

    expscore = pd.concat([
        pd.DataFrame(bim.iloc[keep_snps_indices, [0,1,3]]),
        pd.DataFrame(res)], axis=1)
    expscore.columns = geno_array.colnames[:3] + bim.columns[4:].tolist()

    # output files
    if args.split_output:
        for i in range(len(gsets)):
            gset_name = bim.columns[4+i]
            np.savetxt('{}.{}.M_5_50'.format(args.out, gset_name), M_5_50[i].reshape((1, 1)), fmt='%d')
            bim.iloc[:, range(4) + [4 + i]].to_csv('{}.{}.annot.gz'.format(args.out, gset_name), sep='\t', index=False, compression='gzip')
            expscore.iloc[:, range(3) + [3 + i]].to_csv('{}.{}.l2.ldscore.gz'.format(args.out, gset_name), sep='\t', index=False,
                        float_format='%.5f', compression='gzip')

    else:
        np.savetxt('{}.M_5_50'.format(args.out), M_5_50.reshape((1, len(M_5_50))), fmt='%d')
        bim.to_csv('{}.annot.gz'.format(args.out), sep='\t', index=False, compression='gzip')
        expscore.to_csv('{}.l2.ldscore.gz'.format(args.out), sep='\t', index=False,
                        float_format='%.5f', compression='gzip')


parser = argparse.ArgumentParser()
parser.add_argument('--make-kb-window', default=100, type=int,
                    help='Generate SNP annotation corresponding to x kb window around genes in gene set, '
                         'where x is the input value. Default is 100')
parser.add_argument('--gene-coords', default=os.path.join(dirname, 'data/gene_coords.txt'), type=str,
                    help='File containing the gene locations of files. '
                         'By default, we supply this file in data/gene_locations.txt')
parser.add_argument('--bfile', default=None, type=str,
                    help='Genotypes that are ancestry-matched to GWAS samples')
parser.add_argument('--gene-sets', default=None, type=str,
                    help='File containing gene sets. One gene set per line. First column is gene set name, remaining '
                         'columns are gene names.')
parser.add_argument('--keep', default=os.path.join(dirname, 'data/hm3_snps.txt'), type=str,
                    help='File with SNPs to include in expression score estimation. By default will include Hapmap3 SNPs.'
                         'The file should contain one SNP ID per row.')
parser.add_argument('--out', default=None, type=str,
                    help='Output prefix')
parser.add_argument('--split-output', default=False, action='store_true',
                    help='Output each gene set in its own file.')
parser.add_argument('--gset-start', default=None, type=int,
                    help='(Optional) which line to start reading gene sets in gene set file.')
parser.add_argument('--gset-end', default=None, type=int,
                    help='(Optional) which line to end reading gene sets in gene set file.')

if __name__ == '__main__':

    args = parser.parse_args()
    if not args.bfile:
        raise ValueError('Must specify --bfile')
    if not args.gene_sets:
        raise ValueError('Must specify --gene-sets')
    if not args.out:
        raise ValueError('Must specify --out')
    create_window_ldsc(args)
