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

    M = np.sum(bim.iloc[:,4:].values, axis=0)
    np.savetxt('{}.M_5_50'.format(args.out), M.reshape((1, len(M))), fmt='%d')

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

def create_window_ldsc_batch(args):
    '''
    Create SNP annotation corresponding to x kb window around genes in gene sets. Estimate LD scores using these annotations.
    Analyze in batches of gene sets
    '''
    gsets = read_gene_sets(args.gene_sets)
    gene_coords = pd.read_csv(args.gene_coords, sep='\t')

    # read SNPs
    keep_snps = pd.read_csv(args.keep, header=None)
    geno_fname = args.bfile
    bim = pd.read_csv(geno_fname + '.bim', sep='\t', header=None)  # load bim
    bim = bim.loc[bim[1].isin(keep_snps[0]).values, 0:3]
    bim.columns = ['CHR', 'SNP', 'CM', 'BP']

    # load genotype
    array_indivs = ps.PlinkFAMFile(geno_fname + '.fam')
    array_snps = ps.PlinkBIMFile(geno_fname + '.bim')
    keep_snps_indices = np.where(array_snps.df['SNP'].isin(keep_snps[0]).values)[0]

    geno_array = ld.PlinkBEDFile(geno_fname + '.bed', array_indivs.n, array_snps,
                                 keep_snps=keep_snps_indices)

    block_left = ld.getBlockLefts(geno_array.df[:, 2], 1e6)

    M = []

    # create gene window annot file
    count = 0
    for i in range(0, len(gsets), args.batch_size):
        count += 1
        print('Computing LD scores for gene sets {} to {} (out of {} total)'.format(i+1, min(i+args.batch_size, len(gsets)), len(gsets)))
        temp_gsets = OrderedDict(gsets.items()[i:(i+args.batch_size)])
        new_bim = pd.DataFrame()
        for gset, genes in temp_gsets.items():
            temp_genes = [x for x in genes if x in gene_coords.iloc[:, 2].tolist()]
            toadd_annot = np.zeros(len(bim), dtype=int)
            for gene in temp_genes:
                coord = gene_coords[gene_coords.iloc[:, 2] == gene]
                try:
                    chr = int(coord.iloc[0, 0])
                except:
                    continue
                start = int(coord.iloc[0, 1]) - 1000 * args.make_kb_window
                end = int(coord.iloc[0, 1]) + 1000 * args.make_kb_window
                toadd_annot[(bim['CHR'] == chr).values & (bim['BP'] > start).values & (bim['BP'] < end).values] = 1
            new_bim[gset] = toadd_annot

        # estimate LD scores
        res = geno_array.ldScoreVarBlocks(block_left, c=50, annot=new_bim.values)
        geno_array._currentSNP = 0
        M.extend(np.sum(new_bim, axis=0))

        if i == 0:
            new_bim = pd.concat([bim, new_bim], axis=1)
            new_bim.to_csv('{}.annot.batch{}'.format(args.out, count), sep='\t', index=False)
            expscore = pd.concat([
                pd.DataFrame(geno_array.df[:, :3]),
                pd.DataFrame(res)], axis=1)
            expscore.columns = geno_array.colnames[:3] + new_bim.columns[4:].tolist()
            expscore.to_csv('{}.l2.ldscore.batch{}'.format(args.out, count), sep='\t', index=False, float_format='%.5f')
        else:
            new_bim.to_csv('{}.annot.batch{}'.format(args.out, count), sep='\t', index=False)
            expscore = pd.DataFrame(res, columns=new_bim.columns)
            expscore.to_csv('{}.l2.ldscore.batch{}'.format(args.out, count), sep='\t', index=False, float_format='%.5f')


    subprocess.Popen('paste {} > {}'.format(
        ' '.join(['{}.annot.batch{}'.format(args.out, x) for x in range(1, count + 1)]),
        '{}.annot'.format(args.out)), shell=True)
    subprocess.Popen('paste {} > {}'.format(
        ' '.join(['{}.l2.ldscore.batch{}'.format(args.out, x) for x in range(1, count + 1)]),
        '{}.l2.ldscore'.format(args.out)), shell=True)
    subprocess.Popen(
        'rm {}'.format(' '.join(['{}.annot.batch{}'.format(args.out, x) for x in range(1, count + 1)])),
        shell=True)
    subprocess.Popen(
        'rm {}'.format(' '.join(['{}.l2.ldscore.batch{}'.format(args.out, x) for x in range(1, count + 1)])),
        shell=True)

    subprocess.Popen('gzip {}.l2.ldscore'.format(args.out), shell=True)
    subprocess.Popen('gzip {}.annot'.format(args.out), shell=True)
    np.savetxt('{}.M_5_50'.format(args.out), np.array(M).reshape((1, len(M))), fmt='%d')


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
parser.add_argument('--batch-size', default=None, type=int,
                    help='Analyze gene sets in batches of input size x. Useful to save memory if many gene sets are present.')

if __name__ == '__main__':

    args = parser.parse_args()
    if not args.bfile:
        raise ValueError('Must specify --bfile')
    if not args.gene_sets:
        raise ValueError('Must specify --gene-sets')
    if not args.out:
        raise ValueError('Must specify --out')
    if not args.batch_size:
        create_window_ldsc(args)
    else:
        create_window_ldsc_batch(args)