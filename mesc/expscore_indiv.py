'''
Compute expression scores and estimate expression cis-heritability from individual-level expression and genotype data
'''

from __future__ import division
import numpy as np
import pandas as pd
import os
import argparse
import collections
import subprocess
import ldscore as ld
import parse as ps
import sys

class Suppressor(object):

    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = self

    def __exit__(self, type, value, traceback):
        sys.stdout = self.stdout
        if type is not None:
            pass

    def write(self, x): pass

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def flatten_list(x):
    if isinstance(x, collections.Iterable) and not isinstance(x, basestring):
        return [a for i in x for a in flatten_list(i)]
    else:
        return [x]

def sub_chr(s, chr):
    '''Substitute chr for @, else append chr to the end of str.'''
    if '@' not in s:
        s += '@'

    return s.replace('@', str(chr))

def file_len(fname):
    each_chrom = np.zeros(22, dtype=int)
    with open(fname) as f:
        for i, l in enumerate(f):
            if i == 0:
                continue
            l = l.split()
            chr = int(l[2])
            each_chrom[chr-1] += 1
    return each_chrom

def match(a, b):
    return [b.index(x) if x in b else None for x in a]

def get_eqtl_annot(args, gene_name, phenos, start_bp, end_bp, geno_fname, sample_names, chr):
    FNULL = open(os.devnull, 'w')
    pheno_fname = '{}/{}.pheno'.format(args.tmp, gene_name)
    temp_geno_fname = '{}/{}'.format(args.tmp, gene_name)

    temp_pheno = pd.DataFrame([sample_names, sample_names, phenos]).T
    temp_pheno.to_csv(pheno_fname, sep='\t', index=False, header=False)
    try:
        subprocess.check_output(
            [args.plink_path, '--bfile', geno_fname, '--pheno', pheno_fname, '--make-bed', '--out', temp_geno_fname,
             '--chr', chr, '--from-bp', str(start_bp), '--to-bp', str(end_bp), '--extract', args.keep, '--silent',
             '--allow-no-sex'])
    except subprocess.CalledProcessError:
        return 'NO_PLINK'
    subprocess.call(
        [args.plink_path, '--bfile', geno_fname, '--pheno', pheno_fname, '--make-bed', '--out', temp_geno_fname,
         '--chr', chr, '--from-bp', str(start_bp), '--to-bp', str(end_bp), '--extract', args.keep, '--silent',
         '--allow-no-sex'])
    subprocess.call(
        [args.plink_path, '--bfile', temp_geno_fname, '--allow-no-sex', '--make-grm-bin', '--out', temp_geno_fname, '--silent'])

    if args.covariates:
        subprocess.call(
            [args.gcta_path, '--grm', temp_geno_fname, '--pheno', pheno_fname, '--out', temp_geno_fname, '--reml',
             '--reml-no-constrain', '--qcovar', args.covariates, '--reml-lrt', '1'], stdout=FNULL)
    else:
        subprocess.call(
            [args.gcta_path, '--grm', temp_geno_fname, '--pheno', pheno_fname, '--out', temp_geno_fname,
             '--reml', '--reml-no-constrain', '--reml-lrt', '1'], stdout=FNULL)

    hsq_fname = '{}.hsq'.format(temp_geno_fname)

    if os.path.exists(hsq_fname):
        hsq = pd.read_csv(hsq_fname, sep='\t')
        herit = hsq.iloc[3, 1]
        herit_se = hsq.iloc[3, 2]
        herit_p = hsq.iloc[8, 1]

        if herit < 0:
            out = (herit, herit_se, herit_p, np.nan)

        else:
            if args.covariates:
                subprocess.call(
                    [args.plink_path, '--allow-no-sex', '--bfile', temp_geno_fname, '--lasso', str(hsq),
                     '--covar', args.covariates, '--out', temp_geno_fname, '--silent'])
            else:
                subprocess.call(
                    [args.plink_path, '--allow-no-sex', '--bfile', temp_geno_fname, '--lasso', str(herit),
                     '--out', temp_geno_fname, '--silent'])

            lasso = pd.read_csv('{}.lasso'.format(temp_geno_fname), sep='\t')
            lasso_weights = lasso['EFFECT'].values

            if len(lasso_weights) == 0:
                print('Skipping; LASSO did not converge')
                out = (herit, herit_se, herit_p, np.nan)
            else:
                emp_herit = np.sum(np.square(lasso_weights))
                bias = np.sqrt(herit / emp_herit)
                lasso_weights *= bias
                lasso['EFFECT'] = lasso_weights
                out = (herit, herit_se, herit_p, lasso)
    else:
        out = 'NO_GCTA'
    subprocess.Popen('rm {}*'.format(temp_geno_fname), shell=True)
    return out

def get_expression_scores_chr(args):
    expmat = args.expression_matrix
    n_genes = file_len(expmat)
    gene_num = 0
    keep_snps = pd.read_csv(args.keep, header=None)
    if args.columns:
        columns = args.columns.split(',')
        columns = [int(x)-1 for x in columns]
        if len(columns) != 5:
            raise ValueError('Must specify 5 column indices with --columns')
    else:
        columns = range(5)

    for i in range(1, 23):
        print('Analyzing chromosome {}'.format(i))
        all_lasso = []
        all_herit = []

        geno_fname = sub_chr(args.bfile_chr, i)

        bim = pd.read_csv(geno_fname + '.bim', sep='\t', header=None)
        bim = bim.loc[bim[1].isin(keep_snps[0]),0:3]
        bim.columns = ['CHR', 'SNP', 'CM', 'BP']
        chrom_gene_num = 0
        with open(expmat) as f:
            for j, line in enumerate(f):
                line = line.split()
                if j == 0:
                    sample_names = line[columns[4]:]
                    continue
                gene = line[columns[0]]
                gene_name = line[columns[1]]
                chr = line[columns[2]]
                start_bp = int(line[columns[3]]) - 5e5
                end_bp = int(line[columns[3]]) + 5e5
                phenos = line[columns[4]:]
                if int(chr) != i:
                    continue
                gene_num += 1
                chrom_gene_num += 1
                print('Estimating eQTL effect sizes for gene {} of {} ({} of {} total): {}'.format(chrom_gene_num, n_genes[i-1], gene_num, sum(n_genes), gene_name))
                herit = get_eqtl_annot(args, gene_name, phenos, start_bp, end_bp, geno_fname, sample_names, chr)
                if not herit:
                    print('Skipping; GCTA did not converge')
                    all_herit.append([gene, gene_name, i, np.nan, np.nan, np.nan])
                else:
                    all_herit.append([gene, gene_name, i, herit[0], herit[1], herit[2]])
                    if herit[0] > 0:
                        all_lasso.append((gene, gene_name, herit[0], herit[3]))
                    else:
                        print('Skipping; estimated h2cis < 0')

        if len(all_lasso) == 0:
            print('No genes on chromosome {} in expression file; continuing'.format(i))
            continue

        print('Computing expression scores for chromosome {}'.format(i))
        array_indivs = ps.PlinkFAMFile(geno_fname + '.fam')
        array_snps = ps.PlinkBIMFile(geno_fname + '.bim')
        keep_snps_indices = np.where(array_snps.df['SNP'].isin(keep_snps[0]))[0]
        keep_indiv_indices = np.where(array_indivs.df['IID'].isin(sample_names))[0]

        with Suppressor():
            geno_array = ld.PlinkBEDFile(geno_fname + '.bed', array_indivs.n, array_snps, keep_indivs=keep_indiv_indices,
                                     keep_snps=keep_snps_indices)

        lasso_herits = [x[2] for x in all_lasso]
        g_annot = np.zeros((len(all_lasso), 5), dtype=int)
        eqtl_annot = np.zeros((len(geno_array.df), 5))
        gene_bins = pd.qcut(np.array(lasso_herits), 5, labels=range(5)).astype(int)
        g_bin_names = ['Cis_herit_bin_{}'.format(x) for x in range(1, 6)]
        for j in range(0, len(all_lasso)):
            g_annot[j, gene_bins[j]] = 1
            snp_indices = match(all_lasso[j][3]['SNP'].tolist(), geno_array.df[:,1].tolist())
            eqtl_annot[snp_indices, gene_bins[j]] += np.square(all_lasso[j][3]['EFFECT'].values)

        block_left = ld.getBlockLefts(geno_array.df[:,2], 1e6)
        res = geno_array.ldScoreVarBlocks(block_left, c=50, annot=eqtl_annot)
        g_annot_final = pd.DataFrame(np.c_[[x[0] for x in all_lasso], [x[1] for x in all_lasso], g_annot])
        g_annot_final.columns = ['Gene', 'Gene_symbol'] + g_bin_names
        g_annot_final.to_csv('{}.{}.gannot.gz'.format(args.out, i), sep='\t', index=False, compression='gzip')

        all_herit = pd.DataFrame.from_records(all_herit, columns=['Gene','Gene_symbol','Chrom','h2cis','h2cis_se','h2cis_p'])
        matched_herit = all_herit.loc[all_herit['Gene'].isin(g_annot_final['Gene']),'h2cis'].values
        G = np.sum(g_annot, axis=0)
        ave_cis_herit = np.dot(matched_herit, g_annot) / G

        np.savetxt('{}.{}.G'.format(args.out, i), G.reshape((1, len(G))), fmt='%d')
        np.savetxt('{}.{}.ave_h2cis'.format(args.out, i), ave_cis_herit.reshape((1, len(ave_cis_herit))), fmt="%.5f")

        expscore = pd.DataFrame(np.c_[geno_array.df[:,:3], res])
        expscore.columns = geno_array.colnames[:3] + g_bin_names

        for name in g_bin_names:
            expscore[name] = expscore[name].astype(float)

        expscore.to_csv('{}.{}.expscore.gz'.format(args.out, i), sep='\t', index=False, compression='gzip', float_format='%.5f')

        all_herit.to_csv('{}.{}.hsq'.format(args.out, i), sep='\t', index=False, float_format='%.5f')
        print('Done chromosome {}'.format(i))
        if i == 22:
            print('All done!')

def get_expression_scores(args):
    expmat = args.expression_matrix
    n_genes = file_len(expmat)
    all_herit = []
    gene_num = 0
    keep_snps = pd.read_csv(args.keep, header=None)

    all_lasso = []
    geno_fname = args.bfile

    bim = pd.read_csv(geno_fname + '.bim', sep='\t', header=None)
    bim = bim.loc[bim[1].isin(keep_snps[0]), 0:3]
    bim.columns = ['CHR', 'SNP', 'CM', 'BP']

    if args.columns:
        columns = args.columns.split(',')
        columns = [int(x)-1 for x in columns]
        if len(columns) != 5:
            raise ValueError('Must specify 5 column indices with --columns')
    else:
        columns = range(5)

    with open(expmat) as f:
        for j, line in enumerate(f):
            line = line.split()
            if j == 0:
                sample_names = line[columns[4]:]
                continue
            gene = line[columns[0]]
            gene_name = line[columns[1]]
            chr = line[columns[2]]
            start_bp = int(line[columns[3]]) - 5e5
            end_bp = int(line[columns[3]]) + 5e5
            phenos = line[columns[4]:]

            gene_num += 1
            print('Estimating eQTL effect sizes for gene {} of {}: {}'.format(gene_num, sum(n_genes), gene_name))
            herit = get_eqtl_annot(args, gene_name, phenos, start_bp, end_bp, geno_fname, sample_names, chr)
            if herit == 'NO_PLINK':
                print('Skipping; Genotype file does not contain any SNPs around gene (probably wrong chromosome)')
            elif herit == 'NO_GCTA':
                print('Skipping; GCTA did not converge')
                all_herit.append([gene, gene_name, chr, np.nan, np.nan, np.nan])
            else:
                all_herit.append([gene, gene_name, chr, herit[0], herit[1], herit[2]])
                if herit[0] > 0:
                    all_lasso.append((gene, gene_name, herit[0], herit[3]))
                else:
                    print('Skipping; estimated h2cis < 0')

    if len(all_lasso) == 0:
        print('Error: No genes in expression file matching genotype file')
        exit(1)

    print('Computing expression scores')
    array_indivs = ps.PlinkFAMFile(geno_fname + '.fam')
    array_snps = ps.PlinkBIMFile(geno_fname + '.bim')
    keep_snps_indices = np.where(array_snps.df['SNP'].isin(keep_snps[0]))[0]
    keep_indiv_indices = np.where(array_indivs.df['IID'].isin(sample_names))[0]

    with Suppressor():
        geno_array = ld.PlinkBEDFile(geno_fname + '.bed', array_indivs.n, array_snps,
                                     keep_indivs=keep_indiv_indices,
                                     keep_snps=keep_snps_indices)

    lasso_herits = [x[2] for x in all_lasso]
    g_annot = np.zeros((len(all_lasso), 5), dtype=int)
    eqtl_annot = np.zeros((len(geno_array.df), 5))
    gene_bins = pd.qcut(np.array(lasso_herits), 5, labels=range(5)).astype(int)
    g_bin_names = ['Cis_herit_bin_{}'.format(x) for x in range(1, 6)]
    for j in range(0, len(all_lasso)):
        g_annot[j, gene_bins[j]] = 1
        snp_indices = match(all_lasso[j][3]['SNP'].tolist(), geno_array.df[:, 1].tolist())
        eqtl_annot[snp_indices, gene_bins[j]] += np.square(all_lasso[j][3]['EFFECT'].values)

    block_left = ld.getBlockLefts(geno_array.df[:, 2], 1e6)
    res = geno_array.ldScoreVarBlocks(block_left, c=50, annot=eqtl_annot)
    g_annot_final = pd.DataFrame(np.c_[[x[0] for x in all_lasso], [x[1] for x in all_lasso], g_annot])
    g_annot_final.columns = ['Gene', 'Gene_symbol'] + g_bin_names
    g_annot_final.to_csv('{}.gannot.gz'.format(args.out), sep='\t', index=False, compression='gzip')

    all_herit = pd.DataFrame.from_records(all_herit, columns=['Gene', 'Gene_symbol', 'Chrom', 'h2cis', 'h2cis_se', 'h2cis_p'])
    matched_herit = all_herit.loc[all_herit['Gene'].isin(g_annot_final['Gene']), 'h2cis'].values
    G = np.sum(g_annot, axis=0)
    ave_cis_herit = np.dot(matched_herit, g_annot) / G

    np.savetxt('{}.G'.format(args.out), G.reshape((1, len(G))), fmt='%d')
    np.savetxt('{}.ave_h2cis'.format(args.out), ave_cis_herit.reshape((1, len(ave_cis_herit))), fmt="%.5f")

    expscore = pd.DataFrame(np.c_[geno_array.df[:, :3], res])
    expscore.columns = geno_array.colnames[:3] + g_bin_names

    for name in g_bin_names:
        expscore[name] = expscore[name].astype(float)
    expscore.to_csv('{}.expscore.gz'.format(args.out), sep='\t', index=False, compression='gzip', float_format='%.5f')

    all_herit.to_csv('{}.hsq'.format(args.out), sep='\t', index=False, float_format='%.5f')
    print('All done!')
