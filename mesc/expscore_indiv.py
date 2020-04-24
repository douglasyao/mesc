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

def file_len(fname, input_chr, chr_idx):
    '''
    Get number of genes in gene expression file on input chromosome
    '''
    count = 0
    with open(fname) as f:
        for i, l in enumerate(f):
            if i == 0:
                continue
            l = l.split()
            try:
                chr = int(l[chr_idx])
            except:
                continue
            if chr == input_chr:
                count += 1
    return count

def get_eqtl_annot(args, gene_name, phenos, start_bp, end_bp, geno_fname, sample_names, chr):
    '''
    Create cis-region genotype file, estimate eQTL effects sizes using LASSO, and estimate expression cis-heritability using REML
    :return herit: REML-estimated h2cis
    :return herit_se: h2cis SE
    :return herit_p: h2cis p-value
    :return lasso: eQTL effect size estimates from LASSO
    '''

    FNULL = open(os.devnull, 'w')
    pheno_fname = '{}/{}.pheno'.format(args.tmp, gene_name)
    temp_geno_fname = '{}/{}'.format(args.tmp, gene_name)

    # making temporary phenotype file
    temp_pheno = pd.DataFrame([sample_names, sample_names, phenos]).T
    temp_pheno.to_csv(pheno_fname, sep='\t', index=False, header=False)

    # create cis-region genotype file
    try:
        subprocess.check_output(
            [args.plink_path, '--bfile', geno_fname, '--pheno', pheno_fname, '--make-bed', '--out', temp_geno_fname,
             '--chr', str(chr), '--from-bp', str(start_bp), '--to-bp', str(end_bp), '--extract', args.keep, '--silent',
             '--allow-no-sex'], stderr=FNULL)

    except subprocess.CalledProcessError:
        subprocess.Popen('rm {}*'.format(temp_geno_fname), shell=True)
        return 'NO_PLINK'

    # make grm (for REML)
    subprocess.call(
        [args.plink_path, '--bfile', temp_geno_fname, '--allow-no-sex', '--make-grm-bin', '--out', temp_geno_fname, '--silent'])

    # estimate h2cis using REML
    subprocess.call(
        [args.gcta_path, '--grm', temp_geno_fname, '--pheno', pheno_fname, '--out', temp_geno_fname,
         '--reml', '--reml-no-constrain', '--reml-lrt', '1'], stdout=FNULL, stderr=FNULL)

    hsq_fname = '{}.hsq'.format(temp_geno_fname)

    if os.path.exists(hsq_fname):
        hsq = pd.read_csv(hsq_fname, sep='\t')
        herit = hsq.iloc[3, 1]
        herit_se = hsq.iloc[3, 2]
        herit_p = hsq.iloc[8, 1]
    else:
        herit = np.nan
        herit_se = np.nan
        herit_p = np.nan

    if np.isnan(herit) or herit < 0:
        print('Warning: h2cis < 0 or not converged. Gene will be excluded from expression score calculation.')
        # still estimate LASSO effs for genes w/ h2cis < 0 for purpose of --save-estimates flag

    # set LASSO tuning parameter to estimated herit, otherwise 0.05
    if not np.isnan(herit):
        tune = herit
    else:
        tune = 0.05

    # estimate causal eQTL effect sizes using LASSO
    subprocess.call(
        [args.plink_path, '--allow-no-sex', '--bfile', temp_geno_fname, '--lasso', str(tune),
         '--out', temp_geno_fname, '--silent'], stdout=FNULL, stderr=FNULL)

    if os.path.exists('{}.lasso'.format(temp_geno_fname)):
        lasso = pd.read_csv('{}.lasso'.format(temp_geno_fname), sep='\t')
    else:
        print('Skipping; LASSO did not converge')
        subprocess.Popen('rm {}*'.format(temp_geno_fname), shell=True)
        return (herit, herit_se, herit_p, np.nan)

    lasso_weights = lasso['EFFECT'].values
    emp_herit = np.sum(np.square(lasso_weights))
    if not np.isnan(herit):
        if herit <= 0 or emp_herit <= 0:
            bias = 0
        else:
            bias = np.sqrt(herit / emp_herit)
        lasso['CORR_EFFECT'] = lasso_weights * bias
    else:
        lasso['CORR_EFFECT'] = np.nan

    lasso['GENE'] = gene_name
    if lasso.shape[0] == 0:
        lasso = np.nan
    out = (herit, herit_se, herit_p, lasso)
    subprocess.Popen('rm {}*'.format(temp_geno_fname), shell=True)
    return out

def get_expression_scores(args):
    expmat = args.expression_matrix
    if args.columns:
        columns = args.columns.split(',')
        columns = [int(x)-1 for x in columns]
        if len(columns) != 4:
            raise ValueError('Must specify 4 column indices with --columns')
    else:
        columns = range(4)
    n_genes = file_len(expmat, args.chr, columns[1])
    gene_num = 0
    keep_snps = pd.read_csv(args.keep, header=None)

    print('Analyzing chromosome {}'.format(args.chr))
    all_lasso = []
    all_herit = []

    geno_fname = args.bfile
    bim = pd.read_csv(geno_fname + '.bim', sep='\t', header=None)
    bim = bim.loc[bim[1].isin(keep_snps[0]),0:3]
    bim.columns = ['CHR', 'SNP', 'CM', 'BP']

    if args.covariates:
        covar = pd.read_csv(args.covariates, delim_whitespace=True)

    with open(expmat) as f:

        # compute h2cis and estimate LASSO effect sizes for all genes
        for j, line in enumerate(f):
            line = line.split()
            if j == 0:
                sample_names = line[columns[3]:]
            gene = line[columns[0]]
            try:
                chr = int(line[columns[1]])
            except:
                continue
            if chr != args.chr:
                continue
            start_bp = max(1, int(line[columns[2]]) - 5e5)
            end_bp = int(line[columns[2]]) + 5e5
            phenos = line[columns[3]:]

            # regress out covariates
            if args.covariates:
                phenos = np.array([float(x) for x in phenos])
                res = np.linalg.lstsq(covar.iloc[:,2:].values, phenos, rcond=None)
                phenos -= np.dot(covar.iloc[:,2:].values, res[0])
                phenos = phenos.tolist()

            gene_num += 1
            print('Estimating eQTL effect sizes for gene {} of {}: {}'.format(gene_num, n_genes, gene))
            herit = get_eqtl_annot(args, gene, phenos, start_bp, end_bp, geno_fname, sample_names, chr)

            if herit == 'NO_PLINK':
                print('Skipping; no SNPS around gene')
            else:
                all_herit.append([gene, args.chr, herit[0], herit[1], herit[2]])
                if isinstance(herit[3], pd.DataFrame):
                    all_lasso.append((gene, herit[0], herit[3]))

        # load genotypes
        array_indivs = ps.PlinkFAMFile(geno_fname + '.fam')
        array_snps = ps.PlinkBIMFile(geno_fname + '.bim')
        keep_snps_indices = np.where(array_snps.df['SNP'].isin(keep_snps[0]))[0]
        keep_indiv_indices = np.where(array_indivs.df['IID'].isin(sample_names))[0]
        with Suppressor():
            geno_array = ld.PlinkBEDFile(geno_fname + '.bed', array_indivs.n, array_snps,
                                         keep_indivs=keep_indiv_indices,
                                         keep_snps=keep_snps_indices)

        # SNP indices for fast merging
        snp_indices = dict(zip(geno_array.df[:, 1].tolist(), range(len(geno_array.df))))

        # exclude genes with h2cis < 0 or not converged
        all_lasso_temp = [x for x in all_lasso if not np.isnan(x[1])]
        all_lasso_temp = [x for x in all_lasso_temp if x[1] > 0]

        # create gene annotation files
        lasso_herits = [x[1] for x in all_lasso_temp]
        g_annot = np.zeros((len(all_lasso_temp), 5), dtype=int)
        eqtl_annot = np.zeros((len(geno_array.df), 5))
        gene_bins = pd.qcut(np.array(lasso_herits), 5, labels=range(5)).astype(int)
        g_bin_names = ['Cis_herit_bin_{}'.format(x) for x in range(1, 6)]
        for j in range(0, len(all_lasso_temp)):
            g_annot[j, gene_bins[j]] = 1
            snp_idx = [snp_indices[x] for x in all_lasso_temp[j][2]['SNP'].tolist()]
            eqtl_annot[snp_idx, gene_bins[j]] += np.square(all_lasso_temp[j][2]['CORR_EFFECT'].values)

        # estimate expression scores
        if not args.skip_expscore_estimation:
            print('Computing expression scores'.format(args.chr))

            block_left = ld.getBlockLefts(geno_array.df[:,2], 1e6)

            # estimate expression scores
            res = geno_array.ldScoreVarBlocks(block_left, c=50, annot=eqtl_annot)
            expscore = pd.DataFrame(np.c_[geno_array.df[:, :3], res])
            expscore.columns = geno_array.colnames[:3] + g_bin_names

            for name in g_bin_names:
                expscore[name] = expscore[name].astype(float)

            expscore.to_csv('{}.{}.expscore.gz'.format(args.out, args.chr), sep='\t', index=False, compression='gzip',
                            float_format='%.5f')

        g_annot_final = pd.DataFrame(np.c_[[x[0] for x in all_lasso_temp], g_annot])
        g_annot_final.columns = ['Gene'] + g_bin_names
        g_annot_final.to_csv('{}.{}.gannot.gz'.format(args.out, args.chr), sep='\t', index=False, compression='gzip')

        all_herit = pd.DataFrame.from_records(all_herit, columns=['Gene','Chrom','h2cis','h2cis_se','h2cis_p'])
        matched_herit = all_herit.loc[all_herit['Gene'].isin(g_annot_final['Gene']),'h2cis'].values
        G = np.sum(g_annot, axis=0)
        ave_cis_herit = np.dot(matched_herit, g_annot) / G

        np.savetxt('{}.{}.G'.format(args.out, args.chr), G.reshape((1, len(G))), fmt='%d')
        np.savetxt('{}.{}.ave_h2cis'.format(args.out, args.chr), ave_cis_herit.reshape((1, len(ave_cis_herit))), fmt="%.5f")
        all_herit.to_csv('{}.{}.hsq'.format(args.out, args.chr), sep='\t', index=False, float_format='%.5f')

        if args.save_estimates:
            lasso_out = pd.concat([x[2] for x in all_lasso])
            lasso_out = lasso_out[['GENE', 'CHR', 'SNP', 'EFFECT']]
            lasso_out.to_csv('{}.{}.lasso.gz'.format(args.out, args.chr), sep='\t', index=False, compression='gzip', float_format='%.8f')

        print('Done chromosome {}'.format(args.chr))
        print('All done!')

