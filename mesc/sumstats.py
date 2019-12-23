'''
(c) 2014 Brendan Bulik-Sullivan and Hilary Finucane
Modified by Douglas Yao

This module deals with getting all the data needed for MESC from files
into memory and checking that the input makes sense. There is no math here. MESC is implemented in the regressions module.

'''
from __future__ import division
import numpy as np
import pandas as pd
import itertools as it
import parse as ps
import regressions_mesc as reg
import copy
import os
import re


_N_CHR = 22
# complementary bases
COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
# bases
BASES = COMPLEMENT.keys()
# true iff strand ambiguous
STRAND_AMBIGUOUS = {''.join(x): x[0] == COMPLEMENT[x[1]]
                    for x in it.product(BASES, BASES)
                    if x[0] != x[1]}
# SNPS we want to keep (pairs of alleles)
VALID_SNPS = {x for x in map(lambda y: ''.join(y), it.product(BASES, BASES))
              if x[0] != x[1] and not STRAND_AMBIGUOUS[x]}
# T iff SNP 1 has the same alleles as SNP 2 (allowing for strand or ref allele flip).
MATCH_ALLELES = {x for x in map(lambda y: ''.join(y), it.product(VALID_SNPS, VALID_SNPS))
                 # strand and ref match
                 if ((x[0] == x[2]) and (x[1] == x[3])) or
                 # ref match, strand flip
                 ((x[0] == COMPLEMENT[x[2]]) and (x[1] == COMPLEMENT[x[3]])) or
                 # ref flip, strand match
                 ((x[0] == x[3]) and (x[1] == x[2])) or
                 ((x[0] == COMPLEMENT[x[3]]) and (x[1] == COMPLEMENT[x[2]]))}  # strand and ref flip
# T iff SNP 1 has the same alleles as SNP 2 w/ ref allele flip.
FLIP_ALLELES = {''.join(x):
                ((x[0] == x[3]) and (x[1] == x[2])) or  # strand match
                # strand flip
                ((x[0] == COMPLEMENT[x[3]]) and (x[1] == COMPLEMENT[x[2]]))
                for x in MATCH_ALLELES}


def _splitp(fstr):
    flist = fstr.split(',')
    flist = [os.path.expanduser(os.path.expandvars(x)) for x in flist]
    return flist


def _select_and_log(x, ii, log, msg):
    '''Fiter down to rows that are True in ii. Log # of SNPs removed.'''
    new_len = ii.sum()
    if new_len == 0:
        raise ValueError(msg.format(N=0))
    else:
        x = x[ii]
        log.log(msg.format(N=new_len))
    return x


def smart_merge(x, y):
    '''Check if SNP columns are equal. If so, save time by using concat instead of merge.'''
    if len(x) == len(y) and (x.index == y.index).all() and (x.SNP == y.SNP).all():
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True).drop('SNP', 1)
        out = pd.concat([x, y], axis=1)
    else:
        out = pd.merge(x, y, how='inner', on='SNP')
    return out


def _read_ref_ld(args, log):
    '''Read reference LD Scores.'''
    ref_ld, indices, _ = _read_chr_split_files(args.ref_ld_chr, args.ref_ld, log,
                                   'LD Score', ps.ldscore_fromlist, args=args, suffix='ldscore')
    log.log(
        'Read LD Scores for {N} SNPs.'.format(N=len(ref_ld)))
    return ref_ld, indices

def _read_g_ld(args, log):
    g_ld, indices, groups = _read_chr_split_files(args.exp_chr, args.exp, log,
                                 'expression score', ps.ldscore_fromlist, args=args, suffix='expscore')
    log.log(
        'Read expression scores for {N} SNPs.'.format(N=len(g_ld)))
    return g_ld, indices, groups

def _read_annot(args, log, cnames):
    '''Read annot matrix.'''
    try:
        if args.ref_ld is not None:
            overlap_matrix, M_tot = _read_chr_split_files(args.ref_ld_chr, args.ref_ld, log,
                                                          'annot matrix', ps.annot, cnames=cnames, frqfile=args.frqfile)
        elif args.ref_ld_chr is not None:
            overlap_matrix, M_tot = _read_chr_split_files(args.ref_ld_chr, args.ref_ld, log,
                                                      'annot matrix', ps.annot, cnames=cnames, frqfile=args.frqfile_chr)
    except Exception:
        log.log('Error parsing .annot file.')
        raise

    return overlap_matrix, M_tot

def _read_g_annot(args, log, cnames):
    try:
        if args.exp is not None or args.exp_chr is not None:
            overlap_matrix, G_tot = _read_chr_split_files(args.exp_chr, args.exp, log,
                                                          'annot matrix', ps.g_annot, cnames=cnames)
    except Exception:
        log.log('Error parsing .annot file.')
        raise

    return overlap_matrix, G_tot


def _read_M(args, log, n_annot, ref_indices):
    '''Read M (--M, --M-file, etc).'''
    if args.ref_ld:
        M_annot = ps.M_fromlist(
            _splitp(args.ref_ld), ref_indices, common=True)
    elif args.ref_ld_chr:
        M_annot = ps.M_fromlist(
            _splitp(args.ref_ld_chr), ref_indices, num=_N_CHR, common=True)
    try:
        M_annot = np.array(M_annot).reshape((1, n_annot))
    except ValueError as e:
        raise ValueError(
            '# terms in --M must match # of LD Scores in --ref-ld.\n' + str(e.args))

    return M_annot

def _read_G_and_ave_h2_cis(args, log, g_indices):

    if args.exp:
        G_annot, h2_cis_annot = ps.G_and_ave_h2_cis_fromlist(_splitp(args.exp), g_indices)
    elif args.exp_chr:
        G_annot, h2_cis_annot = ps.G_and_ave_h2_cis_fromlist(_splitp(args.exp_chr), g_indices, num=_N_CHR)

    return G_annot, h2_cis_annot


def _read_w_ld(args, log):
    '''Read regression SNP LD.'''
    if (args.w_ld and ',' in args.w_ld) or (args.w_ld_chr and ',' in args.w_ld_chr):
        raise ValueError(
            '--w-ld must point to a single fileset (no commas allowed).')
    w_ld, _, _ = _read_chr_split_files(args.w_ld_chr, args.w_ld, log,
                                 'regression weight LD Score', ps.ldscore_fromlist, args=args, suffix='weight')
    if len(w_ld.columns) != 2:
        raise ValueError('--w-ld may only have one LD Score column.')
    w_ld.columns = ['SNP', 'LD_weights']  # prevent colname conflicts w/ ref ld
    log.log(
        'Read regression weight LD Scores for {N} SNPs.'.format(N=len(w_ld)))
    return w_ld


def _read_chr_split_files(chr_arg, not_chr_arg, log, noun, parsefunc, **kwargs):
    '''Read files split across 22 chromosomes (annot, ref_ld, w_ld).'''
    try:
        if not_chr_arg:
            log.log('Reading {N} from {F} ...'.format(F=not_chr_arg, N=noun))
            out = parsefunc(_splitp(not_chr_arg), **kwargs)
        elif chr_arg:
            f = ps.sub_chr(chr_arg, '[1-22]')
            log.log('Reading {N} from {F} ...'.format(F=f, N=noun))
            out = parsefunc(_splitp(chr_arg), num=_N_CHR, **kwargs)
    except ValueError as e:
        log.log('Error parsing {N}.'.format(N=noun))
        raise e

    return out


def _read_sumstats(args, log, fh, alleles=False, dropna=False):
    '''Parse summary statistics.'''
    log.log('Reading summary statistics from {S} ...'.format(S=fh))
    sumstats = ps.sumstats(fh, alleles=alleles, dropna=dropna)
    log_msg = 'Read summary statistics for {N} SNPs.'
    log.log(log_msg.format(N=len(sumstats)))
    m = len(sumstats)
    sumstats = sumstats.drop_duplicates(subset='SNP')
    if m > len(sumstats):
        log.log(
            'Dropped {M} SNPs with duplicated rs numbers.'.format(M=m - len(sumstats)))

    return sumstats


def _check_ld_condnum(args, log, ref_ld):
    '''Check condition number of LD Score matrix.'''
    if len(ref_ld.shape) >= 2:
        cond_num = int(np.linalg.cond(ref_ld))
        if cond_num > 100000:
            if args.invert_anyway:
                warn = "WARNING: LD Score matrix condition number is {C}. "
                warn += "Inverting anyway because the --invert-anyway flag is set."
                log.log(warn.format(C=cond_num))
            else:
                warn = "WARNING: LD Score matrix condition number is {C}. "
                warn += "Remove collinear LD Scores. "
                raise ValueError(warn.format(C=cond_num))


def _check_variance(log, M_annot, ref_ld):
    '''Remove zero-variance LD Scores.'''
    ii = ref_ld.ix[:, 1:].var() == 0  # NB there is a SNP column here
    if ii.all():
        raise ValueError('All LD Scores have zero variance.')
    else:
        log.log('Removing partitioned LD Scores with zero variance.')
        ii_snp = np.array([True] + list(~ii))
        ii_m = np.array(~ii)
        ref_ld = ref_ld.ix[:, ii_snp]
        M_annot = M_annot[:, ii_m]

    return M_annot, ref_ld, ii


def _warn_length(log, sumstats):
    if len(sumstats) < 200000:
        log.log(
            'WARNING: number of SNPs less than 200k; this is almost always bad.')

def _merge_and_log(ld, sumstats, noun, log):
    '''Wrap smart merge with log messages about # of SNPs.'''
    sumstats = smart_merge(ld, sumstats)
    msg = 'After merging GWAS summary statistics with {F}, {N} SNPs remain.'
    if len(sumstats) == 0:
        raise ValueError(msg.format(N=len(sumstats), F=noun))
    else:
        log.log(msg.format(N=len(sumstats), F=noun))

    return sumstats


def _read_ld_sumstats(args, log, fh, alleles=False, dropna=True):
    sumstats = _read_sumstats(args, log, fh, alleles=alleles, dropna=dropna)
    ref_ld, ref_indices = _read_ref_ld(args, log)
    g_ld, g_indices, g_groups = _read_g_ld(args, log)

    n_annot = len(ref_ld.columns) - 1
    n_g_annot = len(g_ld.columns) - 1
    M_annot = _read_M(args, log, n_annot, ref_indices)
    G_annot, ave_h2_cis_annot = _read_G_and_ave_h2_cis(args, log, g_indices)

    M_annot, ref_ld, novar_cols = _check_variance(log, M_annot, ref_ld)
    w_ld = _read_w_ld(args, log)
    sumstats = _merge_and_log(g_ld, sumstats, 'expression scores', log)
    sumstats = _merge_and_log(ref_ld, sumstats, 'LD scores', log)
    sumstats = _merge_and_log(sumstats, w_ld, 'LD weights', log)
    w_ld_cname = sumstats.columns[-1]
    ref_ld_cnames = ref_ld.columns[1:len(ref_ld.columns)]
    g_ld_cnames = g_ld.columns[1:len(g_ld.columns)]

    return M_annot, G_annot, ave_h2_cis_annot, w_ld_cname, ref_ld_cnames, g_ld_cnames, sumstats, novar_cols, n_annot, n_g_annot, ref_indices, g_indices, g_groups


def array_to_string(x):
    '''Get rid of brackets and trailing whitespace in numpy arrays.'''
    return ' '.join(map(str, x)).replace('[', '').replace(']', '').strip()

def estimate_h2med(args, log):
    '''Estimate h2med and partitioned h2med.'''
    args = copy.deepcopy(args)

    M_annot, G_annot, ave_h2_cis_annot, w_ld_cname, ref_ld_cnames, g_ld_cnames, sumstats, novar_cols, n_annot, n_g_annot, ref_ld_indices, g_ld_indices, g_groups = _read_ld_sumstats(
        args, log, args.h2med)

    ref_ld = np.array(sumstats[ref_ld_cnames])
    g_ld = np.array(sumstats[g_ld_cnames])
    _check_ld_condnum(args, log, ref_ld_cnames)
    _warn_length(log, sumstats)
    n_snp = len(sumstats)
    n_blocks = min(n_snp, 200)
    chisq_max = args.chisq_max

    if args.chisq_max is None:
        chisq_max = max(0.001*sumstats.N.max(), 80)

    s = lambda x: np.array(x).reshape((n_snp, 1))
    chisq = s(sumstats.Z**2)
    if chisq_max is not None:
        ii = np.ravel(chisq < chisq_max)
        sumstats = sumstats.ix[ii, :]
        log.log('Removed {M} SNPs with chi^2 > {C} ({N} SNPs remain)'.format(
                C=chisq_max, N=np.sum(ii), M=n_snp-np.sum(ii)))
        n_snp = np.sum(ii)  # lambdas are late-binding, so this works
        ref_ld = np.array(sumstats[ref_ld_cnames])
        g_ld = np.array(sumstats[g_ld_cnames])
        chisq = chisq[ii].reshape((n_snp, 1))

    g_list = np.array(g_groups)
    g_groups = pd.Series(g_groups, dtype='category')
    cis_indices = [i for i, x in enumerate(g_groups == 'Cis_herit_bin') if x]
    cis_g_ld_cnames = g_ld_cnames[cis_indices]
    cis_G = G_annot[:, cis_indices]
    cis_ave_h2_cis = ave_h2_cis_annot[:, cis_indices]
    cis_g_ld = g_ld[:, cis_indices]
    cis_hsqhat = reg.H2med(chisq, ref_ld, cis_g_ld, s(sumstats[w_ld_cname]), s(sumstats.N),
                     M_annot, cis_G, cis_ave_h2_cis, n_blocks=n_blocks)
    cis_groups = g_list[cis_indices]
    g_annot_cnames = [[y + 2 for y in x] for x in g_ld_indices]
    g_overlap_matrix, G_tot = _read_g_annot(args, log, g_annot_cnames)
    cis_overlap_matrix = g_overlap_matrix[np.ix_(cis_indices, cis_indices)]
    cis_results = cis_hsqhat._g_overlap_output(cis_g_ld_cnames, cis_overlap_matrix, cis_G, G_tot, cis_ave_h2_cis, cis_groups)
    cis_results['Gene_category'] = ['h2cis_bin_{}'.format(i+1) for i in range(cis_results.shape[0])]

    for i in g_groups.cat.categories:
        if i == 'Cis_herit_bin':
            continue
        temp_indices = [j for j, x in enumerate(g_groups == i) if x]
        temp_G = G_annot[:, cis_indices + temp_indices]
        temp_ave_h2_cis = ave_h2_cis_annot[:, cis_indices + temp_indices]
        temp_g_ld = g_ld[:, cis_indices + temp_indices]
        temp_g_ld_cnames = g_ld_cnames[cis_indices + temp_indices]
        temp_groups = g_list[cis_indices + temp_indices]
        temp_g_overlap_matrix = g_overlap_matrix[np.ix_(cis_indices + temp_indices, cis_indices + temp_indices)]

        hsqhat = reg.H2med(chisq, ref_ld, temp_g_ld, s(sumstats[w_ld_cname]), s(sumstats.N),
                     M_annot, temp_G, temp_ave_h2_cis, n_blocks=n_blocks)

        g_results = hsqhat._g_overlap_output(temp_g_ld_cnames, temp_g_overlap_matrix, temp_G, G_tot, temp_ave_h2_cis, temp_groups)
        g_results = g_results.iloc[-1:,:]
        g_results['Gene_category'] = re.sub('_Cis_herit_bin', '', g_results['Gene_category'].values[0])
        cis_results = cis_results.append(g_results)

    df_all = pd.DataFrame({
        'Quantity': ['h2med', 'h2nonmed', 'h2'],
        'Estimate': np.array([cis_hsqhat.tot_g, cis_hsqhat.tot_dir, cis_hsqhat.tot]),
        'SE(Estimate)': np.array([cis_hsqhat.tot_g_se, cis_hsqhat.tot_dir_se, cis_hsqhat.tot_se]),
        'Estimate_over_h2': np.array([cis_hsqhat.tot_g_prop, cis_hsqhat.tot_dir_prop, 1]),
        'SE(Estimate_over_h2)': np.array([cis_hsqhat.tot_g_prop_se, cis_hsqhat.tot_dir_prop_se, 0])
    }, columns=['Quantity', 'Estimate', 'SE(Estimate)', 'Estimate_over_h2', 'SE(Estimate_over_h2)'])

    cis_results = cis_results.fillna('NA')
    df_all = df_all.fillna('NA')

    cis_results.to_csv(args.out + '.categories.h2med', sep='\t', index=False)
    df_all.to_csv(args.out + '.all.h2med', sep='\t', index=False)

def _filter_alleles(alleles):
    '''Remove bad variants (mismatched alleles, non-SNPs, strand ambiguous).'''
    ii = alleles.apply(lambda y: y in MATCH_ALLELES)
    return ii


def _check_arg_len(x, n):
    x, m = x
    if len(x) != n:
        raise ValueError(
            '{M} must have the same number of arguments as --rg/--h2.'.format(M=m))
