#!/usr/bin/env python
'''
Mediated Expression SCore Regression (MESC)
Adapted from LD Score Regression software (https://github.com/bulik/ldsc)
2019 Douglas Yao
'''

from __future__ import division
import mesc.expscore_indiv as ind
import mesc.expscore_sumstat as ss
import mesc.parse as ps
import mesc.sumstats as sumstats
import numpy as np
import pandas as pd
import subprocess
import time, sys, traceback, argparse
import os

dirname = os.path.dirname(__file__)


try:
    x = pd.DataFrame({'A': [1, 2, 3]})
    x.sort_values(by='A')
except AttributeError:
    raise ImportError('MESC requires pandas version >= 0.17.0')

MASTHEAD = "*********************************************************************\n"
MASTHEAD += "* Mediated Expression Score Regression (MESC)\n"
MASTHEAD += "* by Douglas Yao 2019 \n"
MASTHEAD += "*********************************************************************\n"
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('precision', 4)
pd.set_option('max_colwidth', 1000)
np.set_printoptions(linewidth=1000)
np.set_printoptions(precision=4)


def sec_to_str(t):
    '''Convert seconds to days:hours:minutes:seconds'''
    [d, h, m, s, n] = reduce(lambda ll, b: divmod(ll[0], b) + ll[1:], [(t, 1), 60, 60, 24])
    f = ''
    if d > 0:
        f += '{D}d:'.format(D=d)
    if h > 0:
        f += '{H}h:'.format(H=h)
    if m > 0:
        f += '{M}m:'.format(M=m)

    f += '{S}s'.format(S=s)
    return f


def _remove_dtype(x):
    '''Removes dtype: float64 and dtype: int64 from pandas printouts'''
    x = str(x)
    x = x.replace('\ndtype: int64', '')
    x = x.replace('\ndtype: float64', '')
    return x


class Logger(object):
    '''
    Lightweight logging.
    TODO: replace with logging module

    '''

    def __init__(self, fh):
        self.log_fh = open(fh, 'wb')

    def log(self, msg):
        '''
        Print to log file and stdout with a single command.

        '''
        print >> self.log_fh, msg
        print(msg)


def __filter__(fname, noun, verb, merge_obj):
    merged_list = None
    if fname:
        f = lambda x, n: x.format(noun=noun, verb=verb, fname=fname, num=n)
        x = ps.FilterFile(fname)
        c = 'Read list of {num} {noun} to {verb} from {fname}'
        print(f(c, len(x.IDList)))
        merged_list = merge_obj.loj(x.IDList)
        len_merged_list = len(merged_list)
        if len_merged_list > 0:
            c = 'After merging, {num} {noun} remain'
            print(f(c, len_merged_list))
        else:
            error_msg = 'No {noun} retained for analysis'
            raise ValueError(f(error_msg, 0))

        return merged_list


parser = argparse.ArgumentParser()
parser.add_argument('--out', default=None, type=str,
                    help='Output filename prefix.')

# Compute expression scores from individual level data
# Required flags
parser.add_argument('--compute-expscore-indiv', default=False, action='store_true',
                    help='Estimate expression scores from individual-level expression and genotype data')
parser.add_argument('--plink-path', default=None, type=str,
                    help='Path to plink')
parser.add_argument('--expression-matrix', default=None, type=str,
                    help='Gene by sample expression matrix')
parser.add_argument('--exp-bfile', default=None, type=str,
                    help='Sample genotypes to go along with gene expression matrix. Prefix for PLINK .bed/.bim/.fam file.'
                         'Can only analyze one chromosome at a time, so must be split by chromosome.')
parser.add_argument('--chr', default=None, type=int,
                    help='Which chromosome input genotypes lie on. Can only analyze one chromosome at a time')
parser.add_argument('--geno-bfile', default=None, type=str,
                    help='Genotypes used to compute expression scores, which should be ancestry-matched to GWAS samples.'
                         'We recommend using 1000 Genomes.')

# Optional flags
parser.add_argument('--covariates', default=None, type=str,
                    help='Optional gene expression covariates (in PLINK format)')
parser.add_argument('--gcta-path', default=os.path.join(dirname, 'gcta_nr_robust'), type=str,
                    help='Path to GCTA')
parser.add_argument('--tmp', default=os.path.join(dirname, 'tmp'), type=str,
                    help='Directory to store temporary files')
parser.add_argument('--keep', default=os.path.join(dirname, 'data/hm3_snps.txt'), type=str,
                    help='File with SNPs to include in expression score estimation. '
                    'The file should contain one SNP ID per row.')
parser.add_argument('--est-lasso-only', default=False, action='store_true',
                    help='Skip expression score estimation (only estimate eQTL effect sizes using LASSO and '
                         'h2cis using REML)')

# Compute expression scores from summary statistics
# Required flags
parser.add_argument('--compute-expscore-sumstat', default=False, action='store_true',
                    help='Estimate expression scores from eQTL summary statistics')
parser.add_argument('--eqtl-sumstat', default=None, type=str,
                    help='eQTL summary statistic data.')
# Optional flags
parser.add_argument('--columns', default=None, type=str,
                    help='List of indices separated by commas.')
parser.add_argument('--num-bins', default=5, type=int,
                    help='Number of expression cis-heritability bins. Default 5.')


# Estimate mediated heritability
# Required flags
parser.add_argument('--h2med', default=None, type=str,
                    help='Filename for a .sumstats[.gz] file for MESC. '
                         '--h2med requires at minimum also setting the --exp or --exp-chr.')
parser.add_argument('--exp', default=None, type=str,
                    help='Filename prefix for expression scores. MESC will automatically append .expscore/.expscore.gz.')
parser.add_argument('--exp-chr', default=None, type=str,
                    help='Same as --exp, but will read files split into 22 chromosomes in the same '
                         'manner as --exp-chr.')
# Optional flags
parser.add_argument('--ref-ld', default=None, type=str,
                    help='Use --ref-ld to tell MESC which LD Scores to use as the predictors in the regression. '
                         'MESC will automatically append .l2.ldscore/.l2.ldscore.gz to the filename prefix.')
parser.add_argument('--ref-ld-chr', default=None, type=str,
                    help='Same as --ref-ld, but will automatically concatenate .l2.ldscore files split '
                         'across 22 chromosomes. MESC will automatically append .l2.ldscore/.l2.ldscore.gz '
                         'to the filename prefix. If the filename prefix contains the symbol @, LDSC will '
                         'replace the @ symbol with chromosome numbers. Otherwise, MESC will append chromosome '
                         'numbers to the end of the filename prefix.'
                         'Example 1: --ref-ld-chr ld/ will read ld/1.l2.ldscore.gz ... ld/22.l2.ldscore.gz'
                         'Example 2: --ref-ld-chr ld/@_kg will read ld/1_kg.l2.ldscore.gz ... ld/22_kg.l2.ldscore.gz')
parser.add_argument('--w-ld', default=None, type=str,
                    help='Filename prefix for file with LD Scores with sum r^2 taken over SNPs included '
                         'in the regression. MESC will automatically append .l2.ldscore/.l2.ldscore.gz.')
parser.add_argument('--w-ld-chr', default=None, type=str,
                    help='Same as --w-ld, but will read files split into 22 chromosomes in the same '
                         'manner as --ref-ld-chr.')
parser.add_argument('--frqfile', default=None, type=str,
                    help='')
parser.add_argument('--frqfile-chr', default=os.path.join(dirname, 'data/frq_common/1000G.EUR.QC.'), type=str,
                    help='Prefix for --frqfile files split over chromosome.')
parser.add_argument('--ref-ld-remove-annot', default=None, type=str,
                    help='LD scores to remove.')
parser.add_argument('--ref-ld-keep-annot', default=None, type=str,
                    help='LD scores to keep.')
parser.add_argument('--exp-remove-annot', default=None, type=str,
                    help='Expression scores to remove.')
parser.add_argument('--exp-keep-annot', default=None, type=str,
                    help='Expression scores to keep.')
parser.add_argument('--chisq-max', default=None, type=float,
                    help='Max chi^2.')


if __name__ == '__main__':

    args = parser.parse_args()
    log = Logger(args.out + '.log')
    try:
        defaults = vars(parser.parse_args(''))
        opts = vars(args)
        non_defaults = [x for x in opts.keys() if opts[x] != defaults[x]]
        header = MASTHEAD
        header += "Call: \n"
        header += './run_mesc.py \\\n'
        options = ['--' + x.replace('_', '-') + ' ' + str(opts[x]) + ' \\' for x in non_defaults]
        header += '\n'.join(options).replace('True', '').replace('False', '')
        header = header[0:-1] + '\n'
        log.log(header)
        log.log('Beginning analysis at {T}'.format(T=time.ctime()))
        start_time = time.time()

        if not args.out:
            raise ValueError('Must specify --out')

        if args.compute_expscore_indiv and args.compute_expscore_sumstat:
            raise ValueError('Cannot set both --compute-expscore-indiv and --compute-expscore-sumstat')

        if args.compute_expscore_indiv:
            if args.plink_path is None:
                raise ValueError('Must specify --plink-path with --compute-expscore-indiv')
            if args.expression_matrix is None:
                raise ValueError('Must specify --expression-matrix with --compute-expscore-indiv')
            if not args.exp_bfile:
                raise ValueError('Must specify --exp_bfile with --compute-expscore-indiv')
            if not args.chr:
                raise ValueError('Must specify --chr with --compute-expscore-indiv')
            if not args.geno_bfile:
                raise ValueError('Must specify --geno-bfile with --compute-expscore-indiv')
            subprocess.call(['mkdir', '-p', args.tmp])
            ind.get_expression_scores(args)

        elif args.compute_expscore_sumstat:
            if not args.eqtl_sumstat:
                raise ValueError('Must specify --eqtl-sumstat with --compute-expscore-sumstat')
            if not args.ref_ld_chr:
                args.ref_ld_chr = os.path.join(dirname, 'data/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC.')
            ss.get_expression_scores(args)

        # summary statistics
        elif args.h2med:
            if not (args.exp or args.exp_chr):
                raise ValueError('Must specify --exp or --exp-chr with --h2med')
            if args.exp and args.exp_chr:
                raise ValueError('Cannot set both --exp and --exp-chr')
            if not (args.ref_ld_chr or args.ref_ld):
                args.ref_ld_chr = os.path.join(dirname, 'data/baselineLD_v2.0_pruned/baselineLD.')
            if args.ref_ld_chr and args.ref_ld:
                raise ValueError('Cannot set both --ref-ld and --ref-ld-chr')
            if not (args.w_ld or args.w_ld_chr):
                args.w_ld_chr = os.path.join(dirname, 'data/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC.')
            if args.w_ld and args.w_ld_chr:
                raise ValueError('Cannot set both --w-ld and --w-ld-chr')
            sumstats.estimate_h2med(args, log)

        else:
            print('Error: no analysis selected.')
            print('mesc.py -h describes options.')
    except Exception:
        ex_type, ex, tb = sys.exc_info()
        log.log(traceback.format_exc(ex))
        raise
    finally:
        log.log('Analysis finished at {T}'.format(T=time.ctime()))
        time_elapsed = round(time.time() - start_time, 2)
        log.log('Total time elapsed: {T}'.format(T=sec_to_str(time_elapsed)))
