'''
Compute expression scores and estimate expression cis-heritability from eQTL summary statistics
'''

from __future__ import division
import numpy as np
import pandas as pd
import parse as ps
import scipy.stats
import regressions_ldsc as reg
import sys


def sub_chr(s, chr):
    '''Substitute chr for @, else append chr to the end of str.'''
    if '@' not in s:
        s += '@'

    return s.replace('@', str(chr))

def check_order_and_get_len(cismat):
    print('Checking that genes and chromosomes are sorted')
    each_chrom = np.zeros(22, dtype=int)
    chr = set()
    genes = set()
    with open(cismat) as f:
        for i, l in enumerate(f):
            l = l.split()
            if i == 0:
                if len(l) < 8:
                    raise ValueError('Not enough columns')
                continue
            elif i == 1:
                prev_gene = l[0]
                prev_chr = int(l[4])
                genes.add(prev_gene)
                chr.add(prev_chr)
                each_chrom[prev_chr - 1] += 1
            else:
                curr_gene = l[0]
                curr_chr = int(l[4])
                if curr_gene == prev_gene:
                    continue
                elif curr_chr == prev_chr:
                    prev_gene = curr_gene
                    if prev_gene in genes:
                        raise ValueError('Genes not sorted. eQTLs for each gene must be in one continguous block.')
                    genes.add(prev_gene)
                    each_chrom[prev_chr - 1] += 1
                else:
                    prev_gene = curr_gene
                    if prev_gene in genes:
                        raise ValueError('Genes not sorted. eQTLs for each gene must be in one continguous block.')
                    genes.add(prev_gene)
                    prev_chr = curr_chr
                    if prev_chr in chr:
                        raise ValueError('Chromosomes not sorted. eQTLs on each chromosome must be in one contiguous block.')
                    chr.add(prev_chr)
                    each_chrom[prev_chr - 1] += 1
    return each_chrom, i

def get_snp_list(cismat, keep, keep_chr):
    snp_list = []
    snp_only = set()
    with open(cismat) as f:
        for j, line in enumerate(f):
            if j == 0:
                continue
            line = line.split()
            snp = line[3]
            chr = int(line[4])
            if chr != keep_chr:
                continue
            bp = int(line[5])
            if snp not in snp_only:
                snp_list.append([chr, snp, bp])
                snp_only.add(snp)
    snp_list = pd.DataFrame.from_records(snp_list, columns=['CHR', 'SNP', 'BP'])
    snp_list = snp_list.loc[snp_list['SNP'].isin(keep)]
    snp_list = snp_list.sort_values(['CHR','BP'])
    return snp_list

def read_ldscore(args, chr):
    geno_name = sub_chr(args.ref_ld_chr, chr) + '.l2.ldscore'
    s, comp = ps.which_compression(geno_name)
    ref_ld = pd.read_csv(geno_name + s, sep='\t', compression=comp)
    return ref_ld

def estimate_expression_cis_herit(ref_ld, frq, zscores):
    n_snp = len(zscores)
    s = lambda x: np.array(x).reshape((n_snp, 1))

    start_snp = zscores['SNP'].values[0]
    end_snp = zscores['SNP'].values[-1]

    start_snp_idx = int(np.where(ref_ld['SNP'] == start_snp)[0])
    end_snp_idx = int(np.where(ref_ld['SNP'] == end_snp)[0])

    temp_zscores = zscores[zscores['SNP'].isin(frq['SNP'])]
    temp_start_snp = temp_zscores['SNP'].values[0]
    temp_end_snp = temp_zscores['SNP'].values[-1]
    temp_start_snp_idx = int(np.where(frq['SNP'] == temp_start_snp)[0])
    temp_end_snp_idx = int(np.where(frq['SNP'] == temp_end_snp)[0])

    M_annot = np.array(temp_end_snp_idx - temp_start_snp_idx).reshape((1, 1))

    if np.array_equal(ref_ld.iloc[start_snp_idx:end_snp_idx+1, 1].values, zscores['SNP'].values):
        temp_ref_ld = ref_ld.iloc[start_snp_idx:end_snp_idx+1, :]
        hsqhat = reg.Hsq(s(zscores['Z'].values ** 2), s(temp_ref_ld.iloc[:, 3]), s(temp_ref_ld.iloc[:, 3]),
                         s(zscores['N']), M_annot)
    else:
        all = pd.merge(ref_ld, zscores, on='SNP')
        hsqhat = reg.Hsq(s(all['Z'].values ** 2), s(all['L2']), s(all['L2']),
                         s(all['N']), M_annot)

    herit = hsqhat.tot
    herit_se = hsqhat.tot_se
    herit_p = scipy.stats.norm.sf(abs(herit / herit_se))

    return herit, herit_se, herit_p

def match(a, b):
    return [b.index(x) if x in b else None for x in a]

def get_expression_scores(args):
    cismat = args.eqtl_sumstat
    gene_num = 0
    n_genes, n_lines = check_order_and_get_len(cismat)
    if args.columns:
        columns = args.columns.split(',')
        columns = [int(x)-1 for x in columns]
        if len(columns) != 8:
            raise ValueError('Must specify 8 column indices with --columns')
    else:
        columns = range(8)

    with open(cismat) as f:
        for i, line in enumerate(f):
            line = line.split()
            if i == 0:
                colnames = ['GENE','GENE_SYMBOL','GENE_LOC','SNP','CHROM','BP','N','Z']
                continue
            elif i == 1:
                temp_gene_mat = [line]
                prev_gene = line[0]
                prev_gene_symb = line[1]
                prev_chr = int(line[4])
                print('Analyzing chromosome {}'.format(prev_chr))
                chrom_gene_num = 0
                ref_ld = read_ldscore(args, prev_chr)
                snps = get_snp_list(cismat, ref_ld['SNP'], prev_chr)
                ref_ld = ref_ld[ref_ld['SNP'].isin(snps['SNP'])]
                frq = pd.read_table(sub_chr(args.frqfile_chr, prev_chr) + '.frq', delim_whitespace=True)
                frq = frq[frq['MAF'] > 0.05]
                all_summ = []
                all_herit = []

            else:
                curr_gene = line[0]
                curr_gene_symb = line[1]
                curr_chr = int(line[4])
                if curr_gene == prev_gene and i < n_lines:
                    temp_gene_mat.append(line)
                    continue

                if i == n_lines:
                    temp_gene_mat.append(line)

                gene_num += 1
                chrom_gene_num += 1
                print('Read gene {} of {} ({} of {} total): {}'.format(chrom_gene_num, n_genes[prev_chr - 1], gene_num,
                                                                       sum(n_genes), prev_gene))

                temp_gene_mat = pd.DataFrame.from_records(temp_gene_mat)
                temp_gene_mat = temp_gene_mat[columns]
                temp_gene_mat.columns = colnames
                for col in [2,4,5,6]:
                    temp_gene_mat.iloc[:,col] = temp_gene_mat.iloc[:,col].astype(int)
                temp_gene_mat.iloc[:,7] = temp_gene_mat.iloc[:,7].astype(float)
                temp_gene_mat = temp_gene_mat.drop_duplicates(subset='SNP')
                temp_gene_mat = temp_gene_mat.sort_values('BP')
                temp_gene_mat = temp_gene_mat.loc[temp_gene_mat['SNP'].isin(snps['SNP']),:]

                if len(temp_gene_mat) < 10:
                    print('<10 SNPs around gene; skipping')
                else:
                    herit = estimate_expression_cis_herit(ref_ld, frq, temp_gene_mat)
                    temp_gene_mat['EFF'] = temp_gene_mat['Z'].values**2 / temp_gene_mat['N'].values - 1 / temp_gene_mat['N'].values
                    temp_gene_mat = temp_gene_mat[['SNP','BP','EFF']]
                    all_herit.append([prev_gene, prev_gene_symb, prev_chr, herit[0], herit[1], herit[2]])
                    all_summ.append([prev_gene, prev_gene_symb, herit[0], temp_gene_mat])

                if curr_chr == prev_chr and i < n_lines:
                    temp_gene_mat = [line]
                    prev_gene = curr_gene
                    prev_gene_symb = curr_gene_symb
                    continue

                print('Computing expression scores for chromosome {}'.format(prev_chr))
                eqtl_herits = [x[2] for x in all_summ]
                g_annot = np.zeros((len(all_summ), 5), dtype=int)
                eqtl_annot = np.zeros((len(snps), 5))
                gene_bins = pd.qcut(np.array(eqtl_herits), 5, labels=range(5)).astype(int)
                g_bin_names = ['Cis_herit_bin_{}'.format(x) for x in range(1, 6)]
                for j in range(0, len(all_summ)):
                    g_annot[j, gene_bins[j]] = 1
                    start_snp = all_summ[j][3]['SNP'].values[0]
                    end_snp = all_summ[j][3]['SNP'].values[-1]
                    start_snp_idx = int(np.where(snps['SNP'] == start_snp)[0])
                    end_snp_idx = int(np.where(snps['SNP'] == end_snp)[0])

                    if np.array_equal(snps.iloc[start_snp_idx:end_snp_idx+1, 1].values, all_summ[j][3]['SNP'].values):
                        eqtl_annot[start_snp_idx:end_snp_idx+1, gene_bins[j]] += all_summ[j][3]['EFF'].values
                    else:
                        snp_indices = match(all_summ[j][3]['SNP'].tolist(), snps.iloc[:, 1].tolist())
                        eqtl_annot[snp_indices, gene_bins[j]] += all_summ[j][3]['EFF'].values

                g_annot_final = pd.DataFrame(np.c_[[x[0] for x in all_summ], [x[1] for x in all_summ], g_annot])
                g_annot_final.columns = ['Gene', 'Gene_symbol'] + g_bin_names
                g_annot_final.to_csv('{}.{}.gannot.gz'.format(args.out, prev_chr), sep='\t', index=False, compression='gzip')

                all_herit = pd.DataFrame.from_records(all_herit,
                                                      columns=['Gene', 'Gene_symbol', 'Chrom', 'h2cis', 'h2cis_se',
                                                               'h2cis_p'])
                matched_herit = all_herit.loc[all_herit['Gene'].isin(g_annot_final['Gene']), 'h2cis'].values
                G = np.sum(g_annot, axis=0)
                ave_cis_herit = np.dot(matched_herit, g_annot) / G

                np.savetxt('{}.{}.G'.format(args.out, prev_chr), G.reshape((1, len(G))), fmt='%d')
                np.savetxt('{}.{}.ave_h2cis'.format(args.out, prev_chr), ave_cis_herit.reshape((1, len(ave_cis_herit))),
                           fmt="%.5f")

                expscore = pd.DataFrame(np.c_[snps.values, eqtl_annot])
                expscore.columns = snps.columns.tolist() + g_bin_names
                for name in g_bin_names:
                    expscore[name] = expscore[name].astype(float)
                expscore.to_csv('{}.{}.expscore.gz'.format(args.out, prev_chr), sep='\t', index=False, compression='gzip', float_format='%.5f')

                all_herit.to_csv('{}.{}.hsq'.format(args.out, prev_chr), sep='\t', index=False, float_format='%.5f')
                print('Done chromosome {}'.format(prev_chr))

                if i == n_lines:
                    print('All done!')
                    sys.exit(0)

                prev_chr = curr_chr
                prev_gene = curr_gene
                prev_gene_symb = curr_gene_symb
                temp_gene_mat = [line]
                print('Analyzing chromosome {}'.format(prev_chr))
                chrom_gene_num = 0
                ref_ld = read_ldscore(args, prev_chr)
                snps = get_snp_list(cismat, ref_ld['SNP'], prev_chr)
                ref_ld = ref_ld[ref_ld['SNP'].isin(snps['SNP'])]
                frq = pd.read_table(sub_chr(args.frqfile_chr, prev_chr) + '.frq', delim_whitespace=True)
                frq = frq[frq['MAF'] > 0.05]
                all_summ = []
                all_herit = []




