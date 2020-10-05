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
from collections import defaultdict, OrderedDict


def sub_chr(s, chr):
    '''Substitute chr for @, else append chr to the end of str.'''
    if '@' not in s:
        s += '@'

    return s.replace('@', str(chr))

def check_order_and_get_len(cismat, columns):
    '''
    Check that genes and chromosome are sorted
    '''
    print('Checking that genes and chromosomes are sorted')
    each_chrom = np.zeros(22, dtype=int)
    chr = set()
    genes = set()
    with open(cismat) as f:
        for i, l in enumerate(f):
            l = l.split()
            if i == 0:
                if len(l) < 7:
                    raise ValueError('Not enough columns')
                continue
            elif i == 1:
                prev_gene = l[columns[0]]
                prev_chr = int(l[columns[3]])
                genes.add(prev_gene)
                chr.add(prev_chr)
                each_chrom[prev_chr - 1] += 1
            else:
                curr_gene = l[columns[0]]
                curr_chr = int(l[columns[3]])
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

def get_snp_list(cismat, keep, keep_chr, columns):
    print('Getting SNP list')
    snp_list = []
    snp_only = set()
    with open(cismat) as f:
        for j, line in enumerate(f):
            if j == 0:
                continue
            line = line.split()
            snp = line[columns[2]]
            try:
                chr = int(line[columns[3]])
            except:
                raise ValueError('Please remove non-autosomal genes from cis-eQTL file')
            if chr != keep_chr:
                continue
            bp = int(line[columns[4]])
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

def estimate_expression_cis_herit(ref_ld, frq, zscores, ref_ld_indices):
    '''
    Estimate h2cis using LD score regression
    '''

    n_snp = len(zscores)
    s = lambda x: np.array(x).reshape((n_snp, 1))

    start_snp = zscores['SNP'].values[0]
    end_snp = zscores['SNP'].values[-1]

    start_snp_idx = ref_ld_indices[start_snp]
    end_snp_idx = ref_ld_indices[end_snp]

    i = 0
    while start_snp not in frq:
        i += 1
        start_snp = zscores['SNP'].values[i]
    temp_start_snp_idx = frq[start_snp]

    i = -1
    while end_snp not in frq:
        i -= 1
        end_snp = zscores['SNP'].values[i]
    temp_end_snp_idx = frq[end_snp]

    M_annot = np.array(temp_end_snp_idx - temp_start_snp_idx + 1).reshape((1, 1))

    if np.array_equal(ref_ld.iloc[start_snp_idx:end_snp_idx+1, 1].values, zscores['SNP'].values):
        temp_ref_ld = ref_ld.iloc[start_snp_idx:end_snp_idx+1, :]
        hsqhat = reg.Hsq(s(zscores['Z'].values ** 2), s(temp_ref_ld.iloc[:, 3]), s(temp_ref_ld.iloc[:, 3]),
                         s(zscores['N']), M_annot, n_blocks=10)
    else:
        indices = [ref_ld_indices[x] for x in zscores['SNP']]
        temp_ref_ld = ref_ld.iloc[indices, :]
        hsqhat = reg.Hsq(s(zscores['Z'].values ** 2), s(temp_ref_ld.iloc[:, 3]), s(temp_ref_ld.iloc[:, 3]),
                         s(zscores['N']), M_annot, n_blocks=10)

    herit = hsqhat.tot
    herit_se = hsqhat.tot_se
    herit_p = scipy.stats.norm.sf(abs(herit / herit_se))

    return herit, herit_se, herit_p

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

def get_expression_scores(args):
    cismat = args.eqtl_sumstat
    gene_num = 0
    if args.columns:
        columns = args.columns.split(',')
        columns = [int(x)-1 for x in columns]
        if len(columns) != 7:
            raise ValueError('Must specify 7 column indices with --columns')
    else:
        columns = range(7)

    n_genes, n_lines = check_order_and_get_len(cismat, columns)

    if args.gene_sets is not None:
        gsets = read_gene_sets(args.gene_sets)
        print('{} gene sets read from {}'.format(len(gsets), args.gene_sets))

    keep_snps = pd.read_csv(args.keep, header=None)

    with open(cismat) as f:
        for i, line in enumerate(f):
            line = line.split()
            if i == 0:
                colnames = ['GENE','GENE_LOC','SNP','CHROM','BP','N','Z']
                continue
            elif i == 1:
                temp_gene_mat = [line]
                prev_gene = line[columns[0]]
                prev_chr = int(line[columns[3]])
                print('Analyzing chromosome {}'.format(prev_chr))
                chrom_gene_num = 0
                ref_ld = read_ldscore(args, prev_chr)
                snps = get_snp_list(cismat, ref_ld['SNP'], prev_chr, columns)
                snp_indices = dict(zip(snps['SNP'].tolist(), range(len(snps))))
                ref_ld = ref_ld[ref_ld['SNP'].isin(snps['SNP'])]
                ref_ld_indices = dict(zip(ref_ld['SNP'].tolist(), range(len(ref_ld))))
                frq = pd.read_csv(sub_chr(args.frqfile_chr, prev_chr) + '.frq', delim_whitespace=True)
                frq = frq[frq['MAF'] > 0.05]
                frq = dict(zip(frq['SNP'].tolist(), range(len(frq))))
                all_summ = []
                all_herit = []

            else:
                curr_gene = line[columns[0]]
                curr_chr = int(line[columns[3]])
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
                for col in [1,3,4,5]:
                    try:
                        temp_gene_mat.iloc[:,col] = temp_gene_mat.iloc[:,col].astype(int)
                    except Exception as e:
                        raise ValueError(str(e) + '\nPlease double check that the proper columns in the eQTL file are specified.')
                try:
                    temp_gene_mat.iloc[:,6] = temp_gene_mat.iloc[:,6].astype(float)
                except Exception as e:
                    raise ValueError(str(e) + '\nPlease double check that the proper columns in the eQTL file are specified.')
                temp_gene_mat = temp_gene_mat.drop_duplicates(subset='SNP')
                temp_gene_mat = temp_gene_mat.sort_values('BP')
                temp_gene_mat = temp_gene_mat.loc[temp_gene_mat['SNP'].isin(snps['SNP']),:]

                if len(temp_gene_mat) <= 10:
                    print('<= 10 SNPs around gene; skipping')
                elif np.sum(temp_gene_mat['SNP'].isin(frq)) == 0:
                    print('No SNPs match reference list; skipping')
                else:
                    herit = estimate_expression_cis_herit(ref_ld, frq, temp_gene_mat, ref_ld_indices)
                    temp_gene_mat['EFF'] = temp_gene_mat['Z'].values**2 / temp_gene_mat['N'].values - 1 / temp_gene_mat['N'].values
                    temp_gene_mat = temp_gene_mat[['SNP','BP','EFF']]
                    all_herit.append([prev_gene, prev_chr, herit[0], herit[1], herit[2]])
                    all_summ.append([prev_gene, herit[0], temp_gene_mat])

                if curr_chr == prev_chr and i < n_lines:
                    temp_gene_mat = [line]
                    prev_gene = curr_gene
                    continue

                print('Computing expression scores for chromosome {}'.format(prev_chr))
                genes = [x[0] for x in all_summ]
                eqtl_herits = [x[1] for x in all_summ]

                # gene sets specified
                if args.gene_sets is not None:

                    # get gset names
                    gset_names = ['Cis_herit_bin_{}'.format(x) for x in range(1, args.num_bins + 1)]
                    for k in gsets.keys():
                        gset_names.extend(['{}_Cis_herit_bin_{}'.format(k, x) for x in range(1, args.num_gene_bins + 1)])

                    # create dict indicating membership of each gene in each gene set
                    gene_gset_dict = defaultdict(list)
                    gene_bins = pd.qcut(np.array(eqtl_herits), args.num_bins, labels=range(1, args.num_bins + 1)).astype(int)
                    all_gset_gene_bins = pd.qcut(np.array(eqtl_herits), args.num_gene_bins, labels=range(1, args.num_gene_bins + 1)).astype(int)
                    for ix, gene in enumerate(genes):
                        gene_gset_dict[gene].append('Cis_herit_bin_{}'.format(gene_bins[ix]))
                    for k, v in gsets.items():
                        print('Computing expression scores for gene set: {}'.format(k))
                        v = [x for x in v if x in genes]
                        if len(v) > 0:
                            gene_indices = np.where(np.isin(genes, v))[0]
                            if len(v) == 1:
                                gset_gene_bins = all_gset_gene_bins[gene_indices]
                            else:
                                gset_gene_bins = pd.qcut(np.array(eqtl_herits)[gene_indices], args.num_gene_bins, labels=range(1, args.num_gene_bins + 1)).astype(int)
                            for ix, gene in enumerate(v):
                                gene_gset_dict[gene].append('{}_Cis_herit_bin_{}'.format(k, gset_gene_bins[ix]))

                    eqtl_annot = np.zeros((len(snps), len(gset_names)))
                    g_annot = np.zeros((len(all_summ), len(gset_names)))
                    gset_indices = dict(zip(gset_names, range(len(gset_names))))

                    # populate expscore and gannot
                    for j in range(0, len(all_summ)):
                        gset_idx = [gset_indices[x] for x in gene_gset_dict[genes[j]]]
                        g_annot[j, gset_idx] = 1
                        start_snp = all_summ[j][2]['SNP'].values[0]
                        end_snp = all_summ[j][2]['SNP'].values[-1]
                        start_snp_idx = snp_indices[start_snp]
                        end_snp_idx = snp_indices[end_snp]

                        if np.array_equal(snps.iloc[start_snp_idx:end_snp_idx + 1, 1].values,
                                          all_summ[j][2]['SNP'].values):
                            for gix in gset_idx:
                                eqtl_annot[start_snp_idx:end_snp_idx + 1, gix] += all_summ[j][2]['EFF'].values
                        else:
                            snp_idx = [snp_indices[x] for x in all_summ[j][2]['SNP'].tolist()]
                            for gix in gset_idx:
                                eqtl_annot[snp_idx, gix] += all_summ[j][2]['EFF'].values

                    # create G and ave h2cis files
                    G = np.sum(g_annot, axis=0)
                    ave_cis_herit = np.divide(np.dot(g_annot.T, eqtl_herits), G, out=np.zeros_like(G), where=G!=0)
                    g_annot = g_annot.astype(int)

                # gene sets not specified
                else:
                    eqtl_herits = [x[1] for x in all_summ]
                    g_annot = np.zeros((len(all_summ), args.num_bins), dtype=int)
                    eqtl_annot = np.zeros((len(snps), args.num_bins))
                    gene_bins = pd.qcut(np.array(eqtl_herits), args.num_bins, labels=range(args.num_bins)).astype(int)
                    gset_names = ['Cis_herit_bin_{}'.format(x) for x in range(1, args.num_bins+1)]
                    for j in range(0, len(all_summ)):
                        g_annot[j, gene_bins[j]] = 1
                        start_snp = all_summ[j][2]['SNP'].values[0]
                        end_snp = all_summ[j][2]['SNP'].values[-1]
                        start_snp_idx = snp_indices[start_snp]
                        end_snp_idx = snp_indices[end_snp]

                        if np.array_equal(snps.iloc[start_snp_idx:end_snp_idx+1, 1].values, all_summ[j][2]['SNP'].values):
                            eqtl_annot[start_snp_idx:end_snp_idx+1, gene_bins[j]] += all_summ[j][2]['EFF'].values
                        else:
                            snp_idx = [snp_indices[x] for x in all_summ[j][2]['SNP'].tolist()]
                            eqtl_annot[snp_idx, gene_bins[j]] += all_summ[j][2]['EFF'].values

                    G = np.sum(g_annot, axis=0)
                    ave_cis_herit = []
                    for j in range(5):
                        temp_herits = np.array(eqtl_herits)[np.where(gene_bins == j)[0]]
                        ave_cis_herit.append(np.median(temp_herits))

                g_annot_final = pd.DataFrame(np.c_[[x[0] for x in all_summ], g_annot])
                g_annot_final.columns = ['Gene'] + gset_names
                g_annot_final.to_csv('{}.{}.gannot.gz'.format(args.out, prev_chr), sep='\t', index=False, compression='gzip')

                all_herit = pd.DataFrame.from_records(all_herit,
                                                      columns=['Gene', 'Chrom', 'h2cis', 'h2cis_se',
                                                               'h2cis_p'])

                np.savetxt('{}.{}.G'.format(args.out, prev_chr), G.reshape((1, len(G))), fmt='%d')
                np.savetxt('{}.{}.ave_h2cis'.format(args.out, prev_chr),
                           np.array(ave_cis_herit).reshape((1, len(ave_cis_herit))),
                           fmt="%.5f")

                expscore = pd.concat([
                    pd.DataFrame(snps.values),
                    pd.DataFrame(eqtl_annot)], axis=1)
                expscore.columns = snps.columns.tolist() + gset_names
                expscore = expscore.loc[expscore['SNP'].isin(keep_snps[0]).values]
                expscore.to_csv('{}.{}.expscore.gz'.format(args.out, prev_chr), sep='\t', index=False, compression='gzip', float_format='%.5f')

                all_herit.to_csv('{}.{}.hsq'.format(args.out, prev_chr), sep='\t', index=False, float_format='%.5f')
                print('Done chromosome {}'.format(prev_chr))

                if i == n_lines:
                    print('All done!')
                    sys.exit(0)

                prev_chr = curr_chr
                prev_gene = curr_gene
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




