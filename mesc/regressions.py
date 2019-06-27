'''
(c) 2014 Brendan Bulik-Sullivan and Hilary Finucane

Estimators of heritability and genetic correlation.

Shape convention is (n_snp, n_annot) for all classes.
Last column = intercept.

'''
from __future__ import division
import numpy as np
import pandas as pd
from scipy.stats import norm, chi2
import jackknife as jk
from irwls import IRWLS
from scipy.stats import t as tdist
from collections import namedtuple
np.seterr(divide='raise', invalid='raise')

s = lambda x: remove_brackets(str(np.matrix(x)))


def update_separators(s, ii):
    '''s are separators with ii masked. Returns unmasked separators.'''
    maplist = np.arange(len(ii))[np.squeeze(ii)]
    mask_to_unmask = lambda i: maplist[i]
    t = np.apply_along_axis(mask_to_unmask, 0, s[1:-1])
    t = np.hstack(((0), t, (len(ii))))
    return t


def p_z_norm(est, se):
    '''Convert estimate and se to Z-score and P-value.'''
    try:
        Z = est / se
    except (FloatingPointError, ZeroDivisionError):
        Z = float('inf')

    P = chi2.sf(Z ** 2, 1, loc=0, scale=1)  # 0 if Z=inf
    return P, Z


def remove_brackets(x):
    '''Get rid of brackets and trailing whitespace in numpy arrays.'''
    return x.replace('[', '').replace(']', '').strip()


def append_intercept(x):
    '''
    Appends an intercept term to the design matrix for a linear regression.

    Parameters
    ----------
    x : np.matrix with shape (n_row, n_col)
        Design matrix. Columns are predictors; rows are observations.

    Returns
    -------
    x_new : np.matrix with shape (n_row, n_col+1)
        Design matrix with intercept term appended.

    '''
    n_row = x.shape[0]
    intercept = np.ones((n_row, 1))
    x_new = np.concatenate((x, intercept), axis=1)
    return x_new


def remove_intercept(x):
    '''Removes the last column.'''
    n_col = x.shape[1]
    return x[:, 0:n_col - 1]


def gencov_obs_to_liab(gencov_obs, P1, P2, K1, K2):
    '''
    Converts genetic covariance on the observed scale in an ascertained sample to genetic
    covariance on the liability scale in the population

    Parameters
    ----------
    gencov_obs : float
        Genetic covariance on the observed scale in an ascertained sample.
    P1, P2 : float in (0,1)
        Prevalences of phenotypes 1,2 in the sample.
    K1, K2 : float in (0,1)
        Prevalences of phenotypes 1,2 in the population.

    Returns
    -------
    gencov_liab : float
        Genetic covariance between liabilities in the population.

    Note: if a trait is a QT, set P = K = None.

    '''
    c1 = 1
    c2 = 1
    if P1 is not None and K1 is not None:
        c1 = np.sqrt(h2_obs_to_liab(1, P1, K1))
    if P2 is not None and K2 is not None:
        c2 = np.sqrt(h2_obs_to_liab(1, P2, K2))

    return gencov_obs * c1 * c2


def h2_obs_to_liab(h2_obs, P, K):
    '''
    Converts heritability on the observed scale in an ascertained sample to heritability
    on the liability scale in the population.

    Parameters
    ----------
    h2_obs : float
        Heritability on the observed scale in an ascertained sample.
    P : float in (0,1)
        Prevalence of the phenotype in the sample.
    K : float in (0,1)
        Prevalence of the phenotype in the population.

    Returns
    -------
    h2_liab : float
        Heritability of liability in the population.

    '''
    if np.isnan(P) and np.isnan(K):
        return h2_obs
    if K <= 0 or K >= 1:
        raise ValueError('K must be in the range (0,1)')
    if P <= 0 or P >= 1:
        raise ValueError('P must be in the range (0,1)')

    thresh = norm.isf(K)
    conversion_factor = K ** 2 * \
        (1 - K) ** 2 / (P * (1 - P) * norm.pdf(thresh) ** 2)
    return h2_obs * conversion_factor


class LD_Score_Regression(object):

    def __init__(self, y, x, g_ld, w, N, M, G, ave_h2_cis, n_blocks, intercept=None, slow=False, step1_ii=None, old_weights=False):
        for i in [y, x, g_ld, w, M, N, G, ave_h2_cis]:
            try:
                if len(i.shape) != 2:
                    raise TypeError('Arguments must be 2D arrays.')
            except AttributeError:
                raise TypeError('Arguments must be arrays.')

        n_snp, self.n_only_annot = x.shape
        _, self.n_g_annot = g_ld.shape
        self.n_annot = self.n_only_annot + self.n_g_annot
        if any(i.shape != (n_snp, 1) for i in [y, w, N]):
            raise ValueError(
                'N, weights and response (z1z2 or chisq) must have shape (n_snp, 1).')
        if M.shape != (1, self.n_only_annot):
            raise ValueError('M must have shape (1, n_annot).')

        M_tot = float(np.sum(M))
        G_tot = float(np.sum(G))
        G_mult = np.multiply(G, ave_h2_cis).reshape((1, self.n_g_annot))
        M_combined = np.hstack((M, G_mult)).reshape((1, self.n_annot))

        x_tot = np.sum(x, axis=1).reshape((n_snp, 1))
        self.constrain_intercept = intercept is not None
        self.intercept = intercept
        self.n_blocks = n_blocks
        tot_agg = self.aggregate(y, x_tot, N, M_tot, intercept)
        initial_w = self._update_weights(
            x_tot, w, N, M_tot, tot_agg, intercept)
        x = np.hstack((x, g_ld))
        Nbar = np.mean(N)  # keep condition number low
        x = np.multiply(N, x) / Nbar
        if not self.constrain_intercept:
            x, x_tot = append_intercept(x), append_intercept(x_tot)
            yp = y
        else:
            yp = y - intercept
            self.intercept_se = 'NA'
        del y
        self.twostep_filtered = None
        if step1_ii is not None and self.constrain_intercept:
            raise ValueError(
                'twostep is not compatible with constrain_intercept.')
        elif step1_ii is not None and self.n_annot > 1:
            raise ValueError(
                'twostep not compatible with partitioned LD Score yet.')
        elif step1_ii is not None:
            n1 = np.sum(step1_ii)
            self.twostep_filtered = n_snp - n1
            x1 = x[np.squeeze(step1_ii), :]
            yp1, w1, N1, initial_w1 = map(
                lambda a: a[step1_ii].reshape((n1, 1)), (yp, w, N, initial_w))
            update_func1 = lambda a: self._update_func(
                a, x1, w1, N1, M_tot, Nbar, ii=step1_ii)
            step1_jknife = IRWLS(
                x1, yp1, update_func1, n_blocks, slow=slow, w=initial_w1)
            step1_int, _ = self._intercept(step1_jknife)
            yp = yp - step1_int
            x = remove_intercept(x)
            x_tot = remove_intercept(x_tot)
            update_func2 = lambda a: self._update_func(
                a, x_tot, w, N, M_tot, Nbar, step1_int)
            s = update_separators(step1_jknife.separators, step1_ii)
            step2_jknife = IRWLS(
                x, yp, update_func2, n_blocks, slow=slow, w=initial_w, separators=s)
            c = np.sum(np.multiply(initial_w, x)) / \
                np.sum(np.multiply(initial_w, np.square(x)))
            jknife = self._combine_twostep_jknives(
                step1_jknife, step2_jknife, M_tot, c, Nbar)
        elif old_weights:
            initial_w = np.sqrt(initial_w)
            x = IRWLS._weight(x, initial_w)
            y = IRWLS._weight(yp, initial_w)
            jknife = jk.LstsqJackknifeFast(x, y, n_blocks)
        else:
            update_func = lambda a: self._update_func(
                a, x_tot, w, N, M_tot, Nbar, intercept)
            jknife = IRWLS(
                x, yp, update_func, n_blocks, slow=slow, w=initial_w)

        self.coef, self.coef_cov, self.coef_se = self._coef(jknife, Nbar)
        self.coef_dir, self.coef_dir_cov, self.coef_dir_se = self.coef[:self.n_only_annot], self.coef_cov[:self.n_only_annot,:self.n_only_annot], self.coef_se[:self.n_only_annot]
        self.coef_g, self.coef_g_cov, self.coef_g_se = self.coef[self.n_only_annot:], self.coef_cov[self.n_only_annot:,self.n_only_annot:], self.coef_se[self.n_only_annot:]

        self.cat, self.cat_cov, self.cat_se = self._cat(jknife, M_combined, Nbar, self.coef, self.coef_cov)
        self.cat_dir, self.cat_dir_cov, self.cat_dir_se = self.cat[:,:self.n_only_annot], self.cat_cov[:self.n_only_annot,:self.n_only_annot], self.cat_se[:self.n_only_annot]
        self.cat_g, self.cat_g_cov, self.cat_g_se = self.cat[:,self.n_only_annot:], self.cat_cov[self.n_only_annot:,self.n_only_annot:], self.cat_se[self.n_only_annot:]

        self.tot, self.tot_cov, self.tot_se = self._tot(self.cat, self.cat_cov)
        self.tot_dir, self.tot_dir_cov, self.tot_dir_se = self._tot(self.cat_dir, self.cat_dir_cov)
        self.tot_g, self.tot_g_cov, self.tot_g_se = self._tot(self.cat_g, self.cat_g_cov)

        self.prop, self.prop_cov, self.prop_se = self._prop(jknife, M_combined, Nbar, self.cat, self.tot, self.n_annot)
        self.prop_dir, self.prop_dir_cov, self.prop_dir_se = self._prop(jknife, M, Nbar, self.cat_dir, self.tot_dir, self.n_only_annot)
        self.prop_g, self.prop_g_cov, self.prop_g_se = self._prop(jknife, G_mult, Nbar, self.cat_g, self.tot_g, self.n_only_annot, g_annot=True)
        self.tot_dir_prop, self.tot_dir_prop_cov, self.tot_dir_prop_se = self._tot(self.prop[:,:self.n_only_annot], self.prop_cov[:self.n_only_annot, :self.n_only_annot])
        self.tot_g_prop, self.tot_g_prop_cov, self.tot_g_prop_se = self._tot(self.prop[:,self.n_only_annot:], self.prop_cov[self.n_only_annot:,self.n_only_annot:])

        self.enrichment_dir, self.enrichment_dir_se, self.M_prop_dir = self._enrichment(M, M_tot, self.prop_dir, self.prop_dir_se)
        self.enrichment_g_herit, self.enrichment_g_herit_se, _ = self._enrichment(G, G_tot, self.prop_g, self.prop_g_se)

        # self.enrichment_g_eff, _, self.enrichment_g_eff_se, = self._enrichment_g(G, G_tot, ave_h2_cis, total_ave_h2_cis, self.prop_g, self.prop_g_cov)

        if not self.constrain_intercept:
            self.intercept, self.intercept_se = self._intercept(jknife)

        self.jknife = jknife
        self.tot_delete_values = self._delete_vals_tot(jknife, Nbar, M_combined)
        self.part_delete_values = self._delete_vals_part(jknife, Nbar, M_combined)
        if not self.constrain_intercept:
            self.intercept_delete_values = jknife.delete_values[
                :, self.n_annot]

        self.M = M

    @classmethod
    def aggregate(cls, y, x, N, M, intercept=None):
        if intercept is None:
            intercept = cls.__null_intercept__

        num = M * (np.mean(y) - intercept)
        denom = np.mean(np.multiply(x, N))
        return num / denom

    def _update_func(self, x, ref_ld_tot, w_ld, N, M, Nbar, intercept=None, ii=None):
        raise NotImplementedError

    def _delete_vals_tot(self, jknife, Nbar, M):
        '''Get delete values for total h2 or gencov.'''
        n_annot = self.n_annot
        tot_delete_vals = jknife.delete_values[
            :, 0:n_annot]  # shape (n_blocks, n_annot)
        # shape (n_blocks, 1)
        tot_delete_vals = np.dot(tot_delete_vals, M.T) / Nbar
        return tot_delete_vals

    def _delete_vals_part(self, jknife, Nbar, M):
        '''Get delete values for partitioned h2 or gencov.'''
        n_annot = self.n_annot
        return jknife.delete_values[:, 0:n_annot] / Nbar

    def _coef(self, jknife, Nbar):
        '''Get coefficient estimates + cov from the jackknife.'''
        n_annot = self.n_annot
        coef = jknife.est[0, 0:n_annot] / Nbar
        coef_cov = jknife.jknife_cov[0:n_annot, 0:n_annot] / Nbar ** 2
        coef_se = np.sqrt(np.diag(coef_cov))
        return coef, coef_cov, coef_se

    def _cat(self, jknife, M, Nbar, coef, coef_cov):
        '''Convert coefficients to per-category h2 or gencov.'''
        cat = np.multiply(M, coef)
        cat_cov = np.multiply(np.dot(M.T, M), coef_cov)
        cat_se = np.sqrt(np.diag(cat_cov))
        return cat, cat_cov, cat_se

    def _tot(self, cat, cat_cov):
        '''Convert per-category h2 to total h2 or gencov.'''
        tot = np.sum(cat)
        tot_cov = np.sum(cat_cov)
        if tot_cov < 0:
            tot_se = 0
        else:
            tot_se = np.sqrt(tot_cov)
        return tot, tot_cov, tot_se

    def _prop(self, jknife, M, Nbar, cat, tot, n_annot, g_annot=False):
        '''Convert total h2 and per-category h2 to per-category proportion h2 or gencov.'''
        n_blocks = jknife.delete_values.shape[0]
        if g_annot:
            numer_delete_vals = np.multiply(
                M, jknife.delete_values[:, n_annot:self.n_annot]) / Nbar  # (n_blocks, n_annot)
            denom_delete_vals = np.sum(
                numer_delete_vals, axis=1).reshape((n_blocks, 1))
            denom_delete_vals = np.dot(denom_delete_vals, np.ones((1, self.n_annot - n_annot)))
        else:
            numer_delete_vals = np.multiply(
                M, jknife.delete_values[:, 0:n_annot]) / Nbar  # (n_blocks, n_annot)
            denom_delete_vals = np.sum(
                numer_delete_vals, axis=1).reshape((n_blocks, 1))
            denom_delete_vals = np.dot(denom_delete_vals, np.ones((1, n_annot)))

        prop = jk.RatioJackknife(
            cat / tot, numer_delete_vals, denom_delete_vals)
        return prop.est, prop.jknife_cov, prop.jknife_se

    def _enrichment(self, M, M_tot, prop, prop_se):
        '''Compute proportion of SNPs per-category enrichment for h2 or gencov.'''
        M_prop = M / M_tot
        enrichment = np.divide(prop, M_prop)
        enrichment_se = np.divide(prop_se, M_prop)
        return enrichment, enrichment_se, M_prop

    def _enrichment_g(self, g_overlap_matrix, coef, ave_coef, jknife, n_annot, n_g_tot_annot, indices=None):
        n_blocks = jknife.delete_values.shape[0]
        delete_vals = np.dot(g_overlap_matrix, jknife.delete_values[:, n_annot:self.n_annot].T).T
        if indices:
            numer_delete_vals = np.mean(delete_vals[:, indices], axis=1).reshape((n_blocks, 1))
        else:
            numer_delete_vals = delete_vals
        denom_delete_vals = np.mean(delete_vals[:, :n_g_tot_annot], axis=1).reshape((n_blocks, 1))

        if not indices:
            denom_delete_vals = np.dot(denom_delete_vals, np.ones((1, self.n_annot - n_annot)))

        prop = jk.RatioJackknife(coef / ave_coef, numer_delete_vals, denom_delete_vals)

        return prop.est, prop.jknife_cov, prop.jknife_se

    def _ratio_se_delta_method(self, num_mean, num_var, denom_mean, denom_var, cov):
        var = np.array(np.square(num_mean) / np.square(denom_mean) * (
        num_var / np.square(num_mean) + denom_var / np.square(denom_mean) - 2 * cov / (num_mean * denom_mean)))
        var[var < 0] = np.nan
        to_return = np.sqrt(var)
        return to_return

    def _intercept(self, jknife):
        '''Extract intercept and intercept SE from block jackknife.'''
        n_annot = self.n_annot
        intercept = jknife.est[0, n_annot]
        intercept_se = jknife.jknife_se[0, n_annot]
        return intercept, intercept_se

    def _combine_twostep_jknives(self, step1_jknife, step2_jknife, M_tot, c, Nbar=1):
        '''Combine free intercept and constrained intercept jackknives for --two-step.'''
        n_blocks, n_annot = step1_jknife.delete_values.shape
        n_annot -= 1
        if n_annot > 2:
            raise ValueError(
                'twostep not yet implemented for partitioned LD Score.')

        step1_int, _ = self._intercept(step1_jknife)
        est = np.hstack(
            (step2_jknife.est, np.array(step1_int).reshape((1, 1))))
        delete_values = np.zeros((n_blocks, n_annot + 1))
        delete_values[:, n_annot] = step1_jknife.delete_values[:, n_annot]
        delete_values[:, 0:n_annot] = step2_jknife.delete_values -\
            c * (step1_jknife.delete_values[:, n_annot] -
                 step1_int).reshape((n_blocks, n_annot))  # check this
        pseudovalues = jk.Jackknife.delete_values_to_pseudovalues(
            delete_values, est)
        jknife_est, jknife_var, jknife_se, jknife_cov = jk.Jackknife.jknife(
            pseudovalues)
        jknife = namedtuple('jknife',
                            ['est', 'jknife_se', 'jknife_est', 'jknife_var', 'jknife_cov', 'delete_values'])
        return jknife(est, jknife_se, jknife_est, jknife_var, jknife_cov, delete_values)


class Hsq(LD_Score_Regression):

    __null_intercept__ = 1

    def __init__(self, y, x, g_ld, w, N, M, G, ave_h2_cis, n_blocks=200, intercept=None, slow=False, twostep=None, old_weights=False):
        step1_ii = None
        if twostep is not None:
            step1_ii = y < twostep

        LD_Score_Regression.__init__(self, y, x, g_ld, w, N, M, G, ave_h2_cis, n_blocks, intercept=intercept,
                                      slow=slow, step1_ii=step1_ii, old_weights=old_weights)
        self.mean_chisq, self.lambda_gc = self._summarize_chisq(y)
        if not self.constrain_intercept:
            self.ratio, self.ratio_se = self._ratio(
                self.intercept, self.intercept_se, self.mean_chisq)

    def _update_func(self, x, ref_ld_tot, w_ld, N, M, Nbar, intercept=None, ii=None):
        '''
        Update function for IRWLS

        x is the output of np.linalg.lstsq.
        x[0] is the regression coefficients
        x[0].shape is (# of dimensions, 1)
        the last element of x[0] is the intercept.

        intercept is None --> free intercept
        intercept is not None --> constrained intercept
        '''
        hsq = M * x[0][0] / Nbar
        if intercept is None:
            intercept = max(x[0][1])  # divide by zero error if intercept < 0
        else:
            if ref_ld_tot.shape[1] > 1:
                raise ValueError(
                    'Design matrix has intercept column for constrained intercept regression!')

        ld = ref_ld_tot[:, 0].reshape(w_ld.shape)  # remove intercept
        w = self.weights(ld, w_ld, N, M, hsq, intercept, ii)
        return w

    def _summarize_chisq(self, chisq):
        '''Compute mean chi^2 and lambda_GC.'''
        mean_chisq = np.mean(chisq)
        # median and matrix don't play nice
        lambda_gc = np.median(np.asarray(chisq)) / 0.4549
        return mean_chisq, lambda_gc

    def _ratio(self, intercept, intercept_se, mean_chisq):
        '''Compute ratio (intercept - 1) / (mean chi^2 -1 ).'''
        if mean_chisq > 1:
            ratio_se = intercept_se / (mean_chisq - 1)
            ratio = (intercept - 1) / (mean_chisq - 1)
        else:
            ratio = 'NA'
            ratio_se = 'NA'

        return ratio, ratio_se

    def _overlap_output(self, category_names, overlap_matrix, M_annot, M_tot):
        '''LD Score regression summary for overlapping categories.'''
        overlap_matrix_prop = np.zeros([self.n_only_annot,self.n_only_annot])
        for i in range(self.n_only_annot):
            overlap_matrix_prop[i, :] = overlap_matrix[i, :] / M_annot

        prop_hsq_overlap = np.dot(
            overlap_matrix_prop, self.prop_dir.T).reshape((1, self.n_only_annot))
        prop_hsq_overlap_var = np.diag(
            np.dot(np.dot(overlap_matrix_prop, self.prop_dir_cov), overlap_matrix_prop.T))
        prop_hsq_overlap_se = np.sqrt(
            np.maximum(0, prop_hsq_overlap_var)).reshape((1, self.n_only_annot))
        one_d_convert = lambda x: np.array(x).reshape(np.prod(x.shape))
        prop_M_overlap = M_annot / M_tot
        enrichment = prop_hsq_overlap / prop_M_overlap
        enrichment_se = prop_hsq_overlap_se / prop_M_overlap

        overlap_matrix_diff = np.zeros([self.n_only_annot,self.n_only_annot])

        for i in range(self.n_only_annot):
            if not M_tot == M_annot[0,i]:
                overlap_matrix_diff[i, :] = overlap_matrix[i,:]/M_annot[0,i] - M_annot/M_tot

        diff_est = np.dot(overlap_matrix_diff, self.coef_dir)
        diff_cov = np.dot(np.dot(overlap_matrix_diff,self.coef_dir_cov),overlap_matrix_diff.T)
        diff_se = np.sqrt(np.diag(diff_cov))
        diff_p = ['NA' if diff_se[i]==0 else 2*tdist.sf(abs(diff_est[i]/diff_se[i]),self.n_blocks) for i in range(self.n_only_annot)]

        df = pd.DataFrame({
            'Category': category_names,
            'Prop_SNPs': one_d_convert(prop_M_overlap),
            'Prop_h2': one_d_convert(prop_hsq_overlap),
            'Prop_h2_std_error': one_d_convert(prop_hsq_overlap_se),
            'Enrichment_h2': one_d_convert(enrichment),
            'Enrichment_h2_std_error': one_d_convert(enrichment_se),
            'Enrichment_pvalue': diff_p,
            'Coefficient': one_d_convert(self.coef_dir),
            'Coefficient_std_error': one_d_convert(self.coef_dir_se),
        }, columns=['Category', 'Prop_SNPs', 'Prop_h2', 'Prop_h2_std_error', 'Enrichment_h2', 'Enrichment_h2_std_error',
                    'Enrichment_pvalue', 'Coefficient', 'Coefficient_std_error'])

        return df

    def _g_overlap_output(self, category_names, g_overlap_matrix, G_annot, G_tot, ave_h2_cis_annot, g_groups):
        '''Gene LD Score regression summary for overlapping categories.'''
        one_d_convert = lambda x: np.array(x).reshape(np.prod(x.shape))

        g_groups = pd.Series(g_groups, dtype='category')
        category_names = category_names.append(g_groups.cat.categories)
        cis_indices = [i for i, x in enumerate(g_groups == 'Cis_herit_bin') if x]

        g_overlap_matrix_prop = np.zeros([self.n_g_annot, self.n_g_annot])
        for i in range(self.n_g_annot):
            g_overlap_matrix_prop[i, :] = g_overlap_matrix[i, :] / G_annot

        ## Absolute mediated heritability
        hsq = np.dot(g_overlap_matrix_prop, self.cat_g.T).reshape((1, self.n_g_annot))
        hsq_cov = np.dot(np.dot(g_overlap_matrix_prop, self.cat_g_cov), g_overlap_matrix_prop.T)
        hsq_se = np.sqrt(np.maximum(0, np.diag(hsq_cov))).reshape((1, self.n_g_annot))

        ## Proportion of mediated heritability
        hsq_overlap = np.dot(g_overlap_matrix_prop, self.prop_g.T).reshape((1, self.n_g_annot))
        hsq_overlap_cov = np.dot(np.dot(g_overlap_matrix_prop, self.prop_g_cov), g_overlap_matrix_prop.T)
        hsq_overlap_se_jkknife = np.sqrt(np.maximum(0, np.diag(hsq_overlap_cov))).reshape((1, self.n_g_annot))

        tot_cov = np.sum(hsq_cov[:,cis_indices], axis=1)
        hsq_overlap_se = self._ratio_se_delta_method(hsq, np.square(hsq_se), self.tot_g, self.tot_g_cov, tot_cov)

        ## Causal per-gene effects
        g_overlap_matrix_per = np.zeros([self.n_g_annot, self.n_g_annot])
        for i in range(self.n_g_annot):
            g_overlap_matrix_per[i, :] = g_overlap_matrix[i, :] / G_annot[0, i]
        g_causal = np.dot(g_overlap_matrix_per, self.coef_g).reshape((1, self.n_g_annot))
        g_causal_cov = np.dot(np.dot(g_overlap_matrix_per, self.coef_g_cov), g_overlap_matrix_per.T)
        g_causal_se = np.sqrt(np.maximum(0, np.diag(g_causal_cov))).reshape((1, self.n_g_annot))

        ## Total effs
        ## Total per-gene causal effects
        cis_g_causal = g_causal[:, cis_indices]
        cis_g_causal_cov = g_causal_cov[min(cis_indices):max(cis_indices)+1, min(cis_indices):max(cis_indices)+1]
        tot_cis_g_causal, tot_cis_g_causal_cov, _ = self._tot(cis_g_causal, cis_g_causal_cov)
        tot_cis_g_causal /= len(cis_indices)
        tot_cis_g_causal_cov /= np.square(len(cis_indices))

        ## Total per-gene mediated heritability
        per_tot_herit = self.tot_g / G_tot
        per_tot_herit_cov = self.tot_g_cov / np.square(G_tot)

        ## Enrichment of mediated heritability
        g_enrichment_herit, g_enrichment_herit_se_jkknife, _ = self._enrichment(G_annot, G_tot, hsq_overlap, hsq_overlap_se_jkknife)
        g_enrichment_herit, g_enrichment_herit_se, _ = self._enrichment(G_annot, G_tot, hsq_overlap, hsq_overlap_se)

        ## Enrichment of mediated heritability p-value
        overlap_matrix_diff = np.zeros([self.n_g_annot, self.n_g_annot])
        for i in range(self.n_g_annot):
            if not G_tot == G_annot[0, i]:
                overlap_matrix_diff[i, :] = g_overlap_matrix[i, :]/(G_annot*G_annot[0, i]) - 1.0/G_tot

        diff_est = np.dot(overlap_matrix_diff, self.cat_g.T).T
        diff_cov = np.dot(np.dot(overlap_matrix_diff, self.cat_g_cov), overlap_matrix_diff.T)
        diff_se = np.sqrt(np.diag(diff_cov))
        diff_p = ['NA' if diff_se[i] == 0 else 2 * tdist.sf(abs(diff_est[0,i] / diff_se[i]), self.n_blocks) for i in range(self.n_g_annot)]

        ## Enrichment of causal per-gene effects
        g_enrichment_causal, _, g_enrichment_causal_se_jkknife = self._enrichment_g(g_overlap_matrix_per, g_causal, tot_cis_g_causal, self.jknife,
                                                            self.n_only_annot, len(cis_indices))
        g_cov_all = 1.0/len(cis_indices) * np.sum(g_causal_cov[:,cis_indices], axis=1)
        g_enrichment_causal_se = self._ratio_se_delta_method(g_causal, np.square(g_causal_se), tot_cis_g_causal, tot_cis_g_causal_cov, g_cov_all)

        ## Enrichment of causal per-gene effects p-value
        diff_causal_est = np.zeros(self.n_g_annot)
        diff_causal_se = np.zeros(self.n_g_annot)

        for i in range(self.n_g_annot):
            diff_causal_est[i] = g_causal[0, i] - tot_cis_g_causal
            diff_causal_se[i] = np.sqrt(np.maximum(0, np.square(g_causal_se[0, i]) + tot_cis_g_causal_cov - 2.0/len(cis_indices)*np.sum(g_causal_cov[i, cis_indices])))

        diff_causal_p = ['NA' if diff_causal_se[i] == 0 else 2 * tdist.sf(abs(diff_causal_est[i] / diff_causal_se[i]), self.n_blocks) for i in range(self.n_g_annot)]

        for i in g_groups.cat.categories:
            temp_indices = [j for j,x in enumerate(g_groups==i) if x]
            temp_G = np.sum(G_annot[:,temp_indices])
            temp_ave_h2_cis = np.sum(np.multiply(G_annot[:,temp_indices], ave_h2_cis_annot[:,temp_indices])) / temp_G

            G_annot = np.c_[G_annot, temp_G]
            ave_h2_cis_annot = np.c_[ave_h2_cis_annot, temp_ave_h2_cis]

            ## Coefficient
            temp_coef = self.coef_g[temp_indices]
            temp_coef_cov = self.coef_g_cov[min(temp_indices):max(temp_indices)+1, min(temp_indices):max(temp_indices)+1]
            temp_tot_coef, _, temp_tot_coef_se = self._tot(temp_coef, temp_coef_cov)
            self.coef_g = np.append(self.coef_g, temp_tot_coef)
            self.coef_g_se = np.append(self.coef_g_se, temp_tot_coef_se)

            ## Proportion of mediated heritability
            temp_hsq_overlap = hsq_overlap[:,temp_indices]
            temp_hsq_overlap_cov = hsq_overlap_cov[min(temp_indices):max(temp_indices)+1, min(temp_indices):max(temp_indices)+1]
            combined_hsq_overlap, _, combined_hsq_overlap_se_jkknife = self._tot(temp_hsq_overlap, temp_hsq_overlap_cov)

            combined_cov = np.sum(hsq_cov[min(temp_indices):max(temp_indices)+1, :len(cis_indices)])
            combined_mean, combined_var, _ = self._tot(hsq[:,temp_indices], hsq_cov[min(temp_indices):max(temp_indices)+1, min(temp_indices):max(temp_indices)+1])
            combined_hsq_overlap_se = self._ratio_se_delta_method(combined_mean, combined_var, self.tot_g, self.tot_g_cov, combined_cov)

            hsq_overlap = np.c_[hsq_overlap, combined_hsq_overlap]
            hsq_overlap_se = np.c_[hsq_overlap_se, combined_hsq_overlap_se]
            hsq_overlap_se_jkknife = np.c_[hsq_overlap_se_jkknife, combined_hsq_overlap_se_jkknife]

            ## Absolute mediated heritability
            temp_hsq = hsq[:, temp_indices]
            temp_hsq_cov = hsq_cov[min(temp_indices):max(temp_indices) + 1, min(temp_indices):max(temp_indices) + 1]
            combined_hsq, _, combined_hsq_se = self._tot(temp_hsq, temp_hsq_cov)
            hsq = np.c_[hsq, combined_hsq]
            hsq_se = np.c_[hsq_se, combined_hsq_se]

            ## Absolute per-gene mediated heritability
            combined_per_hsq = combined_hsq / temp_G
            combined_per_hsq_se = combined_hsq_se / temp_G

            ## Causal per-gene effects
            temp_causal = g_causal[:,temp_indices]
            temp_causal_cov = g_causal_cov[min(temp_indices):max(temp_indices)+1, min(temp_indices):max(temp_indices)+1]
            combined_causal, _, combined_causal_se = self._tot(temp_causal, temp_causal_cov)
            combined_causal /= len(temp_indices)
            combined_causal_se /= len(temp_indices)
            g_causal = np.c_[g_causal, combined_causal]
            g_causal_se = np.c_[g_causal_se, combined_causal_se]

            ## Enrichment of mediated heritability
            combined_enrichment_herit, combined_enrichment_herit_se_jkknife, _ = self._enrichment(temp_G, G_tot, combined_hsq_overlap, combined_hsq_overlap_se_jkknife)
            combined_enrichment_herit, combined_enrichment_herit_se, _ = self._enrichment(temp_G, G_tot, combined_hsq_overlap, combined_hsq_overlap_se)
            g_enrichment_herit = np.c_[g_enrichment_herit, combined_enrichment_herit]
            g_enrichment_herit_se = np.c_[g_enrichment_herit_se, combined_enrichment_herit_se]
            g_enrichment_herit_se_jkknife = np.c_[g_enrichment_herit_se_jkknife, combined_enrichment_herit_se_jkknife]

            ## Enrichment of causal per-gene effects
            combined_enrichment_causal, _, combined_enrichment_causal_se_jkknife = self._enrichment_g(
                g_overlap_matrix_per, np.array(np.mean(temp_causal)).reshape((1,1)), tot_cis_g_causal, self.jknife, self.n_only_annot,
                len(cis_indices), indices=temp_indices)

            g_cov_all = 1.0 / (len(cis_indices) * len(temp_indices)) * np.sum(g_causal_cov[min(temp_indices):max(temp_indices)+1, :len(cis_indices)])
            combined_enrichment_causal_se = self._ratio_se_delta_method(combined_causal, np.square(combined_causal_se), tot_cis_g_causal,
                                                                 tot_cis_g_causal_cov, g_cov_all)

            g_enrichment_causal = np.c_[g_enrichment_causal, combined_enrichment_causal]
            g_enrichment_causal_se = np.append(g_enrichment_causal_se, combined_enrichment_causal_se)
            g_enrichment_causal_se_jkknife = np.append(g_enrichment_causal_se_jkknife, combined_enrichment_causal_se_jkknife)

            if i != 'Cis_herit_bin':
                diff_herit_est = combined_per_hsq - per_tot_herit
                diff_herit_se = np.sqrt(np.maximum(0, np.square(combined_per_hsq_se) + per_tot_herit_cov - 2.0/(temp_G * G_tot) *
                                        np.sum(hsq_cov[min(temp_indices):max(temp_indices)+1, min(cis_indices):max(cis_indices)+1])))

                diff_causal_est = combined_causal - tot_cis_g_causal
                diff_causal_se = np.sqrt(np.maximum(0,
                        np.square(combined_causal_se) + tot_cis_g_causal_cov - 2.0/(len(cis_indices)*len(temp_indices)) *
                        np.sum(g_causal_cov[min(temp_indices):max(temp_indices)+1, min(cis_indices):max(cis_indices)+1])))

                combined_herit_p = 2 * tdist.sf(abs(diff_herit_est / diff_herit_se), self.n_blocks)
                combined_causal_p = 2 * tdist.sf(abs(diff_causal_est / diff_causal_se), self.n_blocks)
            else:
                combined_herit_p = 'NA'
                combined_causal_p = 'NA'

            diff_p.append(combined_herit_p)
            diff_causal_p.append(combined_causal_p)

        df = pd.DataFrame({
            'Category': category_names,
            'Num_genes': one_d_convert(G_annot),
            'Prop_genes': one_d_convert(G_annot / G_tot),
            'Ave_h2_cis': one_d_convert(ave_h2_cis_annot),
            'h2': one_d_convert(hsq),
            'h2_std_error': one_d_convert(hsq_se),
            'Prop_h2': one_d_convert(hsq_overlap),
            'Prop_h2_std_error_dm': one_d_convert(hsq_overlap_se),
            'Prop_h2_std_error_jkknife': one_d_convert(hsq_overlap_se_jkknife),
            'Enrichment_h2': one_d_convert(g_enrichment_herit),
            'Enrichment_h2_std_error_dm': one_d_convert(g_enrichment_herit_se),
            'Enrichment_h2_std_error_jkknife': one_d_convert(g_enrichment_herit_se_jkknife),
            'Enrichment_h2_pvalue': diff_p,
            'Coefficient': one_d_convert(self.coef_g),
            'Coefficient_std_error': one_d_convert(self.coef_g_se),
        }, columns=['Category', 'Num_genes', 'Prop_genes', 'Ave_h2_cis', 'h2', 'h2_std_error', 'Prop_h2', 'Prop_h2_std_error_dm', 'Prop_h2_std_error_jkknife',
                    'Enrichment_h2', 'Enrichment_h2_std_error_dm', 'Enrichment_h2_std_error_jkknife', 'Enrichment_h2_pvalue', 'Coefficient', 'Coefficient_std_error'])

        df_g = pd.DataFrame({
            'Category': category_names,
            'Enrichment_causal_eff': one_d_convert(g_enrichment_causal),
            'Enrichment_causal_eff_std_error_dm': one_d_convert(g_enrichment_causal_se),
            'Enrichment_causal_eff_std_error_jkknife': one_d_convert(g_enrichment_causal_se_jkknife),
            'Enrichment_causal_eff_pvalue': diff_causal_p,
        }, columns=['Category', 'Enrichment_causal_eff', 'Enrichment_causal_eff_std_error_dm','Enrichment_causal_eff_std_error_jkknife','Enrichment_causal_eff_pvalue'])

        return df, df_g

    def summary(self, ref_ld_colnames=None, P=None, K=None, overlap=False):
        '''Print summary of the LD Score Regression.'''
        if P is not None and K is not None:
            T = 'Liability'
            c = h2_obs_to_liab(1, P, K)
        else:
            T = 'Observed'
            c = 1

        out = ['Total ' + T + ' scale h2: ' +
               s(c * self.tot) + ' (' + s(c * self.tot_se) + ')']
        if self.n_annot > 1:
            if ref_ld_colnames is None:
                ref_ld_colnames = ['CAT_' + str(i)
                                   for i in xrange(self.n_annot)]

            out.append('Categories: ' + ' '.join(ref_ld_colnames))

            if not overlap:
                out.append(T + ' scale h2: ' + s(c * self.cat))
                out.append(T + ' scale h2 SE: ' + s(c * self.cat_se))
                out.append('Proportion of SNPs: ' + s(self.M_prop))
                out.append('Proportion of h2g: ' + s(self.prop))
                out.append('Enrichment: ' + s(self.enrichment))
                out.append('Coefficients: ' + s(self.coef))
                out.append('Coefficient SE: ' + s(self.coef_se))

        out.append('Lambda GC: ' + s(self.lambda_gc))
        out.append('Mean Chi^2: ' + s(self.mean_chisq))
        if self.constrain_intercept:
            out.append(
                'Intercept: constrained to {C}'.format(C=s(self.intercept)))
        else:
            out.append(
                'Intercept: ' + s(self.intercept) + ' (' + s(self.intercept_se) + ')')
            if self.mean_chisq > 1:
                if self.ratio < 0:
                    out.append(
                      'Ratio < 0 (usually indicates GC correction).')
                else:
                    out.append(
                      'Ratio: ' + s(self.ratio) + ' (' + s(self.ratio_se) + ')')
            else:
                out.append('Ratio: NA (mean chi^2 < 1)')

        return remove_brackets('\n'.join(out))

    def _update_weights(self, ld, w_ld, N, M, hsq, intercept, ii=None):
        if intercept is None:
            intercept = self.__null_intercept__

        return self.weights(ld, w_ld, N, M, hsq, intercept, ii)

    @classmethod
    def weights(cls, ld, w_ld, N, M, hsq, intercept=None, ii=None):
        '''
        Regression weights.

        Parameters
        ----------
        ld : np.matrix with shape (n_snp, 1)
            LD Scores (non-partitioned).
        w_ld : np.matrix with shape (n_snp, 1)
            LD Scores (non-partitioned) computed with sum r^2 taken over only those SNPs included
            in the regression.
        N :  np.matrix of ints > 0 with shape (n_snp, 1)
            Number of individuals sampled for each SNP.
        M : float > 0
            Number of SNPs used for estimating LD Score (need not equal number of SNPs included in
            the regression).
        hsq : float in [0,1]
            Heritability estimate.

        Returns
        -------
        w : np.matrix with shape (n_snp, 1)
            Regression weights. Approx equal to reciprocal of conditional variance function.

        '''
        M = float(M)
        if intercept is None:
            intercept = 1

        hsq = max(hsq, 0.0)
        hsq = min(hsq, 1.0)
        ld = np.fmax(ld, 1.0)
        w_ld = np.fmax(w_ld, 1.0)
        c = hsq * N / M
        het_w = 1.0 / (2 * np.square(intercept + np.multiply(c, ld)))
        oc_w = 1.0 / w_ld
        w = np.multiply(het_w, oc_w)
        return w


class Gencov(LD_Score_Regression):
    __null_intercept__ = 0

    def __init__(self, z1, z2, x, w, N1, N2, M, hsq1, hsq2, intercept_hsq1, intercept_hsq2,
                 n_blocks=200, intercept_gencov=None, slow=False, twostep=None):
        self.intercept_hsq1 = intercept_hsq1
        self.intercept_hsq2 = intercept_hsq2
        self.hsq1 = hsq1
        self.hsq2 = hsq2
        self.N1 = N1
        self.N2 = N2
        y = z1 * z2
        step1_ii = None
        if twostep is not None:
            step1_ii = np.logical_and(z1**2 < twostep, z2**2 < twostep)

        LD_Score_Regression.__init__(self, y, x, w, np.sqrt(N1 * N2), M, n_blocks,
                                     intercept=intercept_gencov, slow=slow, step1_ii=step1_ii)
        self.p, self.z = p_z_norm(self.tot, self.tot_se)
        self.mean_z1z2 = np.mean(np.multiply(z1, z2))

    def summary(self, ref_ld_colnames, P=None, K=None):
        '''Print summary of the LD Score regression.'''
        out = []
        if P is not None and K is not None and\
                all((i is not None for i in P)) and all((i is not None for i in K)):
            T = 'Liability'
            c = gencov_obs_to_liab(1, P[0], P[1], K[0], K[1])
        else:
            T = 'Observed'
            c = 1

        out.append('Total ' + T + ' scale gencov: ' +
                   s(self.tot) + ' (' + s(self.tot_se) + ')')
        if self.n_annot > 1:
            out.append('Categories: ' + str(' '.join(ref_ld_colnames)))
            out.append(T + ' scale gencov: ' + s(c * self.cat))
            out.append(T + ' scale gencov SE: ' + s(c * self.cat_se))
            out.append('Proportion of SNPs: ' + s(self.M_prop))
            out.append('Proportion of gencov: ' + s(self.prop))
            out.append('Enrichment: ' + s(self.enrichment))

        out.append('Mean z1*z2: ' + s(self.mean_z1z2))
        if self.constrain_intercept:
            out.append(
                'Intercept: constrained to {C}'.format(C=s(self.intercept)))
        else:
            out.append(
                'Intercept: ' + s(self.intercept) + ' (' + s(self.intercept_se) + ')')

        return remove_brackets('\n'.join(out))

    def _update_func(self, x, ref_ld_tot, w_ld, N, M, Nbar, intercept=None, ii=None):
        '''
        Update function for IRWLS
        x is the output of np.linalg.lstsq.
        x[0] is the regression coefficients
        x[0].shape is (# of dimensions, 1)
        the last element of x[0] is the intercept.

        '''
        rho_g = M * x[0][0] / Nbar
        if intercept is None:  # if the regression includes an intercept
            intercept = x[0][1]

        # remove intercept if we have one
        ld = ref_ld_tot[:, 0].reshape(w_ld.shape)
        if ii is not None:
            N1 = self.N1[ii].reshape((w_ld.shape))
            N2 = self.N2[ii].reshape((w_ld.shape))
        else:
            N1 = self.N1
            N2 = self.N2

        return self.weights(ld, w_ld, N1, N2, np.sum(M), self.hsq1, self.hsq2, rho_g,
                         intercept, self.intercept_hsq1, self.intercept_hsq2, ii)

    def _update_weights(self, ld, w_ld, sqrt_n1n2, M, rho_g, intercept, ii=None):
        '''Weight function with the same signature for Hsq and Gencov.'''
        w = self.weights(ld, w_ld, self.N1, self.N2, M, self.hsq1, self.hsq2, rho_g,
                         intercept, self.intercept_hsq1, self.intercept_hsq2)
        return w

    @classmethod
    def weights(cls, ld, w_ld, N1, N2, M, h1, h2, rho_g, intercept_gencov=None,
                intercept_hsq1=None, intercept_hsq2=None, ii=None):
        '''
        Regression weights.

        Parameters
        ----------
        ld : np.matrix with shape (n_snp, 1)
            LD Scores (non-partitioned)
        w_ld : np.matrix with shape (n_snp, 1)
            LD Scores (non-partitioned) computed with sum r^2 taken over only those SNPs included
            in the regression.
        M : float > 0
            Number of SNPs used for estimating LD Score (need not equal number of SNPs included in
            the regression).
        N1, N2 :  np.matrix of ints > 0 with shape (n_snp, 1)
            Number of individuals sampled for each SNP for each study.
        h1, h2 : float in [0,1]
            Heritability estimates for each study.
        rhog : float in [0,1]
            Genetic covariance estimate.
        intercept : float
            Genetic covariance intercept, on the z1*z2 scale (so should be Ns*rho/sqrt(N1*N2)).

        Returns
        -------
        w : np.matrix with shape (n_snp, 1)
            Regression weights. Approx equal to reciprocal of conditional variance function.

        '''
        M = float(M)
        if intercept_gencov is None:
            intercept_gencov = 0
        if intercept_hsq1 is None:
            intercept_hsq1 = 1
        if intercept_hsq2 is None:
            intercept_hsq2 = 1

        h1, h2 = max(h1, 0.0), max(h2, 0.0)
        h1, h2 = min(h1, 1.0), min(h2, 1.0)
        rho_g = min(rho_g, 1.0)
        rho_g = max(rho_g, -1.0)
        ld = np.fmax(ld, 1.0)
        w_ld = np.fmax(w_ld, 1.0)
        a = np.multiply(N1, h1 * ld) / M + intercept_hsq1
        b = np.multiply(N2, h2 * ld) / M + intercept_hsq2
        sqrt_n1n2 = np.sqrt(np.multiply(N1, N2))
        c = np.multiply(sqrt_n1n2, rho_g * ld) / M + intercept_gencov
        try:
            het_w = 1.0 / (np.multiply(a, b) + np.square(c))
        except FloatingPointError:  # bizarre error; should never happen
            raise FloatingPointError('Why did you set hsq intercept <= 0?')

        oc_w = 1.0 / w_ld
        w = np.multiply(het_w, oc_w)
        return w


class RG(object):

    def __init__(self, z1, z2, x, w, N1, N2, M, intercept_hsq1=None, intercept_hsq2=None,
                 intercept_gencov=None, n_blocks=200, slow=False, twostep=None):
        self.intercept_gencov = intercept_gencov
        self._negative_hsq = None
        n_snp, n_annot = x.shape
        hsq1 = Hsq(np.square(z1), x, w, N1, M, n_blocks=n_blocks, intercept=intercept_hsq1,
                   slow=slow, twostep=twostep)
        hsq2 = Hsq(np.square(z2), x, w, N2, M, n_blocks=n_blocks, intercept=intercept_hsq2,
                   slow=slow, twostep=twostep)
        gencov = Gencov(z1, z2, x, w, N1, N2, M, hsq1.tot, hsq2.tot, hsq1.intercept,
                        hsq2.intercept, n_blocks, intercept_gencov=intercept_gencov, slow=slow,
                        twostep=twostep)
        gencov.N1 = None  # save memory
        gencov.N2 = None
        self.hsq1, self.hsq2, self.gencov = hsq1, hsq2, gencov
        if (hsq1.tot <= 0 or hsq2.tot <= 0):
            self._negative_hsq = True
            self.rg_ratio = self.rg = self.rg_se = 'NA'
            self.p = self.z = 'NA'
        else:
            rg_ratio = np.array(
                gencov.tot / np.sqrt(hsq1.tot * hsq2.tot)).reshape((1, 1))
            denom_delete_values = np.sqrt(
                np.multiply(hsq1.tot_delete_values, hsq2.tot_delete_values))
            rg = jk.RatioJackknife(
                rg_ratio, gencov.tot_delete_values, denom_delete_values)
            self.rg_jknife = float(rg.jknife_est)
            self.rg_se = float(rg.jknife_se)
            self.rg_ratio = float(rg_ratio)
            self.p, self.z = p_z_norm(self.rg_ratio, self.rg_se)

    def summary(self, silly=False):
        '''Print output of Gencor object.'''
        out = []
        if self._negative_hsq:
            out.append('Genetic Correlation: nan (nan) (h2  out of bounds) ')
            out.append('Z-score: nan (nan) (h2  out of bounds)')
            out.append('P: nan (nan) (h2  out of bounds)')
            out.append('WARNING: One of the h2\'s was out of bounds.')
            out.append(
                'This usually indicates a data-munging error ' +
                'or that h2 or N is low.')
        elif (self.rg_ratio > 1.2 or self.rg_ratio < -1.2) and not silly:
            out.append('Genetic Correlation: nan (nan) (rg out of bounds) ')
            out.append('Z-score: nan (nan) (rg out of bounds)')
            out.append('P: nan (nan) (rg out of bounds)')
            out.append('WARNING: rg was out of bounds.')
            if self.intercept_gencov is None:
                out.append(
                    'This often means that h2 is not significantly ' +
                    'different from zero.')
            else:
                out.append(
                           'This often means that you have constrained' +
                           ' the intercepts to the wrong values.')
        else:
            out.append(
                'Genetic Correlation: ' + s(self.rg_ratio) +
                ' (' + s(self.rg_se) + ')')
            out.append('Z-score: ' + s(self.z))
            out.append('P: ' + s(self.p))
        return remove_brackets('\n'.join(out))
