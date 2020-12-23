from torch.utils.data import Dataset
import time
import h5py
from scipy.io import wavfile
import torch.nn as nn
import os
import random
import glob
import soundfile as sf
from scipy import signal
import torch.nn.functional as F
from scipy import linalg as la
from numpy.random import randn
import h5py as h5
import scipy.linalg as linalg
from scipy.linalg import eigvalsh, inv
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


class RandomSampleDataset(Dataset):
    def __init__(
        self,
        f,
        sample_length_range,
        n_blocks,
        sample_per_epoch,
        min_frames_per_utt,
        meta_data_file=None,
        balance_class=True,
        min_utts_per_spk=8,
    ):
        self.n_blocks = n_blocks
        if meta_data_file:
            with h5py.File(meta_data_file, 'r') as f_meta:
                self.df = self.get_df(f_meta, min_utts_per_spk, min_frames_per_utt)
        else:
            self.df = self.get_df(f, min_utts_per_spk, min_frames_per_utt)

        self.smp_per_epo = sample_per_epoch
        self.mfccs = f["mfcc"]
        self.balance_class = balance_class
        self.smp_len_min, self.smp_len_max = sample_length_range

    def sample(self):
        np.random.seed()
        sample_length = np.random.randint(low=self.smp_len_min, high=self.smp_len_max)
        if self.balance_class:
            prob = (1 / self.df["spk_ids"].nunique() / self.df["n_utts"]).values
            idxes = np.random.choice(len(self.df), p=prob, size=self.smp_per_epo)
            self.spk_ids, self.utt_ids, self.positions = self.sample_segments(
                self.df.iloc[idxes], sample_length
            )
        else:
            idxes = np.random.randint(len(self.df), size=self.smp_per_epo)
            self.spk_ids, self.utt_ids, self.positions = self.sample_segments(
                self.df.iloc[idxes], sample_length
            )

    def sample_original(self):
        np.random.seed()
        sample_length = np.random.randint(low=self.smp_len_min, high=self.smp_len_max)
        if self.balance_class:
            # remove augment data
            mask = ~self.df.utt_ids.str.endswith(('babble', 'music', 'noise', 'reverb'))
            df = self.df[mask]
            utt_counts = df.spk_ids.value_counts()
            df["n_utts"] = df.spk_ids.map(utt_counts)

            prob = (1 / df["spk_ids"].nunique() / df["n_utts"]).values
            idxes = np.random.choice(len(df), p=prob, size=self.smp_per_epo)
            self.spk_ids, self.utt_ids, self.positions = self.sample_segments(
                df.iloc[idxes], sample_length
            )
        else:
            raise NotImplementedError

    def __getitem__(self, i):
        spk_id = self.spk_ids[i]
        utt_id = self.utt_ids[i]
        sub_positions = self.positions[i]
        mfccs = []
        for start, end in sub_positions:
            mfccs.append(self.mfccs[start:end])
        mfccs = np.concatenate(mfccs).T
        return torch.tensor(mfccs), torch.tensor(spk_id), utt_id

    def __len__(self):
        return self.smp_per_epo

    def sample_segments(self, df_smp, sample_length):
        spk_ids = df_smp.spk_ids.values
        utt_ids = df_smp.utt_ids.values
        positions = []
        for start, end in df_smp[["starts", "ends"]].values:
            sub_positions = []
            for _ in range(self.n_blocks):
                smp_start = np.random.randint(low=start, high=end - sample_length)
                smp_end = smp_start + sample_length
                sub_positions.append([smp_start, smp_end])
            positions.append(sub_positions)
        positions = np.array(positions)
        return spk_ids, utt_ids, positions

    @staticmethod
    def get_df(f, min_utts_per_spk, min_frames_per_utt):
        df = pd.DataFrame(
            {
                "spk_ids": f["spk_ids"][:],
                "utt_ids": f["utt_ids"][:],
                "starts": f["positions"][:, 0],
                "ends": f["positions"][:, 1],
            }
        )
        df = df[(df.ends - df.starts) > min_frames_per_utt]
        utt_counts = df.spk_ids.value_counts()
        df["n_utts"] = df.spk_ids.map(utt_counts)
        df = df[df.n_utts > min_utts_per_spk]
        df["spk_ids"] = LabelEncoder().fit_transform(df["spk_ids"])
        # df['utt_ids'] = LabelEncoder().fit_transform(df['utt_ids'])
        return df



def extract_collate(batch):
    assert len(batch) == 1
    x = batch[0][0]
    spk_id = batch[0][1]
    utt_id = batch[0][2]
    return [x[None, ...], spk_id, utt_id]


class ExtractDataset(Dataset):
    def __init__(self, f, meta_data=None):
        self.mfccs = f['mfcc']
        if meta_data:
            with h5py.File(meta_data, 'r') as f_meta:
                self.positions = f_meta['positions'][:]
                self.spk_ids = f_meta['spk_ids'][:]
                self.utt_ids = f_meta['utt_ids'][:]
        else:
            self.positions = f['positions'][:]
            self.spk_ids = f['spk_ids'][:]
            self.utt_ids = f['utt_ids'][:]

    def __getitem__(self, idx):
        start, end = self.positions[idx]
        spk_id = self.spk_ids[idx]
        utt_id = self.utt_ids[idx]
        mfcc = self.mfccs[start:end].T
        return torch.tensor(mfcc), spk_id, utt_id

    def __len__(self):
        return len(self.positions)


class OuterDataset(Dataset):
    def __init__(self, file_lst, logger, inner_batch_size, padding=False):
        self.batch_size = inner_batch_size
        self.logger = logger
        if not padding:
            self.file_lst = file_lst
        else:
            self.file_lst = np.tile(file_lst, 30).tolist()
        # self.file_lst = np.tile(file_lst, 20).tolist()[:total_step]

    def __getitem__(self, idx):
        t0 = time.time()
        with h5py.File(self.file_lst[idx], 'r') as f:
            data = {
                'positions': f['positions'][:],
                'spk_ids': f['spk_ids'][:],
                'utt_ids': f['utt_ids'][:],
                'mfcc': f['mfcc'][:],
            }
        with open(self.logger, 'a') as f:
            f.write(f'disk time is {time.time()-t0}\n')
        inner = InnerDataset(data)
        inner_loader = torch.utils.data.DataLoader(inner,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   drop_last=True,
                                                   )
        return inner_loader

    def __len__(self):
        return len(self.file_lst)


class InnerDataset(Dataset):
    def __init__(self, data):
        assert type(data) in [dict, h5py._hl.files.File]
        self.mfccs = data['mfcc']
        self.positions = data['positions'][:]
        self.spk_ids = data['spk_ids'][:]
        self.utt_ids = data['utt_ids'][:]

    def __getitem__(self, idx):
        start, end = self.positions[idx]
        spk_id = self.spk_ids[idx]
        utt_id = self.utt_ids[idx]
        mfcc = self.mfccs[start:end].T
        return torch.tensor(mfcc), torch.tensor(spk_id), utt_id

    def __len__(self):
        return len(self.positions)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


unicode = h5py.special_dtype(vlen=str)


def h5list2dict(files):
    with h5py.File(files[0], 'r') as f:
        attributs = list(f.keys())

    data = {attribut: [] for attribut in attributs}

    for file in files:
        with h5py.File(file, 'r') as f:
            assert set(attributs) == set(list(f.keys()))
            for attribut in attributs:
                data[attribut].append(f[attribut][...])

    data = {attribut: np.concatenate(data[attribut]) for attribut in attributs}
    return data


def h52dict(file):
    if type(file) is str:
        return h5single2dict(file)
    elif type(file) is list:
        return h5list2dict(file)
    else:
        raise NotImplementedError




class Score:
    def __init__(self, enroll, test, ndx_file,
                 comp_minDCF=True,
                 score_using_spk_ids=False,
                 average_by=None,
                 top_scores=200,
                 preserve_trial_order=False,
                 cohort=None, transforms=None, group_infos=None,
                 blind_trial=False, save_scores_to=None):
        # print('loading cohort ivc from {}'.format(cohort))
        # Todo implement score snorm
        # Todo xvector average
        self.top_scores = top_scores
        self.comp_minDCF = comp_minDCF
        self.preserve_trial_order = preserve_trial_order
        if (type(test) is str) or (type(test) is list):
            self.test = h52dict(test)
            print('loading test ivc from {}'.format(test))
        else:
            self.test = test

        if type(enroll) is str or (type(enroll) is list):
            self.enroll = h52dict(enroll)
            print('loading enroll ivc from {}'.format(enroll))
        else:
            self.enroll = enroll

        if average_by == 'spk_ids' or score_using_spk_ids:
            self.enroll = average_xvec(self.enroll, 'spk_ids')
            # self.test = average_xvec(self.test, 'spk_ids')
        elif average_by == 'utt_ids':
            self.enroll = average_xvec(self.enroll, 'spk_path')
            self.test = average_xvec(self.test, 'spk_path')
        else:
            print('no average')


        #Todo temp need fix
        if score_using_spk_ids:
            # self.test['spk_ids'] = self.test['spk_path']
            self.enroll['spk_path'] = self.enroll['spk_ids']
        else:
            self.enroll['spk_ids'] = self.enroll['spk_path']
        # self.test['spk_ids'] = self.test['spk_path']
        self.test['spk_ids'] = self.test['spk_path']
        # breakpoint()
        if cohort:
            # self.cohort = h52dict(cohort_file)

            if type(cohort) is str or (type(cohort) is list):
                self.cohort = h52dict(cohort)
                print('loading cohort_file ivc from {}'.format(cohort))
            else:
                self.cohort = cohort

        #Todo temp need fix
        if cohort:
            self.cohort['spk_ids'] = self.cohort['spk_path']


        if transforms:
            for transform in transforms:
                self.enroll['X'] = transform.transform(self.enroll['X'])
                self.test['X'] = transform.transform(self.test['X'])
                if cohort:
                    self.cohort['X'] = transform.transform(self.cohort['X'])
        if group_infos:
            for kind, file in group_infos.items():
                data = getattr(self, kind)
                group_info = pd.read_csv(file, sep='\t', header=None, names=['utt_id', 'group'])
                group_info = group_info.set_index('utt_id').group
                data['group'] = pd.Series(data['spk_path']).map(group_info)
                mask = (data['group']).isna()
                data['group'] = data['group'].values
                print(f'total {mask.sum()} NAN in found in {kind}')
                setattr(self, kind, mask_dict(data, ~mask))

        self.enroll['X_origin'] = self.enroll['X'].copy()
        self.test['X_origin'] = self.test['X'].copy()
        if cohort:
            self.cohort['X_origin'] = self.cohort['X'].copy()
        # Todo add format checking
        self.blind_trial = blind_trial
        if not blind_trial:
            # self.ndx = pd.read_csv(ndx_file, sep='\t', usecols=['enroll', 'test', 'label'])
            # self.ndx['label'] = self.ndx.label.map({'target': 1, 'nontarget': 0})
            self.ndx = (
                pd.read_csv(ndx_file, sep='\t', dtype=str,
                            usecols=['modelid', 'segmentid', 'targettype'])
                  .rename(columns={'modelid': 'enroll', 'segmentid': 'test', 'targettype': 'label'})
            )
            self.ndx['label'] = self.ndx.label.map({'target': 1, 'nontarget': 0})
            if preserve_trial_order:
                self.ndx = self.ndx.groupby(['enroll', 'test']).apply(lambda x: x.index.values).reset_index().rename(
                    {0: 'dup_index'}, axis=1)
            else:
                self.ndx = self.ndx.sort_values(by=['enroll', 'test'])
        else:
            # self.ndx = pd.read_csv(ndx_file, sep='\t', usecols=[0, 1], names=['enroll', 'test'])
            self.ndx = (
                pd.read_csv(ndx_file, sep='\t',   dtype=str,
                usecols=['modelid', 'segmentid'])
                  .rename(columns={'modelid': 'enroll', 'segmentid': 'test'})
            )
            if preserve_trial_order:
                self.ndx = self.ndx.groupby(['enroll', 'test']).apply(lambda x: x.index.values).reset_index().rename(
                    {0: 'dup_index'}, axis=1)
            else:
                self.ndx = self.ndx.sort_values(by=['enroll', 'test'])
        self.save_scores_to = save_scores_to

    def batch_plda_score(self, pq):
        score = BatchPLDAScore(self.enroll, self.test, pq=pq)
        
        self.ndx['scores'] = score.score(self.ndx)
        if hasattr(self, 'cohort'):
            normalizer = ScoreNormalizer(self.enroll, self.test, self.cohort, pq, top_scores=self.top_scores)
            self.ndx['scores'] = normalizer.normalize_scores(self.ndx)
        if not self.blind_trial:
            eer = comp_eer(self.ndx.scores, self.ndx.label)
            print(f'EER by PLDA scoreis {eer:.3f}')
            self.reset_X()

        if self.comp_minDCF:
            minDCF = compute_c_norm(self.ndx.scores, self.ndx.label)
            print(f'minDCF is  {minDCF:.3f}')

        if self.save_scores_to:
            print(self.save_scores_to)
            self.save_scores()

        # return self
        return eer

    def batch_cosine_score(self):
        score = BatchCosineScore(self.enroll, self.test)
        self.ndx['scores'] = score.score(self.ndx)
        if hasattr(self, 'cohort'):
            normalizer = ScoreNormalizer(self.enroll, self.test, self.cohort, top_scores=self.top_scores)
            self.ndx['scores'] = normalizer.normalize_scores(self.ndx)

        if self.preserve_trial_order:
            self.ndx = self.ndx.explode('dup_index').set_index('dup_index').sort_index()


        if self.save_scores_to:
            self.save_scores()
        if not self.blind_trial:
            # breakpoint()
            eer = comp_eer(self.ndx.scores, self.ndx.label)
            print(f'EER by cosine score is {eer:.3f}')
            if self.comp_minDCF:
                minDCF = compute_c_norm(self.ndx.scores, self.ndx.label)
                print(f'minDCF is  {minDCF:.3f}')
            return eer

    def plda_score(self, pq):
        scores = np.zeros(len(self.ndx))
        tgt_dict = build_id_dict(self.enroll['X'], self.enroll['spk_ids'])
        tst_dict = build_id_dict(self.test['X'], self.test['spk_ids'])

        for i, (tgt_id, tst_name) in enumerate(self.ndx[['enroll', 'test']].values):
            scores[i] = _plda_score_scores_averge(
                X=tgt_dict[tgt_id],
                y=tst_dict[tst_name].squeeze(),
                P=pq['P'],
                Q=pq['Q'],
                const=pq['const']
            )

            if i % 100000 == 0:
                print(f'{i}/{scores.shape[0]}, {tgt_id}, {tst_name}, {scores[i]}')

        self.ndx['scores'] = scores
        eer = comp_eer(scores, self.ndx.label)
        print(f'EER by PLDA scoreis {eer:.3f}')
        if self.comp_minDCF:
            minDCF = compute_c_norm(scores, self.ndx.label)
            print(f'minDCF is  {minDCF:.3f}')
        self.reset_X()
        return self

    def cosine_score(self):
        self.enroll['X'] = lennorm(self.enroll['X'])
        self.test['X'] = lennorm(self.test['X'])

        scores = np.zeros(len(self.ndx))
        tgt_dict = build_id_dict(self.enroll['X'], self.enroll['spk_ids'])
        tst_dict = build_id_dict(self.test['X'], self.test['spk_ids'])
        for i, (enroll_id, test_id) in enumerate(self.ndx[['enroll', 'test']].values):
            # breakpoint()
            scores[i] = tgt_dict[enroll_id].mean(0) @ tst_dict[test_id].squeeze().T
            if i % 100000 == 0:
                print(f'{i}/{scores.shape[0]}, {enroll_id}, {test_id}, {scores[i]}')

        self.ndx['scores'] = scores
        if not self.blind_trial:
            eer = comp_eer(scores, self.ndx.label)
            print(f'EER by cosine scoreis {eer:.3f}')

            if self.comp_minDCF:
                minDCF = compute_c_norm(scores, self.ndx.label)
                print(f'minDCF is  {minDCF:.3f}')
        # self.reset_X()
            return eer

    def save_scores(self):
        if not self.blind_trial:
            self.ndx_copy = self.ndx.copy()
            self.ndx_copy['label'] = self.ndx.label.map({1: 'target', 0: 'nontarget'})
            self.ndx_copy[['enroll', 'test', 'scores', 'label']].to_csv(self.save_scores_to, sep='\t', index=None)
            print(self.save_scores_to)
        elif self.blind_trial & self.preserve_trial_order:
            self.ndx['scores'] = MinMaxScaler().fit_transform(self.ndx['scores'].values.reshape(-1, 1))[:, 0]
            self.ndx[['enroll', 'test', 'scores']].to_csv(self.save_scores_to, sep='\t', index=None)
            print(self.save_scores_to)
        else:
            self.ndx[['enroll', 'test', 'scores']].to_csv(self.save_scores_to, sep='\t', index=None)
            print(self.save_scores_to)
        # self.ndx[['scores', 'label']].to_csv(scores_file, sep='\t', index=None)

    def reset_X(self):
        self.enroll['X'] = self.enroll['X_origin'].copy()
        self.test['X'] = self.test['X_origin'].copy()
        if hasattr(self, 'cohort'):
            self.cohort['X'] = self.cohort['X_origin'].copy()
        return self


def average_xvec(data, by):
    X, spk_ids, spk_path = [], [], []
    for uni in np.unique(data[by]):
        mask = data[by] == uni
        X.append(data['X'][mask].mean(0))
        spk_ids.append(data['spk_ids'][mask][0])
        spk_path.append(data['spk_path'][mask][0])
    return {
        'X': np.stack(X),
        'spk_ids': np.stack(spk_ids),
        'spk_path': np.stack(spk_path)
    }



def h5single2dict(file):
    with h5py.File(file, 'r') as f:
        return {name: f[name][...] for name in f}


def lennorm(X):
    return X / la.norm(X, axis=1)[:, None]

class BatchCosineScore:
    def __init__(self, enroll, test, debug=False):
        # Donot do any sorting inside this class
        enroll_stat = {'x_avg': {}}
        enroll['X'] = lennorm(enroll['X'])
        test['X'] = lennorm(test['X'])

        # enroll_stat will be retrieved by key
        for uni_id in np.unique(enroll['spk_ids']):
            mask = enroll['spk_ids'] == uni_id
            enroll_stat['x_avg'][uni_id] = enroll['X'][mask].mean(0)
        # test_stat will be retrieved by mask
        # test['xtQx'] = row_wise_dot(test['X'], test['X'].dot(pq['Q']))
        # test['Px'] = test['X'].dot(pq['P'])

        self.enroll_stat = enroll_stat
        self.test_stat = sort_dict(test, 'spk_ids')

        self.debug = debug

    def score(self, ndx):
        # warnings.warn('ndx has to be sorted other wise batch score will fail')
        # Todo check whether ndx is sorted
        # ndx = ndx.sort_values(['enroll', 'test'])
        scores = []
        for enroll_id, test_ids in ndx.groupby('enroll').test:
            x_avg = self.enroll_stat['x_avg'][enroll_id]
            mask = pd.Series(self.test_stat['spk_ids']).isin(test_ids)

            # ndx need to be sort enroll first and test second
            # so the test_ids is in order the test file will aligh with ndx
            if self.debug:
                assert np.all(self.test_stat['spk_ids'][mask] == test_ids)

            score = self.test_stat['X'][mask] @ x_avg.T

            scores.append(score)
        scores = np.concatenate(scores)
        return scores


class BatchPLDAScore:
    def __init__(self, enroll, test, pq, debug=False):
        # Donot do any sorting inside this class
        enroll['xtQx'] = row_wise_dot(enroll['X'], enroll['X'].dot(pq['Q']))
        enroll_stat = {'x_avg': {}, 'xtQx_avg': {}}

        # enroll_stat will be retrieved by key
        for uni_id in np.unique(enroll['spk_ids']):
            mask = enroll['spk_ids'] == uni_id
            enroll_stat['x_avg'][uni_id] = enroll['X'][mask].mean(0)
            enroll_stat['xtQx_avg'][uni_id] = enroll['xtQx'][mask].mean(0)

        # test_stat will be retrieved by mask
        test['xtQx'] = row_wise_dot(test['X'], test['X'].dot(pq['Q']))
        test['Px'] = test['X'].dot(pq['P'])

        self.enroll_stat = enroll_stat
        self.test_stat = sort_dict(test, 'spk_ids')

        self.const = pq['const']
        self.debug = debug

    def score(self, ndx):
        warnings.warn('ndx has to be sorted other wise batch score will fail')
        # Todo check whether ndx is sorted
        # ndx = ndx.sort_values(['enroll', 'test'])
        scores = []
        for enroll_id, test_ids in ndx.groupby('enroll').test:
            x_avg = self.enroll_stat['x_avg'][enroll_id]
            mask = pd.Series(self.test_stat['spk_ids']).isin(test_ids)

            # ndx need to be sort enroll first and test second
            # so the test_ids is in order the test file will aligh with ndx
            if self.debug:
                assert np.all(self.test_stat['spk_ids'][mask] == test_ids)

            xtPx = self.test_stat['Px'][mask] @ x_avg.T
            score = 0.5 * self.enroll_stat['xtQx_avg'][enroll_id] \
                    + xtPx + 0.5 * self.test_stat['xtQx'][mask] + self.const

            scores.append(score)
        scores = np.concatenate(scores)
        return scores


def row_wise_dot(x, y):
    return np.einsum('ij,ij->i', x, y)



def sort_dict(my_dict, field):
    # Warning this is totally wrong
    idx = np.argsort(my_dict[field])
    for key, val in my_dict.items():
        my_dict[key] = val[idx]
    return my_dict


def build_id_dict(X, spk_ids):
    return {spk_id: X[spk_ids == spk_id] for spk_id in np.unique(spk_ids)}


def _plda_score_scores_averge(X, y, P, Q, const):
    return (
        0.5 * ravel_dot(X.T @ X, Q) / X.shape[0]
        + kernel_dot(y, P, X.sum(0).T) / X.shape[0]
        + 0.5 * kernel_dot(y, Q, y.T)
        + const
    )


def ravel_dot(X, Y):
    return X.ravel() @ Y.ravel()


def kernel_dot(x, kernel, y):
    return x @ kernel @ y


class ScoreNormalizer:
    def __init__(self, enroll, test, cohort, pq=None, top_scores=200):
        self.enroll = enroll
        self.test = test
        self.cohort = cohort
        self.pq = pq
        self.top_scores = top_scores

    def normalize_scores(self, ndx):
        znorm = self._normalize_scores('enroll', ndx)
        snorm = self._normalize_scores('test', ndx)
        return (znorm + snorm) / 2

    def _normalize_scores(self, enroll_or_test, ndx):
        enroll = getattr(self, enroll_or_test)
        if 'group' in self.cohort.keys():
            print('using group info')
            stats = []
            for group in np.unique(self.cohort['group']):
                mask = enroll['group'] == group
                enroll_masked = mask_dict(enroll, mask)
                mask = self.cohort['group'] == group
                cohort_masked = mask_dict(self.cohort, mask)
                stats.append(self.get_norm_stat(enroll_masked, cohort_masked))
            stats = pd.concat(stats)
        else:
            stats = self.get_norm_stat(enroll, self.cohort)

        scores_normed = []
        for enroll_id, scores in ndx.groupby(enroll_or_test).scores:
            temp = (scores - stats.loc[enroll_id]['mean']) / stats.loc[enroll_id]['std']
            scores_normed.append(temp)
        # Todo this score may not aligh with ndx
        scores_normed = pd.concat(scores_normed).sort_index()
        return scores_normed

    def get_norm_stat(self, enroll, cohort):
        trial = array_product(np.unique(enroll['spk_ids']),
                              np.unique(cohort['spk_ids']))
        ndx = pd.DataFrame(data=trial, columns=['enroll', 'test'])
        ndx = ndx.sort_values(['enroll', 'test'])
        # Todo call batch PLDA Score need ndx sorted
        if self.pq:
            ndx['scores'] = BatchPLDAScore(enroll, cohort, self.pq).score(ndx)
        else:
            ndx['scores'] = BatchCosineScore(enroll, cohort).score(ndx)
        stat = ndx.groupby('enroll').scores.apply(get_first_n, self.top_scores)
        return stat


def get_first_n(group, n):
    df = group.sort_values(ascending=False).iloc[:n]
    return df.agg([np.mean, np.std])


def array_product(x, y):
    return np.array(np.meshgrid(x, y)).T.reshape(-1, 2)


def mask_dict(dict_, mask):
    return {key: val[mask] for key, val in dict_.items()}


def comp_eer(scores, labels):
    fnr, fpr = _compute_pmiss_pfa(scores, labels)
    eer = _comp_eer(fnr, fpr)
    return eer * 100


def compute_c_norm(scores, labels, p_target=0.01, c_miss=1, c_fa=1):
    fnr, fpr = _compute_pmiss_pfa(scores, labels)
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    c_det, _ = min(dcf), np.argmin(dcf)
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))

    return c_det/c_def


def _comp_eer(fnr, fpr):
    diff_pm_fa = fnr - fpr
    x1 = np.flatnonzero(diff_pm_fa >= 0)[0]
    x2 = np.flatnonzero(diff_pm_fa < 0)[-1]
    a = ( fnr[x1] - fpr[x1] ) / ( fpr[x2] - fpr[x1] - ( fnr[x2] - fnr[x1] ) )

    return fnr[x1] + a * ( fnr[x2] - fnr[x1] )


def _compute_pmiss_pfa(scores, labels):
    tgt_scores = scores[labels == 1] # target trial scores
    imp_scores = scores[labels == 0] # impostor trial scores

    resol = max([np.count_nonzero(labels == 0), np.count_nonzero(labels == 1), 1.e6])
    edges = np.linspace(np.min(scores), np.max(scores), int(resol))

    fnr = _compute_norm_counts(tgt_scores, edges, )
    fpr = 1 - _compute_norm_counts(imp_scores, edges, )

    return fnr, fpr


def _compute_norm_counts(scores, edges):
    score_counts = np.histogram(scores, bins=edges)[0].astype('f')
    norm_counts = np.cumsum(score_counts)/score_counts.sum()
    return norm_counts



class WavRandomSampleDataset(Dataset):
    def __init__(
        self,
        wav_dirs,
        sample_length_range,
        sample_per_epoch,
        min_duration,
        noise_dir,
        aug_config={},
        balance_class=True,
        min_utts_per_spk=8,
        trans_config={},
    ):
        # self.df = process_df(wav_dirs, min_utts_per_spk, min_duration)
        self.df = get_wav_info_multiple_dir(wav_dirs, min_utts_per_spk, min_duration, True)
        self.smp_per_epo = sample_per_epoch
        self.balance_class = balance_class
        self.smp_len_min, self.smp_len_max = sample_length_range
        if noise_dir:
            self.distort = config_distortions(
                reverb_irfiles=f'{noise_dir}/simulated_rirs',
                noises_dir=f'{noise_dir}/noise',
                music_dir=f'{noise_dir}/music',
                overlap_dir=f'{noise_dir}/speech',
                **aug_config,
            )
        else:
            self.distort = None
        # print(self.distort)
        # print(self.trans)
        self.trans = config_trans(**trans_config)

    def sample(self):
        np.random.seed()
        sample_length = np.random.randint(low=self.smp_len_min, high=self.smp_len_max)
        prob = (1 / self.df["spk_ids"].nunique() / self.df["n_utts"]).values
        # warnings.warn('temp solution to dived prob by sum')
        # prob /= prob.sum()
        idxes = np.random.choice(len(self.df), p=prob, size=self.smp_per_epo)
        self.df_smp = self.sample_segments(self.df.iloc[idxes].copy(), sample_length)  # Deep copy

    def __getitem__(self, i):
        df = self.df_smp.iloc[i]
        # _, wav_data = wavfile.read(df['files'])
        _, wav_data = wavfile.read(df['full_paths'])
        # from core.tools import ForkedPdb; ForkedPdb().set_trace()
        # wav_sampled = wav_data[df['starts']:df['ends']][None, :].astype(np.float32)
        wav_sampled = wav_data[df['starts']:df['ends']].astype(np.float32)
        # wav_sampled = torch.tensor(wav_sampled)
        if self.distort:
            wav_sampled = self.distort(wav_sampled)
        # print(type(wav_sampled))
        if self.trans:
            wav_sampled = self.trans(wav_sampled)
        return wav_sampled, torch.tensor(df['spk_ids']), df['utt_ids']

    def __len__(self):
        return self.smp_per_epo

    def sample_segments(self, df_smp, sample_length):
        starts, ends = [], []
        for n_frames in df_smp['ns_frames'].values:
        # for n_frames in df_smp['durations'].values:
            smp_start = np.random.randint(low=0, high=n_frames - sample_length)
            starts.append(smp_start)
            ends.append(smp_start + sample_length)
        df_smp.loc[:, 'starts'] = np.array(starts)
        df_smp.loc[:, 'ends'] = np.array(ends)
        return df_smp


class WavExtractDset(Dataset):
    def __init__(self, wav_dir, trans_config={}):
        self.wav_dir = wav_dir

        if type(self.wav_dir) is list:
            self.df = get_wav_info_multiple_dir(wav_dir)
            for wav_dir_ in wav_dir:
                if not wav_dir_.endswith(('dev', 'test', 'eval')):
                    raise ValueError('not ends with dev or test')

                if not os.path.exists(f'{wav_dir_}/ids_info.tsv'):
                    raise ValueError(f'ids_info.tsv not exist in {wav_dir_}, create one first')
        else:
            self.df = read_wav_info(wav_dir)
            # if not wav_dir.endswith(('dev', 'test', 'eval')):
            #     raise ValueError('not ends with dev or test')

            if not os.path.exists(f'{wav_dir}/ids_info.tsv'):
                raise ValueError(f'ids_info.tsv not exist in {wav_dir}, create one first')

        self.trans = config_trans(**trans_config)

    def __getitem__(self, i):
        # df = self.df.loc[self.files[i]]
        df = self.df.iloc[i]
        # file = self.wav_dir + '/' + self.files[i]
        _, wav = wavfile.read(df['full_paths'])
        # wav, _ = sf.read(df['full_paths'])
        # print(df['full_paths'])
        # wav = torch.tensor(wav.astype(np.float32))
        wav = wav.astype(np.float32)
        if self.trans:
            wav = self.trans(wav)
        return wav, df.spk_ids, df.utt_ids

    def __len__(self):
        return len(self.df)


def read_wav_info(wav_dir):
    print('read_wav_info(): %s' % wav_dir)
    if not os.path.exists(f'{wav_dir}/ids_info.tsv'):
        raise ValueError(f'ids_info.tsv not exist in {wav_dir}, create one first')
    df = pd.read_csv(f'{wav_dir}/ids_info.tsv', sep='\t', dtype={'spk_ids': str})
    files = [
        (file, os.path.relpath(file, wav_dir))
        for file in glob.glob(f'{wav_dir}/**/*.wav', recursive=True)
    ]
    if len(files) == 0:
        raise ValueError('empty dir')

    df_file = pd.DataFrame(files, columns=['full_paths', 'relative_paths'])
    df = df.set_index('relative_paths').join(df_file.set_index('relative_paths'))
    # from core.tools import ForkedPdb; ForkedPdb().set_trace()
    if df.isna().any().any():
        raise ValueError('df contains NA')
    return df


def get_wav_info_multiple_dir(wav_dirs, min_utts_per_spk=0, min_duration=0, digitalize_spk_ids=True):
    dfs = [process_df(wav_dir, min_utts_per_spk, min_duration, False)
           for wav_dir in wav_dirs]
    df = pd.concat(dfs)
    if digitalize_spk_ids:
        df.loc[:, "spk_ids"] = LabelEncoder().fit_transform(df["spk_ids"])
    return df


def process_df(wav_dir, min_utts_per_spk=0, min_duration=0, digitalize_spk_ids=True):
    df = read_wav_info(wav_dir)
    df = df[(df.durations) > min_duration]
    utt_counts = df.spk_ids.value_counts()
    df.loc[:, "n_utts"] = df.spk_ids.map(utt_counts)
    df = df[df.n_utts > min_utts_per_spk]
    print(f'total session is {len(df)}')
    if digitalize_spk_ids:
        df.loc[:, "spk_ids"] = LabelEncoder().fit_transform(df["spk_ids"])
    return df


import torchaudio
import torch


class Fbank(torch.nn.Module):
    def __init__(self, n_mels=80):
        super(Fbank, self).__init__()
        self.MelSpectrogram = torchaudio.transforms.MelSpectrogram(n_mels=n_mels, hop_length=160)

    def forward(self, waveform):
        mel_specgram = self.MelSpectrogram(waveform)
        log_offset = 1e-6
        mel_specgram = torch.log(mel_specgram + log_offset)
        return mel_specgram - mel_specgram.mean(-1, keepdim=True)


# Todo implement double masking for freq
class SpectrumAug(torch.nn.Module):
    def __init__(self, max_freq_mask_len=27, max_time_mask_len=100, n_freq_mask=1):
        super().__init__()
        self.freq_masker = FrequencyMasking(max_freq_mask_len, iid_masks=True, n_freq_mask=n_freq_mask)
        self.time_masker = TimeMasking(max_time_mask_len,  iid_masks=True)

    def forward(self, spectrum):
        if self.training:
            return self.time_masker(self.freq_masker(spectrum[:, None, ...])).squeeze()
        else:
            return spectrum


class _AxisMasking(torch.nn.Module):
    def __init__(self, mask_param, axis, iid_masks, n_mask=1):
        super().__init__()
        self.mask_param = mask_param
        self.axis = axis
        self.iid_masks = iid_masks
        self.n_mask = n_mask

    def forward(self, specgram, mask_value=0.0):
        if self.iid_masks and specgram.dim() == 4:
            for _ in range(self.n_mask):
                specgram = mask_along_axis_iid(specgram, self.mask_param, mask_value, self.axis + 1)
            return specgram
        else:
            return NotImplementedError


class FrequencyMasking(_AxisMasking):

    def __init__(self, freq_mask_param, iid_masks, n_freq_mask):
        super(FrequencyMasking, self).__init__(freq_mask_param, 1, iid_masks, n_mask=n_freq_mask)


class TimeMasking(_AxisMasking):
    def __init__(self, time_mask_param: int, iid_masks: bool = False) -> None:
        super(TimeMasking, self).__init__(time_mask_param, 2, iid_masks)


def mask_along_axis_iid(
        specgrams,
        mask_param,
        mask_value,
        axis,
):

    if axis != 2 and axis != 3:
        raise ValueError('Only Frequency and Time masking are supported')

    value = torch.rand(specgrams.shape[:2]) * mask_param
    min_value = torch.rand(specgrams.shape[:2]) * (specgrams.size(axis) - value)

    # Create broadcastable mask
    mask_start = (min_value.long())[..., None, None].float()
    mask_end = (min_value.long() + value.long())[..., None, None].float()
    mask = torch.arange(0, specgrams.size(axis)).float()

    # Per batch example masking
    specgrams = specgrams.transpose(axis, -1)
    specgrams.masked_fill_((mask >= mask_start).cuda() & (mask < mask_end).cuda(), mask_value)
    specgrams = specgrams.transpose(axis, -1)

    return specgrams


class AMLinear(nn.Module):
    def __init__(self, in_features, n_cls, m=0.35, s=30):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, n_cls))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m
        self.n_cls = n_cls
        self.s = s

    def forward(self, x, labels):
        w = self.weight
        ww = w.renorm(2, 1, 1e-5).mul(1e5)
        x = F.normalize(x, dim=1)

        cos_theta = torch.mm(x, ww)
        cos_theta = torch.clamp(cos_theta, -1, 1)
        phi = cos_theta - self.m
        # labels_one_hot = torch.zeros(len(labels), self.n_cls, device=device).scatter_(1, labels.unsqueeze(1), 1.)
        labels_one_hot = torch.zeros(len(labels), self.n_cls, device=labels.get_device()).scatter_(1, labels.unsqueeze(1), 1.)

        adjust_theta = self.s * torch.where(torch.eq(labels_one_hot, 1), phi, cos_theta)
        return adjust_theta, cos_theta


def cnn_bn_relu(indim, outdim, kernel_size, stride=1, dilation=1, padding=0):
    return nn.Sequential(
            nn.Conv1d(indim, outdim, kernel_size, stride=stride, dilation=dilation, padding=padding),
            nn.BatchNorm1d(outdim),
            torch.nn.ReLU(),
        )


class StatsPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        mean = x.mean(-1, keepdim=True)
        var = torch.sqrt((x - mean).pow(2).mean(-1) + 1e-5)
        return torch.cat([mean.squeeze(-1), var], -1)


class InsertSilent:
    NotImplementedError('to be done')


class SimpleAdditive(object):
    def __init__(
        self, noises_dir, snr_levels=(0, 5, 10), cache=False,
    ):
        self.noises_dir = noises_dir
        self.snr_levels = snr_levels
        self.noises = glob.glob(f'{noises_dir}/**/*.wav', recursive=True)
        self.nidxs = list(range(len(self.noises)))
        if len(self.noises) == 0:
            raise ValueError("[!] No noises found in {}".format(noises_dir))
        else:
            print("[*] Found {} noise files".format(len(self.noises)))
        self.eps = 1e-22
        if cache:
            self.cache = {}
            for noise in self.noises:
                self.load_noise(noise)

    def sample_noise(self):
        if len(self.noises) == 1:
            return self.noises[0]
        else:
            idx = np.random.randint(0, len(self.noises))
            # idx = random.choice(self.nidxs)
            return self.noises[idx]

    def load_noise(self, filename):
        if hasattr(self, "cache") and filename in self.cache:
            return self.cache[filename]
        else:
            nwav, rate = sf.read(filename)
            if hasattr(self, "cache"):
                self.cache[filename] = nwav
        return nwav.astype(np.float32)

    def compute_SNR_K(self, signal, noise, snr):
        Ex = np.dot(signal, signal)
        En = np.dot(noise, noise)
        if En > 0:
            K = np.sqrt(Ex / ((10 ** (snr / 10.0)) * En))
        else:
            K = 1.0
        return K, Ex, En

    def norm_energy(self, osignal, ienergy, eps=1e-14):
        oenergy = np.dot(osignal, osignal)
        return np.sqrt(ienergy / (oenergy + eps)) * osignal

    # @profile
    def __call__(self, wav):
        """ Add noise to clean wav """
        beg_i = 0
        end_i = wav.shape[0]
        sel_noise = self.load_noise(self.sample_noise())
        if len(sel_noise) < len(wav):
            # pad noise
            P = len(wav) - len(sel_noise)
            sel_noise = np.pad(sel_noise, (0, P))
            # mode='reflect').view(-1).data.numpy()
        T = end_i - beg_i
        # TODO: not pre-loading noises from files?
        if len(sel_noise) > T:
            n_beg_i = np.random.randint(0, len(sel_noise) - T)
        else:
            n_beg_i = 0
        noise = sel_noise[n_beg_i:n_beg_i + T]
        # randomly sample the SNR level
        snr = random.choice(self.snr_levels)
        K, Ex, En = self.compute_SNR_K(wav, noise, snr)
        scaled_noise = K * noise
        if En > 0:
            noisy_wav = wav + scaled_noise
            noisy_wav = self.norm_energy(noisy_wav, Ex)
        else:
            noisy_wav = wav
        return noisy_wav

    def __repr__(self):
        attrs = "(noises_dir={})".format(self.noises_dir)
        return self.__class__.__name__ + attrs


# class SingleChunkWav(object):
#     def __init__(self, chunk_size, random_scale=True):
#         self.chunk_size = chunk_size
#         self.random_scale = random_scale
#
#     @staticmethod
#     def assert_format(x):
#         # assert it is a waveform and pytorch tensor
#         assert isinstance(x, torch.Tensor), type(x)
#
#     def select_chunk(self, wav):
#         chksz = self.chunk_size
#         idx = np.random.randint(0, wav.size(0) - self.chunk_size)
#         return wav[idx : idx + chksz]
#
#     def __call__(self, wav):
#         self.assert_format(wav)
#         chunk = self.select_chunk(wav)
#         if self.random_scale:
#             chunk = norm_and_scale(chunk)
#         return chunk
#
#     def __repr__(self):
#         return f"{self.__class__.__name__} ({self.chunk_size})"
#


class Reverb(object):
    def __init__(self, ir_dir, max_reverb_len=24000, cache=False):
        self.ir_files = glob.glob(f"{ir_dir}/**/*.wav", recursive=True)
        assert len(self.ir_files) > 0, f"not file in {self.ir_files}"
        print(f"Found {len(self.ir_files)}")

        self.ir_idxs = list(range(len(self.ir_files)))
        # self.IR, self.p_max = self.load_IR(ir_file, ir_fmt)
        self.max_reverb_len = max_reverb_len
        if cache:
            self.cache = {}
            for ir_file in self.ir_files:
                self.load_IR(ir_file)

    def load_IR(self, ir_file):
        # ir_file = os.path.join(self.data_root, ir_file)
        # print('loading ir_file: ', ir_file)
        if hasattr(self, "cache") and ir_file in self.cache:
            return self.cache[ir_file]
        else:
            IR, _ = sf.read(ir_file)
            IR = IR[: self.max_reverb_len]
            if np.max(IR) > 0:
                IR = IR / np.abs(np.max(IR))
            p_max = np.argmax(np.abs(IR))
            if hasattr(self, "cache"):
                self.cache[ir_file] = (IR, p_max)
            return IR, p_max

    def shift(self, xs, n):
        e = np.empty_like(xs)
        if n >= 0:
            e[:n] = 0.0
            e[n:] = xs[:-n]
        else:
            e[n:] = 0.0
            e[:n] = xs[-n:]
        return e

    def sample_IR(self):
        if len(self.ir_files) == 0:
            return self.ir_files[0]
        else:
            idx = random.choice(self.ir_idxs)
            return self.ir_files[idx]

    ##@profile
    def __call__(self, wav):
        # sample an ir_file
        ir_file = self.sample_IR()
        IR, p_max = self.load_IR(ir_file)
        IR = IR.astype(np.float32)
        Ex = np.dot(wav, wav)
        wav = wav.astype(np.float32)
        # print(IR.shape)
        # wav = wav.reshape(-1)

        rev = signal.convolve(wav, IR, mode="full")
        # with cupy.cuda.Device(1):
        # import numpy as np
        # import cupy
        # import cusignal
        # import scipy.signal
        # wav = np.random.randn(4*16000).astype(np.float32)
        # IR = np.random.randn(24000).astype(np.float32)
        # %timeit rev = scipy.signal.convolve(wav, IR)
        # %timeit rev = cupy.asnumpy(cusignal.convolve(cupy.asarray(wav), cupy.asarray(IR), method='direct'))
        Er = np.dot(rev, rev)
        # IR delay compensation
        rev = self.shift(rev, -p_max)
        if Er > 0:
            Eratio = np.sqrt(Ex / Er)
        else:
            Eratio = 1.0
            # rev = rev / np.max(np.abs(rev))

        # Trim rev signal to match clean length
        rev = rev[: wav.shape[0]]
        rev = Eratio * rev
        # rev = torch.FloatTensor(rev)
        return rev

    def __repr__(self):
        if len(self.ir_files) > 3:
            attrs = "(ir_files={} ...)".format(self.ir_files[:3])
        else:
            attrs = "(ir_files={})".format(self.ir_files)
        return self.__class__.__name__ + attrs


# class Reverb(object):
#     def __init__(self, ir_dir, max_reverb_len=24000, cache=False):
#         self.ir_files = glob.glob(f"{ir_dir}/**/*.wav", recursive=True)
#         assert len(self.ir_files) > 0, f"not file in {self.ir_files}"
#         print(f"Found {len(self.ir_files)}")
#
#         self.ir_idxs = list(range(len(self.ir_files)))
#         # self.IR, self.p_max = self.load_IR(ir_file, ir_fmt)
#         self.max_reverb_len = max_reverb_len
#         if cache:
#             self.cache = {}
#             for ir_file in self.ir_files:
#                 self.load_IR(ir_file)
#
#     def load_IR(self, ir_file):
#         # ir_file = os.path.join(self.data_root, ir_file)
#         # print('loading ir_file: ', ir_file)
#         if hasattr(self, "cache") and ir_file in self.cache:
#             return self.cache[ir_file]
#         else:
#             IR, _ = sf.read(ir_file)
#             IR = IR[: self.max_reverb_len]
#             if np.max(IR) > 0:
#                 IR = IR / np.abs(np.max(IR))
#             p_max = np.argmax(np.abs(IR))
#             if hasattr(self, "cache"):
#                 self.cache[ir_file] = (IR, p_max)
#             return IR, p_max
#
#     def shift(self, xs, n):
#         e = np.empty_like(xs)
#         if n >= 0:
#             e[:n] = 0.0
#             e[n:] = xs[:-n]
#         else:
#             e[n:] = 0.0
#             e[:n] = xs[-n:]
#         return e
#
#     def sample_IR(self):
#         if len(self.ir_files) == 0:
#             return self.ir_files[0]
#         else:
#             idx = random.choice(self.ir_idxs)
#             return self.ir_files[idx]
#
#     ##@profile
#     def __call__(self, wav):
#         # sample an ir_file
#         ir_file = self.sample_IR()
#         IR, p_max = self.load_IR(ir_file)
#         IR = IR.astype(np.float32)
#         Ex = np.dot(wav, wav)
#         # wav = wav.astype(np.float32).reshape(-1)
#         wav = wav.reshape(-1)
#         rev = signal.convolve(wav, IR, mode="full").reshape(-1)
#         Er = np.dot(rev, rev)
#         # IR delay compensation
#         rev = self.shift(rev, -p_max)
#         if Er > 0:
#             Eratio = np.sqrt(Ex / Er)
#         else:
#             Eratio = 1.0
#             # rev = rev / np.max(np.abs(rev))
#
#         # Trim rev signal to match clean length
#         rev = rev[: wav.shape[0]]
#         rev = Eratio * rev
#         # rev = torch.FloatTensor(rev)
#         return rev
#
#     def __repr__(self):
#         if len(self.ir_files) > 3:
#             attrs = "(ir_files={} ...)".format(self.ir_files[:3])
#         else:
#             attrs = "(ir_files={})".format(self.ir_files)
#         return self.__class__.__name__ + attrs


# class OldBable(SimpleAdditive):
#     def __init__(
#         self,
#         noises_dir,
#         cache=False,
#         snr_levels=(15, 20, 25,),
#         n_noise=1,
#         noise_transform=None,
#     ):
#         super().__init__(noises_dir, snr_levels, cache=cache)
#         self.noises_dir = noises_dir
#         self.snr_levels = snr_levels
#         self.nidxs = list(range(len(self.noises)))
#         if len(self.noises) < min(n_noise):
#             raise ValueError(f"not enough files in {noises_dir}")
#         else:
#             print("[*] Found {} noise files".format(len(self.noises)))
#         self.n_noise = n_noise
#
#         # additional out_transform to include potential distortions
#         self.noise_transform = noise_transform
#
#     @property
#     def n_babble(self):
#         return np.random.choice(self.n_noise)
#
#     def __call__(self, wav):
#         # compute shifts of signal
#         for i in range(self.n_babble):
#             shift = np.random.randint(0, int(0.75 * len(wav)))
#             sel_noise = self.load_noise(self.sample_noise())
#             T = len(wav) - shift
#             if len(sel_noise) < T:
#                 # pad noise
#                 P = T - len(sel_noise)
#                 sel_noise = np.pad(sel_noise, (0, P))
#                 n_beg_i = 0
#             elif len(sel_noise) > T:
#                 n_beg_i = np.random.randint(0, len(sel_noise) - T)
#             else:
#                 n_beg_i = 0
#             noise = sel_noise[n_beg_i : n_beg_i + T]
#             if self.noise_transform is not None:
#                 noise = self.noise_transform(noise)
#
#             # now len(noise)=T
#             pad_len = len(wav) - len(noise)
#             # apply padding to equal length now
#             noise = np.pad(noise, (pad_len, 0))
#
#             # randomly sample the SNR level
#             snr = random.choice(self.snr_levels)
#             K, Ex, En = self.compute_SNR_K(wav, noise, snr)
#             scaled_noise = K * noise
#             noisy = wav + scaled_noise
#             # noisy = self.norm_energy(noisy, Ex)
#             wav = self.norm_energy(noisy, Ex)
#         # noisy = wav
#         # x_ = torch.FloatTensor(noisy)
#         return wav
#
#     def __repr__(self):
#         if self.noise_transform is None:
#             attrs = "(noises_dir={})".format(self.noises_dir)
#         else:
#             attrs = "(noises_dir={}, " "noise_transform={})".format(
#                 self.noises_dir, self.noise_transform.__repr__()
#             )
#         return self.__class__.__name__ + attrs


class Bable(SimpleAdditive):
    def __init__(
        self,
        noises_dir,
        cache=False,
        snr_levels=(15, 20, 25,),
        n_noise=1,
        noise_transform=None,
    ):
        super().__init__(noises_dir, snr_levels, cache=cache)
        self.noises_dir = noises_dir
        self.snr_levels = snr_levels
        self.nidxs = list(range(len(self.noises)))
        if len(self.noises) < min(n_noise):
            raise ValueError(f"not enough files in {noises_dir}")
        else:
            print("[*] Found {} noise files".format(len(self.noises)))
        self.n_noise = n_noise

        # additional out_transform to include potential distortions
        self.noise_transform = noise_transform

    @property
    def n_babble(self):
        return np.random.choice(self.n_noise)

    def __call__(self, wav):
        # compute shifts of signal
        noise_all = 0
        for i in range(self.n_babble):
            shift = np.random.randint(0, int(0.75 * len(wav)))
            sel_noise = self.load_noise(self.sample_noise())
            T = len(wav) - shift
            if len(sel_noise) < T:
                # pad noise
                P = T - len(sel_noise)
                sel_noise = np.pad(sel_noise, (0, P))
                n_beg_i = 0
            elif len(sel_noise) > T:
                n_beg_i = np.random.randint(0, len(sel_noise) - T)
            else:
                n_beg_i = 0
            noise = sel_noise[n_beg_i : n_beg_i + T]
            # if self.noise_transform is not None:
            #     noise = self.noise_transform(noise)

            # now len(noise)=T
            pad_len = len(wav) - len(noise)
            # apply padding to equal length now
            noise = np.pad(noise, (pad_len, 0))
            noise_all += noise

        if self.noise_transform is not None:
            noise_all = self.noise_transform(noise_all)

        # randomly sample the SNR level
        snr = random.choice(self.snr_levels)
        K, Ex, En = self.compute_SNR_K(wav, noise_all, snr)
        scaled_noise = K * noise_all
        noisy = wav + scaled_noise
        # noisy = self.norm_energy(noisy, Ex)
        wav = self.norm_energy(noisy, Ex)
        # noisy = wav
        # x_ = torch.FloatTensor(noisy)
        return wav

    def __repr__(self):
        if self.noise_transform is None:
            attrs = "(noises_dir={})".format(self.noises_dir)
        else:
            attrs = "(noises_dir={}, " "noise_transform={})".format(
                self.noises_dir, self.noise_transform.__repr__()
            )
        return self.__class__.__name__ + attrs


# class NewBable(SimpleAdditive):
#     def __init__(
#         self,
#         noises_dir,
#         cache=False,
#         snr_levels=(15, 20, 25,),
#         n_noise=1,
#         noise_transform=None,
#     ):
#         super().__init__(noises_dir, snr_levels, cache=cache)
#         self.noises_dir = noises_dir
#         self.snr_levels = snr_levels
#         self.nidxs = list(range(len(self.noises)))
#         if len(self.noises) < min(n_noise):
#             raise ValueError(f"not enough files in {noises_dir}")
#         else:
#             print("[*] Found {} noise files".format(len(self.noises)))
#         self.n_noise = n_noise
#
#         # additional out_transform to include potential distortions
#         self.noise_transform = noise_transform
#
#     @property
#     def n_babble(self):
#         return np.random.choice(self.n_noise)
#
#     def __call__(self, wav):
#         # compute shifts of signal
#         scaled_noise = 0
#         for i in range(self.n_babble):
#             shift = np.random.randint(0, int(0.75 * len(wav)))
#             sel_noise = self.load_noise(self.sample_noise())
#             T = len(wav) - shift
#             if len(sel_noise) < T:
#                 # pad noise
#                 P = T - len(sel_noise)
#                 sel_noise = np.pad(sel_noise, (0, P))
#                 n_beg_i = 0
#             elif len(sel_noise) > T:
#                 n_beg_i = np.random.randint(0, len(sel_noise) - T)
#             else:
#                 n_beg_i = 0
#             noise = sel_noise[n_beg_i : n_beg_i + T]
#             if self.noise_transform is not None:
#                 noise = self.noise_transform(noise)
#
#             # now len(noise)=T
#             pad_len = len(wav) - len(noise)
#             # apply padding to equal length now
#             noise = np.pad(noise, (pad_len, 0))
#
#             # randomly sample the SNR level
#             snr = random.choice(self.snr_levels)
#             K, Ex, En = self.compute_SNR_K(wav, noise, snr)
#             scaled_noise += K * noise
#
#         noisy = wav + scaled_noise
#         # noisy = self.norm_energy(noisy, Ex)
#         wav = self.norm_energy(noisy, Ex)
#         # noisy = wav
#         # x_ = torch.FloatTensor(noisy)
#         return wav
#
#     def __repr__(self):
#         if self.noise_transform is None:
#             attrs = "(noises_dir={})".format(self.noises_dir)
#         else:
#             attrs = "(noises_dir={}, " "noise_transform={})".format(
#                 self.noises_dir, self.noise_transform.__repr__()
#             )
#         return self.__class__.__name__ + attrs


class Transforms:
    def __init__(self, transforms):
        assert isinstance(transforms, list), type(transforms)
        self.transforms = transforms

    def __call__(self, tensor):
        x = tensor
        for tran in self.transforms:
           x = tran(x)
        # return torch.FloatTensor(x.astype(np.float32))
        return torch.tensor(x.astype(np.float32))

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class PCompose(object):
    def __init__(self, transforms, probs=0.4, report=False):
        assert isinstance(transforms, list), type(transforms)
        self.transforms = transforms
        self.probs = probs
        self.report = report
        if isinstance(probs, list):
            assert len(transforms) == len(probs), "{} != {}".format(
                len(transforms), len(probs)
            )

    # @profile
    def __call__(self, tensor):
        x = tensor
        for ti, transf in enumerate(self.transforms):
            if isinstance(self.probs, list):
                prob = self.probs[ti]
            else:
                prob = self.probs
            if random.random() < prob:
                x = transf(x)
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for ti, t in enumerate(self.transforms):
            if isinstance(self.probs, list):
                prob = self.probs[ti]
            else:
                prob = self.probs
            format_string += "\n"
            format_string += "    {0}".format(t)
            format_string += " >> p={}".format(prob)
        format_string += "\n)"
        return format_string


class Clipping(object):
    def __init__(
        self, clip_factors=[0.3, 0.4, 0.5],
    ):
        self.clip_factors = clip_factors

    # @profile
    def __call__(self, wav):
        # cf = np.random.choice(self.clip_factors, 1)
        cf = random.choice(self.clip_factors)
        clip = np.maximum(wav, cf * np.min(wav))
        clip = np.minimum(clip, cf * np.max(wav))
        return clip

    def __repr__(self):
        attrs = "(clip_factors={})".format(self.clip_factors)
        return self.__class__.__name__ + attrs


class BandDrop(object):

    def __init__(self, filter_dir,):

        filt_files = glob.glob(f'{filter_dir}/*.npy')
        print(f'Found {len(filt_files)} filt_files in {filter_dir}')
        assert len(filt_files) > 0, 'empty dir'

        self.filt_files = filt_files
        assert isinstance(filt_files, list), type(filt_files)
        assert len(filt_files) > 0, len(filt_files)
        self.filt_idxs = list(range(len(filt_files)))

    def load_filter(self, filt_file):
        filt_coeff = np.load(filt_file)
        filt_coeff = filt_coeff / np.abs(np.max(filt_coeff))

        return filt_coeff

    def shift(self, xs, n):
        e = np.empty_like(xs)
        if n >= 0:
            e[:n] = 0.0
            e[n:] = xs[:-n]
        else:
            e[n:] = 0.0
            e[:n] = xs[-n:]
        return e

    def sample_filt(self):
        if len(self.filt_files) == 0:
            return self.filt_files[0]
        else:
            idx = random.choice(self.filt_idxs)
            return self.filt_files[idx]

    ##@profile
    def __call__(self, wav):
        # sample a filter
        filt_file = self.sample_filt()
        filt_coeff = self.load_filter(filt_file)
        filt_coeff = filt_coeff.astype(np.float32)
        Ex = np.dot(wav, wav)
        wav = wav.astype(np.float32).reshape(-1)
        sig_filt = signal.convolve(wav, filt_coeff, mode='full').reshape(-1)

        sig_filt = self.shift(sig_filt, -round(filt_coeff.shape[0] / 2))

        sig_filt = sig_filt[:wav.shape[0]]

        # sig_filt=sig_filt/np.max(np.abs(sig_filt))

        Efilt = np.dot(sig_filt, sig_filt)
        # Ex = np.dot(wav, wav)
        if Efilt > 0:
            Eratio = np.sqrt(Ex / Efilt)
        else:
            Eratio = 1.0
            sig_filt = wav

        sig_filt = Eratio * sig_filt
        return sig_filt

    def __repr__(self):
        if len(self.filt_files) > 3:
            attrs = '(filt_files={} ...)'.format(self.filt_files[:3])
        else:
            attrs = '(filt_files={})'.format(self.filt_files)
        return self.__class__.__name__ + attrs


class RandomScale:
    def __call__(self, wav):
        # assert isinstance(wav, torch.Tensor), type(wav)
        # wav = wav / torch.max(torch.abs(wav))
        # wav = wav / torch.clamp(torch.max(torch.abs(wav)), min=1e-4)
        wav_max = np.clip(np.max(np.abs(wav)), a_min=1e-4, a_max=None)
        wav = wav / wav_max
        return wav * 1.5 * np.random.rand(1)

    def __repr__(self):
        return self.__class__.__name__

class PartialExtractDataset(Dataset):
    def __init__(self, df, mfcc):
        self.mfccs = mfcc
        self.positions = df[['starts', 'ends']].values
        self.spk_ids = df['spk_ids'].values
        self.utt_ids = df['utt_ids'].values

    def __getitem__(self, idx):
        start, end = self.positions[idx]
        spk_id = self.spk_ids[idx]
        utt_id = self.utt_ids[idx]
        mfcc = self.mfccs[start:end].T
        return torch.tensor(mfcc), spk_id, utt_id

    def __len__(self):
        return len(self.positions)

# class FBanks(object):
#
#     def __init__(self, n_filters=40, n_fft=512, hop=160,
#                  win=400, rate=16000,
#                  name='fbank',
#                  device='cpu'):
#         self.n_fft = n_fft
#         self.n_filters = n_filters
#         self.rate = rate
#         self.hop = hop
#         self.name = name
#         self.win = win
#         self.name = name
#
#     # @profile
#     def __call__(self, wav):
#         max_frames = wav.shape[0] // self.hop
#         winlen = (float(self.win) / self.rate)
#         winstep = (float(self.hop) / self.rate)
#         X = logfbank(wav, self.rate, winlen, winstep,
#                      self.n_filters, self.n_fft).T
#
#         return X
#
#     def __repr__(self):
#         attrs = '(n_fft={}, n_filters={}, ' \
#                 'hop={}, win={}'.format(self.n_fft,
#                                         self.n_filters,
#                                         self.hop,
#                                         self.win)
#         return self.__class__.__name__ + attrs

class FBank:
    def __call__(self, wav):
        # return torchaudio.compliance.kaldi.fbank(waveform=wav, snip_edges=False, high_freq=7600, num_mel_bins=40).T
        return torchaudio.compliance.kaldi.fbank(waveform=torch.tensor(wav[None, :]), snip_edges=False, high_freq=7600, num_mel_bins=40).numpy().T

class NormMax:
    def __call__(self, wav):
        # return torch.clamp(wav / torch.max(torch.abs(wav)), min=1e-4)
        # return wav / torch.max(torch.abs(wav))
        # return wav / torch.clamp(torch.max(torch.abs(wav)),  min=1e-4)
        return wav / np.clip(np.max(np.abs(wav)), a_min=1e-4, a_max=None)

# class CMVN:
#     def __init__(self, win=300):
#         self.win = win
#
#     def __call__(self, mfcc):
#         idx = np.arange(self.win, mfcc.shape[-1], self.win)
#         mfcc_split = np.array_split(mfcc, idx, axis=-1)
#         # breakpoint()
#         return self.subtract_mean(mfcc_split)
#
#     def subtract_mean(self, mfccs):
#         return np.concatenate([mfcc - mfcc.mean(-1, keepdims=True) for mfcc in mfccs], axis=-1)


class CMVN:
    def __call__(self, mfcc):
        return mfcc - mfcc.mean(-1, keepdims=True)
        # return (mfcc - mfcc.mean(-1, keepdims=True)) / np.clip(mfcc.var(-1, keepdims=True), a_min=1e-5, a_max=None)

    def __repr__(self):
        return self.__class__.__name__ + 'all_length'

def config_trans(features=None, normalize_by_max=False, cmvn=False):
    trans = []
    if normalize_by_max:
        trans.append(NormMax())
    elif features == 'fbank':
        trans.append(FBank())
        if cmvn:
            trans.append(CMVN())
    else:
        print('no feature')
    return Transforms(trans)


def translate_prob2other_ino(prob, translat2):
    snr_dict = {
        0.: 25,
        0.1: 20,
        0.2: 15,
        0.3: 10,
        0.4: 8,
        0.5: 6,
        0.6: 4,
        0.7: 3,
        0.8: 2,
        0.9: 1,
        1.0: 0,
    }
    mask_value_frequency_dict = {
        0.: 1,
        0.1: 5,
        0.2: 10,
        0.3: 15,
        0.4: 20,
        0.5: 25,
        0.6: 30,
        0.7: 35,
        0.8: 40,
        0.9: 45,
        1.0: 50,
    }
    # mask_value_time_dict = {
    #     0.: 1,
    #     0.1: 5,
    #     0.2: 10,
    #     0.3: 15,
    #     0.4: 20,
    #     0.5: 25,
    #     0.6: 30,
    #     0.7: 35,
    #     0.8: 40,
    #     0.9: 45,
    #     1.0: 50,
    # }
    if translat2 == 'SNR':
        return (snr_dict[prob], snr_dict[prob]+5)
    elif translat2 == 'max_mask_value_freq':
        return mask_value_frequency_dict[prob]


def config_distortions(
    reverb_irfiles=None,
    music_dir=None,
    overlap_dir=None,
    noises_dir=None,
    reverb_p=0.,
    music_p=0,
    noises_p=0.,
    overlap_p=0.,
    random_scale_p=0,
    music_snrs=(10, 15, 20,),
    noises_snrs=range(0, 15),
    overlap_snrs=(15, 20, 25,),
    reverb_snrs=None,
    n_noise=(5,),
    overlap_reverb=True,
    cache=False,
    clip_factors=(),
    clip_p=0,
    bandrop_p=0,
    bandrop_dir=None,
    max_ir_length=24000,
):

    if not type(music_snrs) in [tuple, list]:
        music_snrs = (music_snrs,)

    if not type(noises_snrs) in [tuple, list]:
        noises_snrs = (noises_snrs,)

    if not type(overlap_snrs) in [tuple, list]:
        overlap_snrs = (overlap_snrs,)
    ## deal with SNR using probability input
    trans = []
    probs = []
    # # Reverb can be shared in two different stages of the pipeline
    if random_scale_p > 0.0:
        print(print(f"random_scale_p is {random_scale_p}"))
        trans.append(RandomScale())
        probs.append(random_scale_p)

    reverb = Reverb(reverb_irfiles, cache=False, max_reverb_len=max_ir_length)

    if reverb_p > 0.0 and reverb_irfiles is not None:
        print(f"reverb_p is {reverb_p}")
        trans.append(reverb)
        probs.append(reverb_p)
    #
    if overlap_p > 0.0 and overlap_dir is not None:
        print(f"overlap_p is {overlap_p}")
        if type(overlap_snrs) is float:
            overlap_snrs = translate_prob2other_ino(overlap_snrs, translat2='SNR')
        noise_trans = reverb if overlap_reverb else None
        trans.append(
            Bable(
                overlap_dir,
                snr_levels=overlap_snrs,
                noise_transform=noise_trans,
                n_noise=n_noise,
                cache=cache,
            )
        )
        probs.append(overlap_p)

    if music_p > 0.0 and music_dir is not None:
        print(f"music_p is {music_p}")
        if type(music_snrs) is float:
            music_snrs = translate_prob2other_ino(music_snrs, translat2='SNR')
        trans.append(Bable(music_dir, snr_levels=music_snrs, n_noise=(1,), cache=cache))
        probs.append(music_p)

    if noises_p > 0.0 and noises_dir is not None:
        print(f"noises_p is {noises_p}")
        if type(noises_snrs) is float:
            noises_snrs = translate_prob2other_ino(noises_snrs, translat2='SNR')
        trans.append(SimpleAdditive(noises_dir, noises_snrs, cache=cache))
        probs.append(noises_p)

    if clip_p > 0.0 and len(clip_factors) > 0:
        trans.append(Clipping(clip_factors))
        probs.append(clip_p)

    if bandrop_p > 0. and bandrop_dir is not None:
        trans.append(BandDrop(bandrop_dir))
        probs.append(bandrop_p)

    # print(trans)
    if len(trans) > 0:
        return PCompose(trans, probs=probs)
    else:
        return None

def dict2h5(in_dict, file, dataset='', mode='a'):
    with h5py.File(file, mode) as f:
        for key, val in in_dict.items():
            if val.dtype == np.object:
                f[dataset + key] = val.astype(unicode)
            else:
                f[dataset + key] = val



def averge_xvectors(data, by):
    X, spk_ids, spk_path = [], [], []
    for i, uni in enumerate(np.unique(data[by])):
        mask = data[by] == uni
        x = np.mean(data['X'][mask], 0)
        X.append(x)
        spk_path.append(data['spk_path'][mask][0])
        spk_ids.append(data['spk_ids'][mask][0])
    return {'X': np.stack(X),
            'spk_ids': np.stack(spk_ids).astype(unicode),
            'spk_path': np.stack(spk_path).astype(unicode)}



from scipy.linalg import sqrtm
import math


class Coral:
    def __init__(self):
        pass

    def fit_transform(self, X_src, X_tgt, mode='both'):
        S_inv_sqrt = sqrtm(inv(np.cov(X_src.T)))
        T_sqrt = sqrtm(np.cov(X_tgt.T))
        transfrom = S_inv_sqrt.dot(T_sqrt)
        X_src = X_src.dot(transfrom)
        return X_src


# class Whiten:
#     def __init__(self):
#         pass
#
#     def fit(self, X_tgt):
#         self.matrix = sqrtm(inv(np.cov(X_tgt.T)))
#         return self
#
#     def transform(self, X_src):
#         return X_src.dot(self.matrix)


def get_whiten_matrix(X):
    Xcov = np.cov(X.T)
    d, V = np.linalg.eigh(Xcov)
    D = np.diag(1. / np.sqrt(d))
    return np.dot(V, D)


class Whiten:
    def __init__(self):
        pass

    def fit(self, X):
        Xcov = np.dot(X.T, X) / X.shape[0]

        d, V = np.linalg.eigh(Xcov)

        D = np.diag(1. / np.sqrt(d))
        self.matrix = np.dot(V, D)

        return self

    def transform(self, X_src):
        return X_src.dot(self.matrix)


# class Whiten_diag:
#     def __init__(self):
#         pass
#
#     def fit(self, X):
#         Xcov = np.diag(np.diag(np.dot(X.T, X) / X.shape[0]))
#
#         d, V = np.linalg.eigh(Xcov)
#
#         D = np.diag(1. / np.sqrt(d))
#         self.matrix = np.dot(V, D)
#
#         return self
#
#     def transform(self, X_src):
#         return X_src.dot(self.matrix)


# class Whiten_diag_two:
#     def __init__(self):
#         pass
#
#     def fit(self, X, Y):
#         Xcov = np.diag(np.diag(np.dot(X.T, X) / X.shape[0]))
#         Ycov = np.diag(np.diag(np.dot(Y.T, Y) / Y.shape[0]))
#         cov = (Xcov + Ycov) / 2
#
#         d, V = np.linalg.eigh(cov)
#
#         D = np.diag(1. / np.sqrt(d))
#         self.matrix = np.dot(V, D)
#
#         return self
#
#     def transform(self, X_src):
#         return X_src.dot(self.matrix)

class Whiten_multiple:
    def __init__(self):
        pass

    def fit(self, Xs):
        cov = []
        for X in Xs:
            cov.append(np.dot(X.T, X) / X.shape[0])
        cov = np.stack(cov).mean(0)
        # Xcov = np.dot(X.T, X) / X.shape[0]
        # Ycov = np.dot(Y.T, Y) / Y.shape[0]
        # cov = (Xcov + Ycov) / 2

        d, V = np.linalg.eigh(cov)

        D = np.diag(1. / np.sqrt(d))
        self.matrix = np.dot(V, D)

        return self

    def transform(self, X_src):
        return X_src.dot(self.matrix)


class Whiten_two:
    def __init__(self):
        pass

    def fit(self, X, Y):
        Xcov = np.dot(X.T, X) / X.shape[0]
        Ycov = np.dot(Y.T, Y) / Y.shape[0]
        cov = (Xcov + Ycov) / 2

        d, V = np.linalg.eigh(cov)

        D = np.diag(1. / np.sqrt(d))
        self.matrix = np.dot(V, D)

        return self

    def transform(self, X_src):
        return X_src.dot(self.matrix)


def whiten_by_target_data(X_source, X_target):
    # X is a NxM data matrix, where N is the
    # number of examples, M is the number of features

    # compute covarince matrix of target data
    X_target_cov = np.dot(X_target.T, X_target) / X_target.shape[0]

    # compute eigen of the covariance matrix
    # w is the eignvalues, V is the eighvectors matrix
    w, V = np.linalg.eigh(X_target_cov)

    # compute inverse sqare root of eigen value
    D = np.diag(1. / np.sqrt(w))

    # get whiten matrix
    whiten_matrix = np.dot(V, D)

    # whiten source data by whiten_matrix
    return X_source.dot(whiten_matrix)


def whiten_by_target_data_diag(X_source, X_target):
    # X is a NxM data matrix, where N is the
    # number of examples, M is the number of features

    # compute covarince matrix of target data
    X_target_cov = np.diag(np.diag(np.dot(X_target.T, X_target) / X_target.shape[0]))

    # compute eigen of the covariance matrix
    # w is the eignvalues, V is the eighvectors matrix
    w, V = np.linalg.eigh(X_target_cov)

    # compute inverse sqare root of eigen value
    D = np.diag(1. / np.sqrt(w))

    # get whiten matrix
    whiten_matrix = np.dot(V, D)

    # whiten source data by whiten_matrix
    return X_source.dot(whiten_matrix)

# def whiten_by_target_data(X_source, X_target):
#     # X is a NxM data matrix, where N is the
#     # number of examples, M is the number of features
#
#     # compute covarince matrix of target data
#     X_target_center = X_target - X_target.mean(0)
#     X_target_cov = np.dot(X_target_center.T, X_target_center) / X_target_center.shape[0]
#
#     # compute eigen of the covariance matrix
#     # w is the eignvalues, V is the eighvectors matrix
#     w, V = np.linalg.eigh(X_target_cov)
#
#     # compute inverse sqare root of eigen value
#     D = np.diag(1. / np.sqrt(w))
#
#     # get whiten matrix
#     whiten_matrix = np.dot(V, D)
#
#     # whiten source data by whiten_matrix
#     return X_source.dot(whiten_matrix)


# class MomentsMatcher:
#     def __init__(self):
#         pass
#
#     def fit_transform(self, X_src, X_tgt, mode='both'):
#         S_inv_sqrt = sqrtm(inv(np.cov(X_src.T)))
#         T_sqrt = sqrtm(np.cov(X_tgt.T))
#         transfrom = S_inv_sqrt.dot(T_sqrt)
#         X_src = X_src.dot(transfrom)
#         if mode == 'both':
#             return X_src - X_src.mean(0) + X_tgt.mean(0)
#         elif mode == 'second':
#             return X_src
#         else:
#             raise NotImplemented

        # self.target_mean = X_tgt.mean(0)
        # self.train_flag = True
        # return self

    # def fit(self, source, target):
    #     S_inv_sqrt = sqrtm(inv(np.cov(source.T)))
    #     T_sqrt = sqrtm(np.cov(target.T))
    #     self.transfrom = S_inv_sqrt.dot(T_sqrt)
    #
    #     self.target_mean = target.mean(0)
    #     self.train_flag = True
    #     return self
    #
    # def transform(self, source):
    #     return source.dot(self.transfrom)
    #
    # def fit_transform(self, source, target):
    #     self.fit(source, target)
    #     return self.transform(source)


# class CORAL:
#     def __init__(self):
#         pass
#
#     def fit(self, source, target):
#         S_inv_sqrt = sqrtm(inv(np.cov(source.T)))
#         T_sqrt = sqrtm(np.cov(target.T))
#         self.transfrom = S_inv_sqrt.dot(T_sqrt)
#         self.train_flag = True
#         return self
#
#     def transform(self, source):
#         return source.dot(self.transfrom)
#
#     def fit_transform(self, source, target):
#         self.fit(source, target)
#         return self.transform(source)


# class IDVC:
#     def __init__(self, n_components=None):
#         self.n_components = n_components
#
#     def fit(self, X, y):
#         means = []
#         for y_uni in np.unique(y):
#             means.append(X[y == y_uni].mean(0))
#         means = np.stack(means)
#         means = means - means.mean(0)
#         _, _, Vh = la.svd(means, full_matrices=False)
#         self.proj = np.eye(X.shape[-1]) - Vh.T @ Vh
#         if self.n_components is not None:
#             self.pca = PCA().fit(X @ self.proj)
#         return self
#
#     def transform(self, X):
#         if self.n_components is not None:
#             return self.pca.transform(X @ self.proj)
#         else:
#             return X @ self.proj
#
#     def fit_transform(self, X, y):
#         self.fit(X, y)
#         return self.transform(X)


class ClassBalanceDemean:
    def __init__(self):
        pass

    def fit(self, X, y):
        means = []
        for uni in np.unique(y):
            means.append(X[y == uni].mean(0))
        self.mean = np.stack(means).mean(0)
        return self

    def transform(self, X):
        return X - self.mean

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


class Demean:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.mean = np.mean(X, 0)
        return self

    def transform(self, X):
        return X - self.mean

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class Lennorm:
    def __init__(self):
        pass

    def fit(self, x=None, y=None):
        return self

    def transform(self, X):
        return lennorm(X)

    def fit_transform(self, X, y=None):
        self.fit()
        return self.transform(X)


class NewLennorm:
    def __init__(self):
        pass

    def fit(self, x=None, y=None):
        return self

    def transform(self, X):
        return lennorm(X) / math.sqrt(X.shape[-1])

    def fit_transform(self, X, y=None):
        self.fit()
        return self.transform(X)



class PLDA:
    def __init__(self, transforms=None, dev_files=None,n_fac=150, n_iter=20, print_llh=False):
        self.dev_files = dev_files if type(dev_files) is list else [dev_files]
        self.n_fac = n_fac
        self.n_iter = n_iter
        self.llh_lst = [-np.inf]
        self.transforms = transforms
        self.print_llh = print_llh

    @property
    def prec(self):
        return inv(self.sigma)

    def fit(self, X=None, spk_ids=None):
        if X is None:
            X, spk_ids, utt_ids = self.load_data()
        spk_ids = LabelEncoder().fit_transform(spk_ids)
        print(f'n_spk is {len(np.unique(spk_ids))}')

            # _, inv_idx, counts_uni = np.unique(spk_ids, return_counts=True, return_inverse=True)
            # counts = counts_uni[inv_idx]
            # mask = counts >= 8
            # print(f'filter {(counts_uni < 8).sum()}/{len(counts_uni)}')
            # X, spk_ids = X[mask], spk_ids[mask]

        # for transform in self.transforms:
        #     if hasattr(transform, 'train_flag'):
        #         X = transform.transform(X)
        #         print('warning one transform is not fitted by internal data')
        #     else:
        #         X = transform.fit_transform(X, spk_ids)
        if self.transforms:
            for transform in self.transforms:
                X = transform.fit_transform(X, spk_ids)

        print('preprocessing done')

        self.W= randn(self.n_fac, X.shape[-1])
        self.sigma = abs(randn()) * np.eye(X.shape[-1])
        xstat = suff_xstats(X, spk_ids)
        for i in range(self.n_iter):
            print(f'iteration {i}')
            zstat = self.e_step(xstat)
            if self.print_llh:
                self.comp_llh(xstat, zstat, mode='elbo')
            self.m_step(xstat, zstat)
        return self

    def load_data(self):
        X, spk_ids, utt_ids = [], [], []
        for file in self.dev_files:
            print('load dev data from {}'.format(file))
            with h5.File(file, 'r') as f:
                X.append(f['X'][:])
                spk_ids.append(f['spk_ids'][:])
                utt_ids.append(f['spk_path'][:])
        X = np.concatenate(X, axis=0)
        spk_ids = np.concatenate(spk_ids, axis=0)
        utt_ids = np.concatenate(utt_ids, axis=0)
        return X, spk_ids, utt_ids

    # def load_data(self):
    #     X, spk_ids, durations = [], [], []
    #     for file in self.dev_files:
    #         print('load dev data from {}'.format(file))
    #         with h5.File(file, 'r') as f:
    #             X.append(f['X'][:])
    #             spk_ids.append(f['spk_ids'][:])
    #             durations.append(f['durations'][:])
    #     X = np.concatenate(X, axis=0)
    #     spk_ids = np.concatenate(spk_ids, axis=0)
    #     durations = np.concatenate(durations, axis=0)
    #     return X, spk_ids, durations


    def e_step(self, x):
        WtP = self.prec @ self.W.T
        WtPW = self.W @ WtP
        eigval, eigvec = linalg.eigh(WtPW)
        inv_eigvec = inv(eigvec)
        n_id = len(x['ns_obs'])
        mu_post = np.zeros((n_id, self.n_fac))
        sigma_post = np.zeros((n_id, self.n_fac, self.n_fac))
        # sigma_eigval_sum = np.zeros(self.n_fac)
        for i_id, (X_homo_sum, n_ob) in enumerate(zip(x['homo_sums'], x['ns_obs'])):
            # sigma_post[i_id] = inv(np.eye(self.n_fac) + n_ob * WtPW)
            # inv_diag = np.diag(1 / (1 + n_ob * eigval))
            inv_diag = (1 / (1 + n_ob * eigval))[:, None]
            # Todo this step can be further speed up
            sigma_post[i_id] = eigvec @ (inv_diag * inv_eigvec)
            # sigma_eigval_sum += inv_diag * n_ob
            mu_post[i_id] = X_homo_sum @ WtP @ sigma_post[i_id]
        # Todo look for a more efficent way to compute outer product
        # sigma_post_sum = eigvec @ np.diag(sigma_eigval_sum) @ inv_eigvec

        # mu_mom2s = np.einsum('Bi,Bj->Bij', mu_post, mu_post) + sigma_post

        mu_mom2_sum = mu_post.T @ (mu_post * x['ns_obs'][:, None]) + (x['ns_obs'][:, None, None] * sigma_post).sum(0)

        # mu_mom2_sum = mu_post.T @ (mu_post * x['ns_obs'][:, None]) + sigma_post_sum
        # return {'mom1s': mu_post, 'mom2s': mu_mom2s}
        return {'mom1s': mu_post, 'mu_mom2_sum': mu_mom2_sum}

    def m_step(self, x, z):
        # z_mom2s_sum = np.einsum('B,Bij->ij', x['ns_obs'], z['mom2s'])
        z_mom2s_sum = z['mu_mom2_sum']

        xz_cmom = z['mom1s'].T @ x['homo_sums']
        self.W = inv(z_mom2s_sum) @ xz_cmom
        self.sigma = (x['mom2'] - xz_cmom.T @ self.W) / x['ns_obs'].sum()
        # print('diag sigma')
        # self.sigma = np.diag(np.diag(self.sigma))

    # def comp_llh(self, xstat, zstat, mode='elbo', dev=None, spk_ids=None,):
    #     if mode == 'elbo':
    #         llh = self.elbo(xstat, zstat)
    #     else:
    #         llh = exact_marginal_llh(
    #             dev=dev, idens=spk_ids, W=self.W, sigma=self.sigma,)
    #     self._display_llh(llh)

    def elbo(self, xstat, zstat):
        WtPW = self.W @ self.prec @ self.W.T
        return - _ce_cond_xs(xstat, zstat, self.W, self.prec) \
               - _ce_prior(zstat) \
               + _entropy_q(xstat['ns_obs'], WtPW)

    def _display_llh(self, llh):
        self.llh_lst.append(llh)
        if self.llh_lst[-2] == -np.inf:
            print('llh = {:.4f} increased inf\n'.format(llh))
        else:
            margin = self.llh_lst[-1] - self.llh_lst[-2]
            change_percent = 100 * np.abs(margin / self.llh_lst[-2])
            print('llh = {:.4f} {} {:.4f}%\n'.format(
                llh, 'increased' if margin > 0 else 'decreased', change_percent,))

    def comp_pq(self):
        sig_ac = self.W.T @ self.W
        sig_tot = sig_ac + self.sigma
        prec_tot = inv(sig_tot)
        aux = inv(sig_tot - sig_ac @ prec_tot @ sig_ac)
        B0 = np.zeros_like(self.sigma)
        M1 = np.block([[sig_tot, sig_ac], [sig_ac, sig_tot]])
        M2 = np.block([[sig_tot, B0], [B0, sig_tot]])
        P = aux @ sig_ac @ prec_tot
        Q = prec_tot - aux
        const = 0.5 * (-log_det4psd(M1) + log_det4psd(M2))
        return {'P': P, 'Q': Q, 'const': const}

    def comp_pq_using_within(self, within, without):
        sig_ac = within
        sig_tot = within + without
        prec_tot = inv(sig_tot)
        aux = inv(sig_tot - sig_ac @ prec_tot @ sig_ac)
        B0 = np.zeros_like(sig_tot)
        M1 = np.block([[sig_tot, sig_ac], [sig_ac, sig_tot]])
        M2 = np.block([[sig_tot, B0], [B0, sig_tot]])
        P = aux @ sig_ac @ prec_tot
        Q = prec_tot - aux
        const = 0.5 * (-log_det4psd(M1) + log_det4psd(M2))
        return {'P': P, 'Q': Q, 'const': const}

    def save_model(self, save_file_to):
        model = self.comp_pq()
        model_dict = {
            'model': model,
            'transform_lst': self.transforms,
        }
        with open(save_file_to, 'wb') as f:
            pickle.dump(model_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            print('save model to {}'.format(save_file_to))


def suff_xstats(X, spk_ids):
    # X -= X.mean(0)
    mom2 = X.T @ X
    unique_ids, ns_obs = np.unique(spk_ids, return_counts=True)
    homo_sums = np.zeros((unique_ids.shape[0], X.shape[-1]))
    for i_id, unique_id in enumerate(unique_ids):
        homo_sums[i_id] = np.sum(X[spk_ids == unique_id], axis=0)
    return {'mom2': mom2, 'homo_sums': homo_sums, 'ns_obs': ns_obs}


_LOG_2PI = np.log(2 * np.pi)


def _ce_cond_xs(x, z, W, prec):
    dim = prec.shape[-1]
    N = x['ns_obs'].sum()
    xy_cmom = W.T @ z['mom1s'].T @ x['homo_sums']
    z_mom2s_wsum = np.einsum('B,Bij->ij', x['ns_obs'], z['mom2s'])
    dev_mom2 = x['mom2'] - xy_cmom - xy_cmom.T + W.T @ z_mom2s_wsum @ W
    return 0.5 * (N * dim * _LOG_2PI
                  - N * log_det4psd(prec)
                  + ravel_dot(dev_mom2, prec))


def _ce_prior(z):
    n_ids, dim = z['mom1s'].shape
    return 0.5 * (n_ids * dim * _LOG_2PI
                  + np.einsum('Bii->', z['mom2s']))


def _entropy_q(ns_obs, WtPW):
    n_ids = len(ns_obs)
    zdim = WtPW.shape[0]
    # due to the special form of posterior co logdet can be greatly simplified
    eigvals = np.outer(ns_obs, eigvalsh(WtPW)) + 1
    log_det_sum = np.sum(np.log(1 / eigvals))
    return 0.5 * (n_ids * zdim * _LOG_2PI
                  + log_det_sum
                  + n_ids * zdim)


def log_det4psd(sigma):
    return 2 * np.sum(np.log(np.diag(linalg.cholesky(sigma))))




# def exact_marginal_llh(dev, idens, W, sigma):
#     # this is very computation intensive op should only be used to
#     # check whether low-bound is correct on toy data
#     # stake mu is 0, diag of cov is sigma + WWt, off-diag is WWt
#     llh = 0.0
#     unique_ids = np.unique(idens)
#     for unique_id in unique_ids:
#         dev_homo = dev[idens == unique_id]
#         cov = _construct_marginal_cov(W.T @ W, sigma, dev_homo.shape[0])
#         llh += Gauss(cov=cov).log_p(dev_homo.ravel())
#     return llh


def _construct_marginal_cov(heter_cov, noise_cov, n_obs):
    cov = np.tile(heter_cov, (n_obs, n_obs))
    rr, cc = noise_cov.shape
    r, c = 0, 0
    for _ in range(n_obs):
        cov[r:r+rr, c:c+cc] += noise_cov
        r += rr
        c += cc
    return cov



def train_and_score(
            test,
            ndx,
            enroll=None,
            train=None,
            second_order_adapt=True,
            plda_dim=200,
            target=None,
            cohort=None,
            n_iter=20,
            lda_dim=200,
            top_scores=300,
            group_infos=None,
            blind_trial=False,
            slow_score=False,
            return_plda_pq=False,
            **kwargs):
    if not enroll:
        enroll = test
    if train:
        print("using plda backend")
        data_src = h52dict(train)
        if target:
            if type(target) is dict:
                X_tgt = target["X"]
            else:
                X_tgt = h52dict(target)["X"]
            if second_order_adapt:
                data_src["X"] = Coral().fit_transform(data_src["X"], X_tgt)
            data_src["X"] = data_src["X"] - data_src["X"].mean(0) + X_tgt.mean(0)

        transform_lst = [Demean(), LDA(n_components=lda_dim), NewLennorm(), Demean()]
        plda = PLDA(transforms=transform_lst, n_iter=n_iter, n_fac=plda_dim)
                
        pq = plda.fit(data_src["X"], data_src["spk_ids"]).comp_pq()
                
        score = Score(
            enroll=enroll,
            test=test,
            blind_trial=blind_trial,
            group_infos=group_infos,
            transforms=transform_lst,
            cohort=cohort,
            ndx_file=ndx,
            **kwargs
        )
        score.batch_plda_score(pq)
    else:
        print("using cosine backend")
        if target:
            if type(target) is dict:
                data_target = target
            else:
                data_target = h52dict(target)
            transform_lst = [PCA(whiten=True)]
            for transform in transform_lst:
                transform.fit(data_target["X"])
        else:
            transform_lst = None

        score = Score(
            blind_trial=blind_trial,
            enroll=enroll,
            test=test,
            transforms=transform_lst,
            ndx_file=ndx,
            cohort=cohort,
            top_scores=top_scores,
            **kwargs
        )
        if slow_score:
            score.cosine_score()
        else:
            score.batch_cosine_score()
    if return_plda_pq:
        return score.ndx, pq, transform_lst
    else:
        return score.ndx