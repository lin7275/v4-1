import glob
from subprocess import check_call
import pandas as pd
import numpy as np
from multiprocessing import Process
import os
from shutil import copytree, ignore_patterns


def _convert_wav(df):
    # with open(f'error_{os.getpid()}.log', 'w') as f:
    #     for _, row in df.iterrows():
    #         check_call([
    #             'ffmpeg', '-v', '8', '-i', row['read_files'],
    #             '-f', 'wav', '-acodec', 'pcm_s16le',
    #             f"{row['save_files']}",
    #         ], stderr=f)
    for _, row in df.iterrows():
        check_call([
            'ffmpeg', '-v', '8', '-i', row['read_files'],
            '-f', 'wav', '-acodec', 'pcm_s16le',
            f"{row['save_files']}",
        ])


def convert_wav_interal(read_dir, wav_dir, n_jobs):
    files = glob.glob(f'{read_dir}/**/*.m4a', recursive=True)
    if len(files) == 0:
        raise ValueError('there are files exise in wav dir')
    df = pd.DataFrame({'read_files': files})
    df['save_files'] = df.read_files.str.replace('.m4a$', '.wav', regex=True) \
        .str.replace(read_dir, wav_dir)

    dfs = np.array_split(df, n_jobs)
    processes = []
    for i, df in enumerate(dfs):
        p = Process(target=_convert_wav, args=(df,))
        p.start()
        print(f'process {i} has started')
        processes.append(p)

    for p in processes:
        p.join()


def convert_wav(read_dir, save_dir, n_jobs):
    copytree(read_dir, save_dir, ignore=ignore_patterns('*.wav', '*.m4a'))
    convert_wav_interal(read_dir, save_dir, n_jobs)


def down_sample_wav(read_dir, wav_dir, n_jobs):
    files = glob.glob(f'{read_dir}/**/*.wav', recursive=True)
    if len(files) == 0:
        raise ValueError('there is no exise in wav dir')
    df = pd.DataFrame({'read_files': files})
    df['save_files'] = df.read_files.str.replace(read_dir, wav_dir)

    dfs = np.array_split(df, n_jobs)
    processes = []
    for i, df in enumerate(dfs):
        p = Process(target=_down_sample, args=(df,))
        p.start()
        print(f'process {i} has started')
        processes.append(p)

    for p in processes:
        p.join()


def lauch_job(read_dir, wav_dir, filer_pattern, func_call, n_jobs, new_subfix=None):
    files = glob.glob(f'{read_dir}/{filer_pattern}', recursive=True)
    if len(files) == 0:
        raise ValueError('there are files exise in wav dir')
    df = pd.DataFrame({'read_files': files})
    df['save_files'] = df.read_files.str.replace(read_dir, wav_dir)

    if new_subfix:
        df['save_files'] =df['save_files'].str.replace('\..+$', f'.{new_subfix}', regex=True)
    # df['save_files'] = df.read_files.str.replace('.m4a$', '.wav', regex=True) \
    #     .str.replace(read_dir, wav_dir)

    dfs = np.array_split(df, n_jobs)
    processes = []
    for i, df in enumerate(dfs):
        p = Process(target=func_call, args=(df,))
        p.start()
        print(f'process {i} has started')
        processes.append(p)

    for p in processes:
        p.join()

# def _flac2wav(df):
#     for _, row in df.iterrows():
#         check_call([
#             'flac', '-c', '-d', '-s', row['read_files'], row['save_files'],
#         ])

def _down_sample(df):
    for _, row in df.iterrows():
        check_call([
            'sox', row['read_files'], '-r', '8000', row['save_files'],
        ])


def _concat_wav(df):
    with open(f'error_{os.getpid()}.log', 'w') as f:
        for _, row in df.iterrows():
            # print(row['read_parent_dir'])
            # print(row['save_parent_dir'])

            check_call([
                'sox', f"{row['read_parent_dir']}/*.wav", f"{row['save_parent_dir']}/concat.wav"
            ], stderr=f)


def concat_wav_internal(read_wav_dir, concat_wav_dir, n_jobs):
    read_files = glob.glob(f'{read_wav_dir}/**/*.wav', recursive=True)
    if not read_files:
        raise ValueError('empty dir')
    read_parent_dir = pd.Series(read_files).str.rpartition('/')[0].drop_duplicates()
    save_parent_dir = read_parent_dir.str.replace(read_wav_dir, concat_wav_dir)
    df = pd.DataFrame({'read_parent_dir': read_parent_dir.values,
                       'save_parent_dir': save_parent_dir.values})

    dfs = np.array_split(df, n_jobs)
    processes = []
    for i, df in enumerate(dfs):
        p = Process(target=_concat_wav, args=(df,))
        p.start()
        print(f'process {i} has started')
        processes.append(p)

    for p in processes:
        p.join()


def concat_wav(read_wav_dir, concat_wav_dir, n_jobs):
    copytree(read_wav_dir, concat_wav_dir, ignore=ignore_patterns('*.wav', '*.m4a'))
    concat_wav_internal(read_wav_dir, concat_wav_dir, n_jobs)


def downsample_vox(read_wav_dir, concat_wav_dir, n_jobs):
    copytree(read_wav_dir, concat_wav_dir, ignore=ignore_patterns('*.wav', '*.m4a'))
    down_sample_wav(read_wav_dir, concat_wav_dir, n_jobs)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--func', choices=['m4a2wav', 'concat'], required=True)
    parser.add_argument('--read_dir', required=True)
    parser.add_argument('--save2', required=True)
    parser.add_argument('--nj', default=20)
    args = parser.parse_args()
    print(args)
    if args.func == 'm4a2wav':
        convert_wav(args.read_dir, args.save2, args.nj)
    elif args.func == 'concat':
        concat_wav(args.read_dir, args.save2, args.nj)
    else:
        raise ValueError
