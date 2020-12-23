import pandas as pd
import glob
import contextlib
import wave
import numpy as np


def prepare_voice19_dev_info(read_dir, save2=None, trial_file=None):
    assert read_dir.endswith('v19-dev')
    df = pd.DataFrame({'full_paths': glob.glob(f"{read_dir}/**/*.wav", recursive=True)})
    if len(df) == 0:
        raise ValueError('empty dir')
    # df["relative_paths"] = df.full_paths.str.rpartition("/")[2]
    df["relative_paths"] = df.full_paths.str.split(r"v19-dev/").str[-1]
    df["utt_ids"] = df.full_paths.str.rpartition("/")[2]
    df["spk_ids"] = df.full_paths.str.rsplit("/").str[-2]
    df["utt_ids"] = df["spk_ids"] + '-' + df["utt_ids"]
    df = get_wav_duration(df)
    save2 = f'{read_dir}/ids_info.tsv' if not save2 else save2
    df.to_csv(save2, sep='\t', columns=['relative_paths', 'utt_ids', 'spk_ids', 'durations', 'ns_frames'], index=False)
    print(f'save to {save2}')


def prepare_voice19_eval_info(read_dir, save2=None, trial_file=None):
    assert read_dir.endswith('v19-eval')
    df = pd.DataFrame({'full_paths': glob.glob(f"{read_dir}/sid_eval/*.wav")})
    if len(df) == 0:
        raise ValueError('empty dir')
    # df["relative_paths"] = df.full_paths.str.rpartition("/")[2]
    df["relative_paths"] = df.full_paths.str.split(r"v19-eval/").str[-1]
    df["utt_ids"] = df.full_paths.str.rpartition("/")[2]
    df["spk_ids"] = df["utt_ids"]
    # df["utt_ids"] = df["spk_ids"] + '-' + df["utt_ids"]
    df = get_wav_duration(df)
    save2 = f'{read_dir}/ids_info.tsv' if not save2 else save2
    df.to_csv(save2, sep='\t', columns=['relative_paths', 'utt_ids', 'spk_ids', 'durations', 'ns_frames'], index=False)
    print(f'save to {save2}')


# add duration, n_classes, n_utt to used for filter out spk
def prepare_vox(read_dir, save2=None, trial_file=None, save_trial2=None):
    assert read_dir.endswith(('/dev', '/test')), 'should end with dev or test'
    df = pd.DataFrame({'full_paths': glob.glob(f"{read_dir}/**/*.wav", recursive=True)})
    if len(df) == 0:
        raise ValueError('empty dir')
    df['relative_paths'] = df.full_paths.str.split(r'/test/|/dev/').str[-1]
    # df['relative_paths'] = df.full_paths.str.split(r'(?=test)|(?=dev)').str[-2:]
    # df.loc[:, "utt_ids"] = df.full_paths.str.split("/wav/|/acc/").str[-1]
    df.loc[:, "utt_ids"] = df.full_paths.str.split(r"/wav/|/aac/").str[-1]
    # breakpoint()
    df.loc[:, "spk_ids"] = df["utt_ids"].str.partition('/')[0]
    df = get_wav_duration(df)
    save2 = f'{read_dir}/ids_info.tsv' if not save2 else save2
    df.to_csv(save2, sep='\t', columns=['relative_paths', 'utt_ids', 'spk_ids', 'durations', 'ns_frames'], index=False)
    print(f'save to {save2}')
    if trial_file:
        trial = pd.read_csv(trial_file, sep=' ', names=['modelid', 'segmentid', 'targettype'])
        trial['targettype'] = trial['targettype'].map({1: "target", 0: "nontarget"})
        trial.to_csv(save_trial2, sep='\t', index=None)


def get_wav_duration(df):
    durations = []
    ns_frames = []
    for file in df['full_paths']:
        with contextlib.closing(wave.open(file, "r")) as f:
            n_frames = f.getnframes()
            rate = f.getframerate()
            duration = n_frames / float(rate)
            ns_frames.append(n_frames)
            durations.append(duration)
    df['durations'] = np.array(durations)
    df['ns_frames'] = np.array(ns_frames)
    return df


def utt2utt_trial(trial_file, save2, dev):
    df = pd.read_csv(trial_file, sep=' ', names=['modelid', 'segmentid', 'targettype'])
    df['modelid'] = df['modelid'] + '.wav'
    df['segmentid'] = df['segmentid'].str.rpartition('/')[2]
    if dev:
        df['modelid'] = df['modelid'].str.split('-').str[5] + '-' + df['modelid']
        df['segmentid'] = df['segmentid'].str.split('-').str[5] + '-' + df['segmentid']
    # prefix spk_ids to utt_ids
    # df['modelid'] =
    df['targettype'] = df.targettype.map({'tgt': 'target', 'imp': 'nontarget'})
    df.to_csv(save2, sep='\t', index=None)


if __name__ == '__main__':
    ##
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['vox', 'voice19-dev', 'voice19-eval'])
    parser.add_argument('--dataset_dir', required=True)
    args = parser.parse_args()
    print(args)
    if args.dataset == 'voice19-dev':
        assert args.dataset_dir.endswith('v19-dev')
        prepare_voice19_dev_info(args.dataset_dir)
    elif args.dataset == 'voice19-eval':
        assert args.dataset_dir.endswith('v19-eval')
        prepare_voice19_eval_info(args.dataset_dir)
    elif args.dataset == 'vox':
        assert args.dataset_dir.endswith(('dev', 'test'))
        prepare_vox(args.dataset_dir)
    else:
        raise ValueError


    # prepare_vox('/home12a/wwlin/corpus/vox1_fixed/test')
    # prepare_vox('/home12a/wwlin/corpus/vox1_fixed/dev')
    # prepare_vox('/home12a/wwlin/corpus/vox2_wav/dev')
    # prepare_vox('/home12a/wwlin/corpus/vox2_wav/test')
    #
    # utt2utt_trial('/home7b/wwlin/corpus/voice19/v19-dev/sid_dev_lists_and_keys/dev-trial-keys.lst',
    #               'docs/new_voice19_dev_trial.tsv', dev=True)
    # prepare_voice19_dev_info('/home7b/wwlin/corpus/voice19/v19-dev')
    # prepare_voice19_eval_info('/home7b/wwlin/corpus/voice19/v19-eval')
