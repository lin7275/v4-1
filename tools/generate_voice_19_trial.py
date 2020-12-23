import pandas as pd


def utt2utt_trial(trial_file, save2, type):
    df = pd.read_csv(trial_file, sep=' ', names=['modelid', 'segmentid', 'targettype'])
    df['modelid'] = df['modelid'] + '.wav'
    df['segmentid'] = df['segmentid'].str.rpartition('/')[2]
    if type=='dev':
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
    parser.add_argument('--orgin_trial', required=True)
    parser.add_argument('--new_trial', required=True)
    parser.add_argument('--type', choices=['dev', 'eval'], required=True)
    args = parser.parse_args()

    utt2utt_trial(args.orgin_trial, args.new_trial, args.type)