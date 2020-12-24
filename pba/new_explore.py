# a reasonable SNR level should be between 0 to 15dB
import random
import numpy as np


pname_list = [
    "noises_p",
    "reverb_p",
    "music_p",
    "overlap_p",
    "noises_snrs",
    "music_snrs",
    "overlap_snrs",
    # 'bandrop_p'
]


def convert2old(aug_config):
    return {
        'reverb_p': aug_config['reverb']['p'],
        'music_p': aug_config['music']['p'],
        'noises_p': aug_config['noises']['p'],
        'overlap_p': aug_config['overlap']['p'],
        'music_snrs': aug_config['music']['snr'],
        'overlap_snrs': aug_config['overlap']['snr'],
        'noises_snrs': aug_config['noises']['snr'],
        "spec_aug_time_mask_size": aug_config["spec_aug"]['time_mask_size'],
        "spec_aug_freq_mask_size": aug_config["spec_aug"]['freq_mask_size'],
    }

def convert2new(aug_config):
    return {
        'reverb': {'p': aug_config['reverb_p'], 'snr': 0,},
        'music': {'p': aug_config['music_p'], 'snr': aug_config['music_snrs']},
        'overlap': {'p': aug_config['overlap_p'], 'snr': aug_config['overlap_snrs']},
        'noises': {'p': aug_config['noises_p'], 'snr': aug_config['noises_snrs']},
        "spec_aug": {"time_mask_size": aug_config["spec_aug_time_mask_size"],
                     "freq_mask_size": aug_config["spec_aug_freq_mask_size"]},
    }


def explore(aug_config_new):
    for key, item in aug_config_new.items():
        #ax_freq_mask_len=27, max_time_mask_len=100
        if key == "spec_aug":
            pass
            # if random.random() < 0.5:
            # # only doing resample for spec_aug
            # #     item['time_mask_size'] = int(np.random.choice([50, 80, 120, 150], p=[0.25, 0.25, 0.25, 0.25]))
            # #     item['freq_mask_size'] = int(np.random.choice([10, 15, 20, 25], p=[0.25, 0.25, 0.25, 0.25]))
            #     item['time_mask_size'] = int(np.random.choice([30, 50, 70, 80], p=[0.25, 0.25, 0.25, 0.25]))
            #     item['freq_mask_size'] = int(np.random.choice([10, 15, 20, 25], p=[0.25, 0.25, 0.25, 0.25]))
        else:
            inc_p = float(np.random.choice([0.0, 0.1, 0.15, 0.2, 0.25], p=[0.2, 0.2, 0.2, 0.2, 0.2]))
            inc_snr = int(np.random.choice([2, 3, 4, 5, 6], p=[0.2, 0.2, 0.2, 0.2, 0.2]))
            # amt = int(amt)
            if random.random() < 0.5:
                item['p'] = float(round(max(0, item['p'] - inc_p), 3))
                # breakpoint()
                item['snr'] = int(max(0, item['snr'] - inc_snr))
            else:
                item['p'] = float(round(max(0, item['p'] + inc_p), 3))
                # breakpoint()
                item['snr'] = int(max(0, item['snr'] + inc_snr))

            if item['p'] < 0:
                item['p'] = 0
            elif item['p'] > 1:
                item['p'] = 1

            if item['snr'] < 0:
                item['snr'] = 0


if __name__ == '__main__':
    aug_config_old = {
        'reverb_p': 0.2,
        'music_p': 0.2,
        'noises_p': 0.2,
        'overlap_p': 0.2,
        'music_snrs': 5,
        'overlap_snrs': 10,
        'noises_snrs': 15,
    }
    new_config = convert2new(aug_config_old)
    print(new_config)
    old_config = convert2old(new_config)
    print(old_config)
    new_config = convert2new(old_config)
    explore(new_config)
    print(new_config)
