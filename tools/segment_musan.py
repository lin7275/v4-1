import shutil
import subprocess
import glob
import os


def segment_musan(musan_dir, save_dir, duration=5):
    if os.path.exists(save_dir):
        raise ValueError(f'{save_dir} exist')
    else:
        os.mkdir(save_dir)
    segment_wav(f'{musan_dir}/speech', f'{save_dir}/speech', duration)
    segment_wav(f'{musan_dir}/music', f'{save_dir}/music', duration)
    print('done segmentation')
    shutil.copytree(f'{musan_dir}/noise', f'{save_dir}/noise')
    # shutil.copytree(f'{musan_dir}/simulated_rirs', f'{save_dir}/simulated_rirs')



def segment_wav(read_dir, save_dir, duration):
    shutil.copytree(read_dir, save_dir, ignore=shutil.ignore_patterns('*.wav'))
    for file in glob.glob(f"{read_dir}/**/*.wav", recursive=True):
        save2 = file.replace(read_dir, save_dir)
        subprocess.check_call([
            'ffmpeg', '-i', file, '-f', 'segment',
            '-segment_time', str(duration),
            '-c', 'copy', save2.replace('.wav', '')+'%09d.wav'
        ], stdout=subprocess.DEVNULL)
        # subprocess.check_call([
        #     'ffmpeg', '-i', file, '-f', 'segment',
        #     '-segment_time', str(duration), '-c', 'copy', save2.replace('.wav', '')+'%09d.wav'
        # ])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--musan_dir', required=True)
    parser.add_argument('--save_dir', required=True)
    args = parser.parse_args()
    print(args)
    segment_musan(args.musan_dir, args.save_dir)