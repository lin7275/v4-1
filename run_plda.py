from tools.extract import Extractor
from lib import train_and_score
from lib import h52dict
import argparse
from pathlib import Path
import os

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', required=True)
parser.add_argument('--model_file', required=True)
parser.add_argument('--model', choices=['densenet121_1d', 'xvector'], required=True)
parser.add_argument('--extract', default='N')
args = parser.parse_args()

if args.model == 'xvector':
    args.model = 'Xvector' 

proj = str(Path(args.model_file).parents[2])
model_name = str(Path(args.model_file)).split('/')[2]
h5dir = f'{proj}/h5/{model_name}'

extractor = Extractor(args.model_file,
                      model=args.model,
                      allow_overwrite=True,
                      gpu_id=0)
if not os.path.exists(h5dir):
    print(f'Creating folder {h5dir}')
    os.makedirs(h5dir)

if args.extract is 'Y':
    extractor.extract(f'{args.corpus}/voice19/v19-dev', f'{h5dir}/v19-dev.h5')
    extractor.extract(f'{args.corpus}/voice19/v19-eval', f'{h5dir}/v19-eval.h5')
    extractor.extract(f'{args.corpus}/vox1_concat/dev', f'{h5dir}/vox1.h5')
    extractor.extract(f'{args.corpus}/vox2_concat/dev', f'{h5dir}/vox2.h5')


cohort = h52dict([f'{h5dir}/v19-dev.h5'])
paras = {
    'enroll': f'{h5dir}/v19-eval.h5',
    'test': f'{h5dir}/v19-eval.h5',
    'train': [f'{h5dir}/vox1.h5', f'{h5dir}/vox2.h5'],
    'cohort': cohort,
    'lda_dim': 120 if (args.model == 'Xvector' or args.model == 'densenet121_1d') else 250,
    'plda_dim': 120 if (args.model == 'Xvector' or args.model == 'densenet121_1d') else 250,
    'ndx': f'{args.corpus}/voice19_eval.tsv'
}
df = train_and_score(**paras)


