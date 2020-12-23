from lib import *
import h5py
import numpy as np
import pandas as pd
import time

from torch.utils.data import DataLoader
from lib import unicode, extract_collate,PartialExtractDataset
import GPUtil
import os
import yaml
from torch.multiprocessing import Process
from sklearn.decomposition import PCA
import torch
import shutil
import random
import string


from tools.choose_model import choose_model


class Extractor:
    def __init__(
        self,
        model_file,
        demean_mfcc=False,
        embedding_layer=None,
        model=None,
        allow_overwrite=False,
        trans_config={},
        distort_config=None,
        gpu_id=None,
        using_cpu=False,
        max_len=1000000000000,
        num_workers=1,
    ):
        if not os.path.exists(f"{os.path.dirname(model_file)}/model_config.yaml"):
            raise ValueError(
                f"can not find model cfg in {os.path.dirname(model_file)}/model_config.yaml"
            )
        with open(f"{os.path.dirname(model_file)}/model_config.yaml", "r") as f:
            model_config = yaml.safe_load(f)
            if "embedding_layer" in model_config:
                del model_config["embedding_layer"]
            if model is not None:
                model_config['model'] = model

        self.model = choose_model(model_config, embedding_layer=embedding_layer)
        self.model_file = model_file
        self.mode = "a" if not allow_overwrite else "w"
        self.allow_overwrite = allow_overwrite
        checkpoint = torch.load(self.model_file, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if not os.path.exists(f"{os.path.dirname(model_file)}/trans_config.yaml"):
            print('no trans cfg found use cli input cfg')
            self.trans_config = trans_config
        else:
            with open(f"{os.path.dirname(model_file)}/trans_config.yaml", "r") as f:
                self.trans_config = yaml.safe_load(f)

        self.gpu_id = gpu_id
        self.distort = None if distort_config is None else config_distortions(**distort_config)
        # self.trans = config_trans(**trans_config) if trans_config else None
        self.using_cpu = using_cpu
        self.max_len = max_len
        self.demean_mfcc = demean_mfcc
        self.num_workers = num_workers

    def eval(
        self,
        test_file,
        trial,
        tmp_dir=None,
        enroll_file=None,
        target_file=None,
        cohort_file=None,
        remove_tmp_folder=True,
        world_size=1,
        score_para={},
    ):
        if not tmp_dir:
            tmp_dir = 'tmp' + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
            os.mkdir(tmp_dir)
        elif not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        else:
            raise ValueError(f"{tmp_dir} exist and will be removed may be wrong")

        self.sequential_extract(
            test_file, f"{tmp_dir}/test.h5"
        )
        if enroll_file:
            self.sequential_extract(
                enroll_file, f"{tmp_dir}/enroll.h5"
            )
            enroll_embedding = f"{tmp_dir}/enroll.h5"
        else:
            enroll_embedding = f"{tmp_dir}/test.h5"
        time.sleep(10)
        if cohort_file:
            # self.parallel_extract(cohort_file, f"{tmp_dir}", world_size)
            self.sequential_extract(cohort_file, f"{tmp_dir}/cohort")

            # data = h52dict(glob.glob(f"{tmp_dir}/parallel_job/xvectors_*.h5"))
            data = h52dict(f"{tmp_dir}/cohort")
            data = averge_xvectors(data, "spk_ids")
            dict2h5(data, f"{tmp_dir}/cohort.h5")
            cohort_file = f"{tmp_dir}/cohort.h5"
        else:
            cohort_file = None

        if target_file:
            self.sequential_extract(
                target_file, f"{tmp_dir}/target.h5"
            )
            data_target = h52dict(f"{tmp_dir}/target.h5")
            transform_lst = [PCA(whiten=True)]
            for transform in transform_lst:
                transform.fit_transform(data_target["X"])
        else:
            transform_lst = None

        score = Score(
            enroll=enroll_embedding,
            test=f"{tmp_dir}/test.h5",
            ndx_file=trial,
            cohort=cohort_file,
            transforms=transform_lst,
            # save_scores_to=f"{tmp_dir}/scores.tsv",
            **score_para,
        )
        # breakpoint()
        eer = score.batch_cosine_score()
        # breakpoint()
        print(f"EER is {eer}")
        if remove_tmp_folder:
            shutil.rmtree(tmp_dir)
        return score.ndx

    def sequential_extract(self, extract_file, save_embedding_to):
        if not self.using_cpu:
            if self.gpu_id is not None:
                gpu_id = self.gpu_id
            else:
                gpu_id = GPUtil.getAvailable(
                    maxMemory=0.02, order="last", limit=1,
                )[0]
            device = torch.device(gpu_id)
        else:
            device = torch.device('cpu')

        p = Process(
            target=self.extract_single, args=(extract_file, save_embedding_to, device),
        )
        p.start()
        p.join()

    def parallel_extract(self, extract_file, save_to_dir, world_size):
        if not os.path.exists(f"{save_to_dir}/parallel_job"):
            os.mkdir(f"{save_to_dir}/parallel_job")
        else:
            raise ValueError(f"dir {save_to_dir}/parallel_job exists")
        with h5py.File(extract_file, "r") as f:
            df = pd.DataFrame(
                {
                    "starts": f["positions"][:, 0],
                    "ends": f["positions"][:, 1],
                    "spk_ids": f["spk_ids"][:],
                    "utt_ids": f["utt_ids"][:],
                }
            )

        # np.random.shuffle(df.values)
        dfs = np.array_split(df, world_size)

        gpu_ids = GPUtil.getAvailable(maxMemory=0.02, limit=world_size)
        devices = [torch.device(gpu_id) for gpu_id in gpu_ids]

        processes = []
        for rank, device in enumerate(devices):
            p = Process(
                target=self._parallel_extract,
                args=(
                    extract_file,
                    f"{save_to_dir}/parallel_job/xvectors_{rank}.h5",
                    dfs[rank],
                    device,
                ),
            )
            p.start()
            print(f"process {rank} has started")
            processes.append(p)

        for p in processes:
            p.join()

    def _parallel_extract(self, extract_file, save_to, df, device):
        self.model = self.model.to(device)
        with h5py.File(extract_file, "r") as f:
            dset = PartialExtractDataset(df=df, mfcc=f["mfcc"])
            loader = torch.utils.data.DataLoader(
                dset,
                batch_size=1,
                shuffle=False,
                collate_fn=extract_collate,
                num_workers=1,
            )
            with h5py.File(save_to, self.mode) as fw:
                fw["X"], fw["spk_ids"], fw["spk_path"], fw["n_frames"] = self._extract(
                    loader, device
                )
                print(f"saving xvector to {save_to}")

    def extract(self, extract_file, save_embedding_to, world_size=1):
        if world_size > 1:
            self.parallel_extract(extract_file, save_embedding_to, world_size)
        else:
            if save_embedding_to.endswith('.h5'):
                self.sequential_extract(extract_file, save_embedding_to)
            else:
                save2 = save_embedding_to + '/' + os.path.basename(extract_file)
                self.sequential_extract(extract_file, save2)

    def extract_single(self, wav_dir, save_xvec_to, device):
        self.model = self.model.to(device)
        print(f"reading wav from {wav_dir}")
        dset = WavExtractDset(wav_dir, trans_config=self.trans_config)
        loader = torch.utils.data.DataLoader(
                dset, batch_size=1, shuffle=False, collate_fn=extract_collate, num_workers=16,
            )
        with h5py.File(save_xvec_to, "w") as fw:
            fw["X"], fw["spk_ids"], fw["spk_path"], fw["n_frames"] = self._extract(loader, device)

    def _extract(self, loader, device):
        self.model.eval()
        X, spk_ids, utt_ids, n_frames = [], [], [], []
        with torch.no_grad():
            for batch_idx, (mfcc, spk_id, utt_id) in enumerate(loader):
                if batch_idx % 5000 == 0:
                    print(f"{batch_idx}/{len(loader)}")
                #
                # mfcc = mfcc / torch.max(torch.abs(mfcc))
                # mfcc = mfcc * (torch.rand(1)+0.5)

                mfcc = mfcc.to(device)
                x = self.model.extract(mfcc)

                X.append(x)
                spk_ids.append(spk_id)
                utt_ids.append(utt_id)
                n_frames.append(mfcc.shape[-1])

            X = torch.cat(X).to("cpu").numpy()
            spk_ids = np.array(spk_ids)
            utt_ids = np.array(utt_ids)
            n_frames = np.array(n_frames)
        return X, spk_ids.astype(unicode), utt_ids.astype(unicode), n_frames


