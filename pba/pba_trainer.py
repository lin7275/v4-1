from random import shuffle

#import sys
#sys.path.append("/home2a/mwmak/so/spkver/voxceleb/v4-1")

import GPUtil
import torch.distributed as dist
import torch.optim as optim
import yaml
from torch.multiprocessing import Process
import torch.nn.functional as F
import json
from lib import get_wav_info_multiple_dir, OuterDataset, WavRandomSampleDataset, AverageMeter
from lib import h52dict, Score, WavExtractDset, extract_collate
from pba.new_explore import convert2old, convert2new, explore
from tools.choose_model import choose_model
import os
import random
import torch
import time
import warnings
from sklearn.decomposition import PCA
import pandas as pd
import h5py
import numpy as np

unicode = h5py.special_dtype(vlen=str)

pname_list_save = [
    "noises_p",
    "reverb_p",
    "music_p",
    "overlap_p",
    "noises_snrs",
    "music_snrs",
    "overlap_snrs",
    "spec_aug_time_mask_size",
    "spec_aug_freq_mask_size"
    # 'bandrop_p'
]


pname_list_read = [
    "noises_p",
    "reverb_p",
    "music_p",
    "overlap_p",
    "noises_snrs",
    "music_snrs",
    "overlap_snrs",
    # 'bandrop_p'
]



class PBATrainer:
    def __init__(
            self,
            start_epoch, process_rank, n_process,
            n_epoch,
            lr,
            batch_size,
            project_dir,
            model_config,
            min_utts_per_spk,
            wav_dirs,
            min_duration,
            gpu_ids=None,
            trans_config={},
            aug_config={},
            noise_dir=None,
            only_keep_two_model=False,
            checkpoint_warmrestart=False,
            cooling_epoch=None,
            optimizer_choice='adam',
            scheduler=None,
            scheduler_config=None,
            sample_per_epoch=None,
            sample_length_range=None,
            n_blocks_in_samples=1,
            checkpoint_interval=20,
            valid_enroll=None,
            valid_test=None,
            valid_target=None,
            valid_trial_list=None,
            world_size=None,
            weight_decay=0,
            checkpoint=None,
            score_paras=None,
    ):
        self.process_rank = process_rank
        self.n_process = n_process
    
        if not os.path.exists(f"{project_dir}"):
            os.mkdir(project_dir)
        if not os.path.exists(f"{project_dir}/tmp"):
            os.mkdir(project_dir + "/tmp")
        if not os.path.exists(f"{project_dir}/models"):
            os.mkdir(project_dir + "/models")
        if not os.path.exists(f"{project_dir}/loggers"):
            os.mkdir(project_dir + "/loggers")
        if not os.path.exists(f"{project_dir}/share_file"):
            os.mkdir(project_dir + "/share_file")
        if not os.path.exists(f"{project_dir}/h5"):
            os.mkdir(project_dir + "/h5")
        if not os.path.exists(f"{project_dir}/h5/first_layer"):
            os.mkdir(project_dir + "/h5/first_layer")
        if not os.path.exists(f"{project_dir}/h5/last_layer"):
            os.mkdir(project_dir + "/h5/last_layer")

        self.trans_config = trans_config
        self.aug_config = aug_config
        self.noise_dir = noise_dir
        self.train_egs_dir = None
        self.wav_dirs = wav_dirs
        # training paras
        if not cooling_epoch:
            self.cooling_epoch = n_epoch
        else:
            print('with cooling')
            self.cooling_epoch = cooling_epoch
        self.score_paras = score_paras
        self.checkpoint_warmrestart = checkpoint_warmrestart

        self.scheduler_choice = scheduler
        self.scheduler_config = scheduler_config

        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.optimizer_choice = optimizer_choice
        self.lr = lr
        self.weight_decay = weight_decay
        self.world_size = world_size
        self.only_keep_two_model = only_keep_two_model
        self.checkpoint_interval = checkpoint_interval

        # loader paras
        self.sample_length_range = sample_length_range
        self.min_utts_per_spk = min_utts_per_spk
        self.n_blocks_in_samples = n_blocks_in_samples
        self.sample_per_epoch = sample_per_epoch

        # files
        self.project_dir = project_dir
        self.logger_dir = f"{project_dir}/loggers"
        self.checkpoint = checkpoint
        self.save_checkpoint_to = f"{project_dir}/models/model"
        self.init_file = f"{project_dir}/share_file/{random.random()}"
        self.valid_enroll, self.valid_test = valid_enroll, valid_test
        self.valid_target = valid_target
        self.valid_trial_list = valid_trial_list
        self.min_duration = min_duration
        # model_config["n_classes"] = self.get_n_classes(wav_dir,
        #                                                min_utts_per_spk=min_utts_per_spk,
        #                                                min_duration=min_duration)
        # Todo write a loop to concat all dir and pass df to child process
        model_config["n_classes"] = get_wav_info_multiple_dir(wav_dirs, min_utts_per_spk=min_utts_per_spk,
                                                              min_duration=min_duration).spk_ids.nunique()


        self.model_config = model_config
        # self.model = model(**model_config)

        if checkpoint:
            print(f"loading from {checkpoint}")
        else:
            # breakpoint()
            with open(f"{project_dir}/models/model_config.yaml", "w") as f:
                yaml.dump(model_config, f, default_flow_style=False)

            with open(f"{project_dir}/models/trans_config.yaml", "w") as f:
                yaml.dump(trans_config, f, default_flow_style=False)

        # self.model_config = model_config
        self.model = choose_model(model_config)

        with open(f"{project_dir}/models/model.log", "w") as f:
            print(self.model, file=f)

        self.info = {}
        self.gpu_ids = gpu_ids

        self.meta_project_dir = os.path.dirname(self.project_dir)
        self.previous_rank = 99999
        self.start_epoch = start_epoch


    def general_train(self):
        if self.world_size == 1:
            self.single_train()
        else:
            self.dist_train()

    def dist_train(self):
        print('Running dist_train()')
        gpu_ids_avail = GPUtil.getAvailable(maxMemory=0.02, limit=8)
        shuffle(gpu_ids_avail)
        gpu_ids = gpu_ids_avail[: self.world_size]
        assert len(gpu_ids) == self.world_size, "not enough GPUs"
        processes = []
        for rank, gpu_id in enumerate(gpu_ids):
            p = Process(target=self._dist_train, args=(rank, gpu_id))
            p.start()
            print(f"process {rank} has started")
            processes.append(p)

        for p in processes:
            p.join()

    def init_model(self, gpu_id=0):
        torch.cuda.set_device(gpu_id)
        self.device = torch.device(torch.cuda.current_device())
        self.model = self.model.to(self.device)
        if self.checkpoint:
            self.load_checkpoint(self.checkpoint)
        else:
            self.epoch = 0
            self.model = self.model.cuda()
            self.init_optimizer()

    def _dist_train(self, rank, gpu_id):
        torch.cuda.set_device(gpu_id)
        self.device = torch.device(torch.cuda.current_device())

        if self.checkpoint:
            self.load_checkpoint(self.checkpoint)
        else:
            self.epoch = 0
            self.model = self.model.cuda()
            self.init_optimizer()

        #
        # self.model = self.model.to(self.device)

        self.gpu_id = gpu_id
        # self.init_model(gpu_id)
        self.init_dist(rank)
        self.batch_size = self.batch_size // self.world_size
        self.rank = rank
        t0 = time.time()
        # if self.train_file:
        #     self.train()
        # elif self.train_egs_dir:
        #     local_file_lst = np.array_split(self.file_lst, self.world_size)[rank].tolist()
        #     self.egs_train(local_file_lst)
        # else:
        #     raise NotImplementedError
        # if self.train_file:
        #     self.train()
        if self.train_egs_dir:
            local_file_lst = np.array_split(self.file_lst, self.world_size)[
                rank
            ].tolist()
            self.egs_train(local_file_lst)
        else:
            self.train()
        if rank == 0:
            self.save_model(f"{self.save_checkpoint_to}_final")
        print(f"train total time is {time.time() - t0}")

    def single_train(self):
        print('Running single_train()')
        if self.gpu_ids is None:
            self.gpu_ids = GPUtil.getAvailable(maxMemory=0.02, limit=8)
        gpu_ids = self.gpu_ids[0]
        self.init_model(gpu_ids)
        self.rank = 0
        self.train()
        self.save_model(f"{self.save_checkpoint_to}_final")

    def egs_train(self, file_list):
        outer = OuterDataset(
            file_list,
            logger=f"{self.logger_dir}/data_{self.rank}.log",
            inner_batch_size=self.batch_size // self.world_size,
        )
        warnings.warn("outer_loader is not shuffled")
        outer_loader = torch.utils.data.DataLoader(
            outer, batch_size=1, shuffle=False, collate_fn=unpack_list, num_workers=1
        )
        for epoch in range(self.epoch, self.epoch + self.n_epoch):
            for step, inner_load in enumerate(outer_loader):
                self.epoch = epoch
                self.info["tot_loss"] = AverageMeter()
                self.info["acc"] = AverageMeter()
                self.end = time.time()
                self.model.train()
                self._train(inner_load)
                if self.rank == 0:
                    print(f"step {step+1}")
                    if bool(step % self.checkpoint_interval == 0) & bool(
                        self.valid_trial_list
                    ):
                        self.eval_openset()
                        self.save_model(
                            f"{self.save_checkpoint_to}_{self.epoch}_{step}.tar"
                        )

    def train(self):
        # self.fbank = Fbank(40).cuda()
        # self.spec_aug = SpectrumAug(max_time_mask_len=self.aug_config["spec_aug_time_mask_size"],
        #                             max_freq_mask_len=self.aug_config["spec_aug_freq_mask_size"])
        if hasattr(self.model, 'frame_layers'):
            self.model.frame_layers[1].freq_masker.mask_param = self.aug_config["spec_aug_freq_mask_size"]
            self.model.frame_layers[1].time_masker.time_param = self.aug_config["spec_aug_time_mask_size"]
        elif hasattr(self.model, 'trans'):
            self.model.trans[1].freq_masker.mask_param = self.aug_config["spec_aug_freq_mask_size"]
            self.model.trans[1].time_masker.time_param = self.aug_config["spec_aug_time_mask_size"]
        else:
            print('no using spec aug')
            # raise ValueError
        dset = WavRandomSampleDataset(
            self.wav_dirs,
            # aug_config=self.aug_config,
            aug_config={k: self.aug_config[k] for k in pname_list_read},
            noise_dir=self.noise_dir,
            sample_per_epoch=self.sample_per_epoch // self.world_size,
            balance_class=True,
            min_duration=self.min_duration,
            min_utts_per_spk=self.min_utts_per_spk,
            sample_length_range=self.sample_length_range,
            trans_config=self.trans_config,
        )
        self.loader = torch.utils.data.DataLoader(
            dset, batch_size=self.batch_size, shuffle=False, num_workers=16
        )
        for epoch in range(self.epoch, self.epoch + self.n_epoch):
            self.loader.dataset.sample()

            self.epoch = epoch
            self.info["tot_loss"] = AverageMeter()
            self.info["acc"] = AverageMeter()
            self.end = time.time()
            self.model.train()
            self._train(self.loader)

            self.info["time"] = time.time() - self.end
            self.end = time.time()
            self.scheduler.step()

            message = (
                f"epoch {self.epoch} "
                f"time {self.info['time']:.0f} "
                f"loss {self.info['tot_loss'].avg:.3f} "
                f"acc {self.info['acc'].avg:.3f}\n"
            )
            with open(f"{self.logger_dir}/training.log", "a") as f:
                f.write(message)
            print(message, end="")

            if self.epoch >= self.start_epoch:
                if ((self.epoch + 1) % self.checkpoint_interval) == 0 and bool(
                    self.valid_trial_list
                ):
                    self.save_model(f"{self.save_checkpoint_to}_{self.epoch}.tar")
                    print("save model")
                    self.eval_openset()

                # if self.epoch == 0 & bool(self.valid_trial_list):
                #     self.eval_openset()

    def _train(self, loader):
        for step, (mfcc, spk_ids, _) in enumerate(loader):
            self.optimizer.zero_grad()

            mfcc, spk_ids = mfcc.cuda(), spk_ids.cuda()
            # mfcc = self.spec_aug(self.fbank(mfcc))
            #Todo spec aug
            logit, logit_nomargin = self.model(mfcc, spk_ids)
            loss = F.cross_entropy(logit, spk_ids)

            loss.backward()
            self.optimizer.step()

            acc = (
                logit_nomargin.max(-1)[1].eq(spk_ids).sum().item() / len(spk_ids) * 100
            )
            self.info["tot_loss"].update(loss.item(), spk_ids.shape[0])
            self.info["acc"].update(acc, spk_ids.shape[0])

    def eval_openset(self):
        self.sequential_extract(self.valid_test, f"{self.project_dir}/tmp/test.h5")
        if self.valid_enroll:
            self.sequential_extract(
                self.valid_enroll, f"{self.project_dir}/tmp/enroll.h5"
            )
            enroll_embedding = f"{self.project_dir}/tmp/enroll.h5"
        else:
            enroll_embedding = f"{self.project_dir}/tmp/test.h5"

        if self.valid_target:
            self.sequential_extract(
                self.valid_target, f"{self.project_dir}/tmp/target.h5"
            )
            data_target = h52dict(f"{self.project_dir}/tmp/target.h5")
            transform_lst = [PCA(whiten=True)]
            for transform in transform_lst:
                transform.fit_transform(data_target["X"])
        else:
            transform_lst = None

        if self.score_paras is None:
            self.score_paras = {}
        score = Score(
            comp_minDCF=False,
            enroll=enroll_embedding,
            test=f"{self.project_dir}/tmp/test.h5",
            ndx_file=self.valid_trial_list,
            transforms=transform_lst,
            **self.score_paras,
        )
        eer = score.batch_cosine_score()
        eer = round(eer, 2)
        with open(f"{self.logger_dir}/validation.log", "a") as f:
            f.write(f"{self.epoch} EER is {eer}\n")

        # record all info
        info = {
            'previous_rank': self.previous_rank,
            "eer": eer,
            "noises_p": self.aug_config["noises_p"],
            "reverb_p": self.aug_config["reverb_p"],
            "music_p": self.aug_config["music_p"],
            "overlap_p": self.aug_config["overlap_p"],
            "spec_aug_time_mask_size": int(self.aug_config["spec_aug_time_mask_size"]),
            "spec_aug_freq_mask_size": int(self.aug_config["spec_aug_freq_mask_size"]),
            # "noises_snrs": list(self.aug_config["noises_snrs"]),
            # "music_snrs": list(self.aug_config["music_snrs"]),
            # "overlap_snrs": list(self.aug_config["overlap_snrs"]),
            "noises_snrs": self.aug_config["noises_snrs"],
            "music_snrs": self.aug_config["music_snrs"],
            "overlap_snrs": self.aug_config["overlap_snrs"],
            "process_rank": self.process_rank,
            "epoch": self.epoch,
        }
        print(info)
        with open(f"{self.logger_dir}/info_{self.epoch}.json", "w") as f:
            json.dump(info, f)
        # try:
        #     with open(f"{self.logger_dir}/info_{self.epoch}.json", "w") as f:
        #         json.dump(info, f)
        # except:
        #     breakpoint()
        self.exploit()

    def explore(self):
        new_config = convert2new(self.aug_config)
        explore(new_config)
        if hasattr(self.model, 'frame_layers'):
            self.model.frame_layers[1].freq_masker.mask_param = new_config["spec_aug"]['freq_mask_size']
            self.model.frame_layers[1].time_masker.time_param = new_config["spec_aug"]['time_mask_size']
        elif hasattr(self.model, 'trans'):
            self.model.trans[1].freq_masker.mask_param = new_config["spec_aug"]['freq_mask_size']
            self.model.trans[1].time_masker.time_param = new_config["spec_aug"]['time_mask_size']
        else:
            raise ValueError
        # self.spec_aug = SpectrumAug(max_time_mask_len=new_config["spec_aug"]['time_mask_size'],
        #                             max_freq_mask_len=new_config["spec_aug"]['freq_mask_size'])
        self.aug_config = convert2old(new_config)

        dset = WavRandomSampleDataset(
            self.wav_dirs,
            aug_config={k: self.aug_config[k] for k in pname_list_read},
            noise_dir=self.noise_dir,
            sample_per_epoch=self.sample_per_epoch // self.world_size,
            balance_class=True,
            min_duration=self.min_duration,
            min_utts_per_spk=self.min_utts_per_spk,
            sample_length_range=self.sample_length_range,
            trans_config=self.trans_config,
        )
        self.loader = torch.utils.data.DataLoader(
            dset, batch_size=self.batch_size, shuffle=False, num_workers=16
        )

        aug_info = (
            f"{self.epoch}\nnoises_p is {self.aug_config['noises_p']}\n"
            f"reverb_p is {self.aug_config['reverb_p']}\n"
            f"music_p is {self.aug_config['music_p']}\n"
            f"overlap_p is {self.aug_config['overlap_p']}\n"
        )
        print(aug_info)
        # self.print_augment_info()

    def exploit(self):
        infos = []
        for i in range(self.n_process):
            print_stop = False
            while not os.path.exists(
                f"{self.meta_project_dir}/{str(i)}/loggers/info_{self.epoch}.json"
            ):
                if not print_stop:
                    print(f"waiting for {i}")
                print_stop = True
                time.sleep(3)
            # breakpoint()
            with open(
                f"{self.meta_project_dir}/{str(i)}/loggers/info_{self.epoch}.json"
            ) as f:
                info = json.load(f)
                # info["noises_snrs"] = (self.aug_config["noises_snrs"],)
                # info["music_snrs"] = (self.aug_config["noises_snrs"],)
                # info["overlap_snrs"] = (self.aug_config["noises_snrs"],)
                infos.append(info)
        df = pd.DataFrame(infos).sort_values('eer', ascending=True, ignore_index=True)
        df["perf_rank"] = df.index
        print("done comparison")
        if os.path.exists(f'{self.logger_dir}/all_workers_info.tsv'):
            df_old = pd.read_csv(f'{self.logger_dir}/all_workers_info.tsv', sep='\t')
            df = df.append(df_old)
        df.to_csv(f'{self.logger_dir}/all_workers_info.tsv', sep='\t', index=None)
        # df.to_csv(f'{self.logger_dir}/all_workers_info_{self.epoch}.tsv', sep='\t', index=None)

        # if df.set_index('process_rank').loc[str(self.process_rank)].perf_rank > ((self.n_process // 2) - 1):
        se = df.set_index("process_rank").loc[self.process_rank]
        print(f"process {self.process_rank} current perf rank is {se.perf_rank}")
        if se.perf_rank > ((self.n_process // 2) - 1):
            high_rank_se = (
                df.sort_values("perf_rank")
                .iloc[:self.n_process // 2]
                .sample()
            ).squeeze()
            self.aug_config = high_rank_se[pname_list_save].to_dict()
            # print(self.aug_config)
            self.explore()

            print(f"load model from {str(high_rank_se.process_rank)}")
            self.previous_rank = high_rank_se.process_rank
            self.load_checkpoint(
                f"{self.meta_project_dir}/{str(high_rank_se.process_rank)}/models/model_{self.epoch}.tar"
            )

    def sequential_extract(self, wav_dir, save_xvec_to):
        print(f"reading wav from {wav_dir}")
        dset = WavExtractDset(wav_dir, trans_config=self.trans_config, )
        loader = torch.utils.data.DataLoader(
                dset, batch_size=1, shuffle=False, collate_fn=extract_collate, num_workers=16
            )
        with h5py.File(save_xvec_to, "w") as fw:
            fw["X"], fw["spk_ids"], fw["spk_path"] = self._extraction(loader)

    def _extraction(self, loader):
        self.model.eval()
        X, spk_ids, utt_ids = [], [], []
        with torch.no_grad():
            for batch_idx, (mfcc, spk_id, utt_id) in enumerate(loader):
                # if self.model_choice.endswith('2d'):
                #     mfcc = mfcc[:, None, ...]
                mfcc = mfcc.to(self.device)
                if self.world_size == 1:
                    x = self.model.extract(mfcc)
                else:
                    x = self.model.module.extract(mfcc)
                spk_ids.append(spk_id)
                utt_ids.append(utt_id)
                X.append(x)

            X = torch.cat(X).to("cpu").numpy()
            spk_ids = np.stack(spk_ids).astype(unicode)
            utt_ids = np.stack(utt_ids).astype(unicode)
        return X, spk_ids, utt_ids

    def init_dist(self, rank):
        print("Initialize Process Group...")
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{self.init_file}",
            rank=rank,
            world_size=self.world_size,
        )
        # self.model = self.model.cuda()
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[torch.cuda.current_device()],
            output_device=torch.cuda.current_device(),
            # find_unused_parameters=True,
        )

    # def load_checkpoint(self, checkpoint):
    #     checkpoint = torch.load(
    #         checkpoint,
    #         map_location=torch.device('cpu')
    #     )
    #     self.model.load_state_dict(checkpoint["model_state_dict"])
    #     # self.model = self.model.cuda()
    #     self.init_optimizer()
    #
    #     self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    #     self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    #     self.epoch = checkpoint["epoch"] + 1

    # def load_checkpoint(self, checkpoint):
    #     checkpoint = torch.load(
    #         checkpoint, map_location=torch.device(torch.cuda.current_device())
    #     )
    #     self.model.load_state_dict(checkpoint["model_state_dict"])
    #     self.model = self.model.cuda()
    #     self.init_optimizer()
    #
    #     self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    #     self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    #     self.epoch = checkpoint["epoch"] + 1

    def load_checkpoint(self, checkpoint):
        # checkpoint = torch.load(
        #     checkpoint, map_location=torch.device(torch.cuda.current_device())
        # )
        checkpoint = torch.load(checkpoint, map_location=torch.device("cpu"))
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model = self.model.cuda()
        self.init_optimizer()

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"] + 1

    def save_model(self, save_checkpoint_to):
        # save_checkpoint_to = f"{save_checkpoint_to}.tar"
        if not os.path.exists(os.path.dirname(save_checkpoint_to)):
            os.makedirs(os.path.dirname(save_checkpoint_to))
        torch.save(
            {
                "model_state_dict": self.model.state_dict()
                if self.world_size == 1
                else self.model.module.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "epoch": self.epoch,
                "logger": self.logger_dir,
            },
            save_checkpoint_to,
        )

    @staticmethod
    def get_n_classes(file, min_utts_per_spk, min_frames_per_utt):
        with h5py.File(file, "r") as f:
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
        return df.spk_ids.nunique()

    def init_optimizer(self):
        if self.optimizer_choice == "adam":
            print("using adam")
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_choice == "sgd":
            print("using sgd")
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )
        elif self.optimizer_choice == "adamW":
            print("using adamW")
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
            )

        if self.scheduler_choice == "MultiStepLR":
            print("using MultiStepLR")
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, **self.scheduler_config,
            )
        elif self.scheduler_choice == "CosineAnnealingLR":
            print("CosineAnnealingLR")
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, **self.scheduler_config,
            )
        elif self.scheduler_choice == "ReduceLROnPlateau":
            print("ReduceLROnPlateau")
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, **self.scheduler_config,
            )
        elif self.scheduler_choice == "CosineAnnealingWarmRestarts":
            print("CosineAnnealingWarmRestarts")
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, **self.scheduler_config,
            )
        else:
            print("No scheduler")
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=50, gamma=1
            )


def unpack_list(x):
    return x[0]



def pba(rank, world_size, home, gpu_ids, proj_name, model, interval=40, start_epoch=79):
    # wav_dir = f"{home}/wwlin/corpus"
    wav_dir = home
    PROJECT_NAME = proj_name
    if model == 'xvector':
        model_config = {"model": 'Xvector'}
    elif model == 'densenet121_1d':
        model_config = {"model": 'densenet121_1d', 'growth_rate': 90,
                        "fbank_config": {"n_mels": 80},
                        "specaug_config": {"max_freq_mask_len": 5, "max_time_mask_len": 20}}
    elif model == 'resnet101':
        model_config = {"model": 'resnet101',
                        "fbank_config": {"n_mels": 40}}
    else:
        raise NotImplementedError
    train_para = {
        "project_dir": f"{PROJECT_NAME}/{str(rank)}",
        "noise_dir": f"{wav_dir}/musan_segmented",
        "wav_dirs": [
            f"{wav_dir}/vox1_fixed/dev",
            f"{wav_dir}/vox2_wav/dev",
            f"{wav_dir}/vox2_wav/test",
        ],
        "valid_test": f"{wav_dir}/voice19/v19-dev",
        "valid_trial_list": f"{wav_dir}/voice19_dev.tsv",
        "min_duration": 4.1,
        # "min_duration": 5.1,
        "sample_length_range": [64000, 64001],
        # "sample_length_range": [80000, 80001],
        "sample_per_epoch": int(12e4),
        # "sample_per_epoch": int(12e2),
        # "min_utts_per_spk": 7,
        "min_utts_per_spk": 3,
        "n_epoch": 400,
        # "world_size": 2,
        "world_size": 1,            # So that PBAtrainer.world_size = 1, which only use 1 GPU per pba_trainer.py process
                                    # Setting world_size=1 will cause PBAtrainer.single_train() to run.
        #######################
        #####################
        "batch_size": 64 if model == 'xvector' else 55,
        "lr": 0.01,
        "optimizer_choice": "sgd",
        "weight_decay": 1e-3,
        "scheduler": "CosineAnnealingWarmRestarts",
        "scheduler_config": {"T_0": 200, "T_mult": 1, },
        "model_config": model_config,
        #####################
        "aug_config": {
            "noises_p": random.uniform(0.1, 0.6),
            "reverb_p": random.uniform(0.1, 0.6),
            "music_p": random.uniform(0.1, 0.6),
            "overlap_p": random.uniform(0.1, 0.6),
            "noises_snrs": random.randint(0, 10),
            "music_snrs": random.randint(0, 10),
            "overlap_snrs": random.randint(0, 10),
            "spec_aug_time_mask_size": 5,
            "spec_aug_freq_mask_size": 10,
        },
        "process_rank": rank,
        "n_process": world_size,                # Set the number of processes for the PBA
        "checkpoint_interval": interval,
        "start_epoch": start_epoch,
        "gpu_ids": gpu_ids,
    }

    if not os.path.exists(train_para["project_dir"]):
        os.mkdir(train_para["project_dir"])
    else:
        # raise ValueError("project exist")
        print('warning project already exists')

    with open(train_para["project_dir"] + "/train_config.yaml", "w") as f:
        yaml.dump(train_para, f, default_flow_style=False)

    PBATrainer(**train_para).general_train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--proj', type=str, required=True)
    parser.add_argument('--rank', type=str, required=True)
    parser.add_argument('--ws', type=int, required=True)
    parser.add_argument('--gpu_id', type=int, required=True)
    parser.add_argument('--model', choices=['xvector', 'densenet121_1d', "resnet101"], required=True)
    args = parser.parse_args()
    print(args)

    pba(
        home=args.data_dir,
        rank=args.rank,
        world_size=args.ws,         # Set the number of processes for the PBA
        gpu_ids=[args.gpu_id],
        proj_name=args.proj,
        model=args.model,
        #interval=1, start_epoch=1
    )

    # /home8a/wwlin/corpus
    # /home10b/wwlin/corpus
##############
    # pba(
    #     home='/home12a',
    #     rank='0',
    #     world_size=6,
    #     p=0.1,
    #     interval=40,
    #     gpu_ids = [0],
    #     start_epoch=79,
    #     snr = 1,
    # )

    # pba(
    #     home='/home12a',
    #     rank='1',
    #     world_size=6,
    #     p=0.2,
    #     interval=40,
    #     gpu_ids = [1],
    #     start_epoch=79,
    # snr = 4,
    # )

    # pba(
    #     home='/home10b',
    #     rank='2',
    #     world_size=6,
    #     p=0.3,
    #     interval=40,
    #     start_epoch=79,
    #     gpu_ids=[0],
    # snr = 8,
    # )


    # pba(
    #     home='/home10b',
    #     rank='3',
    #     world_size=6,
    #     p=0.4,
    #     interval=40,
    #     start_epoch=79,
    #     gpu_ids=[1],
    # snr = 12,
    # )

    # pba(
    #     home='/home8a',
    #     rank='4',
    #     world_size=6,
    #     p=0.5,
    #     interval=40,
    #     start_epoch=79,
    #      gpu_ids = [1],
    #     snr=14,
    # )
