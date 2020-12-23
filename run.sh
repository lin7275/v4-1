#!/bin/bash -e

# It is important that all prepared data are under $datadir/ such as /home8a/mwmak/corpus

# Get run level
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <stage>"
    exit
fi
stage=$1

data_dir=/home8a/mwmak/corpus
eval_dir=eval_on_v19
path2vox1=/corpus/voxceleb1
path2vox2=/corpus/voxceleb2
path2voice19=/corpus/voices19c
path2musan=/corpus/musan
path2RIRS_NOISES=/corpus/noiseDB/RIRS_NOISES
PWD=`pwd`

#============================================================================================
# Stage 1: convert m4a in voxceleb2 to wav and concatenate wav in voxceleb1 and 2
#          Insert a wav/ directory to vox1_fixed/dev and vox1_fixed/test so that their
#          directory structure becomes similar to that of Voxceleb2
# This stage will create the following directories under $data_dir:
#    voice19/ vox1_concat/  vox1_fixed/  vox2_concat/  vox2_wav/
#============================================================================================
if [ $stage -eq 1 ]; then
    mkdir -p $data_dir/vox1_fixed
    mkdir -p $data_dir/voice19
    cp -r -p $path2vox1/dev $data_dir/vox1_fixed
    cd $data_dir/vox1_fixed/dev; mkdir -p wav; cd wav; mv ../id* .; cd $PWD
    cp -r -p $path2vox1/test $data_dir/vox1_fixed
    cd $data_dir/vox1_fixed/test; mkdir -p wav; cd wav; mv ../id* .; cd $PWD
    cp -r -p $path2voice19/v19-dev $data_dir/voice19
    cp -r -p $path2voice19/v19-eval $data_dir/voice19
    python3 tools/vox_wav_tools.py --func m4a2wav --read_dir $path2vox2 --save2 $data_dir/vox2_wav
    python3 tools/vox_wav_tools.py --func concat --read_dir $data_dir/vox1_fixed --save2 $data_dir/vox1_concat
    python3 tools/vox_wav_tools.py --func concat --read_dir $data_dir/vox2_wav --save2 $data_dir/vox2_concat
fi


#============================================================================================
# Stage 2: Prepare wav info, generate trials, and prepare augmentation data.
# This step creates the file "ids_info.tsv" under various datasets in $data_dir
#============================================================================================
if [ $stage -eq 2 ]; then

    # 2.1 prepare wav info
    python3 tools/prepare_data.py --dataset voice19-dev --dataset_dir $data_dir/voice19/v19-dev
    python3 tools/prepare_data.py --dataset voice19-eval --dataset_dir $data_dir/voice19/v19-eval
    python3 tools/prepare_data.py --dataset vox --dataset_dir $data_dir/vox1_fixed/test
    python3 tools/prepare_data.py --dataset vox --dataset_dir $data_dir/vox1_fixed/dev
    python3 tools/prepare_data.py --dataset vox --dataset_dir $data_dir/vox2_wav/dev
    python3 tools/prepare_data.py --dataset vox --dataset_dir $data_dir/vox2_wav/test
    python3 tools/prepare_data.py --dataset vox --dataset_dir $data_dir/vox1_concat/test
    python3 tools/prepare_data.py --dataset vox --dataset_dir $data_dir/vox1_concat/dev
    python3 tools/prepare_data.py --dataset vox --dataset_dir $data_dir/vox2_concat/dev
    python3 tools/prepare_data.py --dataset vox --dataset_dir $data_dir/vox2_concat/test

    # 2.2 generate trials
    python3 tools/generate_voice_19_trial.py --orgin_trial $data_dir/voice19/v19-eval/sid_eval_lists_and_keys/eval-trial-keys.lst \
                                          --new_trial $data_dir/voice19_eval.tsv --type eval
    python3 tools/generate_voice_19_trial.py --orgin_trial $data_dir/voice19/v19-dev/sid_dev_lists_and_keys/dev-trial-keys.lst \
                                          --new_trial $data_dir/voice19_dev.tsv --type dev

    # 2.3 prepare augmentation data
    # copy simulate rip in to musan dir
    python3 tools/segment_musan.py --musan_dir $path2musan --save_dir $data_dir/musan_segmented
    cp -r $path2RIRS_NOISES/simulated_rirs $data_dir/musan_segmented

fi

#============================================================================================
# Stage 3: Train PBA
# datadir should contain the following files:
# * datadir/musan_segmented
# * datadir/vox1_fixed/dev
# * datadir/vox2_wav/dev
# * datadir/vox2_wav/dev
# * datadir/voice19_dev_trial.tsv
# Here is a brief explanation of the pba_trainer.py:
#   The number of models (networks) in the population is equal to the number of pba_trainer.py processes,
#   which is also equal to the world_size. Each pba_trainer.py process maintains its own parallel
#   executions of _dist_train() and the accumulation of gradient will be handled by PyTorch's distributed
#   computing module. All parallel processes will be blocked waiting in the exploit() function
#   (via the eval_openset()).
# Note that it is meaningless to set --ws 1, because the exploit() function will always select
# the one and only one network if --ws is 1.
# After training, copy the file 'model_<epoch>.tar', 'model_config.yaml', and 'trans_config.yaml' to
# $eval_dir/models/xvector or $eval_dir/models/densenet
#============================================================================================
if [ $stage -eq 3 ]; then

    n_host=2
    n_gpu_per_host=1
    mkdir -p pba_new
    rm -rf pba_new/* # if you need to rerun it, you have to manually remove all file under pba_new
    mkdir -p logs
    
    # Example for running on a host with 2 GPUs. --model can be either densenet121_d or xvector. Because the PBAtrain.world_size
    # is hardcoded to 1, we need to run two pba_trainer.py processes to use 2 GPUs. "tfenv" is an Anaconda environment. Change it
    # to your environment name. In this example, pba_trainer.py is in $HOME/so/spkver/voxceleb/v4-1/pba directory. Change it
    # to fit your environment.
    if [ $n_host -eq 1 -a $n_gpu_per_host -eq 2 ]; then
	    ssh mwmak@enmcomp8 "bash -ic 'source /usr/local/anaconda3/bin/activate tfenv'; \
                       cd so/spkver/voxceleb/v4-1; export data_dir=/home8a/mwmak/corpus; \
                       export PYTHONPATH=$PWD; \
                       nohup /usr/local/anaconda3/envs/tfenv/bin/python3 pba/pba_trainer.py \
                       --data_dir $data_dir --rank 0 --ws 2 --gpu_id 0 --proj pba_new --model densenet121_1d \
                       &>logs/pba_rank0.log </dev/null" &

	    ssh mwmak@enmcomp8 "bash -ic 'source /usr/local/anaconda3/bin/activate tfenv'; \
                       cd so/spkver/voxceleb/v4-1; export data_dir=/home8a/mwmak/corpus; \
                       export PYTHONPATH=$PWD; \
                       nohup /usr/local/anaconda3/envs/tfenv/bin/python3 pba/pba_trainer.py \
                       --data_dir $data_dir --rank 1 --ws 2 --gpu_id 1 --proj pba_new --model densenet121_1d \
                       &>logs/pba_rank1.log </dev/null" &
    fi

    # Example for running on 2 hosts, each with 1 GPU. Run the following two commands on 2 terminals, each logging
    # into one host with 1 GPU. Note that --gpu_id should be 0 because there is only 1 GPU per host.
    if [ $n_host -eq 2 -a $n_gpu_per_host -eq 1 ]; then
	    ssh mwmak@enmcomp4 "bash -ic 'source /usr/local/anaconda3/bin/activate tfenv'; \
                       cd so/spkver/voxceleb/v4-1; export data_dir=/home8a/mwmak/corpus; \
                       export PYTHONPATH=$PWD; \
                       nohup /usr/local/anaconda3/envs/tfenv/bin/python3 pba/pba_trainer.py \
                       --data_dir $data_dir --rank 0 --ws 2 --gpu_id 0 --proj pba_new --model xvector \
                       &>logs/pba_rank0.log </dev/null" &

	    ssh mwmak@enmcomp13 "bash -ic 'source /usr/local/anaconda3/bin/activate tfenv'; \
                       cd so/spkver/voxceleb/v4-1; export data_dir=/home8a/mwmak/corpus; \
                       export PYTHONPATH=$PWD; \
                       nohup /usr/local/anaconda3/envs/tfenv/bin/python3 pba/pba_trainer.py \
                       --data_dir $data_dir --rank 1 --ws 2 --gpu_id 0 --proj pba_new --model xvector \
                       &>logs/pba_rank1.log </dev/null" &
    fi
    
    # Example for running on 3 hosts (A, B, and C), each with 2 GPU. Run the following 6 commands on 6 terminals.
    # The 1st and 2nd commands should be executed on Host A. The 3rd and 4th commands should be executed on
    # Host B. The 5th and 6th commands should be executed on Host C.
    if [ $n_host -eq 3 -a $n_gpu_per_host -eq 2 ]; then
	python3 pba/pba_trainer.py --data_dir $data_dir --rank 0 --ws 6 --gpu_id 0 --proj pba_new --model densenet121_1d 
	python3 pba/pba_trainer.py --data_dir $data_dir --rank 1 --ws 6 --gpu_id 1 --proj pba_new --model densenet121_1d
	python3 pba/pba_trainer.py --data_dir $data_dir --rank 2 --ws 6 --gpu_id 0 --proj pba_new --model densenet121_1d 
	python3 pba/pba_trainer.py --data_dir $data_dir --rank 3 --ws 6 --gpu_id 1 --proj pba_new --model densenet121_1d
	python3 pba/pba_trainer.py --data_dir $data_dir --rank 4 --ws 6 --gpu_id 0 --proj pba_new --model densenet121_1d 
	python3 pba/pba_trainer.py --data_dir $data_dir --rank 5 --ws 6 --gpu_id 1 --proj pba_new --model densenet121_1d
    fi
fi    

#============================================================================================
# Stage 4: Performance evaluation
# this step includes extraction and plda scoring #
# make sure you have the concat file from step 1 ready
# X-vector: EER on dev = 1.31% and EER on eval = 5.2%.
# Densenet: EER on eval = 4.44% 
# You're likely to get better results if you use larger population
#============================================================================================
if [ $stage -eq 4 ]; then
    #python3 run_plda.py --corpus $data_dir --model_file eval_on_v19/models/xvector/model_399.tar --model xvector --extract Y
    python3 run_plda.py --corpus $data_dir --model_file eval_on_v19/models/densenet/model_399.tar --model densenet121_1d --extract N
fi
