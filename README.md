Neural Collective Entity Linking Based on Recurrent Random Walk Network Learning 
====

Models and results can be found at our IJCAI 2019 paper [Neural Collective Entity Linking Based on Recurrent Random Walk Network Learning]. It achieves the state-of-the-art result on EL task.

Details will be updated soon.

Requirement:
======
	Python: 3.6.3
	PyTorch: 0.3.1 

Input format:
======
We transform the original data into pkl format, if you want the tranform code, please concat me.

pkl data location:
======
link:https://pan.baidu.com/s/17tHxyLAMqdOTozmnQsMQ6w 

Fetch Code:kwvn 

How to run the code?
====
Local:

	python pre_net_xmg.py --cuda_device 0 --nohup regular --epoch 25 --weight_decay 1.28e-5 --LR 0.001 --batch 500 --filter_num 64 --filter_window 3  --local_model_loc model_loc/local/local_regular_new1 --embedding_finetune 1

Global:

	python net_global_train.py --cuda_device 0 --nohup 0.5_0.1_3 --weight_decay 1.28e-5 --LR 0.0005 --local_model_loc model_loc/local/local_regular_new1.938.pkl --global_model_loc model_loc/global/global_model --random_k 3 --lamda 0.5 --flag 4:3:1 --gama 0.1 --batch 200 --epoch 25


Cite: 
========
Please cite our IJCAI 2019 paper:

    @article{xue2018,  
     title={Neural Collective Entity Linking Based on Recurrent Random Walk Network Learning },  
     author={Mengge Xue, Weiming Cai, Jinsong Su and Linfeng Song, Yubin Ge, Yubao Liu, Bin Wang},  
     booktitle={The Program Committee of the 28th International Joint Conference on Artificial Intelligence (IJCAI-19)},
     year={2019}  
    }