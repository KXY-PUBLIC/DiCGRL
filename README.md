# DiCGRL
Codes for paper "Disentangle-based Continual Graph Representation Learning".<br>
Xiaoyu Kou, Yankai Lin, Shaobo Liu, Peng Li, Jie Zhou, Yan Zhang.<br>
In EMNLP 2020
### [arXiv](https://arxiv.org/abs/2010.02565)

## Requirement

* pytorch: 1.6.0
* python: 3.6
* numpy: 1.18.5


## Getting Started

### Installation

Clone this repo.

```bash
git clone https://github.com/KXY-PUBLIC/DiCGRL.git
cd DiCGRL
```

### Dataset

- DiCGRL/data: FB15K-237 and WN18RR are two well-known KG benchmark datasets. 
We randomly split each benchmark dataset into five parts to simulate the real world scenarios, with each part having the ratio of 0.8:0.05:0.05:0.05:0.05 respectively.
- DiCGRL/NE/data: We conduct our experiments on three real-world information networks for node classification task: Cora, CiteSeer and PubMed. 
The nodes, edges and labels in these three citation datasets represent articles, citations and research areas respectively, and their nodes are provided with rich features.
Like KGE datasets, we split each dataset into four parts and the partition ratio is 0.7:0.1:0.1:0.1.


## Training Examples:

For example, training on FB15k-237 datasets:
```
CUDA_VISIBLE_DEVICES=0 nohup python -u multi_run.py --dataset=FB15k-237  --epochs=800 --model_name=TransE_2 --s_N=0 --N=5 --k_factors=8 --embedding_size=25 --top_n=4 --w1=0.3 &> log/multi_TransE_2_fb.out &
```
```
CUDA_VISIBLE_DEVICES=0 nohup python -u multi_run.py --dataset=FB15k-237  --epochs=800 --model_name=ConvKB_2 --s_N=0 --N=5 --k_factors=8 --embedding_size=25 --top_n=4 --w1=0.3 &> log/multi_ConvKB_2_fb.out &
```

Training on WN18RR datasets:

```
CUDA_VISIBLE_DEVICES=0 nohup python -u multi_run.py --dataset=WN18RR  --epochs=800 --model_name=TransE_2 --s_N=0 --N=5 --k_factors=4 --embedding_size=50 --top_n=2 --w1=0.3  &> log/multi_TransE_2_wn.out &
``` 
```
CUDA_VISIBLE_DEVICES=0 nohup python -u multi_run.py --dataset=WN18RR  --epochs=800 --model_name=ConvKB_2 --s_N=0 --N=5 --k_factors=4 --embedding_size=50 --top_n=2 --w1=0.3  &> log/multi_ConvKB_2_wn.out &
``` 

Training on information network datasets:

```
cd NE/
CUDA_VISIBLE_DEVICES=0 nohup python -u multi_run.py --model_name=SpGAT_2 --dataset=cora --k_factors=8 --top_n=4 &> log/multi_SpGAT_2_cora.out &
```


## Cite

Please cite our paper if you use this code in your own work:

```
@article{kou2020DiCGRL,
  title={Disentangle-based Continual Graph Representation Learning},
  author={Kou, Xiaoyu and Lin, Yankai and Liu, Shaobo and Li, Peng and Zhou, Jie and Zhang, Yan},
  journal={EMNLP},
  year={2020}
}
```

