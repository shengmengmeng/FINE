# Revisiting Learning with Noisy Labels: Active Forgetting and Noise Suppression
**Abstract:** Learning with noisy labels (LNL) has received growing attention, with most prior work following the paradigm of clean-sample reliance (e.g., sample selection). However, this reliance also imposes intrinsic limitations, as overfitting to even a few noisy samples is inevitable, creating a major bottleneck for further improvement. This limitation motivates us to go beyond mere clean-sample reliance and explore how to actively forget corrupted knowledge already internalized by models while suppressing further noise assimilation. To this end, we propose FINE, a fundamentally novel perspective for LNL that unifies active ForgettIng via machine unlearning (MU) and Noise supprEssion via negative learning (NL) within a cohesive framework. Specifically, we first reveal two key stages of noise fitting: early-stage generalized learning and later-stage noise overfitting. To actively forget early-stage noise accumulation, we introduce an MU-based module that employs a negative cross-entropy loss to erase corrupted knowledge, while an NL-based module leveraging complementary labels suppresses later-stage overfitting and mitigates reliance on noisy supervision. These modules act synergistically as plug-and-play regularizers, seamlessly integrating into existing baselines. Finally, extensive experiments on both synthetic and real-world noisy benchmarks demonstrate that our FINE consistently boosts robustness and generalization.

# Installation
```
pip install -r requirements.txt
```

# Datasets
We conduct noise robustness experiments on two synthetically corrupted datasets (i.e., CIFAR100N and CIFAR80N) and three real-world datasets (i.e., Web-Aircraft, Web-Car and Web-Bird).
Specifically, we create the closed-set noisy dataset CIFAR100N and the open-set noisy dataset CIFAR80N based on CIFAR100.
To make the open-set noisy dataset CIFAR80N, we regard the last 20 categories in CIFAR100 as out-of-distribution. 
We adopt two classic noise structures: symmetric and asymmetric, with a noise ratio $n \in (0,1)$.

You can download the CIFAR10 and CIFAR100 on [this](https://www.cs.toronto.edu/~kriz/cifar.html).

# Training

An example shell script to run FINE on CIFAR-100N :

```python
python SED_FINE.py --warmup-epoch 200 --epoch 300 --batch-size 128 --lr 0.05 --warmup-lr 0.1  --noise-type symmetric --closeset-ratio 0.2 --lr-decay cosine:200,5e-4,300  --opt sgd --dataset cifar100nc --gpu 4 --momentum-scs 0.999 --momentum-scr 0.99 --aph 0.95 --alpha 1.0 --beta 0.1 --gamma 0.002
```
An example shell script to run SED on CIFAR-80N :

```python
python SED_FINE.py --warmup-epoch 50 --epoch 150 --batch-size 128 --lr 0.05 --warmup-lr 0.05  --noise-type symmetric --closeset-ratio 0.2 --lr-decay cosine:50,5e-4,150  --opt sgd --dataset cifar80no --gpu 4 --momentum-scs 0.999 --momentum-scr 0.95 --aph 0.99 --alpha 1.0 --beta 0.1 --gamma 0.002 
```
Here is an example shell script to run SED on Web-aircraft :

```python
python SED_FINE_web.py --warmup-epoch 5 --epoch 50 --batch-size 32 --lr 0.005  --warmup-lr 0.005  --lr-decay cosine:5,5e-4,50 --weight-decay 5e-4 --seed 123 --opt sgd --dataset web-aircraft --gpu 9 --momentum_scs 0.999 --momentum_scr 0.99 --alpha 1 --aph 0.95 --beta 0.1 --gamma 0.001 --log AIR_BEST
```
