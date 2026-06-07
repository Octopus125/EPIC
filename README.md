# EPIC: Multi-objective Guided Diffusion for Epitope Design in TCR-pMHC Complexes

This repository is the official implementation of EPIC: Multi-objective Guided Diffusion for Epitope Design in TCR-pMHC Complexes. 
![fig1](./fig/epic.png)

## Requirements

To install requirements:

```setup
conda create -n epic python=3.9.12
conda activate epic

pip install -r requirements.txt
```


## Training

To train the generator of EPIC, run this command:

```train
python train_generator.py
```

To train the classifier of EPIC, run this command:

```train
python train_agpep_pred.py --gpus 0,1  # train the antigenicity classifier
python train_pmhc_pred.py --gpus 0,1  # train the pMHC presentation classifier
python train_speci_pred.py --gpus 0,1  # train the TCR-p specificity classifier
```

## Sampling

To sample, run this command:
```sample
python sample.py --mode uncond         # sampling without gradient guidance
python sample.py --mode cond           # sampling with guidance
python sample.py --mode cond_resample  # sampling with guidance and resampling strategy
```

## Evaluation

To evaluate the generated peptides, run:

```eval
python eval.py
```