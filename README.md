# 🩻 MedMod: A Multimodal Benchmark for Clinical Prediction Tasks with Electronic Health Records and Chest X-Ray Scans

Table of contents
=================

<!--ts-->
  * [Background](#Background)
  * [Environment setup](#Environment-setup)
  * [Model training](#Model-training)
  * [Model evaluation](#Model-evaluation)
  * [Citation](#Citation)
   
<!--te-->

Background
============
We follow the data extraction and linking pipeline of the two datasets MIMIC-IV and MIMIC-CXR based on the task definition (i.e., inhospital mortality prediction, clinical conditions, decompensation, length of stay, and radiology).

Environment setup
==================
To run this repo, you must install and run the libraries in the below yml file.

    conda env create -f environment.yml
    conda activate medmod


Training and evaluation framework
====================================

Self-supervised pre-training scripts
-----------------

For pre-training, there are three types of training scripts that have been setup: 
- simclr_trainer.py
- vicreg_trainer.py
- align_trainer.sh


Self-supervised evaluation scripts
------------------
To evaluate the quality of the representations learned using pre-training, several scripts have been implemented:
- (task/train/script.sh) finetune.sh, finetune_cxr.sh, finetune_ehr.sh --> these scripts further fine-tune either the multi-modal architecture or the uni-modal branches.
- (task/train/script.sh) lineareval.sh, lineareval_cxr.sh, lineareval_ehr.sh --> these scripts tune a single layer and freeze the pre-trained encoders for either multi-modal or uni-modal predictions.
- (task/eval/script.sh) lineareval / fteval --> these scripts perform an evaluation run for either fine-tuned or linear classifiers.

All of the scripts above call run_gpu.py

Other useful scripts:
- eval_epoch.sh --> This calls epoch_evaluation.py and evaluates the quality of the representations in terms of AUROC using a linear classifier at each pre-training epoch. It is useful for selecting the epoch that yields the best AUROC on the validation set. It stores results in a csv file.


Citation 
============

If you find MedMod useful for your research and applications, please cite using this BibTeX:
```bibtex

@article{elsharief2025medmod,
  title={MedMod: Multimodal Benchmark for Medical Prediction Tasks with Electronic Health Records and Chest X-Ray Scans},
  author={Elsharief, Shaza and Shurrab, Saeed and Al Jorf, Baraa and L{\'o}pez, L Juli{\'a}n Lechuga and Shamout, Farah E},
  journal={Proceedings of Machine Learning Research},
  volume={287},
  pages={1--23},
  year={2025},
  publisher={ML Research Press}
}
```
