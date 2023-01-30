#Hybrid Loss-driven Multi-source Domain Adversarial Network

This code implements a multi-source learning algorithm for NILM applications
<img src="Plot/framework.png" alt="alt text" width="800" height="300">

# How to use the code and example
Directory tree:

``` bash
├── Data Process
│   ├── redd_process.py
│   ├── refit_process.py
│   └── ukdale_process.py
├── Code
│   ├── HLD_MDAN.ipynb
│   ├── HLD_google.ipynb
│   └── AHMDAN.ipynb
│   ├── DANN+JMMD_MDAN.ipynb
│   └── baseline.ipynb
│   ├── KLD.ipynb
│   └── shiftMode.ipynb
├──Plot
```
## **Create UK-DALE or REDD dataset**
Datasets are built by the power readings of houses 1, 2, and 3 from REDD, building 1, 2 from UK-DALE, and five randomly selected houses from REFIT. For all datasets, we sampled the active load every 8 seconds. The commonly-used appliances were chosen to implement model training, such as dishwasher, fridge, washing machine, kettle, and microwave. For each dataset, 80% of samples were used for model training, and the remaining for testing. The scripts contained in NILM_data_management directory allow the user to create CSV files of training dataset across different application scenarios. Any output CSV file represents a dataset owned by a householder.

## **Benchmarks**
We selected four methods to be the benchmark algorithms for performance comparison. i) Baseline is a basic model trained without domain adaptation. ii) DANN+JMMD [1] is an adversarial neural network conditioned on the joint distribution loss for domain adaptation. It has been empirically proved can alleviate domain shifts in NILM. iii) MDAN [2] achieves MSDA by using 𝑑H distance and an adversarial learning strategy similar to DANN. We adopted the soft version in this paper. iv) AHD-MSDA
[3] is an unsupervised DA method for regression tasks.

## **Training**
The HLD_MDAN.ipynb script provides the main training workflow of HLD_MDAN algorithm for NILM. It integrates with the Seq2point paradigm for NILM data (pairs of multiple samples aggregate data and 1 sample midpoint ground truth). The HLD_google.ipynb is used to implement domain adaptation on the Google platform. We also provide a Dockerfile and original python file of HLD_MDAN to containerized such application for better portability and reproducibility.

Training default parameters:
For the feature extractor, we set the kernel size as 10, 8, 6, 5, and 5, respectively. The filter numbers are 30, 30, 40, 40, and 50. The input of the predictor is 1024. For model training, the learning rate of the Adam optimizer is 0.0001, and the maximum epoch number is 50 with a batch size of 64.

# Dataset
Datasets used can be found:
1. UKDALE: https://jack-kelly.com/data/
2. REDD: http://redd.csail.mit.edu/
3. REFIT: https://www.refitsmarthomes.org/datasets/

# Reference
[1] Yinyan Liu, Li Zhong, Jing Qiu, Junda Lu, and Wei Wang. 2021. Unsupervised Domain Adaptation for Non-Intrusive Load Monitoring Via Adversarial and Joint Adaptation Network. IEEE Transactions on Industrial Informatics (2021).
[2] Han Zhao, Shanghang Zhang, Guanhang Wu, Geoffrey J Gordon, et al. 2018. Multiple source domain adaptation with adversarial learning. (2018).
[3] Guillaume Richard, Antoine de Mathelin, Georges Hébrail, Mathilde Mougeot, and Nicolas Vayatis. 2020. Unsupervised multi-source domain adaptation for regression. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer, 395–411
