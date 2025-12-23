# Certified Defense on the Fairness of Graph Neural Networks

This is the open-source code for KDD'26 Certified Defense on the Fairness of Graph Neural Networks.



## 1.Environment

Experiments are performed on an Nvidia RTX A6000 with Cuda 13.0.

Dependencies can be found in requirements.txt.

Notice: Cuda is enabled for default settings.



## 2.Usage
We have three datasets for experiments, namely German Credit, Recidivism, and Credit Defaulter. To perform training and certified inference, run

```
./effectiveness.sh
```

Alternatively, directly run

```
python gnn_certification.py
```

also gives running under the default parameter settings:

```
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:01<00:00, 138.09it/s]
Time: 1.496720790863037 s
Test set results: loss= 0.6172 accuracy= 0.6640
Statistical Parity:  0.533278777959629
Equality:  0.45588235294117646
Determine the predicted class:                                                                                                                    
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 32.14it/s]
Certify the predicted class:                                                                                                                      
100%|██████████████████████████████████████████████████████████████████| 200/200 [00:40<00:00,  4.98it/s, Rx=12.0459, acc=0.7244, min_fair=0.1461]
Determine the predicted class:                                                                                                                    
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 69.99it/s]
Certify the predicted class:                                                                                                                      
100%|██████████████████████████████████████████████████████████████████| 200/200 [00:40<00:00,  4.97it/s, Rx=12.0459, acc=0.7333, min_fair=0.1953]
Determine the predicted class:                                                                                                                    
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 70.70it/s]
Certify the predicted class:                                                                                                                      
100%|██████████████████████████████████████████████████████████████████| 200/200 [00:40<00:00,  5.00it/s, Rx=12.0459, acc=0.6978, min_fair=0.1170]
Determine the predicted class:                                                                                                                    
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 70.93it/s]
Certify the predicted class:                                                                                                                      
100%|██████████████████████████████████████████████████████████████████| 200/200 [00:39<00:00,  5.01it/s, Rx=12.0459, acc=0.7067, min_fair=0.1787]
Determine the predicted class:                                                                                                                    
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 70.16it/s]
Certify the predicted class:                                                                                                                      
100%|██████████████████████████████████████████████████████████████████| 200/200 [00:40<00:00,  4.99it/s, Rx=12.0459, acc=0.7111, min_fair=0.1811]
Determine the predicted class:                                                                                                                    
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 70.24it/s]
Certify the predicted class:                                                                                                                      
  5%|█████▎                                                                                                     | 5/100 [03:21<1:03:40, 40.22s/it]
  9%|██████                                                             | 18/200 [00:03<00:36,  5.01it/s, Rx=12.0459, acc=0.7067, min_fair=0.2363]
...
```











