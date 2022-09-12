# Optimizing Partial Area Under the Top-k Curve: Theory and Practice

This is a Pytorch implementation of `AUTKC optimization` described in our paper:

>Z. Wang, Q. Xu, Z. Yang, Y. He, X. Cao and Q. Huang. Optimizing Partial Area Under the Top-k Curve: Theory and Practice. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) (Early Access).

And its Arxiv version can be found [here](https://arxiv.org/pdf/2209.01398.pdf).

## Dependencies
Please see the `requirements.yml` in the root folder.


## Data and Train
We conduct the experiments on [Cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html), [Cifar-100](https://www.cs.toronto.edu/~kriz/cifar.html), [Tiny-imagenet-200](https://www.kaggle.com/c/tiny-imagenet), and [Places-365](http://places2.csail.mit.edu/download.html). The method `MemoryDataset.__init__()` in the script `dataset.py` describes how to organize these datasets. What's more, to enable parallel computation, we create a shared memory to store the data. To this end, one should first run `dataset.py` before training or testing the model:
```bash
python dataset.py
```
Note that one should not kill the dataset process. Then, one can train the model via:
```bash
python main.py --dataset "cifar-100" --loss atop --weight_scheme Exp --resume checkpoints/*** 
```

## Citation

```

@article{DBLP:journals/corr/abs-2209-01398,
  author    = {Zitai Wang and
               Qianqian Xu and
               Zhiyong Yang and
               Yuan He and
               Xiaochun Cao and
               Qingming Huang},
  title     = {Optimizing Partial Area Under the Top-k Curve: Theory and Practice},
  journal   = {CoRR},
  volume    = {abs/2209.01398},
  year      = {2022},
}
```