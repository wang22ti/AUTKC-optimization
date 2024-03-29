# Optimizing Partial Area Under the Top-k Curve: Theory and Practice

This is a Pytorch implementation of `AUTKC optimization` described in our paper:

> Zitai Wang, Qianqian Xu, Zhiyong Yang, Yuan He, Xiaochun Cao and Qingming Huang. Optimizing Partial Area Under the Top-k Curve: Theory and Practice. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 45(4): 5053–5069, 2023.

And its Arxiv version can be found [here](https://arxiv.org/pdf/2209.01398.pdf).

The original version of our codes involves a non-standard IO operation, which is described later. Most recently, we provide a new demo that follows the standard learning pipline, which can be found in our [library](https://github.com/statusrank/XCurve/blob/master/example/example_ipynb/cifar_100_AUTKC.ipynb).

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

@article{autkc,
  author    = {Zitai Wang and
               Qianqian Xu and
               Zhiyong Yang and
               Yuan He and
               Xiaochun Cao and
               Qingming Huang},
  title     = {Optimizing Partial Area Under the Top-k Curve: Theory and Practice},
  journal   = {{IEEE} Transactions on Pattern Analysis and Machine Intelligence},
  volume    = {45},
  number    = {4},
  year      = {2023},
  pages     = {5053--5069}
}
```