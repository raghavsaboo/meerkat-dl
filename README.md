# Meerkat Deep Learning Framework

![MeerkatDL](./misc/meerkatdl.png)

A Deep Learning Framework Just for Fun and Education! 🥳

Its purpose is to be a readable and quick to understand reference for building a library like PyTorch with Autodiff written completely in Python.

## Installation

```bash
$ pip install meerkat_dl
```

## Todo

 - [x] Automatic Differentiation
- [x] Computation Graph [Done]
- [ ] Layers:
  - [x] Dense
  - [x] RNN
  - [ ] CNN (2D and 3D)
  - [ ] LSTM
  - [ ] Attention
  - [ ] Dropout
  - [ ] MaxPool
  - [ ] BatchNorm
  - [ ] LayerNorm
- [ ] Initialization
- [x] Activation Functions
  - [x] ReLU
  - [x] TanH
  - [x] Sigmoid
  - [x] Softmax
- [ ] Optimizers
  - [x] SGD
  - [x] Minibatch SGD
  - [ ] Adam
  - [ ] Momentum
  - [ ] Adagrad
  - [ ] RMSProp
  - [ ] Adadelta
- [ ] Learning Rate Scheduling
- [ ] Loss Functions
  - [ ] MSE
  - [ ] CE
  - [ ] SoftMaxCE
  - [ ] Hinge
  - [ ] Margin
  - [ ] Triplet
  - [ ] KL
  - [ ] NLL
  - [ ] MAE
- [ ] Gradient checker
- [ ] Remove duplicate code in Operations forward pass
- [ ] Fix Min, Max, Mean, Flatten operations
- [ ] Add regularization
- [ ] Add reset graph for backward pass
- [ ] Add named parameters and named modules 
- [ ] Write tests
- [ ] Write examples

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`meerkat_dl` was created by Raghav Saboo. It is licensed under the terms of the MIT license.

## References

[Paszke, Adam, et al. "Automatic differentiation in PyTorch." (2017)](https://openreview.net/pdf?id=BJJsrmfCZ)

[Paszke, Adam, et al. "Pytorch: An imperative style, high-performance deep learning library." Advances in neural information processing systems 32 (2019).](https://proceedings.neurips.cc/paper_files/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html)

[Atilim Guns Baydin, et al. "Automatic differentiation in machine learning: a survey" (2018).](https://arxiv.org/pdf/1502.05767.pdf)

## Credits

`meerkat_dl` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
