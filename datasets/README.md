We use the following datasets in our evaluation.

- ImageNet

  You can download it from the [official release](https://www.image-net.org/download.php) or [kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge/data).

- CIFAR10:

  You can download it from the [official release](https://www.cs.toronto.edu/~kriz/cifar.html).

  Note that in our evaluation, we first convert the CIFAR10 data into images and then use these images as test suites. We provide the converted images [here]().

- IMDB:

  The dataset will be auotmatically download in `./data` folder when running `eval_diversity_text.py`.