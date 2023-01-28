PAD_LENGTH = 32
# for text model to unroll layers

PRETRAINED_MODELS = './pretrained_models'

AE_DIR = './adversarial_examples/'

CIFAR10_JPEG_DIR = './datasets/CIFAR10/'
IMAGENET_JPEG_DIR = '/data/yyuanaq/data/IMAGE-NET/ILSVRC/Data/CLS-LOC/'
IMAGENET_LABEL_TO_INDEX = './datasets/ImageNet/ImageNetLabel2Index.json'
# Since we use the pretrained weights provided by pytorch,
# we should use the same `label_to_index` mapping.

BIGGAN_IMAGENET_PROJECT_DIR = './BigGAN-projects/ImageNet'
BIGGAN_CIFAR10_PROJECT_DIR = './BigGAN-projects/CIFAR10'

BIGGAN_CIFAR10_LATENT_DIM = 128
BIGGAN_IMAGENET_LATENT_DIM = 120

STYLE_IMAGE_DIR = './datasets/painting'
STYLE_MODEL_DIR = './pretrained_models/Style'