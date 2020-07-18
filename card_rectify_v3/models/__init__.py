import copy
from models.UNet import UNet, VGGUNet, UNetFCN, UNetSegNet
from models.CRDN import UNetRNN, VGG16RNN, ResNet18RNN, ResNet50RNN, ResNet34RNN, ResNet101RNN, ResNet152RNN, ResNet50UNet, ResNet50FCN


def get_model(model_dict, n_classes, version=None):
    name = model_dict["arch"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")

    if name == ["UNet","UNetFCN","UNetSegNet","VGGUNet"]:
        model = model(n_classes=n_classes, input_channel=3, **param_dict)

    elif name == "UNetRNN":
        model = model(input_channel=3, n_classes=n_classes, kernel_size=3, feature_scale=4, decoder="vanilla", bias=True)

    elif name in ["VGG16RNN","ResNet18RNN", "ResNet34RNN", "ResNet50RNN", "ResNet101RNN", "ResNet152RNN"]:
        model = model(input_channel=3, n_classes=n_classes, kernel_size=3, decoder="GRU", bias=True)

    elif name in ["ResNet50UNet","ResNet50FCN"]:
        model = model(n_classes=n_classes, input_channel=3, **param_dict)

    else:
        model = model(n_classes=n_classes, **param_dict)

    return model


def _get_model_instance(name):
    try:
        return {
            "UNetRNN": UNetRNN,
        }[name]
    except:
        raise ("Model {} not available".format(name))

