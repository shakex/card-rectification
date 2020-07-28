import copy
from models.CRDN import UNetRNN


def get_model(model_dict, n_classes):
    name = model_dict["arch"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")

    model = model(input_channel=3, n_classes=n_classes, kernel_size=3, feature_scale=4, decoder="vanilla", bias=True)

    return model


def _get_model_instance(name):
    try:
        return {
            "UNetRNN": UNetRNN,
        }[name]
    except:
        raise ("Model {} not available".format(name))

