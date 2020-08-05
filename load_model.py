import sys
import torch
from models import get_model
from collections import OrderedDict

model_arch = 'UNetRNN'
model_path = "CRDN1000.pkl"


def load_model():
    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    try:
        model = get_model({'arch': model_arch}, n_classes=2).to(device)
        state = convert_state_dict(torch.load(model_path, map_location=device)["model_state"])
        model.load_state_dict(state)
        model.eval()
        model.to(device)
    except:
        print("Model Error: Model \'" + model_arch + "\' import failed, please check the model file.")
        sys.exit()

    return model, device


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    if not next(iter(state_dict)).startswith("module."):
        return state_dict  # abort if dict is not a DataParallel model_state
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        item_name = k[7:]  # remove `module.`
        new_state_dict[item_name] = v
    return new_state_dict


if __name__ == "__main__":
    trained_model, device = load_model()
