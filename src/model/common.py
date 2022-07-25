from utils.common import get_object


def get_model(model_name, model_kwargs):
    return get_object(model_name, model_kwargs)
