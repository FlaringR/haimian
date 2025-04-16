import importlib

def getattr_nested(_module_src, _model_name):
    module = importlib.import_module(_module_src, package="Rehaimian")
    return getattr(module, _model_name)