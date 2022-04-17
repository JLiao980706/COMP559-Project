import json

model_config = {
    'name': 'ChebNet5',
    'hidden_dims': [16],
    'dropout': 0.5,
    'drop_input': 0.5,
    'drop_name': 'sparse',
    'activation': 'Relu',
    'initializer': 'glorot',
    'bias': False
}

FILENAME = 'basic_cheb5_config.json'
config_string = json.dumps(model_config)
with open(FILENAME, 'w+') as jfile:
    jfile.write(config_string)
 