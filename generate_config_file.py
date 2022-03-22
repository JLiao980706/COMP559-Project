import json

model_config = {
    'name': 'GCN',
    'hidden_dims': [1000, 500],
    'activation': 'Relu'
}

FILENAME = 'basic_gcn_config.json'
config_string = json.dumps(model_config)
with open(FILENAME, 'w+') as jfile:
    jfile.write(config_string)

