import yaml
import json

def getConfigYAML(conf_path:str) -> any:
    with open(conf_path, 'rt') as f:
        config = yaml.safe_load(f.read())
    return config

def getConfigJSON(conf_path:str) -> any:
    with open(conf_path, 'rt') as f:
        config = json.load(f)
    return config


def load_config(config_yaml:str, config_json:str):
    full_config = getConfigYAML(config_yaml)
    model_structs = getConfigJSON(config_json)
    
    ens_config = full_config['ml']['ensemble']
    defaults = ens_config.get('default', {})
    models_config = ens_config.get('models', {})
    force_retrain = full_config['ml'].get('force_retrain', False)

    json_models = model_structs.get('models', {})
    
    processed_models = {}
    
    for name, specific_conf in models_config.items():

        final_conf = defaults.copy()
        
        if 'train' in specific_conf:
            final_conf['train'] = final_conf.get('train', {}).copy()
            final_conf['train'].update(specific_conf['train'])
        
        for k, v in specific_conf.items():
            if k != 'train':
                final_conf[k] = v
        
        if name in json_models and "struct" in json_models[name]:
            final_conf['structure'] = json_models[name]["struct"]
        else:
            final_conf['structure'] = None
                
        processed_models[name] = final_conf
        
    return processed_models, full_config['ml']['device'], force_retrain
