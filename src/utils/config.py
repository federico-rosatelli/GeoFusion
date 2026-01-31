import yaml

def getConfigYAML(conf_path:str) -> any:
    with open(conf_path, 'rt') as f:
        config = yaml.safe_load(f.read())
    return config

def load_config(config_path:str):
    full_config = getConfigYAML(config_path)
    
    ens_config = full_config['ml']['ensemble']
    defaults = ens_config.get('default', {})
    models_config = ens_config.get('models', {})
    force_retrain = full_config['ml'].get('force_retrain', False)
    
    processed_models = {}
    
    for name, specific_conf in models_config.items():

        final_conf = defaults.copy()
        
        if 'train' in specific_conf:
            final_conf['train'] = final_conf.get('train', {}).copy()
            final_conf['train'].update(specific_conf['train'])
        
        for k, v in specific_conf.items():
            if k != 'train':
                final_conf[k] = v
                
        processed_models[name] = final_conf
        
    return processed_models, full_config['ml']['device'], force_retrain
