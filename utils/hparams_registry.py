def default_hparams(ALGORITHM, DATASET, TRIAL_SEED):
    """
    Global registry of hyperparams.
    New algorithms / networks / etc. should add entries here.
    """
    hparams = {}
    
    hparams['aggregator'] = ALGORITHM
    hparams['datasets'] = DATASET
    hparams['trial_seed'] = TRIAL_SEED
    
    SMALL_IMAGES = ['RotatedMNIST', 'ColoredMNIST']

    hparams['batch_size'] = 32
    hparams['holdout_fraction'] = 0.6
    hparams['imbalance_ratio'] = 1.0
    hparams['samplesize_ratio'] = 1.0

    if DATASET in SMALL_IMAGES:
        hparams['lr'] = 1e-3
        hparams['weight_decay'] = 1e-4
        
        hparams['lambda1'] = 0.01
        hparams['lambda2'] = 0.001
        hparams['lambda3'] = 0.001
        
        hparams['nonlinear_classifier'] = True
        
        if DATASET == 'ColoredMNIST':
            hparams['data_dir'] = "./data"
            
            hparams['no_classes'] = 2
            hparams['num_client'] = 10
            hparams['heterogeneity'] = 90
            
            hparams['global_epochs'] = 3
            hparams['local_epochs'] = 150
            
            
        elif DATASET == 'RotatedMNIST':
            hparams['data_dir'] = "./data"
            
            hparams['no_classes'] = 10
            hparams['num_client'] = 6
            hparams['heterogeneity'] = 75
            
            hparams['global_epochs'] = 5
            hparams['local_epochs'] = 150
            
            
    if ALGORITHM == 'fedem':
        hparams['M'] = 3

    return hparams