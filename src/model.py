import segmentation_models_pytorch as smp

def get_model(config):
    """
    Initializes the model architecture defined in config.
    """
    print(f"Initializing {config.ARCHITECTURE} with {config.ENCODER} encoder...")
    
    # MAnet is a SegFormer-like architecture when paired with Mix Transformers (mit_b0)
    if config.ARCHITECTURE == 'MAnet':
        model = smp.MAnet(
            encoder_name=config.ENCODER, 
            encoder_weights=config.ENCODER_WEIGHTS, 
            in_channels=3, 
            classes=1, 
            activation=None # Return logits for numerical stability in loss
        )
        
    elif config.ARCHITECTURE == 'Unet':
        model = smp.Unet(
            encoder_name=config.ENCODER, 
            encoder_weights=config.ENCODER_WEIGHTS, 
            in_channels=3, 
            classes=1, 
            activation=None
        )
    else:
        raise ValueError(f"Architecture {config.ARCHITECTURE} not supported.")
        
    return model
