from .trainer import retrain_model

def on_successful_payment():
    """
    Called AFTER DB commit.
    """
    retrain_model()
