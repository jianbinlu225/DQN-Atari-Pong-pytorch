from trainer import Trainer
from datetime import datetime
import torch

class Config(dict):
    """Simple class that provides dot access to the properties"""
    __getattribute__ = dict.get


if __name__ == "__main__":
    config = Config({
        'environmentName': 'PongNoFrameskip-v4',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'rewardBound': 19,
        'replayBufferCapacity': 10000,
        'agentExplorationProbability': 0.1,
        'learningRate': 1e-4,
        'epsilonStart': 1,
        'epsilonFinal': 0.01,
        'epsilonLastDecayStep': 150000,
        'replayBufferStart': 10000,
        'discountFactor': 0.99,
        'batchSize': 32,
        'syncNetworkFrequency': 1000,
        'log_dir': "./runs"+datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    })
    trainer = Trainer(config)
    trainer.train()
