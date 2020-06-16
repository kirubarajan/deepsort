"""
For running experiments.
"""
from experiments import evaluate, report_metrics, train

if __name__ == "__main__":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train.run()
