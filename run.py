"""Train, adapt, or evaluate the RIFT Q2-upload / affine-Q4-broadcast release."""
from __future__ import annotations
import argparse
from rift.data import CITIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    training = commands.add_parser('train', help='Federated transfer followed by target adaptation')
    training.add_argument('--target', choices=CITIES, required=True)
    training.add_argument('--seed', type=int, default=0)
    training.add_argument('--output', required=True, help='New output directory; existing directories are preserved')
    training.add_argument('--stats-dir', default='data/source_statistics', help='Reusable local source-normalization releases')
    training.add_argument('--rounds', type=int, default=20)
    training.add_argument('--source-rounds', type=int, default=10)
    training.add_argument('--source-steps', type=int, default=4)
    training.add_argument('--local-epochs', type=int, default=2)
    training.add_argument('--adapt-epochs', type=int, default=50)
    training.add_argument('--target-days', type=int, default=3)
    training.add_argument('--batch-size', type=int, default=32)
    training.add_argument('--source-batch', type=int, default=128)
    training.add_argument('--clip', type=float, default=.25)
    training.add_argument('--sigma', type=float, default=.8261)
    training.add_argument('--target-weight', type=float, default=.9)
    training.add_argument('--temperature', type=float, default=1.)
    training.add_argument('--smoothing', type=float, default=.25)
    training.add_argument('--upload-bits', type=int, choices=[2], default=2)
    training.add_argument('--download-bits', type=int, choices=[4], default=4)
    training.add_argument('--smoke', action='store_true', help='Generated-data check only; never a paper result')
    evaluation = commands.add_parser('evaluate', help='Evaluate a saved target without retraining')
    adaptation = commands.add_parser('adapt', help='Restart final target adaptation from a saved model')
    for sub in (evaluation, adaptation):
        sub.add_argument('--checkpoint', required=True)
    adaptation.add_argument('--output', required=True)
    adaptation.add_argument('--adapt-epochs', type=int, default=50)
    for sub in (training, evaluation, adaptation):
        sub.add_argument('--data-root', required=True)
        sub.add_argument('--device', default='cuda:0')
        sub.add_argument('--threads', type=int, default=2)
        sub.add_argument('--split', choices=['dev', 'test'], default='test',
                         help='Evaluation interval (default: held-out test)')
    args = parser.parse_args()
    if args.command == 'train':
        if not 1 <= args.source_rounds <= args.rounds <= 253:
            parser.error('Require 1 <= source-rounds <= rounds <= 253.')
        if min(args.source_steps, args.local_epochs, args.target_days, args.batch_size, args.source_batch) < 1 or args.adapt_epochs < 0:
            parser.error('Epochs, batch sizes and target days must be positive; adapt-epochs may be zero.')
        if args.smoke:
            args.rounds, args.source_rounds, args.source_steps = 3, 2, 1
            args.local_epochs, args.adapt_epochs, args.target_days = 1, 1, 1
            args.source_batch, args.batch_size = 2, 16
            args.split = 'test'
        from rift.engine import train
        train(args)
    else:
        from rift.engine import from_checkpoint
        from_checkpoint(args)


if __name__ == '__main__':
    main()
