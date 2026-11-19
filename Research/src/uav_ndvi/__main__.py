import argparse
from .config import load_config


def main():
    parser = argparse.ArgumentParser(description='UAV AgriMatNet NDVI pipeline')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('preprocess', aliases=['prepare'])
    p.add_argument('--manifest', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--data-root')
    p.add_argument('--reference')
    p.add_argument('--config')
    p = sub.add_parser('examples')
    p.add_argument('--cache', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--config')
    p = sub.add_parser('train')
    p.add_argument('--examples', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--config')
    p = sub.add_parser('evaluate')
    p.add_argument('--run', required=True)
    p.add_argument('--split', choices=['train', 'validation', 'test'], default='test')
    p.add_argument('--out', help='Fresh evaluation directory; defaults to run/evaluation_SPLIT')
    p = sub.add_parser('forecast')
    p.add_argument('--cache', required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--date', required=True)
    p.add_argument('--season', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--model', choices=['linear', 'transformer', 'persistence', 'all'], default='all')
    p.add_argument('--mask', help='Optional single-band aligned mask; positive valid values predict')
    p.add_argument('--classification', action='store_true')
    args = parser.parse_args()
    if args.command in ['preprocess', 'prepare']:
        from .preprocessing import preprocess
        preprocess(args.manifest, args.out, load_config(args.config), args.data_root, args.reference)
    elif args.command == 'examples':
        from .dataset import build_examples
        build_examples(args.cache, args.out, load_config(args.config))
    elif args.command == 'train':
        from .train import train
        train(args.examples, args.out, load_config(args.config))
    elif args.command == 'evaluate':
        from .evaluate import evaluate
        evaluate(args.run, args.split, args.out)
    else:
        from .forecast import forecast
        if args.model == 'all':
            from pathlib import Path
            for name in ['persistence', 'linear', 'transformer']:
                forecast(args.cache, args.run, args.date, args.season, Path(args.out) / name, name, args.mask, args.classification)
        else:
            forecast(args.cache, args.run, args.date, args.season, args.out, args.model, args.mask, args.classification)


if __name__ == '__main__':
    main()
