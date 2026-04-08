from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from data_prep.urban_context import ASSETS, build_context_features, ensure_context_dirs, refresh_source_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description='Download official NYC urban-context layers and aggregate them to H3')
    parser.add_argument('--resolution', type=int, default=9)
    parser.add_argument('--densify-step-m', type=float, default=40.0)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--metadata-only', action='store_true')
    parser.add_argument('--max-rows-per-asset', type=int, default=None)
    parser.add_argument(
        '--assets',
        nargs='+',
        choices=sorted(ASSETS),
        default=list(ASSETS),
        help='Subset of urban-context assets to download and aggregate',
    )
    args = parser.parse_args()

    ensure_context_dirs()
    manifest_path = refresh_source_manifest(force=args.force)
    print(f'Wrote source manifest to {manifest_path}')

    if args.metadata_only:
        return

    output_path = build_context_features(
        resolution=args.resolution,
        densify_step_m=args.densify_step_m,
        force_download=args.force,
        max_rows_per_asset=args.max_rows_per_asset,
        asset_keys=args.assets,
    )
    print(f'Wrote urban-context features to {output_path}')


if __name__ == '__main__':
    main()
