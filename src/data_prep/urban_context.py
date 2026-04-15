from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import h3
import ijson
import pandas as pd
import requests
from shapely import wkt
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from spatial.h3_utils import haversine_m, polyline_to_h3_cells

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESOLUTION = 9
DEFAULT_DENSIFY_STEP_M = 40.0
DOWNLOAD_CHUNK_BYTES = 1 << 20
SOCRATA_PAGE_SIZE = 20_000
USER_AGENT = 'rendezvous-urban-context/1.0'

RAW_DIR = ROOT / 'data' / 'urban_context' / 'raw'
PROCESSED_DIR = ROOT / 'data' / 'urban_context' / 'processed'


@dataclass(frozen=True)
class UrbanContextAsset:
    key: str
    dataset_id: str
    title: str
    download_url: str
    storage_name: str
    docs_url: str
    format: str

    def raw_path(self) -> Path:
        return RAW_DIR / self.storage_name

    def metadata_path(self) -> Path:
        return RAW_DIR / f'{self.key}.metadata.json'

    def download_state_path(self) -> Path:
        return RAW_DIR / f'{self.key}.download.json'


ASSETS: dict[str, UrbanContextAsset] = {
    'street_centerline': UrbanContextAsset(
        key='street_centerline',
        dataset_id='inkn-q76z',
        title='Centerline',
        download_url='https://data.cityofnewyork.us/resource/inkn-q76z.csv?$select=the_geom,trafdir,status',
        storage_name='street_centerline.csv',
        docs_url='https://data.cityofnewyork.us/d/inkn-q76z',
        format='csv',
    ),
    'sidewalk_centerline': UrbanContextAsset(
        key='sidewalk_centerline',
        dataset_id='fytp-pq92',
        title='Sidewalk Centerline',
        download_url='https://data.cityofnewyork.us/resource/fytp-pq92.csv?$select=the_geom',
        storage_name='sidewalk_centerline.csv',
        docs_url='https://data.cityofnewyork.us/d/a9xv-vek9',
        format='csv',
    ),
    'building_footprints': UrbanContextAsset(
        key='building_footprints',
        dataset_id='5zhs-2jue',
        title='Building Footprints (Map)',
        download_url='https://data.cityofnewyork.us/resource/5zhs-2jue.csv?$select=the_geom,shape_area,height_roof',
        storage_name='building_footprints.csv',
        docs_url='https://data.cityofnewyork.us/d/jh45-qr5r',
        format='csv',
    ),
    'pluto': UrbanContextAsset(
        key='pluto',
        dataset_id='64uk-42ks',
        title='Primary Land Use Tax Lot Output (PLUTO)',
        download_url='https://data.cityofnewyork.us/resource/64uk-42ks.csv?$select=latitude,longitude,numfloors,bldgarea,unitstotal,yearbuilt',
        storage_name='pluto.csv',
        docs_url='https://data.cityofnewyork.us/d/64uk-42ks',
        format='csv',
    ),
    'elevation_points': UrbanContextAsset(
        key='elevation_points',
        dataset_id='9uxf-ng6q',
        title='NYC Planimetric Database: Elevation Points',
        download_url='https://data.cityofnewyork.us/resource/9uxf-ng6q.csv?$select=the_geom,elevation',
        storage_name='elevation_points.csv',
        docs_url='https://data.cityofnewyork.us/d/szwg-xci6',
        format='csv',
    ),
}


def ensure_context_dirs() -> tuple[Path, Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DIR, PROCESSED_DIR


def get_asset(key: str) -> UrbanContextAsset:
    try:
        return ASSETS[key]
    except KeyError as exc:
        raise ValueError(f'Unknown urban context asset {key!r}. Expected one of: {sorted(ASSETS)}') from exc


def download_asset(key: str, *, force: bool = False, timeout_s: int = 120) -> Path:
    ensure_context_dirs()
    asset = get_asset(key)
    target = asset.raw_path()
    if target.exists() and not force and _download_is_complete(asset):
        return target

    session = requests.Session()
    session.headers.update({'User-Agent': USER_AGENT})
    metadata = session.get(f'https://data.cityofnewyork.us/api/views/{asset.dataset_id}.json', timeout=timeout_s)
    metadata.raise_for_status()
    asset.metadata_path().write_text(json.dumps(metadata.json(), indent=2), encoding='utf-8')

    rows_downloaded, page_count = _download_csv_pages(session, asset, target, timeout_s=timeout_s)
    asset.download_state_path().write_text(
        json.dumps(
            {
                'complete': True,
                'download_url': asset.download_url,
                'row_count': rows_downloaded,
                'page_count': page_count,
                'page_size': SOCRATA_PAGE_SIZE,
                'downloaded_at': datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding='utf-8',
    )
    return target


def refresh_source_manifest(*, force: bool = False, timeout_s: int = 60) -> Path:
    ensure_context_dirs()
    manifest: dict[str, dict[str, object]] = {}
    session = requests.Session()
    session.headers.update({'User-Agent': USER_AGENT})
    for key, asset in ASSETS.items():
        metadata_path = asset.metadata_path()
        if force or not metadata_path.exists():
            response = session.get(f'https://data.cityofnewyork.us/api/views/{asset.dataset_id}.json', timeout=timeout_s)
            response.raise_for_status()
            metadata_path.write_text(json.dumps(response.json(), indent=2), encoding='utf-8')
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        download_state = {}
        download_state_path = asset.download_state_path()
        if download_state_path.exists():
            download_state = json.loads(download_state_path.read_text(encoding='utf-8'))
        manifest[key] = {
            'dataset_id': asset.dataset_id,
            'title': asset.title,
            'docs_url': asset.docs_url,
            'download_url': asset.download_url,
            'raw_path': str(asset.raw_path()),
            'download_complete': bool(download_state.get('complete', False)),
            'download_row_count': int(download_state.get('row_count', 0) or 0),
            'download_page_count': int(download_state.get('page_count', 0) or 0),
            'rows_updated_at': metadata.get('rowsUpdatedAt'),
            'view_last_modified': metadata.get('viewLastModified'),
            'attribution': metadata.get('attribution'),
            'description': metadata.get('description', ''),
        }

    manifest_path = PROCESSED_DIR / 'urban_context_sources.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest_path


def build_context_features(
    *,
    resolution: int = DEFAULT_RESOLUTION,
    densify_step_m: float = DEFAULT_DENSIFY_STEP_M,
    force_download: bool = False,
    max_rows_per_asset: int | None = None,
    asset_keys: list[str] | None = None,
) -> Path:
    ensure_context_dirs()
    selected_keys = asset_keys or list(ASSETS)
    for key in selected_keys:
        download_asset(key, force=force_download)
    refresh_source_manifest(force=force_download)

    frames: list[pd.DataFrame] = []
    if 'street_centerline' in selected_keys:
        frames.append(_aggregate_street_centerline(get_asset('street_centerline').raw_path(), resolution, densify_step_m, max_rows_per_asset))
    if 'sidewalk_centerline' in selected_keys:
        frames.append(
            _aggregate_csv_lines(
                get_asset('sidewalk_centerline').raw_path(),
                resolution,
                densify_step_m,
                max_rows_per_asset,
                prefix='sidewalk',
            )
        )
    if 'building_footprints' in selected_keys:
        frames.append(_aggregate_csv_polygons(get_asset('building_footprints').raw_path(), resolution, max_rows_per_asset, prefix='building'))
    if 'pluto' in selected_keys:
        frames.append(_aggregate_pluto(get_asset('pluto').raw_path(), resolution, max_rows_per_asset))
    if 'elevation_points' in selected_keys:
        frames.append(_aggregate_csv_points(get_asset('elevation_points').raw_path(), resolution, max_rows_per_asset, prefix='elevation'))

    merged = _merge_feature_frames(frames)
    merged['resolution'] = resolution
    output_path = PROCESSED_DIR / f'urban_context_h3_res{resolution}.parquet'
    merged.to_parquet(output_path, index=False)
    return output_path


def _aggregate_street_centerline(path: Path, resolution: int, densify_step_m: float, max_rows: int | None) -> pd.DataFrame:
    stats: dict[str, dict[str, float]] = {}
    processed = 0
    for chunk in pd.read_csv(path, chunksize=5_000):
        for row in chunk.itertuples(index=False):
            if max_rows is not None and processed >= max_rows:
                return _stats_to_frame(stats)
            processed += 1
            geometry_text = getattr(row, 'the_geom', None)
            if not isinstance(geometry_text, str) or not geometry_text:
                continue
            cells, length_m = _line_cells_from_geometry(wkt.loads(geometry_text), resolution, densify_step_m)
            if not cells:
                continue
            length_share = length_m / max(len(cells), 1)
            one_way = 1.0 if str(getattr(row, 'trafdir', '')).strip().upper() in {'FT', 'TF'} else 0.0
            for cell in cells:
                bucket = stats.setdefault(cell, {'street_segment_count': 0.0, 'street_length_m': 0.0, 'street_one_way_count': 0.0})
                bucket['street_segment_count'] += 1.0
                bucket['street_length_m'] += length_share
                bucket['street_one_way_count'] += one_way
    return _stats_to_frame(stats)


def _aggregate_geojson_lines(path: Path, resolution: int, densify_step_m: float, max_rows: int | None, *, prefix: str) -> pd.DataFrame:
    stats: dict[str, dict[str, float]] = {}
    for idx, feature in enumerate(_iter_geojson_features(path), start=1):
        if max_rows is not None and idx > max_rows:
            break
        geometry = shape(feature['geometry'])
        cells, length_m = _line_cells_from_geometry(geometry, resolution, densify_step_m)
        if not cells:
            continue
        length_share = length_m / max(len(cells), 1)
        for cell in cells:
            bucket = stats.setdefault(cell, {f'{prefix}_segment_count': 0.0, f'{prefix}_length_m': 0.0})
            bucket[f'{prefix}_segment_count'] += 1.0
            bucket[f'{prefix}_length_m'] += length_share
    return _stats_to_frame(stats)


def _aggregate_csv_lines(path: Path, resolution: int, densify_step_m: float, max_rows: int | None, *, prefix: str) -> pd.DataFrame:
    stats: dict[str, dict[str, float]] = {}
    processed = 0
    for chunk in pd.read_csv(path, chunksize=5_000):
        for row in chunk.itertuples(index=False):
            if max_rows is not None and processed >= max_rows:
                return _stats_to_frame(stats)
            processed += 1
            geometry_text = getattr(row, 'the_geom', None)
            if not isinstance(geometry_text, str) or not geometry_text:
                continue
            cells, length_m = _line_cells_from_geometry(wkt.loads(geometry_text), resolution, densify_step_m)
            if not cells:
                continue
            length_share = length_m / max(len(cells), 1)
            for cell in cells:
                bucket = stats.setdefault(cell, {f'{prefix}_segment_count': 0.0, f'{prefix}_length_m': 0.0})
                bucket[f'{prefix}_segment_count'] += 1.0
                bucket[f'{prefix}_length_m'] += length_share
    return _stats_to_frame(stats)


def _aggregate_geojson_polygons(path: Path, resolution: int, max_rows: int | None, *, prefix: str) -> pd.DataFrame:
    stats: dict[str, dict[str, float]] = {}
    for idx, feature in enumerate(_iter_geojson_features(path), start=1):
        if max_rows is not None and idx > max_rows:
            break
        geometry = shape(feature['geometry'])
        if geometry.is_empty:
            continue
        centroid = geometry.centroid
        cell = h3.latlng_to_cell(float(centroid.y), float(centroid.x), resolution)
        bucket = stats.setdefault(cell, {f'{prefix}_count': 0.0, f'{prefix}_area_m2': 0.0})
        bucket[f'{prefix}_count'] += 1.0
        bucket[f'{prefix}_area_m2'] += _approx_area_m2(geometry)
    return _stats_to_frame(stats)


def _aggregate_csv_polygons(path: Path, resolution: int, max_rows: int | None, *, prefix: str) -> pd.DataFrame:
    stats: dict[str, dict[str, float]] = {}
    processed = 0
    for chunk in pd.read_csv(path, chunksize=5_000):
        for row in chunk.itertuples(index=False):
            if max_rows is not None and processed >= max_rows:
                return _stats_to_frame(stats)
            processed += 1
            geometry_text = getattr(row, 'the_geom', None)
            if not isinstance(geometry_text, str) or not geometry_text:
                continue
            geometry = wkt.loads(geometry_text)
            if geometry.is_empty:
                continue
            centroid = geometry.centroid
            cell = h3.latlng_to_cell(float(centroid.y), float(centroid.x), resolution)
            bucket = stats.setdefault(
                cell,
                {
                    f'{prefix}_count': 0.0,
                    f'{prefix}_area_m2': 0.0,
                    f'{prefix}_height_sum': 0.0,
                },
            )
            bucket[f'{prefix}_count'] += 1.0
            bucket[f'{prefix}_area_m2'] += _safe_float(getattr(row, 'shape_area', 0.0)) or _approx_area_m2(geometry)
            bucket[f'{prefix}_height_sum'] += _safe_float(getattr(row, 'height_roof', 0.0))
    return _stats_to_frame(stats)


def _aggregate_geojson_points(path: Path, resolution: int, max_rows: int | None, *, prefix: str) -> pd.DataFrame:
    stats: dict[str, dict[str, float]] = {}
    for idx, feature in enumerate(_iter_geojson_features(path), start=1):
        if max_rows is not None and idx > max_rows:
            break
        geometry = shape(feature['geometry'])
        if geometry.is_empty:
            continue
        cell = h3.latlng_to_cell(float(geometry.y), float(geometry.x), resolution)
        bucket = stats.setdefault(cell, {f'{prefix}_point_count': 0.0})
        bucket[f'{prefix}_point_count'] += 1.0
    return _stats_to_frame(stats)


def _aggregate_csv_points(path: Path, resolution: int, max_rows: int | None, *, prefix: str) -> pd.DataFrame:
    stats: dict[str, dict[str, float]] = {}
    processed = 0
    for chunk in pd.read_csv(path, chunksize=20_000):
        for row in chunk.itertuples(index=False):
            if max_rows is not None and processed >= max_rows:
                return _stats_to_frame(stats)
            processed += 1
            geometry_text = getattr(row, 'the_geom', None)
            if not isinstance(geometry_text, str) or not geometry_text:
                continue
            geometry = wkt.loads(geometry_text)
            if geometry.is_empty:
                continue
            cell = h3.latlng_to_cell(float(geometry.y), float(geometry.x), resolution)
            bucket = stats.setdefault(cell, {f'{prefix}_point_count': 0.0, f'{prefix}_value_sum': 0.0})
            bucket[f'{prefix}_point_count'] += 1.0
            bucket[f'{prefix}_value_sum'] += _safe_float(getattr(row, 'elevation', 0.0))
    return _stats_to_frame(stats)


def _aggregate_pluto(path: Path, resolution: int, max_rows: int | None) -> pd.DataFrame:
    stats: dict[str, dict[str, float]] = {}
    processed = 0
    for chunk in pd.read_csv(path, chunksize=25_000):
        chunk = chunk.dropna(subset=['latitude', 'longitude'])
        for row in chunk.itertuples(index=False):
            if max_rows is not None and processed >= max_rows:
                return _stats_to_frame(stats)
            processed += 1
            cell = h3.latlng_to_cell(float(row.latitude), float(row.longitude), resolution)
            bucket = stats.setdefault(
                cell,
                {
                    'pluto_lot_count': 0.0,
                    'pluto_numfloors_sum': 0.0,
                    'pluto_bldgarea_sum': 0.0,
                    'pluto_unitstotal_sum': 0.0,
                },
            )
            bucket['pluto_lot_count'] += 1.0
            bucket['pluto_numfloors_sum'] += _safe_float(row.numfloors)
            bucket['pluto_bldgarea_sum'] += _safe_float(row.bldgarea)
            bucket['pluto_unitstotal_sum'] += _safe_float(row.unitstotal)
    return _stats_to_frame(stats)


def _merge_feature_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=['h3_cell'])
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on='h3_cell', how='outer')
    merged = merged.fillna(0.0)

    if 'pluto_lot_count' in merged.columns:
        denominator = merged['pluto_lot_count'].replace(0.0, 1.0)
        merged['pluto_mean_numfloors'] = merged['pluto_numfloors_sum'] / denominator
        merged['pluto_mean_bldgarea'] = merged['pluto_bldgarea_sum'] / denominator
        merged['pluto_mean_unitstotal'] = merged['pluto_unitstotal_sum'] / denominator
    else:
        merged['pluto_mean_numfloors'] = 0.0
        merged['pluto_mean_bldgarea'] = 0.0
        merged['pluto_mean_unitstotal'] = 0.0

    if 'building_count' in merged.columns:
        building_denominator = merged['building_count'].replace(0.0, 1.0)
        merged['building_mean_height'] = _series_or_zeros(merged, 'building_height_sum') / building_denominator
    else:
        merged['building_mean_height'] = 0.0

    merged['street_complexity'] = (_series_or_zeros(merged, 'street_segment_count') / 12.0).clip(0.0, 1.0)
    merged['building_intensity'] = (_series_or_zeros(merged, 'building_count') / 25.0).clip(0.0, 1.0)
    merged['building_height_proxy'] = (
        0.6 * (merged['pluto_mean_numfloors'] / 20.0).clip(0.0, 1.0)
        + 0.4 * (merged['building_mean_height'] / 80.0).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    merged['elevation_complexity'] = (_series_or_zeros(merged, 'elevation_point_count') / 40.0).clip(0.0, 1.0)

    street_length = _series_or_zeros(merged, 'street_length_m').replace(0.0, 1.0)
    sidewalk_length = _series_or_zeros(merged, 'sidewalk_length_m')
    merged['sidewalk_access_score'] = (sidewalk_length / street_length).clip(0.0, 1.0)
    merged['urban_clutter_index'] = (
        0.35 * merged['building_height_proxy']
        + 0.25 * merged['building_intensity']
        + 0.20 * merged['street_complexity']
        + 0.20 * merged['elevation_complexity']
    ).clip(0.0, 1.0)
    return merged.sort_values('h3_cell').reset_index(drop=True)


def _iter_geojson_features(path: Path) -> Iterator[dict[str, object]]:
    with path.open('rb') as handle:
        for feature in ijson.items(handle, 'features.item'):
            geometry = feature.get('geometry')
            if geometry is None:
                continue
            yield feature


def _download_is_complete(asset: UrbanContextAsset) -> bool:
    if not asset.raw_path().exists():
        return False
    state_path = asset.download_state_path()
    if not state_path.exists():
        return False
    try:
        state = json.loads(state_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return False
    return bool(state.get('complete'))


def _download_csv_pages(
    session: requests.Session,
    asset: UrbanContextAsset,
    target: Path,
    *,
    timeout_s: int,
) -> tuple[int, int]:
    temp_target = target.with_suffix(target.suffix + '.tmp')
    total_rows = 0
    page_count = 0
    header: str | None = None
    offset = 0

    with temp_target.open('w', encoding='utf-8', newline='') as handle:
        while True:
            page_url = _with_socrata_paging(asset.download_url, limit=SOCRATA_PAGE_SIZE, offset=offset)
            response = session.get(page_url, timeout=timeout_s)
            response.raise_for_status()
            text = response.text
            lines = text.splitlines()
            if not lines:
                break
            page_header = lines[0]
            data_lines = lines[1:]
            if header is None:
                header = page_header
                handle.write(page_header + '\n')
            elif page_header != header:
                raise RuntimeError(f'Urban context download header changed mid-stream for {asset.key}')

            if not data_lines:
                break

            handle.write('\n'.join(data_lines))
            handle.write('\n')
            total_rows += len(data_lines)
            page_count += 1
            if len(data_lines) < SOCRATA_PAGE_SIZE:
                break
            offset += SOCRATA_PAGE_SIZE

    temp_target.replace(target)
    return total_rows, page_count


def _with_socrata_paging(url: str, *, limit: int, offset: int) -> str:
    parsed = urlsplit(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query['$limit'] = str(limit)
    query['$offset'] = str(offset)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment))


def _line_cells_from_geometry(geometry: BaseGeometry, resolution: int, densify_step_m: float) -> tuple[tuple[str, ...], float]:
    coords: list[tuple[float, float]] = []
    total_length_m = 0.0
    if geometry.geom_type == 'LineString':
        coords = [(float(lat), float(lng)) for lng, lat in geometry.coords]
        total_length_m = _polyline_length_m(coords)
    elif geometry.geom_type == 'MultiLineString':
        parts = []
        for line in geometry.geoms:
            part = [(float(lat), float(lng)) for lng, lat in line.coords]
            if part:
                parts.append(part)
                total_length_m += _polyline_length_m(part)
        cells: list[str] = []
        seen: set[str] = set()
        for part in parts:
            for cell in polyline_to_h3_cells(part, resolution=resolution, step_m=densify_step_m):
                if cell not in seen:
                    seen.add(cell)
                    cells.append(cell)
        return tuple(cells), total_length_m
    else:
        return tuple(), 0.0

    if not coords:
        return tuple(), 0.0
    return tuple(polyline_to_h3_cells(coords, resolution=resolution, step_m=densify_step_m)), total_length_m


def _polyline_length_m(polyline: list[tuple[float, float]]) -> float:
    if len(polyline) < 2:
        return 0.0
    total = 0.0
    for idx in range(1, len(polyline)):
        total += haversine_m(polyline[idx - 1], polyline[idx])
    return total


def _approx_area_m2(geometry: BaseGeometry) -> float:
    if geometry.is_empty:
        return 0.0
    centroid = geometry.centroid
    lat_scale = 111_320.0
    lon_scale = lat_scale * math.cos(math.radians(float(centroid.y)))

    def _project_coords(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
        return [(float(lng) * lon_scale, float(lat) * lat_scale) for lng, lat in coords]

    if geometry.geom_type == 'Polygon':
        shell = _project_coords(list(geometry.exterior.coords))
        return abs(_shoelace_area(shell))
    if geometry.geom_type == 'MultiPolygon':
        return sum(_approx_area_m2(part) for part in geometry.geoms)
    return 0.0


def _shoelace_area(coords: list[tuple[float, float]]) -> float:
    if len(coords) < 3:
        return 0.0
    area = 0.0
    for idx in range(len(coords) - 1):
        x1, y1 = coords[idx]
        x2, y2 = coords[idx + 1]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def _stats_to_frame(stats: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for cell, values in stats.items():
        rows.append({'h3_cell': cell, **values})
    return pd.DataFrame(rows)


def _safe_float(value: object) -> float:
    try:
        if value is None or value == '':
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def asset_catalog_rows() -> list[dict[str, object]]:
    return [asdict(asset) for asset in ASSETS.values()]


def _series_or_zeros(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(0.0, index=frame.index)
