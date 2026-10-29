"""Numerical contracts for the saved-product notebook's small helpers."""

import ast
from datetime import datetime, timedelta
import json
from pathlib import Path

import numpy as np
import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / 'examples/Alt_probe_L1_demo.ipynb'
STATES = ('template_background', 'positive', 'negative', 'target_science')
namespace = {'np': np, 'STATES': STATES}
for cell in json.loads(NOTEBOOK.read_text())['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(line for line in cell['source'] if not line.startswith('%'))
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name in ('mean_frames', 'assign_states'):
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(NOTEBOOK), 'exec'), namespace)


@pytest.mark.parametrize('number', [1, 3])
def test_online_mean_and_coverage_exclude_masks_and_nonfinite(number):
    values = np.array([[[1., np.nan, 8.], [4., 5., 6.]],
                       [[3., 4., 9.], [np.nan, 8., 10.]],
                       [[5., 6., 10.], [12., 11., 14.]]])[:number]
    masks = np.array([[[0, 0, 1], [0, 1, 0]], [[0, 0, 1], [0, 1, 0]],
                      [[1, 0, 1], [0, 1, 0]]], dtype=bool)[:number]
    rows = [dict(combine_key=('same level/visit/state/camera',), image=np.ma.array(v, mask=m))
            for v, m in zip(values, masks)]
    before = values.copy(), masks.copy()
    namespace['load_crop'] = lambda row: row['image']
    result, count = namespace['mean_frames'](rows)
    valid = np.isfinite(values) & ~masks
    expected_count = valid.sum(axis=0)
    total = np.where(valid, values, 0).sum(axis=0)
    expected = np.divide(total, expected_count, out=np.zeros_like(total), where=expected_count != 0)
    np.testing.assert_array_equal(count, expected_count)
    np.testing.assert_array_equal(np.ma.getmaskarray(result), count == 0)
    np.testing.assert_allclose(result.data[count > 0], expected[count > 0])
    np.testing.assert_array_equal(values, before[0])
    np.testing.assert_array_equal(masks, before[1])
    with pytest.raises(ValueError, match='homogeneous'):
        namespace['mean_frames'](rows + [dict(rows[0], combine_key=('different camera',))])


def test_current_science_only_and_narrow_legacy_background_inference():
    def row(index, satspots, sign=None):
        time = datetime(2026, 12, 20) + timedelta(seconds=index)
        return dict(visitid='visit', sctsrt=time, ftimeutc=time,
                    satspots=satspots, sign=sign, path=Path('synthetic.fits'))

    science = [row(3, 0)]
    namespace['assign_states'](science, {})
    assert science[0]['state'] == 'target_science'
    legacy = [row(0, 0), row(1, 1, 'positive'), row(2, 1, 'negative'), row(3, 0)]
    known = {}
    namespace['assign_states'](legacy, known)
    assert [r['state'] for r in legacy] == list(STATES)
    background_only = [row(0, 0)]
    namespace['assign_states'](background_only, known)
    assert background_only[0]['state'] == 'template_background'
