"""Regression checks for adaptation provenance and one-time statistics caching."""
import argparse
import contextlib
import io
import json
import multiprocessing as mp
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch
from rift import engine
from rift.data import source_statistics


class TinyCity:
    city, days, day_steps, population, bound = 'fixture', 1, 2, 1, 200.
    array = np.array([[[10.]], [[20.]]], dtype=np.float32)

    def __init__(self, calls=None):
        self.calls = calls

    def values(self, begin, end, source=False):
        if self.calls is not None:
            with self.calls.get_lock():
                self.calls.value += 1
        return self.array[begin:end, :, 0]

    def interval(self, split):
        return (20, 40)


def slow_dump(value, stream, **kwargs):
    text = json.dumps(value, **kwargs)
    stream.write(text[:len(text)//2])
    stream.flush()
    time.sleep(.1)
    stream.write(text[len(text)//2:])


def cache_worker(directory, barrier, calls, queue):
    barrier.wait(timeout=60)
    with patch('rift.data.json.dump', side_effect=slow_dump):
        queue.put(source_statistics(TinyCity(calls), directory))


class ReleaseStateTests(unittest.TestCase):
    def test_parallel_cache_publishes_one_complete_release(self):
        ctx = mp.get_context('spawn')
        calls, queue, barrier = ctx.Value('i', 0), ctx.Queue(), ctx.Barrier(4)
        with tempfile.TemporaryDirectory() as directory:
            workers = [ctx.Process(target=cache_worker, args=(directory, barrier, calls, queue)) for _ in range(4)]
            path = Path(directory)/'fixture.json'
            try:
                for worker in workers:
                    worker.start()
                deadline = time.monotonic() + 90
                while any(worker.is_alive() for worker in workers):
                    if path.exists():
                        json.loads(path.read_text(encoding='utf-8'))
                    self.assertLess(time.monotonic(), deadline, 'Cache workers timed out')
                    time.sleep(.01)
                for worker in workers:
                    worker.join()
                    self.assertEqual(worker.exitcode, 0)
                values = [queue.get(timeout=5) for _ in workers]
                self.assertTrue(all(value == values[0] for value in values))
                self.assertEqual(calls.value, 2)  # One mean pass and one second-moment pass.
                self.assertEqual(json.loads(path.read_text())['statistics'], values[0])
                self.assertEqual(list(Path(directory).glob('*.tmp')), [])
                with patch.object(TinyCity, 'values', side_effect=AssertionError('Cache should be reused')):
                    self.assertEqual(source_statistics(TinyCity(), directory), values[0])
            finally:
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                        worker.join()
                queue.close()

    def test_failed_write_and_corrupt_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'fixture.json'
            with patch('rift.data.json.dump', side_effect=OSError('Injected write failure')):
                with self.assertRaises(OSError):
                    source_statistics(TinyCity(), directory)
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])
            source_statistics(TinyCity(), directory)  # Failed writer released its lock.
            path.write_text('{broken', encoding='utf-8')
            with patch.object(TinyCity, 'values', side_effect=AssertionError('Do not regenerate')):
                with self.assertRaises(json.JSONDecodeError):
                    source_statistics(TinyCity(), directory)
            self.assertEqual(path.read_text(), '{broken')

    def test_readaptation_records_actual_config_and_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            parent, output = Path(directory)/'parent.pt', Path(directory)/'adapted'
            original = dict(seed=7, target_days=3, batch_size=2, adapt_epochs=50,
                            data_root='old_data', split='dev', smoke=True)
            model = torch.nn.Linear(1, 1)
            engine.save_checkpoint(parent, model, 'fixture', [0., 1.], original, 'before_adaptation')
            args = argparse.Namespace(command='adapt', checkpoint=str(parent), output=str(output),
                                      adapt_epochs=100, data_root='new_data', split='test', device='cpu', threads=2)
            with patch.object(engine, 'CityData', return_value=TinyCity()), \
                 patch.object(engine, 'make_model', return_value=model), \
                 patch.object(engine, 'Windows', return_value=[(torch.zeros(1), torch.zeros(1))]), \
                 patch.object(engine, 'adapt') as adaptation, \
                 patch.object(engine, 'evaluate', return_value={'split': 'test', 'interval': [20, 40]}), \
                 contextlib.redirect_stdout(io.StringIO()):
                engine.from_checkpoint(args)
            self.assertEqual(adaptation.call_args.args[-1], 100)
            saved = torch.load(output/'target.pt', weights_only=True)
            for key, value in vars(args).items():
                self.assertEqual(saved['config'][key], value)
            self.assertEqual(saved['config']['evaluation_interval'], [20, 40])
            self.assertEqual(saved['parent_config'], original)
            self.assertEqual(saved['parent_checkpoint'], str(parent.resolve()))
            self.assertEqual(json.loads((output/'config.json').read_text()), saved['config'])
            self.assertEqual(torch.load(parent, weights_only=True)['config'], original)


if __name__ == '__main__':
    unittest.main()
