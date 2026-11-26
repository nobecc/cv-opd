import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from finance_opd.launch import prepare_training

class TrainingHoldTests(unittest.TestCase):
    def test_owned_hold_blocks_before_preflight_or_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / 'owned-run'
            (run / 'configs').mkdir(parents=True)
            (run / 'configs/TRAINING_HOLD.json').write_text('{"reason":"admission method handoff"}')
            output = run / 'training'
            with patch('finance_opd.launch.doctor') as doctor:
                with self.assertRaisesRegex(RuntimeError, 'screening-only'):
                    prepare_training({'output': str(output)})
                doctor.assert_not_called()
            self.assertFalse(output.exists())

    def test_other_run_hold_does_not_block_new_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'old/configs').mkdir(parents=True)
            (root / 'old/configs/TRAINING_HOLD.json').write_text('{}')
            output = root / 'new/training'
            config = {'output': str(output), 'slime': '/fixture/slime'}
            with patch('finance_opd.launch.training_arguments', return_value=[]), \
                 patch('finance_opd.launch.doctor', return_value={'training_ready': True}), \
                 patch('finance_opd.launch.environment', return_value={}):
                command, env = prepare_training(config)
            self.assertTrue((output / 'run_config.json').exists())
            self.assertEqual(env['FINANCE_OPD_CONFIG'], str(output / 'run_config.json'))

if __name__ == '__main__':
    unittest.main()
