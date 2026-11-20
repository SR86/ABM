"""ABM paper reproduction entry point; original draft: tmp/original/abm_experiment.py."""
import sys

from abm.models import ResNet18
from abm.attacks import BackdoorAttack, BadNetsAttack, BlendAttack
from abm.method import ABM, NBAB, evaluate_model
from abm.runner import main

if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
