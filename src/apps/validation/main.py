"""Entry point for the T1–T7 validator app.

Usage:
    python -m rouse_model_python.src.apps.validation.main \\
        --device cpu --is_parallel false --force \\
        --output_dir validation_output
"""

import sys

from rouse_model_python.src.apps.validation.t1_t7_validator_app import T1T7ValidatorApp


def main(argv=None) -> int:
    return T1T7ValidatorApp(argv=argv).run()


if __name__ == "__main__":
    sys.exit(main())
