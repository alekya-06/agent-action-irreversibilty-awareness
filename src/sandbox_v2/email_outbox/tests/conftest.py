
import os
import sys

EMAIL_OUTBOX_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if EMAIL_OUTBOX_DIR not in sys.path:
    sys.path.insert(0, EMAIL_OUTBOX_DIR)
