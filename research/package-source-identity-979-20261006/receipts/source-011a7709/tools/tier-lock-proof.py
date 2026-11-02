#!/usr/bin/env python3
"""Verify an inherited canonical flock FD without unlocking it or trusting env state.

This is a cooperative process-lifetime proof, not authentication against a hostile
same-user process. No third lock path and no environment switch are accepted.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import stat
import sys

LOCKS = ('/tmp/memra-5090.lock', '/tmp/memra-gpu.lock')


def verify(fd, path, owner='collector'):
    if path not in LOCKS or fd < 3:
        raise ValueError('canonical lock path and inherited non-stdio FD required')
    actual = os.fstat(fd)
    canonical = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(canonical.st_mode) or (actual.st_dev, actual.st_ino) != (canonical.st_dev, canonical.st_ino):
        raise ValueError('inherited FD is not the canonical lock inode')
    # An independent open must be excluded. Acquiring an unlocked inherited FD
    # alone would silently bless a descriptor that never carried a lock.
    with Path(path).open('a') as independent:
        try:
            fcntl.flock(independent, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(independent, fcntl.LOCK_UN)
            raise ValueError('canonical inode is not locked')
        # The inherited open-file description must own that exclusion, not an
        # unrelated process/descriptor. Reasserting its lock does not release it.
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('inherited FD does not own the canonical flock') from error
    return {'owner': owner, 'lock': path, 'mechanism': 'inherited-flock-same-open-description',
            'device': actual.st_dev, 'inode': actual.st_ino, 'qualification': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fd', required=True, type=int)
    parser.add_argument('--lock', required=True, choices=LOCKS)
    parser.add_argument('--owner', choices=['collector', 'internal-canonical'], default='collector')
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.fd, args.lock, args.owner)))
    except (OSError, ValueError) as error:
        print('REFUSED: ' + str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
