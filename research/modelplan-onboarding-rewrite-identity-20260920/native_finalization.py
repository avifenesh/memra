"""Shared final-result commit for native qualification runners; no GPU operations."""

import contextlib
import signal


class PublicationCancelled(RuntimeError):
    pass


@contextlib.contextmanager
def publication_guard(check):
    """Do not defer a cancellation through preparation/atomic publication of PASS."""
    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    previous = {sig: signal.getsignal(sig) for sig in signals}
    def interrupted(signum, frame):
        handler = previous[signum]
        try:
            if callable(handler):
                handler(signum, frame)  # Preserve the outer controller's cancellation state.
        finally:
            raise PublicationCancelled(f'result publication cancelled by signal {signum}')
    try:
        for sig in signals:
            signal.signal(sig, interrupted)
        check()
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def publish_result(out, result, *, writer, check_cancelled=lambda: None):
    pending = out / 'result.json.pending'
    if result['status'] != 'passed':
        writer(pending, result)
        pending.replace(out / 'result.json')
        return
    try:
        with publication_guard(check_cancelled):
            writer(pending, result)
            check_cancelled()  # Includes cancellation received while writing the pending file.
            pending.replace(out / 'result.json')
    except PublicationCancelled as error:
        # If a signal interrupted the atomic operation after replacement, do not
        # leave a stale PASS as the final observable result of the cancelled run.
        failed = {**result, 'status': 'failed', 'publication_error': str(error)}
        writer(pending, failed)
        pending.replace(out / 'result.json')
        raise


def write_evidence_manifest(out, *, writer, digest):
    excluded = {out / 'files-sha256.json', out / 'result.json', out / 'result.json.pending'}
    writer(out / 'files-sha256.json', {str(path.relative_to(out)): digest(path)
          for path in sorted(out.rglob('*')) if path.is_file() and path not in excluded})
    return digest(out / 'files-sha256.json')


def finalize_evidence(out, telemetry, telemetry_log, verify, *, writer, digest, cleanups=()):
    errors = []
    if telemetry is not None:
        try:
            telemetry.terminate()
            telemetry.wait(timeout=15)
        except BaseException as error:
            errors.append(f'telemetry cleanup: {type(error).__name__}: {error}')
            try:
                telemetry.kill()
                telemetry.wait(timeout=15)
            except BaseException as reap_error:
                errors.append(f'telemetry kill/reap: {type(reap_error).__name__}: {reap_error}')
    for cleanup in cleanups:
        try:
            cleanup()
        except BaseException as error:
            errors.append(f'artifact cleanup: {type(error).__name__}: {error}')
    try:
        telemetry_log.close()
    except BaseException as error:
        errors.append(f'telemetry log close: {type(error).__name__}: {error}')
    manifest_sha = None
    try:
        manifest_sha = write_evidence_manifest(out, writer=writer, digest=digest)
    except BaseException as error:
        errors.append(f'evidence manifest: {type(error).__name__}: {error}')
    try:
        verify()
    except BaseException as error:
        errors.append(f'final invariants: {type(error).__name__}: {error}')
    return errors, manifest_sha
