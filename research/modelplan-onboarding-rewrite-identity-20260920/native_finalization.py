"""Shared final-result commit for native qualification runners; no GPU operations."""

def publish_result(out, result, *, writer):
    # An interrupted/failed final write leaves the earlier incomplete receipt intact.
    pending = out / 'result.json.pending'
    writer(pending, result)
    pending.replace(out / 'result.json')


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
        excluded = {out / 'files-sha256.json', out / 'result.json', out / 'result.json.pending'}
        writer(out / 'files-sha256.json', {str(path.relative_to(out)): digest(path)
              for path in sorted(out.rglob('*')) if path.is_file() and path not in excluded})
        manifest_sha = digest(out / 'files-sha256.json')
    except BaseException as error:
        errors.append(f'evidence manifest: {type(error).__name__}: {error}')
    try:
        verify()
    except BaseException as error:
        errors.append(f'final invariants: {type(error).__name__}: {error}')
    return errors, manifest_sha
