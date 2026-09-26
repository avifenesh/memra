"""Deterministic source containers; file contents retain their recorded timestamps."""
import gzip
from pathlib import PurePosixPath
import tarfile


def write_archive(destination, files):
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for name, path in sorted(files.items()):
                    logical = PurePosixPath(name)
                    if logical.is_absolute() or ".." in logical.parts or str(logical) != name:
                        raise ValueError("noncanonical archive member")
                    member = archive.gettarinfo(str(path), arcname=name)
                    if not member.isfile():
                        raise ValueError("source archive only accepts regular files")
                    member.uid = member.gid = 0
                    member.uname = member.gname = ""
                    member.mtime = 0
                    member.mode = 0o755 if member.mode & 0o111 else 0o644
                    member.pax_headers = {}
                    with path.open("rb") as stream:
                        archive.addfile(member, stream)
