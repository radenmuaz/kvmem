"""
kvmem/build_standalone.py — bundles kvmem/hmn.py plus whatever internal
kvmem.* modules it imports (currently just kvmem/structured_data.py, found
by scanning `from kvmem.X import ...` lines rather than hardcoded, so this
keeps working if hmn.py grows more internal dependencies later) into ONE
dependency-free .py file, optionally with a config's hp dict embedded — for
copying to a server that doesn't have (or shouldn't need) the `kvmem`
package/repo layout, no PYTHONPATH or `-m kvmem.hmn` package resolution
required.

The bundle's own --device default is 'cuda' (these bundles are built for
copying to a server; override with --default-device cpu/mps at BUILD time
for a bundle meant for local use instead).

Usage:
    # bundle hmn.py + structured_data.py only, config passed at run time:
    python3 -m kvmem.build_standalone --out standalone_hmn.py

    # also embed one config's hp dict as the default (still overridable):
    python3 -m kvmem.build_standalone --config kvmem/configs/hmn_squeeze_sweetspot_n4.py --out standalone_hmn.py

    # then, on the server, with ONLY standalone_hmn.py copied over
    # (--device defaults to cuda already, no flag needed):
    python3 standalone_hmn.py
    # or, to use a DIFFERENT config (if you also copy one over):
    python3 standalone_hmn.py --config some_other_config.py
    # override the device if needed:
    python3 standalone_hmn.py --device cpu

Not itself imported by kvmem/hmn.py or any training path — this is a
one-off build tool, run manually when you actually need a standalone file,
not part of the normal training/eval flow.
"""
from __future__ import annotations

import argparse
import importlib.util
import pprint
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_IMPORT_RE = re.compile(r'^from kvmem\.(\w+) import [^\n]*\n?', re.MULTILINE)
_FUTURE_IMPORT_RE = re.compile(r'^from __future__ import [^\n]*\n?', re.MULTILINE)


def _module_path(name: str) -> Path:
    return REPO_ROOT / 'kvmem' / f'{name}.py'


def _resolve_bundle_order(entry_module: str) -> list[str]:
    """Topological order: dependencies before dependents. DFS post-order
    over `from kvmem.X import ...` lines, entry_module last."""
    visited: set[str] = set()
    order: list[str] = []

    def _visit(name: str):
        if name in visited:
            return
        visited.add(name)
        src = _module_path(name).read_text()
        for dep in _IMPORT_RE.findall(src):
            if dep != name:
                _visit(dep)
        order.append(name)

    _visit(entry_module)
    return order


def _load_config_hp(config_path: str) -> dict:
    spec = importlib.util.spec_from_file_location('_kvmem_bundle_cfg', config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'hp'):
        raise ValueError(f'{config_path!r} must define a module-level `hp` dict')
    return dict(module.hp)


def build(entry_module: str = 'hmn', config_path: str | None = None,
         out_path: str = 'standalone_hmn.py', default_device: str = 'cuda') -> None:
    order = _resolve_bundle_order(entry_module)

    parts: list[str] = []
    parts.append('#!/usr/bin/env python3')
    parts.append('"""')
    parts.append('AUTO-GENERATED standalone bundle - DO NOT EDIT DIRECTLY.')
    parts.append(f'Built by kvmem/build_standalone.py from: {", ".join(f"kvmem/{m}.py" for m in order)}')
    if config_path:
        parts.append(f'Embedded default config: {config_path}')
    parts.append('Regenerate with: python3 -m kvmem.build_standalone ...')
    parts.append('"""')
    # `from __future__ import annotations` must be the first statement in
    # the FILE (Python disallows it mid-file) — every source module below
    # has its own copy stripped out and this single one emitted instead.
    parts.append('from __future__ import annotations')
    parts.append('')

    entry_src = None
    for name in order:
        src = _module_path(name).read_text()
        # Strip intra-package imports — every kvmem.* module this bundle
        # depends on is concatenated ABOVE this point already (topological
        # order), so the symbols are already in scope without an import.
        src = _IMPORT_RE.sub('', src)
        src = _FUTURE_IMPORT_RE.sub('', src)
        if name == entry_module:
            entry_src = src
            continue
        parts.append(f'# {"=" * 20} kvmem/{name}.py {"=" * 20}')
        parts.append(src)

    assert entry_src is not None
    # Strip the entry module's own `if __name__ == '__main__': main()` —
    # replaced below with a combined entry point that supports an embedded
    # default config in addition to the original --config flag.
    entry_src = re.sub(
        r"\n\nif __name__ == '__main__':\n    main\(\)\n?$", '\n', entry_src)
    # This bundle is built for copying to a server — default --device to
    # `default_device` (cuda by default) instead of hmn.py's own local-dev
    # default (cpu), in BOTH the entry module's own main() (used when no
    # config is embedded, or via --config override) and the wrapper below.
    entry_src = entry_src.replace(
        "p.add_argument('--device',     default='cpu')",
        f"p.add_argument('--device',     default={default_device!r})")
    parts.append(f'# {"=" * 20} kvmem/{entry_module}.py {"=" * 20}')
    parts.append(entry_src)

    if config_path:
        hp = _load_config_hp(config_path)
        parts.append(f'# {"=" * 20} embedded config: {config_path} {"=" * 20}')
        parts.append(f'_EMBEDDED_HP = {pprint.pformat(hp, sort_dicts=False)}')

    parts.append(f'# {"=" * 20} combined CLI entry point {"=" * 20}')
    if config_path:
        parts.append(f'''
if __name__ == '__main__':
    _p = argparse.ArgumentParser()
    _p.add_argument('--config', default=None,
                    help='optional: path to a different config.py (must define `hp`); '
                         'defaults to the config embedded at build time')
    _p.add_argument('--device', default={default_device!r})
    _p.add_argument('--pretrained', default=None)
    _p.add_argument('--log-dir', default='logs')
    _args = _p.parse_args()

    hp = load_config(_args.config) if _args.config else dict(_EMBEDDED_HP)
    if _args.pretrained:
        hp['_pretrained_ckpt'] = _args.pretrained
    train(hp, log_base=_args.log_dir, device_str=_args.device)
'''.strip('\n'))
    else:
        parts.append("if __name__ == '__main__':\n    main()")

    out = '\n\n'.join(parts) + '\n'
    Path(out_path).write_text(out)
    print(f'wrote {out_path} ({len(out):,} bytes, modules: {", ".join(order)}'
         f'{f", config: {config_path}" if config_path else ""})')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--entry', default='hmn', help='module name (without .py) to bundle, default hmn')
    p.add_argument('--config', default=None, help='optional config.py to embed as the default hp')
    p.add_argument('--out', default='standalone_hmn.py')
    p.add_argument('--default-device', default='cuda',
                   help="--device default baked into the bundle's own CLI (default 'cuda', "
                        "since these bundles are built for copying to a server; override with "
                        "'cpu'/'mps' for a bundle meant for local use)")
    args = p.parse_args()
    build(entry_module=args.entry, config_path=args.config, out_path=args.out,
         default_device=args.default_device)


if __name__ == '__main__':
    main()
