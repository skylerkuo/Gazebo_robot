#!/usr/bin/env python3
"""Create a local SFP0 configuration from a public JSON template."""
import argparse
import json
from pathlib import Path
import re

TEMPLATES = Path(__file__).resolve().parents[1] / 'configs/public'
PATH_SLOT = re.compile(r'\{\{([A-Z][A-Z0-9_]*)\}\}')


def render_config(template, paths):
    required = set()

    def substitute(value):
        if isinstance(value, dict):
            return {key: substitute(item) for key, item in value.items()}
        if isinstance(value, list):
            return [substitute(item) for item in value]
        if isinstance(value, str) and (match := PATH_SLOT.fullmatch(value)):
            key = match[1]
            required.add(key)
            return str(paths[key].expanduser().resolve()) if key in paths else value
        return value

    result = substitute(template)
    missing = required - paths.keys()
    unknown = paths.keys() - required
    if missing:
        raise ValueError('Missing path values: ' + ', '.join(sorted(missing)))
    if unknown:
        raise ValueError('Unused path values: ' + ', '.join(sorted(unknown)))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('template', choices=sorted(p.stem for p in TEMPLATES.glob('*.json')))
    parser.add_argument('--set', action='append', default=[], metavar='KEY=PATH',
                        help='Set a path required by the template. Repeat for additional paths.')
    parser.add_argument('--output', required=True, type=Path,
                        help='Write to a new JSON file. Existing files are preserved.')
    args = parser.parse_args(argv)
    paths = {}
    try:
        for assignment in args.set:
            key, separator, value = assignment.partition('=')
            if not separator or not value.strip() or key in paths:
                raise ValueError('Expected a single nonempty KEY=PATH value per setting')
            paths[key] = Path(value)
        template = json.loads((TEMPLATES / f'{args.template}.json').read_text())
        config = render_config(template, paths)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as output:
            json.dump(config, output, indent=2)
            output.write('\n')
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
