# Spice Personal

Spice Personal is a reference personal decision app/CLI built on top of Spice runtime.

`spice-personal` depends on `spice-runtime`.

## Install

```bash
pip install spice-personal
```

This automatically installs `spice-runtime>=0.1.0,<0.2.0`.

## Quick Start

```bash
spice-personal ask "Should I quit my job?"
```

Optional:

```bash
spice-personal init
spice-personal session
```

## CLI Entry Points

- `spice-personal`
- `spice-model-openrouter`
- `spice-agent-codex`
- `spice-agent-claude-code`

## Repo Layout

- `spice_personal/`: product/CLI code and tests

## License

MIT
