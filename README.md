<div align="center">
  <img src="spice_logo.png" alt="spice" width="500">
  <h1>Spice Personal — Your Decision Assistant</h1>
  <p>
    <a href="https://pypi.org/project/spice-personal/"><img src="https://img.shields.io/pypi/v/spice-personal" alt="PyPI"></a>
    <img src="https://img.shields.io/badge/python-≥3.11-blue" alt="Python">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
    <a href="https://discord.gg/DajVWWNMfE"><img src="https://img.shields.io/badge/Discord-Community-5865F2?style=flat&logo=discord&logoColor=white" alt="Discord"></a>
  </p>
</div>


## 🧭 What is Spice Personal?

Spice Personal is a reference app built on top of:

👉 **[Spice Runtime](https://github.com/Dyalwayshappy/spice)**

It helps you think through decisions — and optionally take action via external agents.

---



## 👉 Try it in seconds:

**Install from source (latest features, for development)**

```bash
# Clone both repos (Spice Personal depends on spice-runtime)
git clone https://github.com/Dyalwayshappy/Spice.git
git clone https://github.com/Dyalwayshappy/Spice_personal.git

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -U pip
pip install -e ./Spice
pip install -e ./Spice_personal
```

**Install from PyPI (stable, recommended)**

```bash
pip install spice-personal
```

This will automatically install spice-runtime.


##  Upgrade to latest version

```bash
pip install -U spice-personal
spice-personal --version
```

### Verify installation

```bash
spice-personal --help
```





---

## 🚀 Quick Start

Spice is a decision-layer runtime.

The easiest way to try Spice is through the reference application: **Spice Personal**.


### 1. Initialize your workspace

```bash
spice-personal init
```

This creates a local workspace at:
> .spice/personal/
and generates a default configuration file.




### 2. Initialize your workspace

```bash
spice-personal init
```

This creates a local personal workspace and generates the default configuration.



### 3. Ask your first question

```bash
spice-personal ask "What should I do next?"
```
Since no model is configured yet, you will see a Decision Card guiding you to the next step:

Setup required (no model configured)

Next:

-> Edit .spice/personal/personal.config.json

-> Then run: spice-personal ask "What should I do next?"



### 4. Connect a model
Edit the generated config file:
> .spice/personal/personal.config.json

Configure your model provider (e.g. OpenRouter) and set your API key:

```bash
export OPENROUTER_API_KEY=...
```


### 5. Run your intent
```bash
spice-personal ask "your intent"
```
Now Spice will produce a real decision, not just a setup guide.

### 6. (Optional) Interactive mode
```bash
spice-personal session
```

### 7. (Optional) Connect external agents

Spice can delegate actions to external agents (e.g. Claude Code, Codex).

This enables:

- gathering real-world evidence
- executing tasks based on decisions
- closing the loop from decision → action

  
To enable this, configure your agent in:
> .spice/personal/personal.config.json


This is where Spice moves beyond reasoning — into action.

Now Spice can:

- search for relevant information

- call external tools(Currently supports wrappers for CodeX and ClaudeCode.)

- and make decisions grounded in real-world signals





## 📁 Project Structure

```
spice-personal/
├── spice_personal/            # 🧭 Personal reference app/CLI
│   ├── cli/                   #    User-facing commands (ask/init/session)
│   ├── app/                   #    Personal orchestration flow
│   ├── advisory/              #    Personal decision/advisory logic
│   ├── execution/             #    Execution intent + evidence round logic
│   ├── executors/             #    Personal executor wiring/factory
│   ├── wrappers/              #    External agent/model wrappers
│   ├── provider_bridges/      #    Provider bridge layer
│   ├── config/                #    Workspace/env config resolution
│   ├── profile/               #    Profile contract + validation
│   ├── domain/                #    Built-in personal domain assets
│   └── tests/                 # ✅ Personal test suite
├── pyproject.toml             # 📦 spice-personal package metadata
├── README.md                  # 📝 Personal product quick start
├── LICENSE                    # ⚖️ MIT
└── .gitignore                 # 🙈 Ignore rules

```

--- 









## ⭐ Star History

<div align="center">
  <a href="https://star-history.com/#Dyalwayshappy/spice_personal&Date">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Dyalwayshappy/spice_personal&type=Date&theme=dark" />
      <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Dyalwayshappy/spice_personal&type=Date" />
      <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Dyalwayshappy/spice_personal&type=Date" style="border-radius: 15px; box-shadow: 0 0 30px rgba(0, 217, 255, 0.3);" />
    </picture>
  </a>
</div>

<p align="center">
  <em>⭐ Star us if you find Spice interesting</em><br><br>
  <img src="https://visitor-badge.laobi.icu/badge?page_id=Dyalwayshappy.spice_personal&style=for-the-badge&color=00d4ff" alt="Views">
</p>


<p align="center">
  <sub>Everyone should have a Spice — a decision brain for thinking and action.</sub>
</p>
