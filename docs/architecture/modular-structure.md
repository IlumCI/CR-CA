# Modular Structure

CR-CA's modular architecture enables flexible composition and extension.

## Module Organization

```
CR-CA/
├── CRCA.py              # Core agent
├── utils/               # Utilities
├── templates/          # Agent templates
├── schemas/            # Data schemas
├── branches/           # Specialized branches
├── features/           # Feature modules
└── architecture/       # Architecture components
```

## Module Dependencies

Modules follow a dependency hierarchy:

$$Core \to Utils \to Templates \to Branches$$

Where each layer depends only on lower layers.

## Mathematical Foundation

Modules communicate through well-defined interfaces:

$$f: M_1 \times M_2 \times \cdots \times M_n \to O$$

Where $M_i$ are input modules and $O$ is the output.

## Next Steps

- [Hybrid Agent](hybrid-agent/overview.md) - Hybrid agent architecture
- [Causal Graphs](causal-graphs.md) - Causal graph architecture
