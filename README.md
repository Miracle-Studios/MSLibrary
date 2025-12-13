<div align="center">

# 🚀 MSLib

**A comprehensive plugin development framework for exteraGram**

[![Version](https://img.shields.io/badge/version-1.1.0--beta-blueviolet.svg)](https://github.com/Miracle-Studios/MSLib/releases)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)]()
[![Telegram](https://img.shields.io/badge/Telegram-@MiracleStudios-0088cc.svg)](https://t.me/MiracleStudios)

</div>

---

## 📌 Overview

**MSLib** is a sophisticated toolkit designed to streamline and accelerate **exteraGram** plugin development. It provides developers with powerful abstractions, utilities, and built-in enhancements that eliminate boilerplate code and enable rapid prototyping of feature-rich Telegram extensions.

Whether you're building simple command handlers or complex plugin ecosystems, MSLib offers the architectural foundations needed for scalable and maintainable plugin development.

## ✨ Key Features

- **🎯 Command Framework** — Declarative command registration with automatic routing and help generation
- **🔄 Auto-Update System** — Seamless plugin updates with intelligent version management and change detection
- **📦 Smart Caching** — High-performance data persistence with compression and serialization support
- **🌍 Internationalization** — Built-in localization for Russian and English with extensible language support
- **🎨 UI Components Library** — Pre-built settings components for consistent plugin configuration interfaces
- **🔌 Integrated Plugins** — Collection of production-ready Telegram improvements (hashtag fixes, call confirmations, etc.)

## 🚀 Quick Start

### Basic Command Handler

```python
from MSLib import command, send_message

@command("hello", "Greet the user")
def hello_handler(message):
    send_message(message.peer_id, "Hello, world! 👋")
```

### Working with Cache

```python
from MSLib import CacheFile

cache = CacheFile("user_data.json", compress=True)
cache.write({"user_id": 123, "preferences": {"theme": "dark"}})
data = cache.read()
```

### Settings UI

```python
from MSLib import Header, Switch

def create_settings(self):
    return [
        Header(text="Plugin Settings"),
        Switch(
            key="feature_enabled",
            text="Enable Feature",
            subtext="Toggle this feature on/off",
            default=True
        )
    ]
```

## 📚 Documentation

Comprehensive guides and API documentation:

| Resource | Description |
|----------|-------------|
| [Getting Started](docs/getting-started.md) | Installation and first steps |
| [API Reference](docs/api-reference.md) | Complete class and function documentation |
| [Integrated Plugins](docs/integrated-plugins.md) | Built-in plugin features |
| [Commands Guide](docs/commands.md) | Command system deep dive |
| [Caching & Storage](docs/cache-storage.md) | Data persistence strategies |

## 🛠️ Requirements

- **exteraGram** 12.0.0 or higher
- **Python** 3.11+

## 💡 Why MSLib?

- **Reduces Boilerplate** — Focus on business logic, not infrastructure
- **Production-Ready** — Battle-tested utilities and patterns
- **Type-Safe** — Clear APIs with predictable behavior
- **Well-Documented** — Comprehensive guides with examples
- **Active Development** — Regular updates and community support

## 🤝 Contributing

We welcome contributions! Please check our [issues page](https://github.com/Miracle-Studios/MSLib/issues) for areas where you can help.

## 📄 License

This project is licensed under the **BSD 3-Clause License** - see the [LICENSE](LICENSE) file for details.

```
Copyright (c) 2024, Miracle Studios
All rights reserved.
```

---

<div align="center">

**v1.1.0-beta** | Crafted with ❤️ by [Miracle Studios](https://t.me/MiracleStudios)  
[Join our channel →](https://t.me/MiracleStudios)

</div>