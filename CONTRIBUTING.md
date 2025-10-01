# Contributing to X-FORECAST

## Development Setup

1. Fork the repository
2. Clone your fork: `git clone <your-fork-url>`
3. Create virtual environment: `python -m venv venv`
4. Activate environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Unix)
5. Install dependencies: `pip install -r requirements.txt`
6. Install in development mode: `pip install -e .`

## Code Standards

- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Write unit tests for new features
- Keep functions focused and modular

## Testing

Run tests before submitting:
```bash
python -m pytest tests/ -v
```

## Pull Request Process

1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes and add tests
3. Ensure all tests pass
4. Update documentation if needed
5. Submit pull request with clear description

## Reporting Issues

- Use GitHub issues for bug reports
- Include steps to reproduce
- Provide sample data if relevant
- Specify Python version and OS