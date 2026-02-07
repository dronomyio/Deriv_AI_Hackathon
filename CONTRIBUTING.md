# Contributing to CortexON

Welcome to the CortexON community! We're thrilled that you're interested in contributing to our project. This document provides guidelines and instructions to help you contribute effectively. Every contribution matters, whether it's fixing a typo, improving documentation, or implementing new features.

## Table of Contents
- [Introduction](#introduction)
- [How to Contribute](#how-to-contribute)
- [Code Style & Standards](#code-style--standards)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Code of Conduct](#code-of-conduct)
- [Additional Resources](#additional-resources)

## Introduction

The project aims to democratize a generalized AI agent, and make capabilities of the likes of OpenAI DeepResearch and ManusAI open-source for all. We believe in the power of community-driven development and welcome contributions from developers of all skill levels and backgrounds. Your contributions help make this project better for everyone.

## How to Contribute

### Types of Contributions
We welcome various types of contributions, including:
- Bug fixes
- New features
- Documentation improvements
- UI/UX enhancements
- Test coverage improvements
- Performance optimizations

### Getting Started

1. **Fork the Repository**
   ```bash
   # Clone your fork locally
   git clone https://github.com/your-username/CortexOn.git
   cd CortexOn

   # Add the upstream repository
   git remote add upstream https://github.com/TheAgenticAI/CortexOn.git
   ```

2. **Create a Branch**
   ```bash
   # Ensure you're up to date
   git checkout main
   git pull upstream main

   # Create and switch to a new branch
   git checkout -b feature/your-feature-name
   ```

3. **Make Your Changes**
   - Write your code
   - Add or update tests as needed
   - Update documentation if required

4. **Submit Your Contribution**
   ```bash
   # Stage and commit your changes
   git add .
   git commit -m "feat: add new feature"

   # Push to your fork
   git push origin feature/your-feature-name
   ```

### Reporting Issues
- Use the GitHub issue tracker to report bugs or suggest features
- Check if a similar issue already exists before creating a new one
- Use the provided issue templates when available
- Include as much relevant information as possible:
  - Steps to reproduce the issue
  - Expected vs actual behavior
  - Screenshots if applicable
  - Environment details

## Code Style & Standards

### Coding Conventions
- Use consistent indentation (2 or 4 spaces)
- Follow the existing code style
- Write clear, descriptive variable and function names
- Add comments for complex logic
- Keep functions focused and concise

### Commit Message Guidelines
We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types include:
- feat: New feature
- fix: Bug fix
- docs: Documentation changes
- style: Code style changes
- refactor: Code refactoring
- test: Adding or updating tests
- chore: Maintenance tasks

Example:
```
feat(auth): add OAuth2 authentication support

Implement OAuth2 authentication using Google provider.
Includes user profile fetching and token refresh logic.

Closes #123
```

## Documentation

- Update README.md if adding new features
- Add JSDoc comments for new functions and classes
- Update API documentation if changing interfaces
- Include examples for new functionality

## Pull Request Process

1. **Before Submitting**
   - Ensure all tests pass
   - Update documentation as needed
   - Rebase your branch on the latest main

2. **PR Guidelines**
   - Link related issues
   - Provide a clear description of changes
   - Include screenshots for UI changes
   - List any breaking changes

3. **Review Process**
   - Two approvals required for merge
   - Address review feedback promptly
   - Keep the PR focused and reasonable in size
   - Be open to suggestions and improvements

4. **After Merge**
   - Delete your branch
   - Update any related issues
   - Help review other PRs

## Code of Conduct

We take our open source community seriously and hold ourselves and other contributors to high standards of communication. By participating and contributing to this project, you agree to uphold our [Code of Conduct](CODE_OF_CONDUCT.md).

The full Code of Conduct document details:
- Our pledge to make participation a harassment-free experience
- Specific examples of encouraged positive behavior
- Unacceptable behavior that will not be tolerated
- The scope of applicability
- Enforcement and reporting procedures
- Detailed enforcement guidelines and consequences

If you witness or experience any violations of our Code of Conduct, please report them to our project team at product@theagentic.ai. All reports will be handled with discretion and confidentiality.

## Additional Resources

- [Project Documentation](docs/README.md)
- [API Reference](docs/api.md)
- Community Channel:
  - [Discord Server](https://discord.gg/f6pXswy2h6)


### License
By contributing to this project, you agree that your contributions will be licensed under its [LICENSE](LICENSE) terms.

---

Thank you for considering contributing to our project! If you have any questions, feel free to reach out to the maintainers or ask in our community channels. 