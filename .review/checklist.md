# Review Checklist

- [ ] SOLID respected
- [ ] Plugin architecture respected
- [ ] No MainWindow business logic
- [ ] No direct plugin dependency
- [ ] No memory leak
- [ ] No race condition
- [ ] No circular dependency
- [ ] Thread safe
- [ ] Local Build & Tests pass (C++)
- [ ] Local Python Lint passes (Ruff)
- [ ] Compatible with GitHub Actions CI workflows (`ci.yml`, `python-ci.yml`)
- [ ] Release changes generate valid MSI Installer (`release.yml`)
- [ ] No duplicated code
