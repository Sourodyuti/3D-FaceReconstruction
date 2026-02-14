# Contributing to 3D Face Reconstruction

Thank you for your interest in contributing to this project! We welcome contributions from the community.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue on GitHub with:

- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, etc.)
- Error messages or logs

### Suggesting Enhancements

For feature requests or enhancements:

- Check if the feature is already in the roadmap
- Describe the use case and benefits
- Provide examples if possible

### Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/3D-FaceReconstruction.git
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the coding style below
   - Add tests if applicable
   - Update documentation

4. **Test your changes**
   ```bash
   python test_setup.py
   python main.py live  # Manual testing
   ```

5. **Commit with clear messages**
   ```bash
   git commit -m "Add feature: description of what you added"
   ```

6. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Coding Style

### Python Style Guide

- Follow PEP 8 guidelines
- Use type hints for function parameters and returns
- Write docstrings for classes and functions
- Keep functions focused and modular

### Example:

```python
def process_landmarks(landmarks: np.ndarray, 
                      scale: float = 1.0) -> np.ndarray:
    """
    Process facial landmarks with scaling
    
    Args:
        landmarks: Input landmarks array (N, 3)
        scale: Scaling factor
    
    Returns:
        Processed landmarks
    """
    processed = landmarks * scale
    return processed
```

### Code Organization

- Place utility functions in `utils.py`
- Geometric operations go in `geometry_engine.py`
- Visualization code goes in `visualization.py`
- Keep `main.py` focused on orchestration

## Testing

- Test on multiple Python versions (3.8, 3.9, 3.10+)
- Verify camera access works
- Test export formats (OBJ, PLY, STL)
- Check memory usage for long-running sessions

## Documentation

When adding features:

- Update README.md if adding user-facing features
- Update USAGE.md with examples
- Add inline comments for complex logic
- Update config.yaml if adding configuration options

## Areas for Contribution

### High Priority

- [ ] Unit tests for core modules
- [ ] Performance optimizations
- [ ] Better error handling and user messages
- [ ] Multi-face support
- [ ] Improved texture mapping algorithms

### Medium Priority

- [ ] GUI interface (Qt/Tkinter)
- [ ] Video file processing
- [ ] Batch processing scripts
- [ ] Docker containerization
- [ ] CI/CD pipeline

### Nice to Have

- [ ] Web interface
- [ ] Mobile app (iOS/Android)
- [ ] Cloud deployment guide
- [ ] Model compression for edge devices
- [ ] Advanced visualization options

## Questions?

Feel free to open an issue for discussion before starting major work!

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
