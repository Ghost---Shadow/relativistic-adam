# RelativisticAdam

[![PyPI version](https://badge.fury.io/py/relativistic-adam.svg)](https://badge.fury.io/py/relativistic-adam)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.9+](https://img.shields.io/badge/pytorch-1.9+-ee4c2c.svg)](https://pytorch.org/)

A PyTorch optimizer that implements a relativistic gradient clipping mechanism, inspired by the theory of special relativity. RelativisticAdam prevents gradient explosion by introducing a configurable "speed limit" for parameter updates, similar to how nothing can exceed the speed of light in physics.

## 🌟 Key Features

- **Physics-Inspired Design**: Applies relativistic mechanics principles to optimization
- **Automatic Gradient Clipping**: No need for manual gradient clipping
- **Smooth & Differentiable**: Unlike hard clipping, provides smooth transitions
- **Drop-in Replacement**: Compatible with existing PyTorch code
- **Multiple Modes**: Global, per-parameter, or per-component scaling
- **Stable Training**: Especially effective with high learning rates or unstable architectures


## 🚀 Installation

```bash
pip install relativistic-adam
```

## 📖 Quick Start

```python
import torch
from relativistic_adam import RelativisticAdam

# Create your model
model = torch.nn.Linear(10, 1)

# Initialize the optimizer
optimizer = RelativisticAdam(
    model.parameters(),
    lr=0.001,
    speed_limit=0.1,  # Maximum update magnitude
    relativistic_mode='per_param'  # 'global', 'per_param', or 'per_component'
)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    loss = your_loss_function(model(input))
    loss.backward()
    optimizer.step()
```

## 🔬 The Physics Behind It

### The Analogy

In special relativity, as an object's velocity approaches the speed of light, its relativistic mass increases, making further acceleration increasingly difficult:

$$m_{rel} = \frac{m_0}{\sqrt{1 - \frac{v^2}{c^2}}}$$

Similarly, RelativisticAdam treats gradient updates as "velocities" and applies a similar scaling:

$$\text{scaled\_update} = \frac{\text{update}}{\sqrt{1 - \left(\frac{\|\text{update}\|}{c}\right)^2}}$$

Where `c` is the configurable "speed limit" for updates.

### Key Properties

1. **Small updates** (‖update‖ << c): Pass through nearly unchanged
2. **Large updates** (‖update‖ ≈ c): Get increasingly dampened
3. **Extreme updates** (‖update‖ > c): Smoothly saturate at the speed limit

## 🎛️ Configuration Options

### Basic Parameters (from Adam)
- `lr` (float, default=1e-3): Learning rate
- `betas` (tuple, default=(0.9, 0.999)): Coefficients for computing running averages
- `eps` (float, default=1e-8): Term added for numerical stability
- `weight_decay` (float, default=0): Weight decay (L2 penalty)

### Relativistic Parameters
- `speed_limit` (float, default=0.1): Maximum allowed update magnitude
- `relativistic_mode` (str, default='per_param'): Scaling mode
  - `'global'`: Single scaling factor for entire model
  - `'per_param'`: Scaling per parameter tensor (recommended)
  - `'per_component'`: Element-wise scaling (finest control)
- `adaptive_speed` (bool, default=False): Enable adaptive speed limit
- `speed_warmup_steps` (int, default=1000): Warmup steps for speed limit

## 📚 Advanced Usage

### With Adaptive Speed Limit

```python
optimizer = RelativisticAdam(
    model.parameters(),
    lr=0.001,
    speed_limit=0.1,
    adaptive_speed=True,
    speed_warmup_steps=1000  # Gradually increase speed limit
)
```

### RelativisticAdamW (with Decoupled Weight Decay)

```python
from relativistic_adam import RelativisticAdamW

optimizer = RelativisticAdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01,
    speed_limit=0.1
)
```

### Fine-Grained Control

```python
optimizer = RelativisticAdam(
    model.parameters(),
    lr=0.001,
    speed_limit=0.01,
    relativistic_mode='per_component'  # Element-wise scaling
)
```

## 🧪 When to Use RelativisticAdam

RelativisticAdam is particularly effective for:

- **High Learning Rates**: Can handle learning rates that would cause standard Adam to explode
- **Deep Networks**: Especially beneficial for very deep architectures
- **RNNs/LSTMs**: Where gradient explosion is common
- **Transformers**: Large models with potential instabilities  
- **Mixed Precision Training**: Where gradient scales can vary dramatically
- **Experimental Architectures**: When you're unsure about gradient stability

## 🔧 Tuning Guidelines

### Speed Limit Selection

- **Conservative** (0.01-0.1): For highly unstable problems
- **Moderate** (0.1-1.0): For standard deep learning tasks
- **Aggressive** (1.0-10.0): When you want minimal intervention

### Mode Selection

- Use `'per_param'` (default) for most cases
- Use `'global'` for uniform clipping across the model
- Use `'per_component'` for finest control (higher computational cost)

## 📈 Comparison with Standard Methods

| Method | Pros | Cons |
|--------|------|------|
| **Gradient Clipping** | Simple, effective | Hard cutoff, not differentiable |
| **Adam** | Adaptive learning rates | Can explode with high LR |
| **AdamW** | Better regularization | Still suffers from explosion |
| **RelativisticAdam** | Smooth clipping, physics-inspired | Additional hyperparameter (speed_limit) |

## 🏃 Running the Demo

```bash
# Basic demo
python demo.py

# The demo will:
# 1. Create a deep network
# 2. Train with very high learning rate (LR=1.0)
# 3. Show Adam exploding while RelativisticAdam remains stable
# 4. Generate comparison plots
```

## 🔍 Implementation Details

The optimizer implements three key components:

1. **Standard Adam Updates**: Maintains moving averages of gradients and squared gradients
2. **Relativistic Scaling**: Applies physics-inspired scaling to prevent explosion
3. **Adaptive Mechanisms**: Optional warmup and adaptive speed limits

### The Core Algorithm

```python
# Standard Adam momentum
m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²

# Compute update
update = lr * m_t / (√v_t + ε)

# Apply relativistic scaling
if ||update|| < speed_limit:
    scaled_update = update / √(1 - (||update||/c)²)
else:
    scaled_update = speed_limit * tanh(||update||/speed_limit)

# Update parameters
θ_t = θ_{t-1} - scaled_update
```

## 📊 Performance

![Gradient Explosion Comparison](high_lr_explosion_comparison.png)

The above figure shows how RelativisticAdam prevents gradient explosion with a high learning rate (LR=1.0) while standard Adam explodes immediately.


## 📝 Citation

If you use RelativisticAdam in your research, please cite:

```bibtex
@software{relativistic_adam,
  title = {RelativisticAdam: A Physics-Inspired Optimizer for Gradient Explosion Prevention},
  author = {Souradeep Nanda},
  year = {2025},
  url = {https://github.com/Ghost---Shadow/relativistic-adam}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the elegant principles of special relativity
- Built on top of PyTorch's optimization framework
- Thanks to the open-source community for feedback and contributions

## 🐛 Troubleshooting

### Common Issues

1. **Still experiencing gradient explosion**: Try reducing the `speed_limit` parameter
2. **Training too slow**: Increase `speed_limit` or use `adaptive_speed=True`
3. **Validation loss not decreasing**: The speed limit might be too restrictive

### FAQ

**Q: How is this different from gradient clipping?**  
A: RelativisticAdam provides smooth, differentiable scaling rather than hard cutoffs, and the scaling is inspired by relativistic physics.

**Q: Can I use this with other optimizers?**  
A: The relativistic scaling mechanism could be adapted to other optimizers. PRs welcome!

**Q: What's the computational overhead?**  
A: Minimal - just computing norms and applying scaling, similar to gradient clipping.

## 📮 Contact

For questions and feedback:
- Open an issue on [GitHub](https://github.com/Ghost---Shadow/relativistic-adam)

---

**Note**: This is an experimental optimizer. While it shows promising results in preventing gradient explosion, it should be thoroughly tested in your specific use case before production deployment.
