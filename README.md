# ppg

Tools for kinetic modeling of dynamic PET data: time-activity curve
handling (`Tac`), arterial input function models (`aif_model`), PET
compartmental models (`pet_model`), and image/text I/O and plotting
helpers (`io`, `util`).

## Install

```bash
pip install -e .
```

For running the test suite:

```bash
pip install -e ".[dev]"
pytest
```

## Usage

```python
from ppg import Tac, aif_model, pet_model

aif = Tac(time, counts, dc=True, h_life=122.24)
model = pet_model.FlowTwo(aif, pet_tac)
```
