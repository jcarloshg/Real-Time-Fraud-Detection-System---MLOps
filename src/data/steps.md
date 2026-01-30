```bash
python3.12 -m venv .venv            # create the env called .venv
source ./.venv/bin/activate         # active the .venv
pip install -r requirements.txt     # install requirements
```

```bash
# Stream first 1000 transactions (for testing)
python3.12 stream_simulator.py --max 1000 --batch-size 50 --delay 0.5

# Or stream all data (this will take a while)
python3.12 stream_simulator.py --batch-size 100 --delay 1

# Or stream with shuffling
python3.12 stream_simulator.py --max 5000 --shuffle
```
