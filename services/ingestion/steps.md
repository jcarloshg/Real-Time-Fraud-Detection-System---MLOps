# Steps for ingestion

```bash
python3 -m venv venv                # create enviroment venv
source ./venv/bin/activate          # active the venv
pip install -r requirements.txt     # install requirements
python stream_simulator.py          # run script
```

## Issues

### psycopg2-binary 2.9.9 doesn't have pre-built

Now we have another compatibility issue - psycopg2-binary 2.9.9 doesn't
have pre-built wheels for Python 3.14, so it's trying to build from source and
needs PostgreSQL development headers.

Install PostgreSQL dev headers (quickest fix)

```bash
sudo apt-get update
sudo apt-get install libpq-dev
```
