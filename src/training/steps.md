```bash
# Create a virtual environment (recommended)
python3.12 -m venv venv
# On Windows: .venv\Scripts\activate
source venv/bin/activate
# Install dependencies
pip install -r requirements.txt

# 1.
python3.12 prepare_data.py

# 2.
python3.12 train_baseline.py

# 3.
python3.12 train_advanced.py

# 4. 
python3.12 register_model.py
```
