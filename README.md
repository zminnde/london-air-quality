# Project Setup

## Prerequisites
- Python 3.13 or higher ([Download](https://python.org))
- Git

## Installation

1. Clone the repository
```bash
git clone https://github.com/zminnde/london-air-quality.git
cd london-air-quality
```

2. Create virtual environment
```bash
python3 -m venv .venv
```

3. Activate virtual environment
```bash
# Mac/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

4. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

5. Run the project
```bash
python3 normalize_data.py
```