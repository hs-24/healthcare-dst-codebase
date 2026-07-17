# Healthcare Supply Network Twin

DSC2205 Group 3 Assignment 2 technical prototype.

This proof of concept compares a centralised digital twin with a fragmented,
lagged spreadsheet baseline across three simulated healthcare warehouse sites.
It uses the public Kaggle **Hospital Supply Chain** inventory dataset as its
operational starting point.

Dataset: https://www.kaggle.com/datasets/vanpatangan/hospital-supply-chain

## Data methodology

The source `inventory_data.csv` provides dated item, stock, minimum-stock,
capacity, cost, average-usage, lead-time, and vendor fields. The prototype
derives the site assignment, UOM, storage zone, batch number, expiry date,
demand variability, and reorder quantity required by the simulation.

Derived fields are deterministic and identified in
`data/processed/hospital_inventory_normalized.csv`. They must not be described
as original hospital records. The source publisher does not identify a named
hospital, so the project describes it as a public hospital supply-chain dataset,
not authenticated operational data from a specific hospital.

## Run the prototype

Install [Ollama](https://ollama.com/download/windows) and download the local
chat model once on each computer:

```powershell
ollama pull qwen2.5:1.5b
```

Ollama normally starts in the background on Windows. The model is stored by
Ollama outside this repository and must not be committed to Git.

```powershell
python -m pip install -r requirements.txt
python run_pilot.py
python export_dashboard_data.py
python server.py
```

Open http://127.0.0.1:5000/.

## Dashboard capabilities

- All-sites KPI and risk comparison
- Site selector with independent live replay state
- 3D stock and storage-zone view
- Resizable operational sidebar with Overview, Inventory, Alerts, Agents, and Evidence tabs
- Manual-versus-digital-twin evidence series
- Batch, expiry, UOM, vendor, capacity, and data-provenance details
- Cross-site transfer recommendations when one site has a safe surplus
- Local read-only inventory chatbot grounded in live dashboard evidence

The chatbot uses Ollama without a paid API. Risk rankings, transfer quantities,
and agent-log answers are returned directly from verified Python data; the
language model explains non-transactional questions and cannot alter inventory.

## Main files

```text
src/digital_twin/hospital_data.py        Dataset validation and enrichment
data/raw/hospital-supply-chain/          Downloaded Kaggle source files
data/processed/                          Normalized site/SKU catalogue
run_pilot.py                             Three-site simulation pipeline
export_dashboard_data.py                Dashboard data contract
server.py                                Multi-site Flask API
web/live_dashboard.html                 Live multi-site interface
```
