# ChakraNetra — Demo & Setup Guide
**Team Techtonic | SIH 2026 (SIH26070)**

---

## Option 1: Online Demo (No commands needed!)

Just open this URL in any browser:

> **https://chakra-netra-p9x4zhl1c-chakra-netra.vercel.app**

- Frontend is hosted on **Vercel** (always on)
- Backend API is hosted on **Render** (may take 30–50 sec to wake up on first visit — open it 1 minute before presenting)

**That's it. No installation, no commands, works on any device.**

---

## Option 2: Local Demo (Run on your own laptop)

Use this when you want to demo offline or from `localhost`.

### Prerequisites (one-time setup)
- Python 3.11+ installed
- Git installed

### Step 1 — Clone the repo (first time only)
```powershell
git clone https://github.com/YOUR_USERNAME/ChakraNetra.git
cd ChakraNetra
```

### Step 2 — Install dependencies (first time only)
```powershell
python -m pip install -r requirements.txt
```

### Step 3 — Start the server
```powershell
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

### Step 4 — Open the dashboard
Open your browser and go to:
> **http://localhost:8000**

---

## If you already have the repo on your laptop

Just run **one command** every time you want to demo:
```powershell
cd "C:\Users\Shashwat Tripathi\OneDrive\Desktop\Hackathon stuff\Prototype_works\ChakraNetra"
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```
Then open: **http://localhost:8000**

To **stop** the server: press `Ctrl + C`

---

## Quick Reference

| Task | Command |
|------|---------|
| Start server | `python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload` |
| Run tests | `python -m pytest tests/ -q` |
| Stop server | `Ctrl + C` |
| Pull latest code | `git pull origin main` |
| Push your changes | `git push origin main` |

---

## Demo Flow (3-minute SIH presentation)

1. **Open** https://chakra-netra-p9x4zhl1c-chakra-netra.vercel.app  
   *(or localhost:8000 for local)*

2. **Page 1 — Command Dashboard**
   - Show the interactive Leaflet map with storm track
   - Point out the forecast track with +24h / +48h / +72h uncertainty circles
   - Show the wind radii (R34 yellow, R50 orange, R64 red)
   - Show the Intelligence Strip: Severity | RI | ERC | Wind Radii

3. **Page 2 — Storm Selector / NIO Archive**
   - Show the full archive of 20 North Indian Ocean cyclones (2018–2023)
   - Click a different storm to load it (e.g., Fani 2019 or Amphan 2020)

4. **Page 3 — RI & ERC Deep Analysis**
   - Explain Dual-Signal RI: TabNet (environmental) + XGBoost (WWLLN lightning)
   - Show the ERC (Eyewall Replacement Cycle) probability

5. **Page 4 — Wind Radii & Risk Model**
   - Show the Rankine Vortex polar diagram
   - Show the Composite Risk Score waterfall

6. **Page 5 — IBTrACS Upload**
   - Show the real IBTrACS data ingestion table with all 20 storms

---

## Architecture Summary (for judges)

```
[ Vercel ]                         [ Render.com ]
  frontend/index.html   ←→   FastAPI (src/api.py)
  Static SPA                       │
  Leaflet maps                HistGradientBoosting
  Canvas charts               12 ML models (sklearn)
  Tailwind CSS                RI: TabNet + XGBoost
                              ERC: 3-component classifier
                              Risk: Rankine vortex model
                              Data: IBTrACS NI basin
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Vercel site loads but no data | Wait 60 sec for Render to wake up, then refresh |
| `uvicorn not found` locally | Run `python -m uvicorn` instead of just `uvicorn` |
| Map doesn't load tiles | Check internet connection (Esri tiles need internet) |
| Port 8000 already in use | Run `python -m uvicorn src.api:app --port 8001` and open `localhost:8001` |
| Tests fail | Run `python -m pip install -r requirements.txt` then retry |
