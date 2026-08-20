# Link Auto Price (Legacy v1)

Earlier iteration of the car price prediction model for Chevrolet/Daewoo/Ravon vehicles. Superseded by the current `link-auto` repo.

## Why this exists
Kept for historical reference — this was an early version of the auto price prediction pipeline.

## Tech stack
Python, LightGBM

## How to run
1. `pip install -r requirements.txt`
2. `python app.py` to serve predictions using `CHEVROLET_DAEWOO_RAVON_LGBM_logtuned.pkl` and `target_encoder.pkl`

**Note:** this is a legacy version — check `link-auto` for the current, actively maintained pipeline.
