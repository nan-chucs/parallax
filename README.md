# PARALLAX

Full-stack geospatial exploration platform for satellite imagery and GIS layers.

## Repository layout

- `backend/` contains the Python backend and its tests.
- `frontend/` is reserved for the future React application.

## Backend development

Run backend commands from `backend/`:

```powershell
python -m pip install -e ".[test]"
python -m pytest
```

The backend package can be imported as `app`.
