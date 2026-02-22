# collision_sim_with_multiple_models_fixed_forced_future20_with_airports_legend.py
# Combined script with airport labels and bottom-left legend outside the plot
# Modified:
#   - Smooth restore back to the original altitude profile (cruise + descent),
#     so AMD-DEL flights return to 35000 m after conflict AND still descend to land.
#   - safer persistent_active logic so descended flights also restore correctly.
#   - harsh-origin cancellations
#   - message panel on the right side (wrapped, narrow, scrollable with mouse wheel)
#   - altitude labels shifted slightly to the right of marker
#   - altitude labels show fid + origin->dest + altitude
#   - simplified GUI: only origin/dest + Add Flight + Done (no heading/model/late_start/FID inputs)
#   - heading and model_key are selected automatically
#   - GEN model renamed to BOM_DEL
#   - messages when flights are added (origin/dest/model), only airport codes (no lat/lon)
#   - messages panel without timestamps, left-aligned text
#   - each conflict pair logs the conflict message only once per conflict episode
#   - removed sep-before/sep-after messages from message box
#   - FIX: avoid KeyError on alt_texts, and avoid re-entering flights after landing
#   - NEW: altitude-adjustment message logged once per pair; removed "removed after 3s"
#          message; removed "timesteps" log line.
#   - NEW: do NOT clear conflict_logged_pairs / alt_adjust_logged_pairs (so each pair
#          prints conflict + altitude adjustment only once for entire sim).
#   - NEW: remove per-flight distance print from log (only short route).
#   - NEW: landed label now shows "fid=<id> Landed".
#   - NEW: console now prints ONLY: models loaded, flights list, landed messages.
#   - NEW: removed restoring messages from message panel.
#   - NEW: airport labels use only short codes; legend airport icon is a triangle.
#   - NEW: block same-origin/same-destination flights (e.g. BOM->BOM) in GUI.
#   - NEW: FID removed from GUI; FIDs auto-assigned for each new flight.
#   - NEW: "Remove selected" button to delete flights from the list in the GUI.
#   - NEW: conflict resolution now applies altitude change only ONCE per flight pair,
#          to prevent AMD<->BOM flights from endlessly climbing/descending.

import os, time, json, csv, ast, textwrap
import numpy as np, pandas as pd, joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from tensorflow.keras.models import load_model
from math import radians, sin, cos, asin

# ---------------- CONFIG ----------------
SEQ_LEN = 10
FUTURE_STEPS = 20
NUM_FEATURES = 5
PLOT_INTERVAL = 0.12

HORIZ_KM = 5.0
ALT_THRESH_M = 300.0

# vertical separation / ramp
PERSIST_ALT = 3000.0
RAMP_UP_STEPS = 20
RAMP_DOWN_STEPS = 20
PASS_INCREASE_COUNT = 6
SINGLE_PLANE_INC_M = 3000.0

# Global cruise constant (single source of truth)
CRUISE_ALT_M = 35000.0
RETURN_SMOOTH_STEPS_AMD_DEL = 20

# landed display time (seconds)
LANDED_SHOW_SEC = 3.0

FEATURES = ["latitude","longitude","altitude","speed","heading"]

# ---------------- Per-route timing tuning ----------------
MIN_TIMESTEPS = 80
MAX_TIMESTEPS = 300
AMD_BOM_FACTOR = 0.6

# ---------------- Small airport map ----------------
AIRPORTS = {
    "AMD": (23.0776, 72.6326),
    "BOM": (19.0896, 72.8656),
    "DELHI": (28.5562, 77.1000),
}

# ---------------- Simple logging for on-figure message panel ----------------
log_messages = []

def add_log(msg: str):
    """
    Log message ONLY to the on-figure Messages panel (no console print).
    Console is reserved for: model loading, flight list, and landed messages.
    """
    log_messages.append(msg)

# ---------------- Helpers ----------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dl = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dl/2)**2
    return 2 * R * asin(np.sqrt(a))

def meters_to_deg_lat(m): return m / 111320.0
def meters_to_deg_lon(m, lat): return m / (111320.0 * cos(radians(lat)) + 1e-12)

def compute_bearing(lat1, lon1, lat2, lon2):
    lat1r, lat2r = radians(lat1), radians(lat2)
    dlon = radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    br = np.degrees(np.arctan2(x, y))
    br = (br + 360.0) % 360.0
    return int(round(br))

def coords_to_code_global(coords):
    for code, c in AIRPORTS.items():
        if abs(coords[0] - c[0]) < 1e-6 and abs(coords[1] - c[1]) < 1e-6:
            return code
    return None

# ---------------- Models / Scalers loading ----------------
models_dir = "models"
candidates = {
    "AMD_BOM": {
        "scaler_X": os.path.join(models_dir, "scaler_X_AMD_BOM.save"),
        "scaler_Y": os.path.join(models_dir, "scaler_Y_AMD_BOM.save"),
        "model":    os.path.join(models_dir, "delta_multi_lstm_AMD_BOM.h5")
    },
    "BOM_DEL": {  
        "scaler_X": os.path.join(models_dir, "scaler_X_BOM_DEL.save"),
        "scaler_Y": os.path.join(models_dir, "scaler_Y_BOM_DEL.save"),
        "model":    os.path.join(models_dir, "delta_multi_lstm_BOM_DEL.h5")
    },
    "AMD_DEL": {
        "scaler_X": os.path.join(models_dir, "scaler_X_amd_del.save"),
        "scaler_Y": os.path.join(models_dir, "scaler_Y_amd_del.save"),
        "model":    os.path.join(models_dir, "delta_multi_lstm_amd_del.h5")
    }
}

model_registry = {}
print("Loading model sets from", models_dir)
for key, paths in candidates.items():
    try:
        sx = joblib.load(paths["scaler_X"])
        sy = joblib.load(paths["scaler_Y"])
        m  = load_model(paths["model"])
        model_registry[key] = {"scaler_X": sx, "scaler_Y": sy, "model": m}
        print(f"Loaded model set '{key}'.")
    except Exception as e:
        print(f"Could not load model set '{key}': {e}")
print("Model registry keys:", list(model_registry.keys()))
if len(model_registry) == 0:
    print("Warning: No models loaded. GUI will still allow plan creation but simulation will fail unless models exist.")

# ---------------- Predictor ----------------
def predict_horizon(seq_scaled, model_key, fid=None):
    reg = model_registry.get(model_key)
    if reg is None:
        raise RuntimeError(f"Model key '{model_key}' not found in model_registry")
    model = reg["model"]
    sx = reg["scaler_X"]
    sy = reg["scaler_Y"]

    try:
        ypred_raw = model.predict(seq_scaled.reshape(1, SEQ_LEN, NUM_FEATURES), verbose=0)[0]
    except Exception as e:
        print(f"❗ predict() failed for model '{model_key}': {e}")
        try:
            last_real = sx.inverse_transform(seq_scaled[-1].reshape(1,-1))[0]
        except Exception:
            last_real = np.zeros(NUM_FEATURES)
        return np.repeat(last_real.reshape(1,-1), FUTURE_STEPS, axis=0)

    flat = np.asarray(ypred_raw).reshape(-1)
    if flat.size == FUTURE_STEPS * NUM_FEATURES:
        n_steps = FUTURE_STEPS
    elif flat.size % NUM_FEATURES == 0:
        n_steps = flat.size // NUM_FEATURES
    else:
        try:
            inv_single = sy.inverse_transform(flat.reshape(1,-1))
            inv_rep = np.repeat(inv_single, FUTURE_STEPS, axis=0)
            last_real = sx.inverse_transform(seq_scaled[-1].reshape(1,-1))[0]
            st = last_real.copy(); out_tmp = []
            for i in range(inv_rep.shape[0]):
                st = st + inv_rep[i]
                out_tmp.append(st.copy())
            out = np.array(out_tmp)
            if out.shape[0] >= FUTURE_STEPS:
                out = out[:FUTURE_STEPS]
            else:
                pad_n = FUTURE_STEPS - out.shape[0]
                pad = np.repeat(out[-1:].copy(), pad_n, axis=0)
                out = np.vstack([out, pad])
            return out
        except Exception:
            last_real = sx.inverse_transform(seq_scaled[-1].reshape(1,-1))[0]
            return np.repeat(last_real.reshape(1,-1), FUTURE_STEPS, axis=0)

    try:
        y_s = flat.reshape(n_steps, NUM_FEATURES)
    except Exception:
        last_real = sx.inverse_transform(seq_scaled[-1].reshape(1,-1))[0]
        return np.repeat(last_real.reshape(1,-1), FUTURE_STEPS, axis=0)

    try:
        deltas = sy.inverse_transform(y_s)
    except Exception:
        deltas = y_s.copy()

    try:
        last_real = sx.inverse_transform(seq_scaled[-1].reshape(1,-1))[0].copy()
    except Exception:
        last_real = np.zeros(NUM_FEATURES)

    out_rows = []
    st = last_real.copy()
    for i in range(deltas.shape[0]):
        st = st + deltas[i]
        out_rows.append(st.copy())
    out = np.array(out_rows)

    if out.shape[0] > FUTURE_STEPS:
        out = out[:FUTURE_STEPS]
    elif out.shape[0] < FUTURE_STEPS:
        if out.shape[0] >= 1:
            pad_n = FUTURE_STEPS - out.shape[0]
            pad = np.repeat(out[-1:].copy(), pad_n, axis=0)
            out = np.vstack([out, pad])
        else:
            out = np.repeat(last_real.reshape(1,-1), FUTURE_STEPS, axis=0)

    # Force straight ground-track increments for lat/lon (toward destination) when fid provided
    if fid is not None:
        try:
            cur_lat = float(last_real[0]); cur_lon = float(last_real[1])
            dest_rows = df_all[df_all["flight_id"] == fid]
            if len(dest_rows) > 0:
                dest_row = dest_rows.sort_values("timestep").iloc[-1]
                dest_lat = float(dest_row["latitude"]); dest_lon = float(dest_row["longitude"])
                remaining_samples = None
                try:
                    rem = len(scaled_groups[fid]) - indices[fid]
                    remaining_samples = int(max(1, rem))
                except Exception:
                    remaining_samples = None
                if remaining_samples is None or remaining_samples <= 0:
                    remaining_samples = max(1, FUTURE_STEPS * 4)
                delta_lat_total = dest_lat - cur_lat
                delta_lon_total = dest_lon - cur_lon
                per_step_lat = delta_lat_total / float(remaining_samples)
                per_step_lon = delta_lon_total / float(remaining_samples)
                steps_idx = np.arange(1, FUTURE_STEPS + 1, dtype=float)
                new_lats = cur_lat + per_step_lat * steps_idx
                new_lons = cur_lon + per_step_lon * steps_idx
                out[:,0] = new_lats
                out[:,1] = new_lons
        except Exception as e:
            print(f"⚠️ straight-track-for-future failed for fid={fid}: {e}")

    try:
        def smooth_col(col_vals, window=3):
            if len(col_vals) <= 1: return col_vals
            w = min(window, len(col_vals)); kern = np.ones(w)/float(w)
            return np.convolve(col_vals, kern, mode='same')
        out[:,2] = smooth_col(out[:,2], window=3)
        out[:,3] = smooth_col(out[:,3], window=3)
        out[:,4] = smooth_col(out[:,4], window=3)
    except Exception:
        pass

    return out

# ---------------- Tkinter GUI ----------------
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

def _parse_loc_entry(s, AIRPORTS):
    if s is None: raise ValueError("Empty")
    s = s.strip()
    if s == "": raise ValueError("Empty")
    code = s.upper()
    if code in AIRPORTS:
        return AIRPORTS[code]
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        arr = ast.literal_eval(s)
        return float(arr[0]), float(arr[1])
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        return float(parts[0]), float(parts[1])
    raise ValueError("Unrecognized location: " + s)

# forced map uses new BOM_DEL key
forced_map = {
    ("AMD",  "BOM"):  "AMD_BOM",
    ("AMD",  "DELHI"):"AMD_DEL",
    ("BOM",  "AMD"):  "AMD_BOM",
    ("BOM",  "DELHI"):"BOM_DEL",
    ("DELHI","AMD"):  "AMD_DEL",
    ("DELHI","BOM"):  "BOM_DEL",
}

def infer_model_for_route(origin, dest):
    """Auto-select model_key based on origin/dest and forced_map."""
    orig_code = coords_to_code_global(origin)
    dest_code = coords_to_code_global(dest)
    if orig_code and dest_code:
        forced = forced_map.get((orig_code, dest_code))
        if forced and forced in model_registry:
            return forced
    if "BOM_DEL" in model_registry:
        return "BOM_DEL"
    if model_registry:
        return list(model_registry.keys())[0]
    return "BOM_DEL"

def _normalize_row(values, AIRPORTS, auto_fid):
    """
    Normalize GUI form values into an internal flight spec.

    - FID is auto-assigned (no GUI field).
    - origin/dest from combo or lat,lon
    - heading automatically computed from origin->dest
    - model_key automatically inferred (no GUI selection)
    - late_start always 0 from GUI (staggering logic modifies later)
    - block same-origin/same-destination flights (e.g. BOM->BOM).
    """
    fid = auto_fid()
    origin = _parse_loc_entry(values["origin"], AIRPORTS)
    dest   = _parse_loc_entry(values["dest"], AIRPORTS)

    # Disallow same origin and destination
    if abs(origin[0] - dest[0]) < 1e-6 and abs(origin[1] - dest[1]) < 1e-6:
        raise ValueError("Origin and destination cannot be the same.")

    heading = compute_bearing(origin[0], origin[1], dest[0], dest[1])
    model_key = infer_model_for_route(origin, dest)
    late_start = 0
    return {
        "fid": fid,
        "origin": origin,
        "dest": dest,
        "heading": heading,
        "model_key": model_key,
        "late_start": late_start
    }

def adjust_for_weather(flights, airport_weather, model_registry, AIRPORTS, forced_model_map):
    if forced_model_map is None:
        raise RuntimeError("forced_model_map must be provided in this version (forced-only model selection).")

    def coords_to_code(coords):
        for code, c in AIRPORTS.items():
            if abs(coords[0] - c[0]) < 1e-6 and abs(coords[1] - c[1]) < 1e-6:
                return code
        return None

    changes = []

    def apply_reroute(f, orig_code, new_dest_code):
        old_dest_code = coords_to_code(f["dest"]) or "Custom"
        old_model = f.get("model_key")
        f["dest"] = AIRPORTS[new_dest_code]
        forced = forced_model_map.get((orig_code, new_dest_code))
        if forced is None:
            add_log(f"⚠️ No forced model mapping for {orig_code}->{new_dest_code}; keeping old model '{old_model}' for fid {f['fid']}.")
            new_model = old_model
        else:
            if forced not in model_registry:
                add_log(f"⚠️ Forced model '{forced}' for {orig_code}->{new_dest_code} not loaded; keeping old model '{old_model}' for fid {f['fid']}.")
                new_model = old_model
            else:
                new_model = forced

        f["model_key"] = new_model
        changes.append({
            "fid": f["fid"],
            "old_dest": old_dest_code,
            "new_dest": new_dest_code,
            "old_model": old_model,
            "new_model": new_model
        })
        add_log(f"↪ Rerouted flight {f['fid']}: {orig_code}->{old_dest_code} -> {orig_code}->{new_dest_code}; model {old_model} -> {new_model}")

    harsh = {code for code, st in airport_weather.items() if str(st).strip().lower() == "harsh"}

    for f in flights:
        orig_code = coords_to_code(f["origin"])
        dest_code = coords_to_code(f["dest"])
        if orig_code is None or dest_code is None:
            continue

        if "AMD" in harsh:
            if orig_code == "BOM" and dest_code == "AMD":
                apply_reroute(f, "BOM", "DELHI")
            elif orig_code == "DELHI" and dest_code == "AMD":
                apply_reroute(f, "DELHI", "BOM")

        if "BOM" in harsh:
            if orig_code == "AMD" and dest_code == "BOM":
                apply_reroute(f, "AMD", "DELHI")
            elif orig_code == "DELHI" and dest_code == "BOM":
                apply_reroute(f, "DELHI", "AMD")

        if "DELHI" in harsh:
            if orig_code == "AMD" and dest_code == "DELHI":
                apply_reroute(f, "AMD", "BOM")
            elif orig_code == "BOM" and dest_code == "DELHI":
                apply_reroute(f, "BOM", "AMD")

    return changes

def gui_build_flights(model_registry, AIRPORTS, initial_list=None):
    initial_list = initial_list or []

    root = tk.Tk()
    root.title("Flight Plan Builder (with Weather)")
    root.geometry("900x480")
    root.resizable(True, True)

    flights = []
    used_fids = set()
    def set_used(f):
        used_fids.add(int(f))
    for r in initial_list:
        if "fid" in r:
            set_used(r["fid"])

    def next_auto_fid():
        i = 0
        while i in used_fids:
            i += 1
        used_fids.add(i)
        return i

    # ---- Layout: left form, mid list, right weather ----
    frm_left = ttk.Frame(root, padding=(8,8)); frm_left.pack(side="left", fill="y")
    frm_mid = ttk.Frame(root, padding=(8,8)); frm_mid.pack(side="left", fill="both", expand=True)
    frm_right = ttk.Frame(root, padding=(8,8)); frm_right.pack(side="right", fill="y")

    # ---- Left: minimal form (origin/dest only) ----
    ttk.Label(frm_left, text="Origin (code or lat,lon)").grid(row=0, column=0, sticky="w")
    ent_origin = ttk.Combobox(frm_left, values=list(AIRPORTS.keys()), width=20)
    ent_origin.set("")
    ent_origin.grid(row=0,column=1, pady=2)

    ttk.Label(frm_left, text="Dest (code or lat,lon)").grid(row=1, column=0, sticky="w")
    ent_dest = ttk.Combobox(frm_left, values=list(AIRPORTS.keys()), width=20)
    ent_dest.set("")
    ent_dest.grid(row=1,column=1, pady=2)

    btn_add = ttk.Button(frm_left, text="Add flight")
    btn_add.grid(row=2, column=0, columnspan=2, pady=(10,2), sticky="we")

    # ---- Middle: list of flights + Done + Remove ----
    ttk.Label(frm_mid, text="Planned Flights").pack(anchor="w")
    listbox = tk.Listbox(frm_mid, selectmode="browse")
    listbox.pack(fill="both", expand=True, padx=4, pady=4)

    frm_actions = ttk.Frame(frm_mid)
    frm_actions.pack(fill="x", pady=(6,0))
    btn_done = ttk.Button(frm_actions, text="Done", width=12)
    btn_done.grid(row=0, column=0, padx=4)
    btn_remove = ttk.Button(frm_actions, text="Remove selected", width=18)
    btn_remove.grid(row=0, column=1, padx=4)

    # ---- Right: airport weather ----
    ttk.Label(frm_right, text="Airport Weather").pack(anchor="w")
    weather_frame = ttk.Frame(frm_right); weather_frame.pack(fill="y", pady=6)
    airport_weather = {}
    for code in AIRPORTS.keys():
        airport_weather[code] = "Good"
    comb_widgets = {}
    for r, code in enumerate(sorted(AIRPORTS.keys())):
        ttk.Label(weather_frame, text=code, width=6).grid(row=r, column=0, padx=2, pady=2)
        cb = ttk.Combobox(weather_frame, values=["Good","Harsh"], width=8)
        cb.set(airport_weather[code])
        cb.grid(row=r, column=1, padx=2, pady=2)
        comb_widgets[code] = cb
        def make_handler(c, comb=cb):
            def handler(event=None):
                airport_weather[c] = comb.get()
            return handler
        cb.bind("<<ComboboxSelected>>", make_handler(code))
        cb.bind("<FocusOut>", make_handler(code))

    # ---- Helper functions ----
    def row_to_text(r):
        oc = coords_to_code_global(r["origin"]) or "Custom"
        dc = coords_to_code_global(r["dest"]) or "Custom"
        return (f"fid={r.get('fid')}  {oc}->{dc}  "
                f"hdg={r.get('heading')}  model={r.get('model_key')}  "
                f"late={r.get('late_start')}")

    def get_form_values():
        return {
            "origin": ent_origin.get().strip(),
            "dest": ent_dest.get().strip()
        }

    def refresh_listbox():
        listbox.delete(0, tk.END)
        for r in flights:
            listbox.insert(tk.END, row_to_text(r))

    def clear_form():
        ent_origin.set("")
        ent_dest.set("")

    def add_flight():
        vals = get_form_values()
        try:
            normalized = _normalize_row(vals, AIRPORTS, next_auto_fid)
        except Exception as e:
            messagebox.showerror("Invalid entry", str(e))
            return
        flights.append(normalized)
        refresh_listbox()
        # log message about added flight (codes only) – panel only
        oc = coords_to_code_global(normalized["origin"]) or "Custom"
        dc = coords_to_code_global(normalized["dest"]) or "Custom"
        add_log(
            f"Added flight fid={normalized['fid']} from {oc} to {dc} "
            f"(model={normalized['model_key']}, late_start={normalized['late_start']})"
        )
        clear_form()

    def remove_selected():
        sel = listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if 0 <= idx < len(flights):
            removed = flights.pop(idx)
            refresh_listbox()
            add_log(f"Removed flight fid={removed['fid']}.")

    def on_select(evt):
        sel = listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        r = flights[idx]
        def find_code(latlon):
            for k,v in AIRPORTS.items():
                if abs(v[0]-latlon[0])<1e-6 and abs(v[1]-latlon[1])<1e-6:
                    return k
            return f"{latlon[0]:.6f},{latlon[1]:.6f}"
        ent_origin.set(find_code(r["origin"]))
        ent_dest.set(find_code(r["dest"]))

    listbox.bind("<<ListboxSelect>>", on_select)

    result = {"done": False, "flights": []}
    def do_done():
        try:
            fids = [int(r["fid"]) for r in flights]
            if len(fids) != len(set(fids)):
                messagebox.showerror("Validation", "Duplicate fid detected.")
                return
        except Exception as e:
            messagebox.showerror("Validation", str(e))
            return
        for code, cb in comb_widgets.items():
            airport_weather[code] = cb.get()
        result["done"] = True
        result["flights"] = flights.copy()
        result["airport_weather"] = airport_weather.copy()
        root.destroy()

    btn_add.configure(command=add_flight)
    btn_done.configure(command=do_done)
    btn_remove.configure(command=remove_selected)

    # preload initial_list if any
    for r in initial_list:
        try:
            origin = tuple(r["origin"])
            dest   = tuple(r["dest"])
            heading = int(r.get("heading", compute_bearing(origin[0], origin[1], dest[0], dest[1])))
            model_key = r.get("model_key") or infer_model_for_route(origin, dest)
            late_start = int(r.get("late_start", 0))
            fid = int(r["fid"])
            norm = {
                "fid": fid,
                "origin": origin,
                "dest": dest,
                "heading": heading,
                "model_key": model_key,
                "late_start": late_start
            }
            flights.append(norm); set_used(fid)
        except Exception as e:
            print("Skipping invalid initial flight:", r, e)
    refresh_listbox()

    root.focus_force()
    root.mainloop()

    if result["done"]:
        return result["flights"], result["airport_weather"]
    else:
        return [], result.get("airport_weather", airport_weather.copy())

# ---------------- Run GUI and then simulator ----------------
flights_spec, airport_weather = gui_build_flights(model_registry, AIRPORTS)
if not flights_spec:
    print("No flights selected. Exiting.")
    raise SystemExit(0)

changes = adjust_for_weather(flights_spec, airport_weather, model_registry, AIRPORTS, forced_model_map=forced_map)
if changes:
    add_log("Weather-based reroutes applied:")
    for ch in changes:
        add_log(f"  fid {ch['fid']}: {ch['old_dest']} -> {ch['new_dest']}, model {ch['old_model']} -> {ch['new_model']}")
else:
    add_log("No weather-based reroutes needed.")

# ---------------- CANCEL FLIGHTS DUE TO HARSH ORIGIN WEATHER ----------------
harsh_origins = {
    code
    for code, st in airport_weather.items()
    if str(st).strip().lower() == "harsh"
}

if harsh_origins:
    cancelled = []
    kept = []
    for spec in flights_spec:
        orig_code = coords_to_code_global(spec["origin"])
        if orig_code in harsh_origins:
            cancelled.append(spec)
        else:
            kept.append(spec)

    flights_spec = kept

    if cancelled:
        add_log("Flights cancelled due to harsh origin weather:")
        for spec in cancelled:
            add_log(f"  fid={spec['fid']} origin={coords_to_code_global(spec['origin'])}")

if not flights_spec:
    add_log("All flights cancelled due to harsh origin weather. Exiting.")
    print("All flights cancelled due to harsh origin weather. Exiting.")
    raise SystemExit(0)

# ---------------- STAGGER LATE-STARTS ----------------
LATE_STEP = 40
origin_groups = {}
for spec in flights_spec:
    code = coords_to_code_global(spec["origin"])
    if code is None:
        continue
    origin_groups.setdefault(code, []).append(spec)

for code, specs in origin_groups.items():
    if len(specs) < 2:
        continue
    add_log(f"Origin '{code}' has {len(specs)} flights -> staggering late_start by {LATE_STEP} per subsequent flight.")
    for idx, spec in enumerate(specs[1:], start=1):
        before = int(spec.get("late_start", 0))
        add_ = LATE_STEP * idx
        spec["late_start"] = before + add_
        add_log(f"  fid {spec['fid']}: late_start {before} -> {spec['late_start']} (added {add_})")

# ---------------- Build df_all ----------------
def route_key(spec):
    a = spec["origin"]; b = spec["dest"]
    def is_same(p, code):
        lat, lon = AIRPORTS[code]
        return abs(p[0]-lat) < 1e-6 and abs(p[1]-lon) < 1e-6
    if (is_same(a, "AMD") and is_same(b, "BOM")) or (is_same(a, "BOM") and is_same(b, "AMD")):
        return "AMD_BOM"
    return None

for spec in flights_spec:
    o = spec["origin"]; d = spec["dest"]
    spec["distance_km"] = haversine_km(o[0], o[1], d[0], d[1])

max_dist = max(spec["distance_km"] for spec in flights_spec) if flights_spec else 1.0
for spec in flights_spec:
    frac = spec["distance_km"] / max_dist if max_dist > 0 else 1.0
    steps = int(round(MIN_TIMESTEPS + frac * (MAX_TIMESTEPS - MIN_TIMESTEPS)))
    rk = route_key(spec)
    if rk == "AMD_BOM" and AMD_BOM_FACTOR is not None and AMD_BOM_FACTOR != 1.0:
        steps = max(MIN_TIMESTEPS, int(round(steps * AMD_BOM_FACTOR)))
    spec["timesteps"] = max(MIN_TIMESTEPS, min(MAX_TIMESTEPS, steps))
    oc = coords_to_code_global(spec["origin"]) or "Custom"
    dc = coords_to_code_global(spec["dest"]) or "Custom"
    # Per-flight log: both panel AND console
    msg_flight = f"Flight {spec['fid']} {oc}->{dc}"
    add_log(msg_flight)
    print(msg_flight)

def generate_flight(fid, origin, dest, steps, heading_base):
    lat1, lon1 = origin
    lat2, lon2 = dest
    lats = np.linspace(lat1, lat2, steps)
    lons = np.linspace(lon1, lon2, steps)
    C = max(3, int(steps * 0.2))
    R = max(3, int(steps * 0.5))
    D = max(3, steps - C - R)
    altitudes = np.concatenate([
        np.linspace(0, CRUISE_ALT_M, C),
        np.full(R, CRUISE_ALT_M),
        np.linspace(CRUISE_ALT_M, 0, D)
    ])
    speeds = np.concatenate([
        np.linspace(0,250,C),
        np.full(R,450),
        np.linspace(450,200,D)
    ])
    headings = np.full(steps, heading_base)
    return pd.DataFrame({
        "flight_id": fid,
        "timestep": np.arange(steps),
        "latitude": lats,
        "longitude": lons,
        "altitude": altitudes,
        "speed": speeds,
        "heading": headings
    })

dfs = []
for spec in flights_spec:
    tsteps = spec.get("timesteps", MAX_TIMESTEPS)
    df = generate_flight(spec["fid"], spec["origin"], spec["dest"], tsteps, spec["heading"])
    if spec.get("late_start", 0):
        df["timestep"] += spec["late_start"]
    df["model_key"] = spec["model_key"]
    dfs.append(df)
df_all = pd.concat(dfs, ignore_index=True)

# ---------------- Prepare scaled arrays ----------------
scaled_groups = {}
original_scaled = {}
seqs = {}
indices = {}
start_times = {}
flight_model_key = {}

# route labels per fid for altitude text: "AMD->BOM"
route_label = {}

for spec in flights_spec:
    fid = spec["fid"]
    g = df_all[df_all["flight_id"]==fid].sort_values("timestep").reset_index(drop=True)
    arr = g[FEATURES].values.astype(float)
    start_times[fid] = g["timestep"].iloc[0]
    model_key = spec["model_key"]
    flight_model_key[fid] = model_key
    if model_key not in model_registry:
        raise RuntimeError(f"Model key '{model_key}' not found in loaded model registry. Make sure model files exist.")
    sx = model_registry[model_key]["scaler_X"]
    scaled = sx.transform(arr)
    scaled_groups[fid] = scaled.copy()
    original_scaled[fid] = scaled.copy()
    original_scaled[fid].setflags(write=False)
    seqs[fid] = None
    indices[fid] = start_times[fid]

    oc = coords_to_code_global(spec["origin"]) or "Custom"
    dc = coords_to_code_global(spec["dest"]) or "Custom"
    route_label[fid] = f"{oc}->{dc}"

flights = [spec["fid"] for spec in flights_spec]
persistent_active = {fid: False for fid in flights}
persistent_offset = {fid: 0.0 for fid in flights}
ramp_remaining = {fid: None for fid in flights}
applied_offset_current = {fid: 0.0 for fid in flights}

landed_frames_remaining = {fid: 0 for fid in flights}
LANDED_SHOW_FRAMES = max(1, int(round(LANDED_SHOW_SEC / PLOT_INTERVAL)))

# track which flights are completely finished (so they don't re-enter)
flight_finished = {fid: False for fid in flights}

dist_history = []
conflict_log = []
conflict_logged_pairs = set()
alt_adjust_logged_pairs = set()   # for altitude-adjustment messages and to prevent repeated adjustments

def add_increment_to_future(fid, increment_m):
    idx = indices[fid]
    if idx >= len(scaled_groups[fid]):
        return
    model_key = flight_model_key[fid]
    sx = model_registry[model_key]["scaler_X"]
    for i in range(idx, len(scaled_groups[fid])):
        base = sx.inverse_transform(scaled_groups[fid][i].reshape(1,-1))[0].copy()
        base[2] = float(base[2] + increment_m)
        scaled_groups[fid][i] = sx.transform(base.reshape(1,-1))[0]

def apply_persistent_general(fid, offset_m):
    idx = indices[fid]
    model_key = flight_model_key[fid]
    sx = model_registry[model_key]["scaler_X"]
    n = len(scaled_groups[fid])
    for i in range(idx, n):
        real = sx.inverse_transform(original_scaled[fid][i].reshape(1,-1))[0].copy()
        real[2] += offset_m
        scaled_groups[fid][i] = sx.transform(real.reshape(1,-1))[0]
    persistent_offset[fid] += offset_m
    applied_offset_current[fid] += offset_m
    persistent_active[fid] = True

def restore_general(fid):
    """
    Smoothly restore from current modified profile back to the ORIGINAL
    altitude profile for this flight (which includes climb, cruise, and descent).
    """
    start = indices[fid]
    model_key = flight_model_key[fid]
    sx = model_registry[model_key]["scaler_X"]
    n = len(scaled_groups[fid])
    for j in range(RETURN_SMOOTH_STEPS_AMD_DEL):
        i = start + j
        if i >= n: break
        cur = sx.inverse_transform(scaled_groups[fid][i].reshape(1,-1))[0].copy()
        orig = sx.inverse_transform(original_scaled[fid][i].reshape(1,-1))[0].copy()
        frac = (j+1)/RETURN_SMOOTH_STEPS_AMD_DEL
        blend = cur + (orig - cur) * frac
        scaled_groups[fid][i] = sx.transform(blend.reshape(1,-1))[0]
    for k in range(start + RETURN_SMOOTH_STEPS_AMD_DEL, n):
        scaled_groups[fid][k] = original_scaled[fid][k].copy()
    applied_offset_current[fid] = 0.0
    persistent_offset[fid] = 0.0
    persistent_active[fid] = False

def ramp_restore_to_cruise(fid, cruise_alt_m=None):
    restore_general(fid)

def resolve_and_start_ramp(fid_a, fid_b, collision_step):
    mk_a = flight_model_key[fid_a]; mk_b = flight_model_key[fid_b]

    if seqs.get(fid_a) is None or seqs.get(fid_b) is None:
        return False

    pa = predict_horizon(seqs[fid_a], flight_model_key[fid_a], fid=fid_a)
    pb = predict_horizon(seqs[fid_b], flight_model_key[fid_b], fid=fid_b)

    def minsep(p,q):
        md_km = float("inf"); md_alt = float("inf")
        for x,y in zip(p,q):
            d = haversine_km(x[0], x[1], y[0], y[1])
            a = abs(x[2] - y[2])
            if d < md_km:
                md_km = d; md_alt = a
        return md_km, md_alt

    before_km, before_alt = minsep(pa, pb)

    target = fid_a if fid_a < fid_b else fid_b

    candidates = [SINGLE_PLANE_INC_M, -SINGLE_PLANE_INC_M]
    best_inc = None
    best_min_alt = -1.0

    targ_pred = pa if target == fid_a else pb
    othr_pred = pb if target == fid_a else pa

    for inc in candidates:
        shifted = targ_pred.copy()
        shifted[:,2] = shifted[:,2] + inc
        min_alt = float("inf")
        for A,B in zip(shifted, othr_pred):
            va = abs(A[2] - B[2])
            if va < min_alt:
                min_alt = va
        if min_alt > best_min_alt:
            best_min_alt = min_alt
            best_inc = inc

    if best_inc is None:
        add_log("No suitable increment found — skipping conflict resolution.")
        return False

    ramp_full = np.linspace(0.0, best_inc, RAMP_UP_STEPS, endpoint=True)
    ramp_steps = (ramp_full[1:] - ramp_full[:-1]).tolist()

    if ramp_remaining.get(target):
        ramp_remaining[target].extend(ramp_steps)
    else:
        ramp_remaining[target] = ramp_steps

    applied_any = False
    if ramp_remaining[target] and len(ramp_remaining[target]) > 0:
        inc0 = ramp_remaining[target].pop(0)
        applied_offset_current[target] += inc0
        add_increment_to_future(target, inc0)
        ni = indices[target]
        if ni < len(scaled_groups[target]):
            model_key = flight_model_key[target]
            sx = model_registry[model_key]["scaler_X"]
            base = sx.inverse_transform(scaled_groups[target][ni].reshape(1,-1))[0].copy()
            base[2] = float(base[2] + inc0)
            scaled_groups[target][ni] = sx.transform(base.reshape(1,-1))[0]
            if seqs.get(target) is not None:
                seqs[target] = np.vstack([seqs[target][1:], scaled_groups[target][ni]])
        applied_any = True

    persistent_offset[target] = persistent_offset.get(target, 0.0) + best_inc

    if (ramp_remaining.get(target) and len(ramp_remaining[target]) > 0) or applied_any:
        persistent_active[target] = True
    else:
        persistent_active[target] = False

    pair_key = tuple(sorted((fid_a, fid_b)))
    if pair_key not in alt_adjust_logged_pairs:
        add_log(f"Altitude adjustment applied to flight {target} for conflict resolution.")
        alt_adjust_logged_pairs.add(pair_key)

    conflict_log.append({
        "time": time.time(),
        "fid_a": fid_a, "fid_b": fid_b,
        "target": target, "best_inc_m": best_inc,
        "before_m": before_km*1000.0, "before_alt_m": before_alt,
        "after_m": None, "after_alt_m": None
    })

    return True

# ---------------- Plot setup ----------------
plt.ion()
fig, ax = plt.subplots(figsize=(12,7))

plt.subplots_adjust(top=0.85, bottom=0.12, right=0.70)

ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.grid(True)

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
actual_lines = {}; pred_lines = {}; markers = {}; alt_texts = {}
for i, fid in enumerate(flights):
    actual_lines[fid], = ax.plot([], [], color=colors[i % len(colors)], linewidth=2)
    pred_lines[fid], = ax.plot([], [], color=colors[i % len(colors)], linestyle='--')
    markers[fid] = ax.scatter([], [], s=90, c=colors[i % len(colors)], edgecolor="black", zorder=5)
    alt_texts[fid] = ax.text(0,0,"", fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

airport_lons = []
airport_lats = []
for code, (lat, lon) in AIRPORTS.items():
    airport_lats.append(lat); airport_lons.append(lon)
ax.scatter(airport_lons, airport_lats, marker='^', s=140, c='black', zorder=15)
for code, (lat, lon) in AIRPORTS.items():
    ax.text(
        lon,
        lat + meters_to_deg_lat(120),
        code,
        fontsize=10, fontweight='bold',
        ha='center', va='bottom', zorder=16,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
    )

all_lats = df_all["latitude"].to_numpy()
all_lons = df_all["longitude"].to_numpy()
ax.set_xlim(all_lons.min()-0.5, all_lons.max()+0.5)
ax.set_ylim(all_lats.min()-0.5, all_lats.max()+0.5)

plt.pause(0.2)

# ---------------- Weather panel (figure top) ----------------
def _build_weather_text(airport_weather_dict):
    lines = []
    for code in sorted(AIRPORTS.keys()):
        status = airport_weather_dict.get(code, "Unknown")
        lines.append(f"{code}: {status}")
    return "   ".join(lines)

fig_text = fig.text(
    0.02, 0.96,
    _build_weather_text(airport_weather if 'airport_weather' in globals() else {k: "Unknown" for k in AIRPORTS}),
    transform=fig.transFigure,
    va='center', ha='left',
    fontsize=10,
    bbox=dict(facecolor='white', alpha=0.95, boxstyle='round,pad=0.4'),
    zorder=30
)

def draw_weather_panel():
    try:
        txt = _build_weather_text(airport_weather)
        fig_text.set_text(txt)
    except Exception as e:
        print(f"Warning updating weather panel: {e}")

# ---------------- Messages panel ----------------
MESSAGE_WRAP_WIDTH = 40
MAX_LOG_VISIBLE_LINES = 30

log_panel_title = fig.text(
    0.72, 0.96,
    "Messages",
    transform=fig.transFigure,
    va='top', ha='left',
    fontsize=11,
    fontweight='bold',
    bbox=dict(facecolor='white', alpha=0.95, boxstyle='round,pad=0.2'),
    zorder=21
)

log_panel = fig.text(
    0.72, 0.93,
    "",
    transform=fig.transFigure,
    va='top', ha='left',
    fontsize=10,
    bbox=dict(facecolor='white', alpha=0.95, boxstyle='round,pad=0.25'),
    zorder=20
)

log_lines_wrapped = []
log_scroll_offset = 0  # 0 = bottom (latest), positive -> scroll up

def update_log_panel():
    global log_lines_wrapped, log_scroll_offset
    try:
        wrapped_lines = []
        for m in log_messages:
            wrapped = textwrap.wrap(m, MESSAGE_WRAP_WIDTH)
            if not wrapped:
                wrapped_lines.append("")
            else:
                wrapped_lines.extend(wrapped)

        log_lines_wrapped = wrapped_lines
        total = len(log_lines_wrapped)

        if total <= MAX_LOG_VISIBLE_LINES:
            start = 0
        else:
            base_start = max(0, total - MAX_LOG_VISIBLE_LINES)
            start = base_start + log_scroll_offset
            start = max(0, min(start, max(0, total - MAX_LOG_VISIBLE_LINES)))

        end = start + MAX_LOG_VISIBLE_LINES
        visible = log_lines_wrapped[start:end]
        log_panel.set_text("\n".join(visible))
    except Exception as e:
        print("Message panel update failed:", e)

def on_scroll(event):
    global log_scroll_offset
    if event.inaxes is not None:
        return
    if event.button == 'up':
        log_scroll_offset -= 3
    elif event.button == 'down':
        log_scroll_offset += 3
    update_log_panel()
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('scroll_event', on_scroll)

# ---------------- Legend ----------------
proxy_actual = Line2D([0], [0], color="black", lw=2, linestyle='-')
proxy_pred = Line2D([0], [0], color="black", lw=2, linestyle='--')
proxy_marker = Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=10)
proxy_airport = Line2D([0], [0], marker='^', linestyle='None', markerfacecolor='black', markeredgecolor='black', markersize=10)
proxy_weather = mpatches.Patch(facecolor='white', edgecolor='black', label='Weather panel (top)')

legend_handles = [proxy_actual, proxy_pred, proxy_marker, proxy_airport, proxy_weather]
legend_labels = ["Actual path (solid)", "Predicted path (dashed)", "Current position marker", "Airport (triangle)", "Weather panel"]

legend = fig.legend(
    legend_handles, legend_labels,
    loc='lower left', bbox_to_anchor=(0.02, 0.02),
    fontsize=9, frameon=True, facecolor='white', framealpha=0.95
)

# ---------------- Draw altitude/status ----------------
def draw_alt_label(fid):
    txt = alt_texts.get(fid)
    if txt is None:
        return

    if seqs.get(fid) is not None:
        model_key = flight_model_key[fid]
        sx = model_registry[model_key]["scaler_X"]
        cur = sx.inverse_transform(seqs[fid][-1].reshape(1,-1))[0]
        lat, lon, alt = cur[0], cur[1], cur[2]
    else:
        lat = lon = alt = 0.0

    dx = meters_to_deg_lon(800.0, lat)
    dy_main = meters_to_deg_lat(800.0)
    dy_landed = meters_to_deg_lat(60.0)

    if (
        seqs.get(fid) is not None and
        indices[fid] >= len(scaled_groups[fid]) and
        landed_frames_remaining.get(fid,0) == 0 and
        not flight_finished.get(fid, False)
    ):
        msg_landed = f"🛬 Flight {fid} landed."
        add_log(msg_landed)
        print(msg_landed)
        landed_frames_remaining[fid] = LANDED_SHOW_FRAMES

    if landed_frames_remaining.get(fid,0) > 0:
        if seqs.get(fid) is not None:
            model_key = flight_model_key[fid]
            sx = model_registry[model_key]["scaler_X"]
            cur = sx.inverse_transform(seqs[fid][-1].reshape(1,-1))[0]
            lat, lon = cur[0], cur[1]
        txt.set_position((lon + dx, lat + dy_landed))
        txt.set_text(f"fid={fid} Landed")
        return

    if seqs.get(fid) is None:
        return

    tag = "(P)" if persistent_active[fid] else ""
    route = route_label.get(fid, "")
    txt.set_position((lon + dx, lat + dy_main))
    txt.set_text(f"fid={fid} {route} {alt:.0f} m {tag}")

# ---------------- Main loop ----------------
sim_time = 0
while True:
    for fid in flights:
        if flight_finished.get(fid, False):
            continue
        if (
            seqs[fid] is None and
            sim_time >= start_times[fid] and
            landed_frames_remaining.get(fid,0) == 0
        ):
            seqs[fid] = scaled_groups[fid][0:SEQ_LEN].copy()
            indices[fid] = SEQ_LEN
            add_log(f"✈ Flight {fid} ENTERED SIM at t={sim_time} (model={flight_model_key[fid]})")

    for fid in flights:
        if seqs.get(fid) is None or flight_finished.get(fid, False):
            continue
        if ramp_remaining[fid] is not None and len(ramp_remaining[fid]) > 0:
            inc = ramp_remaining[fid].pop(0)
            applied_offset_current[fid] += inc
            add_increment_to_future(fid, inc)
            ni = indices[fid]
            if ni < len(scaled_groups[fid]):
                model_key = flight_model_key[fid]
                sx = model_registry[model_key]["scaler_X"]
                base = sx.inverse_transform(scaled_groups[fid][ni].reshape(1,-1))[0].copy()
                base[2] = float(base[2] + inc)
                scaled_groups[fid][ni] = sx.transform(base.reshape(1,-1))[0]
                seqs[fid] = np.vstack([seqs[fid][1:], scaled_groups[fid][ni]])
            if len(ramp_remaining[fid]) == 0:
                ramp_remaining[fid] = None

    if len(flights) >= 2:
        f0, f1 = flights[0], flights[1]
        if (
            seqs.get(f0) is not None and not flight_finished.get(f0, False) and
            seqs.get(f1) is not None and not flight_finished.get(f1, False)
        ):
            mk0 = flight_model_key[f0]; mk1 = flight_model_key[f1]
            sx0 = model_registry[mk0]["scaler_X"]; sx1 = model_registry[mk1]["scaler_X"]
            cur0 = sx0.inverse_transform(seqs[f0][-1].reshape(1,-1))[0]
            cur1 = sx1.inverse_transform(seqs[f1][-1].reshape(1,-1))[0]
            dkm = haversine_km(cur0[0], cur0[1], cur1[0], cur1[1])
            dist_history.append(dkm)

    predictions = {}
    for fid in flights:
        if seqs.get(fid) is None or flight_finished.get(fid, False):
            continue
        if landed_frames_remaining.get(fid,0) > 0:
            continue

        mk = flight_model_key[fid]
        pred = predict_horizon(seqs[fid], mk, fid=fid)
        predictions[fid] = pred
        model_key = flight_model_key[fid]
        sx = model_registry[model_key]["scaler_X"]
        actual_real = sx.inverse_transform(seqs[fid])
        if actual_lines.get(fid) is not None:
            actual_lines[fid].set_data(actual_real[:,1], actual_real[:,0])
        if pred_lines.get(fid) is not None:
            pred_lines[fid].set_data(pred[:,1], pred[:,0])
        if markers.get(fid) is not None:
            try:
                markers[fid].set_offsets([[actual_real[-1,1], actual_real[-1,0]]])
            except Exception:
                pass
        draw_alt_label(fid)

    def compute_predictions_for_active(active_list, predictions_dict):
        for fid in active_list:
            if (
                seqs.get(fid) is None or
                landed_frames_remaining.get(fid,0) > 0 or
                flight_finished.get(fid, False)
            ):
                continue
            mk = flight_model_key[fid]
            try:
                predictions_dict[fid] = predict_horizon(seqs[fid], mk, fid=fid)
            except Exception as e:
                print(f"Prediction recompute failed for fid={fid}: {e}")
        return predictions_dict

    active = [
        fid for fid in flights
        if seqs.get(fid) is not None
        and landed_frames_remaining.get(fid,0) == 0
        and not flight_finished.get(fid, False)
    ]
    predictions = compute_predictions_for_active(active, predictions)

    i = 0
    conflict_found_and_applied = False
    while i < len(active):
        j = i + 1
        while j < len(active):
            fa, fb = active[i], active[j]
            if fa not in predictions or fb not in predictions:
                j += 1
                continue
            pa, pb = predictions[fa], predictions[fb]
            conflict = False; step = None
            for s, (A, B) in enumerate(zip(pa, pb)):
                d = haversine_km(A[0], A[1], B[0], B[1])
                h = abs(A[2] - B[2])
                if d < HORIZ_KM and h < ALT_THRESH_M:
                    conflict = True; step = s; break
            if conflict:
                pair_key = tuple(sorted((fa, fb)))
                if pair_key not in conflict_logged_pairs:
                    add_log(f"⚠️ Predicted conflict between {fa} & {fb} at horizon-step {step}")
                    conflict_logged_pairs.add(pair_key)

                # *** NEW: only apply altitude adjustment once per pair ***
                if pair_key in alt_adjust_logged_pairs:
                    # We've already done a separation maneuver for this pair;
                    # don't stack more altitude changes on top.
                    j += 1
                    continue

                applied = resolve_and_start_ramp(fa, fb, step)
                ax.set_title("Conflict -> Single-plane gradual separation (persistent)", color='red')
                if applied:
                    predictions = compute_predictions_for_active(active, {})
                    conflict_found_and_applied = True
                    i = 0
                    j = i + 1
                    break
            j += 1
        if conflict_found_and_applied:
            conflict_found_and_applied = False
            continue
        i += 1

    try:
        any_conflict = False
        for idx, a in enumerate(active):
            for b in active[idx+1:]:
                pa = predictions.get(a, [])
                pb = predictions.get(b, [])
                for A, B in zip(pa, pb):
                    if (
                        haversine_km(A[0], A[1], B[0], B[1]) < HORIZ_KM
                        and abs(A[2] - B[2]) < ALT_THRESH_M
                    ):
                        any_conflict = True
                        break
                if any_conflict:
                    break
            if any_conflict:
                break

        if not any_conflict:
            ax.set_title("No imminent conflict", color="black")
            for fid in flights:
                if (
                    persistent_active.get(fid) and
                    ramp_remaining.get(fid) is None and
                    not flight_finished.get(fid, False)
                ):
                    try:
                        ramp_restore_to_cruise(fid, cruise_alt_m=CRUISE_ALT_M)
                    except Exception as e:
                        print(f"Warning: ramp_restore_to_cruise failed for fid={fid}: {e}")

    except Exception:
        ax.set_title("No imminent conflict", color="black")

    for fid in flights:
        if (
            seqs.get(fid) is None or
            landed_frames_remaining.get(fid,0) > 0 or
            flight_finished.get(fid, False)
        ):
            continue
        if indices[fid] < len(scaled_groups[fid]):
            seqs[fid] = np.vstack([seqs[fid][1:], scaled_groups[fid][indices[fid]]])
            indices[fid] += 1

    for fid in flights:
        draw_alt_label(fid)

    draw_weather_panel()
    update_log_panel()

    for fid in flights:
        if landed_frames_remaining.get(fid,0) > 0 and not flight_finished.get(fid, False):
            landed_frames_remaining[fid] -= 1
            if landed_frames_remaining[fid] <= 0:
                try:
                    if actual_lines.get(fid) is not None:
                        actual_lines[fid].remove()
                        actual_lines.pop(fid, None)
                    if pred_lines.get(fid) is not None:
                        pred_lines[fid].remove()
                        pred_lines.pop(fid, None)
                    if markers.get(fid) is not None:
                        markers[fid].remove()
                        markers.pop(fid, None)
                    if alt_texts.get(fid) is not None:
                        alt_texts[fid].remove()
                        alt_texts.pop(fid, None)
                except Exception as e:
                    print(f"Warning while final-removing fid={fid}: {e}")
                persistent_active[fid] = False
                persistent_offset[fid] = 0.0
                ramp_remaining[fid] = None
                applied_offset_current[fid] = 0.0
                seqs[fid] = None
                flight_finished[fid] = True

    fig.canvas.draw(); fig.canvas.flush_events()
    time.sleep(PLOT_INTERVAL)

    sim_time += 1

    if all(flight_finished.get(fid, False) for fid in flights):
        break

plt.ioff()
plt.show()
print("Simulation finished.")
