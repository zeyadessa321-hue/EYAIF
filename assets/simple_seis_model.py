"""
seis_gui_advanced.py
Advanced GUI for Seismic Prediction & Energy Estimation
Author: Zeyad Ahmed (template by ChatGPT)
Requirements:
pip install customtkinter numpy pandas scikit-learn joblib matplotlib
"""

import os
import time
import csv
import threading
import numpy as np
import pandas as pd
import customtkinter as ctk
from tkinter import messagebox, filedialog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from joblib import dump, load
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# -------------------------
# Constants & paths
# -------------------------
MODEL_FILE = "seis_advanced_rf.joblib"
LOG_CSV = "seis_runs_advanced.csv"

# -------------------------
# Synthetic dataset generator (more features, more realistic variety)
# -------------------------
def generate_synthetic_dataset(n_samples=12000, random_state=42):
    rng = np.random.RandomState(random_state)
    # features: magnitude_recent, depth_km, ground_accel, plate_velocity, magnetic_uT,
    # freq_hz, temp_variation, animal_index, lunar_phase_idx, atm_pressure, efield_change,
    # co2_ppm, stress_MPa, ground_tilt_deg
    mag = np.clip(rng.normal(2.0, 0.8, size=n_samples) + rng.rand(n_samples)*1.5, 0.0, 9.0)
    depth = np.abs(rng.normal(10, 15, size=n_samples))  # km
    ground_accel = np.abs(rng.normal(0.01, 0.03, size=n_samples))  # m/s^2
    plate_vel = np.abs(rng.normal(5, 3, size=n_samples))  # mm/year converted to approx
    mag_uT = (rng.randn(n_samples) * 0.02)  # µT-level fluctuations
    freq_hz = np.clip(np.abs(rng.normal(1.0, 0.8, size=n_samples)), 0.01, 20.0)
    temp_var = rng.normal(0.2, 0.6, size=n_samples)  # deg C
    animal_idx = np.clip(rng.beta(2,8,size=n_samples), 0, 1)  # 0..1
    # lunar phase index: use circular-like random but with clustering near new/full in some samples
    lunar = rng.choice([-1.0, -0.7, -0.5, -0.2, 1.0, 0.2, 0.5, 0.7], size=n_samples, p=[0.12,0.11,0.13,0.12,0.2,0.1,0.13,0.09])
    atm_pressure = rng.normal(1013, 8, size=n_samples)  # hPa
    efield = (rng.randn(n_samples) * 0.5)  # V/m
    co2 = np.clip(400 + rng.randn(n_samples)*30, 200, 1000)  # ppm
    stress = np.clip(rng.normal(20, 10, size=n_samples), 0.1, 200)  # MPa (toy)
    tilt = rng.normal(0.01, 0.05, size=n_samples)  # degrees

    # combine into DataFrame
    data = np.vstack([mag, depth, ground_accel, plate_vel, mag_uT, freq_hz, temp_var,
                      animal_idx, lunar, atm_pressure, efield, co2, stress, tilt]).T
    cols = ['magnitude_recent','depth_km','ground_accel','plate_velocity','magnetic_uT',
            'freq_hz','temp_variation','animal_index','lunar_index','atm_pressure',
            'efield_change','co2_ppm','stress_MPa','ground_tilt']
    df = pd.DataFrame(data, columns=cols)

    # create probabilistic label — synthetic risk model:
    # more weight to magnitude, deep small negative; higher accel, higher stress, lunar extremes, animal spikes, freq patterns
    score = (0.8 * df['magnitude_recent']) \
            + (5.0 * df['ground_accel'] * 100.0) \
            + (0.03 * df['stress_MPa']) \
            + (1.2 * np.abs(df['lunar_index'])) \
            + (3.0 * df['animal_index']) \
            + (0.5 * (df['co2_ppm'] - 400)/100.0) \
            + (0.2 * np.clip((df['freq_hz']-0.5),0,10))
    # logistic to probability
    prob = 1 / (1 + np.exp(-0.6*(score - 4.0)))
    rng2 = rng.rand(n_samples)
    y = (rng2 < prob).astype(int)

    df['label'] = y
    return df

# -------------------------
# Model train / load functions
# -------------------------
def train_and_save_model(df=None, n_estimators=200):
    if df is None:
        df = generate_synthetic_dataset()
    X = df.drop(columns=['label']).values
    y = df['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=7, stratify=y)
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=12, n_jobs=-1, random_state=0)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_prob)
    dump(clf, MODEL_FILE)
    return clf, auc, X_test, y_test

def load_model_if_exists():
    if os.path.exists(MODEL_FILE):
        try:
            m = load(MODEL_FILE)
            return m
        except Exception as e:
            print("Model load failed:", e)
            return None
    return None

# -------------------------
# Small helper utilities
# -------------------------
def log_run_to_csv(row):
    header = ['timestamp'] + list(row.keys())
    exists = os.path.exists(LOG_CSV)
    with open(LOG_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S")] + list(row.values()))

# -------------------------
# GUI building
# -------------------------
ctk.set_appearance_mode("System")  # "Dark" or "Light"
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("SeisGuard — Advanced Prediction & Energy Estimator")
app.geometry("1200x760")

# frames
frame_top = ctk.CTkFrame(master=app, corner_radius=8)
frame_top.pack(padx=12, pady=10, fill="x")

lbl_title = ctk.CTkLabel(frame_top, text="SeisGuard — Advanced Demo", font=ctk.CTkFont(size=20, weight="bold"))
lbl_title.pack(pady=8)

frame_main = ctk.CTkFrame(master=app, corner_radius=8)
frame_main.pack(padx=12, pady=(0,12), fill="both", expand=True)

left_frame = ctk.CTkFrame(master=frame_main, width=420)
left_frame.pack(side="left", padx=12, pady=12, fill="y")

mid_frame = ctk.CTkFrame(master=frame_main)
mid_frame.pack(side="left", padx=6, pady=12, fill="both", expand=True)

right_frame = ctk.CTkFrame(master=frame_main, width=320)
right_frame.pack(side="right", padx=12, pady=12, fill="y")

# -------------------------
# Left: Inputs (14 fields)
# -------------------------
inputs = {}

def add_input_label_entry(parent, text, var_name, default, step=None, from_=None, to=None):
    label = ctk.CTkLabel(parent, text=text, anchor="w")
    label.pack(padx=8, pady=(8,2), fill="x")
    if isinstance(default, float):
        entry = ctk.CTkEntry(parent)
        entry.insert(0, str(default))
    else:
        entry = ctk.CTkEntry(parent)
        entry.insert(0, str(default))
    entry.pack(padx=8, pady=(0,6), fill="x")
    inputs[var_name] = entry

# Add 14 inputs
add_input_label_entry(left_frame, "Recent Magnitude (Mw)", "magnitude_recent", 2.5)
add_input_label_entry(left_frame, "Depth (km)", "depth_km", 10.0)
add_input_label_entry(left_frame, "Ground acceleration (m/s^2)", "ground_accel", 0.02)
add_input_label_entry(left_frame, "Plate velocity (mm/yr)", "plate_velocity", 5.0)
add_input_label_entry(left_frame, "Magnetic fluctuation (µT)", "magnetic_uT", 0.01)
add_input_label_entry(left_frame, "Seismic freq dominant (Hz)", "freq_hz", 1.0)
add_input_label_entry(left_frame, "Temp variation (°C)", "temp_variation", 0.2)
add_input_label_entry(left_frame, "Animal activity index (0-1)", "animal_index", 0.05)
# lunar: dropdown for clarity
label = ctk.CTkLabel(left_frame, text="Lunar Phase (select)", anchor="w")
label.pack(padx=8, pady=(8,2), fill="x")
lunar_values = ["New Moon","Waxing Crescent","First Quarter","Waxing Gibbous","Full Moon","Waning Gibbous","Last Quarter","Waning Crescent"]
lunar_combo = ctk.CTkComboBox(left_frame, values=lunar_values)
lunar_combo.set("Full Moon")
lunar_combo.pack(padx=8, pady=(0,6), fill="x")
inputs['lunar_phase'] = lunar_combo

add_input_label_entry(left_frame, "Atmospheric pressure (hPa)", "atm_pressure", 1013.25)
add_input_label_entry(left_frame, "Electric field change (V/m)", "efield_change", 0.0)
add_input_label_entry(left_frame, "Gas emission CO2 (ppm)", "co2_ppm", 410.0)
add_input_label_entry(left_frame, "Historical stress (MPa)", "stress_MPa", 20.0)
add_input_label_entry(left_frame, "Ground tilt (deg)", "ground_tilt", 0.01)

# action buttons on left
btn_frame = ctk.CTkFrame(left_frame)
btn_frame.pack(padx=8, pady=12, fill="x")

def on_generate_dataset():
    # small blocking op — run in thread?
    def job():
        df = generate_synthetic_dataset(n_samples=8000, random_state=int(time.time()%100000))
        # save to CSV for inspection
        df.to_csv("synthetic_seismic_dataset.csv", index=False)
        messagebox.showinfo("Dataset", "Synthetic dataset generated and saved as 'synthetic_seismic_dataset.csv'")
    threading.Thread(target=job).start()

def on_train_model():
    def job():
        try:
            btn_train.configure(state="disabled")
            clf, auc, _, _ = train_and_save_model()
            messagebox.showinfo("Model trained", f"Model trained and saved ({MODEL_FILE}). Validation AUC = {auc:.3f}")
        except Exception as e:
            messagebox.showerror("Training error", str(e))
        finally:
            btn_train.configure(state="normal")
    threading.Thread(target=job).start()

btn_generate = ctk.CTkButton(btn_frame, text="Generate synthetic dataset", command=on_generate_dataset)
btn_generate.pack(side="left", padx=6, pady=6, fill="x", expand=True)

btn_train = ctk.CTkButton(btn_frame, text="Train model (fast)", command=on_train_model)
btn_train.pack(side="left", padx=6, pady=6, fill="x", expand=True)

# -------------------------
# Middle: plots & outputs
# -------------------------
# Placeholders for matplotlib figure
fig, ax = plt.subplots(figsize=(7,3))
ax.set_title("Seismogram / Live preview")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
line, = ax.plot([], [], color="#0b6fbf")
ax.grid(alpha=0.3)

canvas = FigureCanvasTkAgg(fig, master=mid_frame)
canvas.get_tk_widget().pack(padx=8, pady=8, fill="both", expand=True)

# result label
result_label = ctk.CTkLabel(mid_frame, text="Prediction: —", font=ctk.CTkFont(size=16, weight="bold"))
result_label.pack(pady=(6,4))

detail_label = ctk.CTkLabel(mid_frame, text="Details will appear here.", wraplength=600, justify="left")
detail_label.pack(pady=(0,8))

# -------------------------
# Right: generator params & controls
# -------------------------
ctk.CTkLabel(right_frame, text="Generator parameters", font=ctk.CTkFont(size=15, weight="bold")).pack(pady=6)
gen_params = {}
def add_gen_param(parent, text, var_name, default):
    lbl = ctk.CTkLabel(parent, text=text, anchor="w")
    lbl.pack(padx=8, pady=(6,2), fill="x")
    ent = ctk.CTkEntry(parent)
    ent.insert(0, str(default))
    ent.pack(padx=8, pady=(0,6), fill="x")
    gen_params[var_name] = ent

add_gen_param(right_frame, "Coil turns (N)", "N", 500)
add_gen_param(right_frame, "Magnetic field B (T)", "B", 0.5)
add_gen_param(right_frame, "Coil area A (m^2)", "A", 0.01)
add_gen_param(right_frame, "Proof mass (kg)", "mass", 2.0)
add_gen_param(right_frame, "Spring k (N/m)", "k", 50.0)

# Buttons: Predict, Log, Load model, Load CSV
def assemble_feature_vector():
    try:
        fv = {}
        fv['magnitude_recent'] = float(inputs['magnitude_recent'].get())
        fv['depth_km'] = float(inputs['depth_km'].get())
        fv['ground_accel'] = float(inputs['ground_accel'].get())
        fv['plate_velocity'] = float(inputs['plate_velocity'].get())
        fv['magnetic_uT'] = float(inputs['magnetic_uT'].get())
        fv['freq_hz'] = float(inputs['freq_hz'].get())
        fv['temp_variation'] = float(inputs['temp_variation'].get())
        fv['animal_index'] = float(inputs['animal_index'].get())
        # lunar mapping
        lun = inputs['lunar_phase'].get()
        phase_map = {"New Moon": -1.0, "Waxing Crescent": -0.7, "First Quarter": -0.5, "Waxing Gibbous": -0.2,
                     "Full Moon": 1.0, "Waning Gibbous": 0.2, "Last Quarter": 0.5, "Waning Crescent": 0.7}
        fv['lunar_index'] = phase_map.get(lun, 0.0)
        fv['atm_pressure'] = float(inputs['atm_pressure'].get())
        fv['efield_change'] = float(inputs['efield_change'].get())
        fv['co2_ppm'] = float(inputs['co2_ppm'].get())
        fv['stress_MPa'] = float(inputs['stress_MPa'].get())
        fv['ground_tilt'] = float(inputs['ground_tilt'].get())
        return fv
    except Exception as e:
        messagebox.showerror("Input error", f"Please ensure all inputs are numeric.\n{e}")
        return None

def compute_energy_estimates(seismo, N, B, A, mass, k):
    # coarse estimate: use peak sample diff as proxy for velocity -> convert with scale
    v_est = np.abs(np.diff(seismo)).max() if len(seismo)>1 else 0.0
    scale = 0.05  # tuning factor (explain in poster)
    v_m_s = v_est * scale
    V_gen = N * B * A * v_m_s
    Rload = 10.0
    P = (V_gen**2) / (Rload + 1e-9)
    return V_gen, P, v_m_s

def on_predict():
    fv = assemble_feature_vector()
    if fv is None:
        return
    # build sample for model order
    feature_order = ['magnitude_recent','depth_km','ground_accel','plate_velocity','magnetic_uT',
                     'freq_hz','temp_variation','animal_index','lunar_index','atm_pressure',
                     'efield_change','co2_ppm','stress_MPa','ground_tilt']
    X = np.array([fv[name] for name in feature_order]).reshape(1,-1)
    mdl = load_model_if_exists()
    if mdl is not None:
        prob = float(mdl.predict_proba(X)[0,1])
    else:
        # fallback heuristic similar to synthetic scoring
        score = (0.8 * fv['magnitude_recent']) + (5.0 * fv['ground_accel'] * 100.0) + (0.03 * fv['stress_MPa']) \
                + (1.2 * abs(fv['lunar_index'])) + (3.0 * fv['animal_index'])
        prob = float(1 / (1 + np.exp(-0.6*(score - 4.0))))

    # simulate a seismogram for visualization (short)
    t = np.linspace(0, 8, 1000)
    base_freq = max(0.1, 0.4 + (fv['magnitude_recent']/9.0)*5.0)
    amp = 0.1 + (fv['magnitude_recent']/9.0)*1.5 + abs(fv['lunar_index'])*0.3 + abs(fv['magnetic_uT'])*2.0 + fv['animal_index']*0.8
    signal = amp * np.sin(2*np.pi*base_freq*t) * np.exp(-t*0.08)
    # add bursts depending on magnitude and plate velocity
    bursts = np.zeros_like(t)
    rng = np.random.RandomState(int(time.time()%10000))
    n_bursts = int(min(6, max(0, int(fv['magnitude_recent']))))
    for _ in range(n_bursts):
        center = rng.randint(0, len(t))
        width = rng.randint(3, 60)
        mag = rng.rand()*amp*2.0
        kernel = mag * np.exp(-((np.arange(len(t))-center)**2)/(2*(width**2)))
        bursts += kernel
    noise = rng.randn(len(t)) * (0.02 + 0.2*fv['ground_accel'])
    seismo = signal + bursts + noise

    # update plot
    line.set_data(t, seismo)
    ax.relim()
    ax.autoscale_view()
    canvas.draw()

    # energy estimate
    try:
        N = float(gen_params['N'].get())
        B = float(gen_params['B'].get())
        A = float(gen_params['A'].get())
        mass = float(gen_params['mass'].get())
        k = float(gen_params['k'].get())
    except:
        N,B,A,mass,k = 500,0.5,0.01,2.0,50.0

    V_gen, P_est, v_ms = compute_energy_estimates(seismo, N, B, A, mass, k)

    # display result
    prob_pct = prob*100
    level = "LOW"
    if prob>=0.7:
        level="HIGH"
    elif prob>=0.4:
        level="MEDIUM"

    result_label.configure(text=f"Prediction: {prob_pct:.1f}%  — Risk level: {level}")
    detail_label.configure(text=f"Estimated open-circuit V = {V_gen:.4f} V | Estimated instantaneous power ≈ {P_est:.6f} W\n"
                                f"Peak velocity proxy ≈ {v_ms:.4f} m/s\n"
                                f"Model used: {'trained RF' if load_model_if_exists() is not None else 'heuristic'}")

    # log run
    row = {
        'magnitude_recent': fv['magnitude_recent'],
        'depth_km': fv['depth_km'],
        'ground_accel': fv['ground_accel'],
        'plate_velocity': fv['plate_velocity'],
        'magnetic_uT': fv['magnetic_uT'],
        'freq_hz': fv['freq_hz'],
        'temp_variation': fv['temp_variation'],
        'animal_index': fv['animal_index'],
        'lunar_phase': inputs['lunar_phase'].get(),
        'lunar_index': fv['lunar_index'],
        'atm_pressure': fv['atm_pressure'],
        'efield_change': fv['efield_change'],
        'co2_ppm': fv['co2_ppm'],
        'stress_MPa': fv['stress_MPa'],
        'ground_tilt': fv['ground_tilt'],
        'probability': prob,
        'V_gen': V_gen,
        'P_est': P_est
    }
    log_run_to_csv(row)

btn_predict = ctk.CTkButton(right_frame, text="Predict", fg_color="#1a73e8", command=on_predict)
btn_predict.pack(padx=12, pady=8, fill="x")

def on_load_model():
    mdl = load_model_if_exists()
    if mdl is not None:
        messagebox.showinfo("Model load", f"Model loaded from {MODEL_FILE}")
    else:
        messagebox.showwarning("Model", "No model found. Train one first (use left panel).")

btn_load_model = ctk.CTkButton(right_frame, text="Load model", command=on_load_model)
btn_load_model.pack(padx=12, pady=(0,8), fill="x")

def on_export_logs():
    if not os.path.exists(LOG_CSV):
        messagebox.showinfo("No logs", "No run logs found yet.")
        return
    fpath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV file","*.csv")], initialfile="seis_runs_advanced.csv")
    if fpath:
        with open(LOG_CSV, 'r') as fr, open(fpath, 'w', newline='') as fw:
            fw.write(fr.read())
        messagebox.showinfo("Exported", f"Logs exported to {fpath}")

btn_export = ctk.CTkButton(right_frame, text="Export logs (CSV)", command=on_export_logs)
btn_export.pack(padx=12, pady=(0,8), fill="x")

def on_open_logs():
    if not os.path.exists(LOG_CSV):
        messagebox.showinfo("No logs", "No run logs yet.")
        return
    df = pd.read_csv(LOG_CSV)
    # show brief stats
    top = tk_top = ctk.CTkToplevel(app)
    top.geometry("800x400")
    top.title("Logged runs")
    txt = ctk.CTkTextbox(top, width=780, height=360)
    txt.pack(padx=10, pady=10, fill="both", expand=True)
    txt.insert("0.0", df.to_string(index=False))

btn_open_logs = ctk.CTkButton(right_frame, text="View logs", command=on_open_logs)
btn_open_logs.pack(padx=12, pady=(0,8), fill="x")

# -------------------------
# Finalize & start
# -------------------------
# load initial model if available
_loaded = load_model_if_exists()
if _loaded is not None:
    print("Initial model available.")

app.mainloop()
