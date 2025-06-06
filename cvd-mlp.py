#Developed by ODAT project
#please see https://odat.info
#please see https://github.com/ODAT-Project
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.calibration import calibration_curve
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index #for Cox C-index
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import gc
import traceback

STYLE_CONFIG = {
    "font_family": "Segoe UI", "font_size_normal": 10, "font_size_header": 14,
    "font_size_section": 12, "bg_root": "#F0F0F0", "bg_widget": "#FFFFFF",
    "bg_entry": "#FFFFFF", "fg_text": "#333333", "fg_header": "#000000",
    "accent_color": "#0078D4", "accent_text_color": "#FFFFFF", "border_color": "#CCCCCC",
    "listbox_select_bg": "#0078D4", "listbox_select_fg": "#FFFFFF",
    "disabled_bg": "#E0E0E0", "disabled_fg": "#A0A0A0", "error_text_color": "#D32F2F",
}

class MLPModel(nn.Module):
    #deep learning model for survival analysis to generate a risk score, a feed-forward neural network that outputs a single log-risk value.
    def __init__(self, input_features, hidden_layers):
        
        super(MLPModel, self).__init__()
        
        layers = []
        
        prev_layer_size = input_features
        
        for layer_size in hidden_layers:
            
            layers.append(nn.Linear(prev_layer_size, layer_size))
            
            layers.append(nn.ReLU())
            
            layers.append(nn.Dropout(0.5))
            
            prev_layer_size = layer_size
        
        layers.append(nn.Linear(prev_layer_size, 1))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def cox_loss(log_risk, times, events):
    events = events.type(torch.bool)
    
    if not torch.any(events):
        return torch.tensor(0.0, requires_grad=True)

    sorted_times, sort_indices = torch.sort(times, descending=True)
    
    sorted_log_risk = log_risk[sort_indices]
    
    sorted_events = events[sort_indices]

    log_sum_exp_risk = torch.logcumsumexp(sorted_log_risk, dim=0)
    
    observed_log_risk = sorted_log_risk[sorted_events]
    
    observed_log_sum_exp_risk = log_sum_exp_risk[sorted_events]
    
    loss = -torch.sum(observed_log_risk - observed_log_sum_exp_risk)
    
    return loss

class DynamicCVDApp:
    DL_RISK_SCORE_COL_NAME = 'MLP_Risk_Score_Covariate' #the deep learning risk score, we print as MLP score

    def __init__(self, root):
        self.root = root
        
        self.root.title("Advanced Dynamic CVD Risk Predictor -- MLP-CPH Enhanced")
        
        self.root.geometry("1450x980")
        
        self.root.configure(bg=STYLE_CONFIG["bg_root"])

        self.data_df = None
        
        self.dl_model = None
        
        self.cph_model = None
        
        self.scaler_dl = None
        
        self.scaler_cph_linear = None
        
        self.num_imputer_dl = None
        
        self.num_imputer_cph_linear = None

        self.trained_dl_feature_names = []
        
        self.trained_cph_linear_feature_names = []
        
        self.all_base_features_for_input = []

        self.trained_feature_medians_dl = {}
        
        self.trained_feature_medians_cph_linear = {}
        
        self.scaled_columns_dl = []
        
        self.scaled_columns_cph_linear = []

        self.train_cph_df_for_metrics = None
        
        self.test_cph_df_for_metrics = None

        self.target_event_col_var = tk.StringVar()
        
        self.time_horizon_var = tk.StringVar(value="5")
        
        self.time_to_event_col_var = tk.StringVar()

        self.learning_rate_var = tk.StringVar(value="0.001")
        
        self.epochs_var = tk.StringVar(value="50")
        
        self.hidden_layers_var = tk.StringVar(value="64,32")
        
        self.cph_penalizer_var = tk.StringVar(value="0.1")

        self.prediction_input_widgets = {}
        
        self.dynamic_input_scrollable_frame = None
        
        self.more_plots_window = None

        self.setup_styles()
        
        self.create_menu()
        
        self.create_main_layout()
        
        self.toggle_train_predict_sections_enabled(data_loaded=False, model_trained=False)

    
    
    def setup_styles(self):
        s = ttk.Style(self.root)
        
        s.theme_use("default")

        font_normal = (STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_normal"])
        
        font_bold = (STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_normal"], "bold")
        
        font_header = (STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_header"], "bold")
        
        font_section = (STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_section"], "bold")

        s.configure(".", font=font_normal, background=STYLE_CONFIG["bg_widget"], foreground=STYLE_CONFIG["fg_text"], bordercolor=STYLE_CONFIG["border_color"], lightcolor=STYLE_CONFIG["bg_widget"], darkcolor=STYLE_CONFIG["bg_widget"])
        
        s.configure("TFrame", background=STYLE_CONFIG["bg_root"])
        
        s.configure("Content.TFrame", background=STYLE_CONFIG["bg_widget"])
        
        s.configure("TLabel", background=STYLE_CONFIG["bg_widget"], foreground=STYLE_CONFIG["fg_text"])
        
        s.configure("Header.TLabel", font=font_header, foreground=STYLE_CONFIG["fg_header"], background=STYLE_CONFIG["bg_root"])
        
        s.configure("Section.TLabel", font=font_section, foreground=STYLE_CONFIG["fg_header"], background=STYLE_CONFIG["bg_widget"])
        
        s.configure("TButton", font=font_bold, padding=6, background=STYLE_CONFIG["accent_color"], foreground=STYLE_CONFIG["accent_text_color"])

        s.map("TButton", background=[("active", STYLE_CONFIG["accent_color"]), ("disabled", STYLE_CONFIG["disabled_bg"])], foreground=[("active", STYLE_CONFIG["accent_text_color"]), ("disabled", STYLE_CONFIG["disabled_fg"])])
        
        s.configure("TEntry", fieldbackground=STYLE_CONFIG["bg_entry"], foreground=STYLE_CONFIG["fg_text"], insertcolor=STYLE_CONFIG["fg_text"])
        
        s.configure("TCombobox", fieldbackground=STYLE_CONFIG["bg_entry"], foreground=STYLE_CONFIG["fg_text"], selectbackground=STYLE_CONFIG["bg_entry"], selectforeground=STYLE_CONFIG["fg_text"], arrowcolor=STYLE_CONFIG["fg_text"])

        self.root.option_add('*TCombobox*Listbox.background', STYLE_CONFIG["bg_entry"])
        
        self.root.option_add('*TCombobox*Listbox.foreground', STYLE_CONFIG["fg_text"])
        
        self.root.option_add('*TCombobox*Listbox.selectBackground', STYLE_CONFIG["listbox_select_bg"])
        
        self.root.option_add('*TCombobox*Listbox.selectForeground', STYLE_CONFIG["listbox_select_fg"])

        s.configure("TScrollbar", background=STYLE_CONFIG["bg_widget"], troughcolor=STYLE_CONFIG["bg_root"], arrowcolor=STYLE_CONFIG["fg_text"])
        
        s.configure("TCheckbutton", background=STYLE_CONFIG["bg_widget"], foreground=STYLE_CONFIG["fg_text"])
        
        s.map("TCheckbutton", indicatorcolor=[("selected", STYLE_CONFIG["accent_color"]), ("!selected", STYLE_CONFIG["border_color"])])
        
        s.configure("TPanedwindow", background=STYLE_CONFIG["bg_root"])
        
        s.configure("TLabelFrame", background=STYLE_CONFIG["bg_widget"], bordercolor=STYLE_CONFIG["border_color"])
        
        s.configure("TLabelFrame.Label", background=STYLE_CONFIG["bg_widget"], foreground=STYLE_CONFIG["fg_header"], font=font_section)
        
        s.configure("TNotebook.Tab", font=font_bold, padding=[5, 2], background=STYLE_CONFIG["bg_root"])
        
        s.map("TNotebook.Tab", background=[("selected", STYLE_CONFIG["accent_color"])], foreground=[("selected", STYLE_CONFIG["accent_text_color"])])

    
    
    def create_menu(self):
        menubar = tk.Menu(self.root, bg=STYLE_CONFIG["bg_widget"], fg=STYLE_CONFIG["fg_text"], activebackground=STYLE_CONFIG["accent_color"], activeforeground=STYLE_CONFIG["accent_text_color"])
        
        file_menu = tk.Menu(menubar, tearoff=0, bg=STYLE_CONFIG["bg_widget"], fg=STYLE_CONFIG["fg_text"], activebackground=STYLE_CONFIG["accent_color"], activeforeground=STYLE_CONFIG["accent_text_color"])
        
        file_menu.add_command(label="Load CSV...", command=self.load_csv_file, accelerator="Ctrl+O")
        
        file_menu.add_separator(); file_menu.add_command(label="About", command=self.show_about_dialog)
        
        file_menu.add_separator(); file_menu.add_command(label="Quit", command=self.root.quit, accelerator="Ctrl+Q")
        
        menubar.add_cascade(label="File", menu=file_menu)
        
        self.root.config(menu=menubar)
        
        self.root.bind_all("<Control-o>", lambda e: self.load_csv_file())
        
        self.root.bind_all("<Control-q>", lambda e: self.root.quit())

    
    
    def create_main_layout(self):
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        train_config_pane = ttk.Frame(main_pane, padding="10", style="Content.TFrame")
        
        main_pane.add(train_config_pane, weight=1)
        
        predict_results_pane = ttk.PanedWindow(main_pane, orient=tk.VERTICAL)
        
        main_pane.add(predict_results_pane, weight=1)
        
        self.prediction_input_outer_frame = ttk.Frame(predict_results_pane, padding="10", style="Content.TFrame")
        
        predict_results_pane.add(self.prediction_input_outer_frame, weight=2)
        
        results_display_frame = ttk.Frame(predict_results_pane, padding="10", style="Content.TFrame")
        
        predict_results_pane.add(results_display_frame, weight=3)
        
        self.create_train_config_widgets(train_config_pane)
        
        self.create_dynamic_prediction_inputs_placeholder(self.prediction_input_outer_frame)
        
        self.create_results_display_widgets(results_display_frame)

    
    
    def log_training_message(self, message, is_error=False):
        
        if not hasattr(self, 'training_log_text') or not self.training_log_text.winfo_exists():
            print(f"LOG {'(Error)' if is_error else ''}: {message}"); return
        try:
            self.training_log_text.configure(state=tk.NORMAL)
            
            tag = "error_tag" if is_error else "normal_tag"
            
            self.training_log_text.tag_configure("error_tag", foreground=STYLE_CONFIG["error_text_color"])
            
            self.training_log_text.tag_configure("normal_tag", foreground=STYLE_CONFIG["fg_text"])
            
            self.training_log_text.insert(tk.END, message + "\n", tag)
            
            self.training_log_text.see(tk.END)
            
            self.training_log_text.configure(state=tk.DISABLED)
            
            self.root.update_idletasks()
        except tk.TclError: print(f"LOG (TCL Error) {'(Error)' if is_error else ''}: {message}")

    
    def create_train_config_widgets(self, parent_frame):
        ttk.Label(parent_frame, text="Model Training Configuration", style="Header.TLabel", background=STYLE_CONFIG["bg_widget"]).pack(pady=(0,10), anchor=tk.W)
        
        load_button = ttk.Button(parent_frame, text="Load CSV File", command=self.load_csv_file)
        
        load_button.pack(pady=5, fill=tk.X)
        
        self.loaded_file_label = ttk.Label(parent_frame, text="No file loaded.")
        
        self.loaded_file_label.pack(pady=(2,5), anchor=tk.W)

        target_config_frame = ttk.LabelFrame(parent_frame, text="Target Variable & Time")
        
        target_config_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(target_config_frame, text="Event Column (1=event, 0=censor):").grid(row=0, column=0, padx=5, pady=3, sticky=tk.W)
        
        self.target_event_selector = ttk.Combobox(target_config_frame, textvariable=self.target_event_col_var, state="readonly", width=28, font=(STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_normal"]))
        
        self.target_event_selector.grid(row=0, column=1, padx=5, pady=3, sticky=tk.EW)
        
        ttk.Label(target_config_frame, text="Time to Event/Censor Column (days):").grid(row=1, column=0, padx=5, pady=3, sticky=tk.W)
        
        self.time_to_event_selector = ttk.Combobox(target_config_frame, textvariable=self.time_to_event_col_var, state="readonly", width=28, font=(STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_normal"]))
        
        self.time_to_event_selector.grid(row=1, column=1, padx=5, pady=3, sticky=tk.EW)
        
        ttk.Label(target_config_frame, text="Prediction Horizon (Years, for results):").grid(row=2, column=0, padx=5, pady=3, sticky=tk.W)
        
        self.time_horizon_entry = ttk.Entry(target_config_frame, textvariable=self.time_horizon_var, width=10, font=(STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_normal"]))
        
        self.time_horizon_entry.grid(row=2, column=1, padx=5, pady=3, sticky=tk.W)
        
        target_config_frame.columnconfigure(1, weight=1)

        dl_fs_frame = ttk.LabelFrame(parent_frame, text="Features for MLP Risk Score Model")
        
        dl_fs_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        ttk.Label(dl_fs_frame, text="Select features for non-linear risk score (MLP):").pack(anchor=tk.W, padx=5, pady=(5,0))
        
        dl_listbox_container = ttk.Frame(dl_fs_frame, style="Content.TFrame")
        
        dl_listbox_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.dl_feature_listbox = tk.Listbox(dl_listbox_container, selectmode=tk.MULTIPLE, exportselection=False, height=6, bg=STYLE_CONFIG["bg_entry"], fg=STYLE_CONFIG["fg_text"], selectbackground=STYLE_CONFIG["listbox_select_bg"], selectforeground=STYLE_CONFIG["listbox_select_fg"], highlightthickness=1, highlightbackground=STYLE_CONFIG["border_color"], font=(STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_normal"]))
        
        dl_feature_listbox_scrollbar = ttk.Scrollbar(dl_listbox_container, orient=tk.VERTICAL, command=self.dl_feature_listbox.yview)
        
        self.dl_feature_listbox.configure(yscrollcommand=dl_feature_listbox_scrollbar.set)
        
        dl_feature_listbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.dl_feature_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        dl_params_frame = ttk.LabelFrame(parent_frame, text="Risk Score Model (MLP) Hyperparameters")
        
        dl_params_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(dl_params_frame, text="Learning Rate:").grid(row=0, column=0, padx=5, pady=3, sticky=tk.W)
        
        self.learning_rate_entry = ttk.Entry(dl_params_frame, textvariable=self.learning_rate_var, width=8, font=(STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_normal"]))
        
        self.learning_rate_entry.grid(row=0, column=1, padx=5, pady=3, sticky=tk.W)
        
        ttk.Label(dl_params_frame, text="Epochs:").grid(row=0, column=2, padx=5, pady=3, sticky=tk.W)
        
        self.epochs_entry = ttk.Entry(dl_params_frame, textvariable=self.epochs_var, width=8, font=(STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_normal"]))
        
        self.epochs_entry.grid(row=0, column=3, padx=5, pady=3, sticky=tk.W)
        
        ttk.Label(dl_params_frame, text="Hidden Layers (e.g. 64,32):").grid(row=1, column=0, columnspan=2, padx=5, pady=3, sticky=tk.W)
        
        self.hidden_layers_entry = ttk.Entry(dl_params_frame, textvariable=self.hidden_layers_var, width=15, font=(STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_normal"]))
        
        self.hidden_layers_entry.grid(row=1, column=2, columnspan=2, padx=5, pady=3, sticky=tk.W)

        cph_fs_frame = ttk.LabelFrame(parent_frame, text="Features for Linear Part of CPH Model")
        
        cph_fs_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        ttk.Label(cph_fs_frame, text="Select features for CPH linear terms:").pack(anchor=tk.W, padx=5, pady=(5,0))
        
        cph_listbox_container = ttk.Frame(cph_fs_frame, style="Content.TFrame")
        
        cph_listbox_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.cph_linear_feature_listbox = tk.Listbox(cph_listbox_container, selectmode=tk.MULTIPLE, exportselection=False, height=5, bg=STYLE_CONFIG["bg_entry"], fg=STYLE_CONFIG["fg_text"], selectbackground=STYLE_CONFIG["listbox_select_bg"], selectforeground=STYLE_CONFIG["listbox_select_fg"], highlightthickness=1, highlightbackground=STYLE_CONFIG["border_color"], font=(STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_normal"]))
        
        cph_linear_feature_listbox_scrollbar = ttk.Scrollbar(cph_listbox_container, orient=tk.VERTICAL, command=self.cph_linear_feature_listbox.yview)
        
        self.cph_linear_feature_listbox.configure(yscrollcommand=cph_linear_feature_listbox_scrollbar.set)
        
        cph_linear_feature_listbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.cph_linear_feature_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        cph_params_frame = ttk.LabelFrame(parent_frame, text="CPH Model Hyperparameters")
        
        cph_params_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(cph_params_frame, text="L2 Penalizer:").grid(row=0, column=0, padx=5, pady=3, sticky=tk.W)
        
        self.cph_penalizer_entry = ttk.Entry(cph_params_frame, textvariable=self.cph_penalizer_var, width=8, font=(STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_normal"]))
        
        self.cph_penalizer_entry.grid(row=0, column=1, padx=5, pady=3, sticky=tk.W)

        self.train_button = ttk.Button(parent_frame, text="Train Hybrid MLP-CPH Model", command=self.train_model_action)
        
        self.train_button.pack(pady=(10,5), fill=tk.X)
        
        ttk.Label(parent_frame, text="Training Log & Report:", style="Section.TLabel").pack(anchor=tk.W, pady=(10,0))
        
        self.training_log_text = scrolledtext.ScrolledText(parent_frame, height=5, wrap=tk.WORD, bg=STYLE_CONFIG["bg_entry"], fg=STYLE_CONFIG["fg_text"], insertbackground=STYLE_CONFIG["fg_text"], font=(STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_normal"]-1), highlightthickness=1, highlightbackground=STYLE_CONFIG["border_color"])
        
        self.training_log_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.training_log_text.configure(state=tk.DISABLED)

    
    
    def create_dynamic_prediction_inputs_placeholder(self, parent_frame):
        ttk.Label(parent_frame, text="Patient Data for Prediction", style="Header.TLabel", background=STYLE_CONFIG["bg_widget"]).pack(pady=(0,10), anchor=tk.W)
        
        self.dynamic_input_canvas = tk.Canvas(parent_frame, borderwidth=0, background=STYLE_CONFIG["bg_widget"], highlightthickness=0)
        
        vsb = ttk.Scrollbar(parent_frame, orient="vertical", command=self.dynamic_input_canvas.yview)
        
        self.dynamic_input_canvas.configure(yscrollcommand=vsb.set); vsb.pack(side="right", fill="y")
        
        self.dynamic_input_canvas.pack(side="left", fill="both", expand=True)
        
        self.dynamic_input_scrollable_frame = ttk.Frame(self.dynamic_input_canvas, style="Content.TFrame")
        
        self.dynamic_input_canvas.create_window((0, 0), window=self.dynamic_input_scrollable_frame, anchor="nw")
        
        self.dynamic_input_scrollable_frame.bind("<Configure>", lambda e: self.dynamic_input_canvas.configure(scrollregion=self.dynamic_input_canvas.bbox("all")))
        
        self.placeholder_pred_label = ttk.Label(self.dynamic_input_scrollable_frame, text="Train a model to enable prediction inputs.", style="TLabel")
        
        self.placeholder_pred_label.pack(padx=10, pady=20)
        
        self.assess_button = ttk.Button(self.dynamic_input_scrollable_frame, text="Assess Risk (CPH)", command=self.assess_risk_action)
        
        self.assess_button.pack_forget()

    
    
    def create_dynamic_prediction_inputs(self):
        if self.dynamic_input_scrollable_frame:
            for widget in self.dynamic_input_scrollable_frame.winfo_children(): widget.destroy()
        
        self.prediction_input_widgets = {}
        
        self.all_base_features_for_input = sorted(list(set(self.trained_dl_feature_names + self.trained_cph_linear_feature_names)))
        
        if not self.all_base_features_for_input:
            self.placeholder_pred_label = ttk.Label(self.dynamic_input_scrollable_frame, text="No features available. Train model.", style="TLabel")
            
            self.placeholder_pred_label.pack(padx=10, pady=20)
            
            if hasattr(self, 'assess_button') and self.assess_button.winfo_exists(): self.assess_button.pack_forget()
            return
        
        for feature_name in self.all_base_features_for_input:
            row_frame = ttk.Frame(self.dynamic_input_scrollable_frame, style="Content.TFrame")
            
            row_frame.pack(fill=tk.X, pady=1, padx=2)
            
            display_name = feature_name if len(feature_name) < 35 else feature_name[:32] + "..."
            
            label = ttk.Label(row_frame, text=f"{display_name}:", width=35, anchor="w"); label.pack(side=tk.LEFT, padx=(0,5))
            
            entry = ttk.Entry(row_frame, font=(STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_normal"]))
            
            default_val = "0"
            
            if feature_name in self.trained_feature_medians_dl: default_val = self.trained_feature_medians_dl.get(feature_name, "0")
            
            elif feature_name in self.trained_feature_medians_cph_linear: default_val = self.trained_feature_medians_cph_linear.get(feature_name, "0")
            
            entry.insert(0, str(default_val)); entry.pack(side=tk.LEFT, expand=True, fill=tk.X)
            
            self.prediction_input_widgets[feature_name] = entry
        
        self.assess_button = ttk.Button(self.dynamic_input_scrollable_frame, text="Assess Risk (CPH)", command=self.assess_risk_action)
        
        self.assess_button.pack(pady=(15,10), fill=tk.X, padx=5)
        
        self.dynamic_input_scrollable_frame.update_idletasks()
        
        self.dynamic_input_canvas.config(scrollregion=self.dynamic_input_canvas.bbox("all"))

    
    
    def create_results_display_widgets(self, parent_frame):
        top_frame = ttk.Frame(parent_frame, style="Content.TFrame"); top_frame.pack(fill=tk.X, pady=5)
        
        pred_res_frame = ttk.LabelFrame(top_frame, text="CPH Prediction Result")
        
        pred_res_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5))
        
        self.risk_prob_label = ttk.Label(pred_res_frame, text="N/A", font=(STYLE_CONFIG["font_family"], 22, "bold"), foreground=STYLE_CONFIG["accent_color"])
        
        self.risk_prob_label.pack(pady=(5,2))
        
        self.risk_interpretation_label = ttk.Label(pred_res_frame, text="Train model & assess.", font=(STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_normal"]))
        
        self.risk_interpretation_label.pack(pady=(0,5))
        
        self.more_plots_button = ttk.Button(top_frame, text="View More Plots", command=self.show_more_plots_window, state=tk.DISABLED)
        
        self.more_plots_button.pack(side=tk.RIGHT, padx=(5,0), pady=10, anchor="ne")

        plot_frame = ttk.LabelFrame(parent_frame, text="Model Performance Visuals"); plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.fig = plt.Figure(figsize=(8, 7), dpi=90, facecolor=STYLE_CONFIG["bg_widget"]); self.fig.subplots_adjust(hspace=0.6, wspace=0.4)
        
        self.ax_cph_coeffs = self.fig.add_subplot(2, 2, 1)
        
        self.ax_risk_dist = self.fig.add_subplot(2, 2, 2)
        
        self.ax_baseline_survival = self.fig.add_subplot(2, 2, 3)
        
        self.ax_survival_curve = self.fig.add_subplot(2, 2, 4)

        for ax in [self.ax_cph_coeffs, self.ax_risk_dist, self.ax_baseline_survival, self.ax_survival_curve]:
            
            ax.tick_params(colors=STYLE_CONFIG["fg_text"]); ax.xaxis.label.set_color(STYLE_CONFIG["fg_text"])
            
            ax.yaxis.label.set_color(STYLE_CONFIG["fg_text"]); ax.title.set_color(STYLE_CONFIG["fg_header"])
            
            ax.set_facecolor(STYLE_CONFIG["bg_entry"])

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame); self.canvas_widget = self.canvas.get_tk_widget()
        
        self.canvas_widget.configure(bg=STYLE_CONFIG["bg_widget"]); self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        self.update_plots(clear_only=True)

    
    
    def toggle_train_predict_sections_enabled(self, data_loaded=False, model_trained=False):
        train_state = tk.NORMAL if data_loaded else tk.DISABLED
        
        predict_state = tk.NORMAL if model_trained else tk.DISABLED
        
        if hasattr(self, 'train_button'): self.train_button.config(state=train_state)
        
        if hasattr(self, 'dl_feature_listbox'): self.dl_feature_listbox.config(state=train_state)
        
        if hasattr(self, 'cph_linear_feature_listbox'): self.cph_linear_feature_listbox.config(state=train_state)
        
        if hasattr(self, 'target_event_selector'): self.target_event_selector.config(state="readonly" if data_loaded else tk.DISABLED)
        
        if hasattr(self, 'time_to_event_selector'): self.time_to_event_selector.config(state="readonly" if data_loaded else tk.DISABLED)
        
        if hasattr(self, 'time_horizon_entry'): self.time_horizon_entry.config(state=tk.NORMAL if data_loaded else tk.DISABLED)
        
        if hasattr(self, 'learning_rate_entry'): self.learning_rate_entry.config(state=tk.NORMAL if data_loaded else tk.DISABLED)
        
        if hasattr(self, 'epochs_entry'): self.epochs_entry.config(state=tk.NORMAL if data_loaded else tk.DISABLED)
        
        if hasattr(self, 'hidden_layers_entry'): self.hidden_layers_entry.config(state=tk.NORMAL if data_loaded else tk.DISABLED)
        
        if hasattr(self, 'cph_penalizer_entry'): self.cph_penalizer_entry.config(state=tk.NORMAL if data_loaded else tk.DISABLED)
        
        if hasattr(self, 'assess_button') and self.assess_button.winfo_exists(): self.assess_button.config(state=predict_state)
        
        if hasattr(self, 'more_plots_button'): self.more_plots_button.config(state=predict_state)
        
        for _feature_name, widget in self.prediction_input_widgets.items():
            if hasattr(widget, 'config'): widget.config(state=tk.NORMAL if model_trained else tk.DISABLED)

    
    
    def downcast_numerics(self, df):
        self.log_training_message("  Attempting to downcast numeric types for memory optimization...")
        
        f_cols = df.select_dtypes('float').columns; i_cols = df.select_dtypes('integer').columns
        
        df[f_cols] = df[f_cols].apply(pd.to_numeric, downcast='float')
        
        df[i_cols] = df[i_cols].apply(pd.to_numeric, downcast='integer')
        
        gc.collect(); return df

    
    
    def _populate_ui_lists_after_load(self, column_names):
        self.dl_feature_listbox.delete(0, tk.END); self.cph_linear_feature_listbox.delete(0, tk.END)
        
        for col_name in column_names:
            self.dl_feature_listbox.insert(tk.END, col_name)
            self.cph_linear_feature_listbox.insert(tk.END, col_name)
        self.target_event_selector['values'] = column_names; self.time_to_event_selector['values'] = column_names
        
        default_target = 'Cardiovascular_mortality'; default_time_col = 'Time_to_CVD_mortality_days'
        
        if default_target in column_names: self.target_event_col_var.set(default_target)
        
        elif column_names: self.target_event_col_var.set(column_names[0])
        
        if default_time_col in column_names: self.time_to_event_col_var.set(default_time_col)
        
        elif 'Time_to_mortality_days' in column_names: self.time_to_event_col_var.set('Time_to_mortality_days')
        
        elif column_names and len(column_names) > 1: self.time_to_event_col_var.set(column_names[1])
        
        self.log_training_message(f"  UI lists populated with {len(column_names)} columns."); self.root.update_idletasks()

    def load_csv_file(self):
        filepath = filedialog.askopenfilename(title="Select CSV File", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        
        if not filepath: self.log_training_message("File loading cancelled by user."); return
        try:
            self.data_df = pd.read_csv(filepath, low_memory=False)
            
            self.data_df = self.downcast_numerics(self.data_df.copy()); gc.collect()
            
            self.loaded_file_label.config(text=f"Loaded: {filepath.split('/')[-1]} ({self.data_df.shape[0]} rows, {self.data_df.shape[1]} cols)")
            
            self.log_training_message(f"Successfully loaded and downcasted {filepath.split('/')[-1]}.")
            
            column_names = sorted([str(col) for col in self.data_df.columns if str(col).strip()])
            
            if not column_names:
                self.log_training_message("No columns found in CSV or header is missing.", is_error=True)
                messagebox.showerror("CSV Error", "No columns detected in CSV."); return
            
            self.root.after(10, self._populate_ui_lists_after_load, column_names)
            
            self.dl_model = None; self.cph_model = None;
            
            self.toggle_train_predict_sections_enabled(data_loaded=True, model_trained=False)
            
            if self.dynamic_input_scrollable_frame:
                for widget in self.dynamic_input_scrollable_frame.winfo_children(): widget.destroy()
            self.placeholder_pred_label = ttk.Label(self.dynamic_input_scrollable_frame, text="Train a model to enable prediction inputs.", style="TLabel")
            
            self.placeholder_pred_label.pack(padx=10, pady=20)
            
            if hasattr(self, 'assess_button') and self.assess_button.winfo_exists(): self.assess_button.pack_forget()
            
            self.update_plots(clear_only=True)
            
            self.risk_interpretation_label.config(text="Data loaded. Configure and train model."); self.risk_prob_label.config(text="N/A")
        except Exception as e:
            self.log_training_message(f"Error loading CSV: {str(e)}", is_error=True); self.log_training_message(traceback.format_exc(), is_error=True)
            
            messagebox.showerror("Error Loading CSV", f"Failed to load or parse CSV file.\nError: {e}")
            
            self.data_df = None; self.toggle_train_predict_sections_enabled(data_loaded=False, model_trained=False)

    def _preprocess_features(self, df_subset, feature_names, stored_medians, scaler_to_use, scaled_cols_list, imputer_to_use, fit_transform=True):
        processed_df = df_subset[feature_names].copy()
        
        for col in processed_df.columns: processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        
        if fit_transform:
            for col in processed_df.columns: stored_medians[col] = processed_df[col].median()
            imputer_to_use.fit(processed_df)
        imputed_values = imputer_to_use.transform(processed_df)
        
        processed_df = pd.DataFrame(imputed_values, columns=processed_df.columns, index=processed_df.index)
        
        if fit_transform:
            scaled_cols_list.clear()
            for col in processed_df.columns:
                if pd.api.types.is_numeric_dtype(processed_df[col]) and processed_df[col].nunique(dropna=False) > 2:
                    scaled_cols_list.append(col)
        
        if scaled_cols_list and len(scaled_cols_list) > 0:
            if fit_transform: scaler_to_use.fit(processed_df[scaled_cols_list])
            processed_df[scaled_cols_list] = scaler_to_use.transform(processed_df[scaled_cols_list])
        
        return processed_df

    def train_model_action(self):
        if self.data_df is None: messagebox.showerror("Error", "No data loaded."); return
        
        selected_dl_indices = self.dl_feature_listbox.curselection()
        
        selected_cph_linear_indices = self.cph_linear_feature_listbox.curselection()
        
        target_event_col = self.target_event_col_var.get()
        
        time_to_event_col = self.time_to_event_col_var.get()

        if not selected_dl_indices: messagebox.showerror("Error", "No features selected for MLP Model."); return
        
        if not target_event_col or not time_to_event_col: messagebox.showerror("Error", "Target and time columns must be selected."); return

        self.trained_dl_feature_names = [self.dl_feature_listbox.get(i) for i in selected_dl_indices]
        
        self.trained_cph_linear_feature_names = [self.cph_linear_feature_listbox.get(i) for i in selected_cph_linear_indices]

        for col_list_name, col_list in [("MLP", self.trained_dl_feature_names), ("CPH", self.trained_cph_linear_feature_names)]:
            if target_event_col in col_list or time_to_event_col in col_list:
                messagebox.showerror("Error", f"Target/Time column cannot be in {col_list_name} feature list."); return
        
        if self.DL_RISK_SCORE_COL_NAME in self.trained_cph_linear_feature_names:
             messagebox.showerror("Error", f"'{self.DL_RISK_SCORE_COL_NAME}' is a reserved column name."); return
        try:
            lr = float(self.learning_rate_var.get())
            
            epochs = int(self.epochs_var.get())
            
            hidden_layers_str = self.hidden_layers_var.get()
            
            hidden_layers = [int(x.strip()) for x in hidden_layers_str.split(',') if x.strip()]
            
            cph_penalizer = float(self.cph_penalizer_var.get())
            
            if lr <= 0 or epochs <= 0 or cph_penalizer < 0: raise ValueError("Hyperparameters out of valid range.")
        except ValueError: messagebox.showerror("Hyperparameter Error", "Invalid hyperparameter values."); return

        self.log_training_message("--- Starting Hybrid Model Training ---")
        
        full_df_processed = self.data_df.copy(); gc.collect()
        
        try:
            full_df_processed[target_event_col] = pd.to_numeric(full_df_processed[target_event_col], errors='raise')
            full_df_processed[time_to_event_col] = pd.to_numeric(full_df_processed[time_to_event_col], errors='raise')
        except Exception as e:
            self.log_training_message(f"Error: Target or Time column non-numeric: {e}", is_error=True); messagebox.showerror("Data Error", f"Target/Time column has non-numeric data: {e}"); return

        initial_rows = len(full_df_processed)
        
        full_df_processed.dropna(subset=[target_event_col, time_to_event_col], inplace=True)
        
        full_df_processed = full_df_processed[full_df_processed[time_to_event_col] > 0]
        
        if len(full_df_processed) < initial_rows: self.log_training_message(f"  Dropped {initial_rows - len(full_df_processed)} rows (NaNs/invalid times in target/time).")
        
        if full_df_processed.empty: messagebox.showerror("Data Error", "No valid data after cleaning."); return

        self.log_training_message("\n--- Stage 1: Training Risk Score Model (MLP) ---")
        
        try:
            X_dl_full = full_df_processed[self.trained_dl_feature_names]
            
            self.scaler_dl = StandardScaler(); self.num_imputer_dl = SimpleImputer(strategy='median')
            
            self.trained_feature_medians_dl = {}; self.scaled_columns_dl = []
            
            X_dl_processed = self._preprocess_features(X_dl_full, self.trained_dl_feature_names, self.trained_feature_medians_dl, self.scaler_dl, self.scaled_columns_dl, self.num_imputer_dl, fit_transform=True)

            y_event = full_df_processed[target_event_col].astype(bool)
            
            y_time = full_df_processed[time_to_event_col].astype(float)

            self.dl_model = MLPModel(len(self.trained_dl_feature_names), hidden_layers)
            
            optimizer = optim.Adam(self.dl_model.parameters(), lr=lr)
            
            dataset = TensorDataset(torch.tensor(X_dl_processed.values, dtype=torch.float32), torch.tensor(y_time.values, dtype=torch.float32), torch.tensor(y_event.values, dtype=torch.float32))
            
            loader = DataLoader(dataset, batch_size=64, shuffle=True)

            self.log_training_message(f"  Training MLP for {epochs} epochs...")
            
            for epoch in range(epochs):
                self.dl_model.train()
                for X_batch, time_batch, event_batch in loader:
                    optimizer.zero_grad()
                    
                    log_risk = self.dl_model(X_batch)
                    
                    loss = cox_loss(log_risk.squeeze(), time_batch, event_batch)
                    
                    if torch.isnan(loss):
                        self.log_training_message(f"  Warning: NaN loss detected at epoch {epoch+1}. Stopping DL training.", is_error=True)
                        break
                    loss.backward()
                    optimizer.step()
                if torch.isnan(loss): break
            self.log_training_message("  MLP model trained.")
            
            self.dl_model.eval()
            with torch.no_grad():
                dl_risk_scores_all_data = self.dl_model(torch.tensor(X_dl_processed.values, dtype=torch.float32)).numpy().flatten()
            
            full_df_processed[self.DL_RISK_SCORE_COL_NAME] = dl_risk_scores_all_data
            
            self.log_training_message(f"  '{self.DL_RISK_SCORE_COL_NAME}' generated for all {len(dl_risk_scores_all_data)} samples.")

        except Exception as e:
            self.log_training_message(f"Error in MLP training: {e}", is_error=True); self.log_training_message(traceback.format_exc(), is_error=True)
            messagebox.showerror("MLP Training Error", f"Failed MLP stage: {e}"); self.dl_model = None; return

        self.log_training_message("\n--- Stage 2: Training Cox Proportional Hazards (CPH) Model ---")
        try:
            cph_features_to_use = [self.DL_RISK_SCORE_COL_NAME]
            
            df_for_cph_fitting = full_df_processed[[self.DL_RISK_SCORE_COL_NAME]].copy()

            if self.trained_cph_linear_feature_names:
                cph_features_to_use.extend(self.trained_cph_linear_feature_names)
                
                X_cph_linear_part = full_df_processed[self.trained_cph_linear_feature_names]
                
                self.scaler_cph_linear = StandardScaler(); self.num_imputer_cph_linear = SimpleImputer(strategy='median')
                
                self.trained_feature_medians_cph_linear = {}; self.scaled_columns_cph_linear = []
                
                X_cph_linear_processed = self._preprocess_features(X_cph_linear_part, self.trained_cph_linear_feature_names, self.trained_feature_medians_cph_linear, self.scaler_cph_linear, self.scaled_columns_cph_linear, self.num_imputer_cph_linear, fit_transform=True)
                
                for col in X_cph_linear_processed.columns:
                    df_for_cph_fitting[col] = X_cph_linear_processed[col]

            self.log_training_message(f"  Features for CPH fitter: {df_for_cph_fitting.columns.tolist()}")

            df_for_cph_fitting[time_to_event_col] = full_df_processed[time_to_event_col].values
            
            df_for_cph_fitting[target_event_col] = full_df_processed[target_event_col].astype(int).values

            train_cph_df, test_cph_df = train_test_split(df_for_cph_fitting, test_size=0.25, random_state=42, stratify=df_for_cph_fitting[target_event_col] if df_for_cph_fitting[target_event_col].nunique() > 1 else None)
            
            if train_cph_df.empty or len(test_cph_df) < 5 :
                 messagebox.showerror("Data Split Error", "CPH training/test set is too small."); return
            
            self.train_cph_df_for_metrics = train_cph_df.copy()
            
            self.test_cph_df_for_metrics = test_cph_df.copy()

            self.cph_model = CoxPHFitter(penalizer=cph_penalizer)
            
            self.cph_model.fit(self.train_cph_df_for_metrics, duration_col=time_to_event_col, event_col=target_event_col)
            
            self.log_training_message("  CPH model trained."); self.log_training_message(f"  CPH Concordance on training: {self.cph_model.concordance_index_:.4f}")
            
            c_index_test = self.cph_model.score(self.test_cph_df_for_metrics, scoring_method="concordance_index")
            
            self.log_training_message(f"  CPH Concordance on test set: {c_index_test:.4f}")
            
            self.generate_training_report(lr, epochs, hidden_layers, cph_penalizer, c_index_test)
            
            self.create_dynamic_prediction_inputs(); self.toggle_train_predict_sections_enabled(data_loaded=True, model_trained=True)
            
            self.update_plots()
        except Exception as e:
            self.log_training_message(f"Error in CPH training: {e}", is_error=True); self.log_training_message(traceback.format_exc(), is_error=True)
            
            messagebox.showerror("CPH Training Error", f"Failed CPH stage: {e}"); self.cph_model = None
            
            self.toggle_train_predict_sections_enabled(data_loaded=True, model_trained=bool(self.dl_model))

    #here we generate a brief report to be saved in report.txt -- useful for users to read that file than looking at the small log box 
    def generate_training_report(self, lr, epochs, hidden_layers, cph_pen, c_index):

            report_lines = []
            
            target_col_name = self.target_event_col_var.get()
            
            time_col_name = self.time_to_event_col_var.get()
            
            report_lines.append("--- Hybrid MLP-CPH Model Training Report ---")
            
            if self.data_df is not None and self.train_cph_df_for_metrics is not None and self.test_cph_df_for_metrics is not None:
                cleaned_rows = self.train_cph_df_for_metrics.shape[0] + self.test_cph_df_for_metrics.shape[0]
                report_lines.append(f"Dataset Shape (after cleaning): ({cleaned_rows}, {self.data_df.shape[1]})")
            else:
                report_lines.append("Dataset Shape: N/A")
                
            
            report_lines.append(f"Target Event Column: '{target_col_name}'")
            
            report_lines.append(f"Time to Event Column: '{time_col_name}'")
            
            report_lines.append("\n--- Stage 1: Risk Score Model (MLP) ---")
            
            report_lines.append(f"  Features used: {len(self.trained_dl_feature_names)}")
            
            report_lines.append(f"  Hyperparameters: Learning Rate={lr}, Epochs={epochs}, HiddenLayers={hidden_layers}")
            
            report_lines.append("\n--- Stage 2: Cox Proportional Hazards (CPH) Model ---")
            
            cph_features_in_model = self.trained_cph_linear_feature_names + [self.DL_RISK_SCORE_COL_NAME]
            
            report_lines.append(f"  Features used (incl. MLP score): {len(cph_features_in_model)}")
            
            report_lines.append(f"  Linear CPH Features: {self.trained_cph_linear_feature_names if self.trained_cph_linear_feature_names else 'None'}")
            
            report_lines.append(f"  CPH L2 Penalizer: {cph_pen}")
            
            if self.cph_model and hasattr(self.cph_model, 'summary') and not self.cph_model.summary.empty:
                report_lines.append("\n--- CPH Model Summary (Hazard Ratios) ---")
                report_lines.append(self.cph_model.summary.to_string())
            else:
                report_lines.append("\nCPH Model Summary not available.")
                
            report_lines.append(f"\n--- Final Model Performance on Test Set ---")
            
            report_lines.append(f"  Concordance Index (C-index): {c_index:.4f}")
            
            report_lines.append("\n--- End of Report ---")
            
            for line in report_lines:
                self.log_training_message(line)
                
            try:
                with open("report.txt", "w", encoding='utf-8') as f_report:
                    f_report.write("\n".join(report_lines))
                self.log_training_message("\n--- Report content also saved to report.txt ---")
            except Exception as e:
                self.log_training_message(f"\n--- Error saving report to file: {e} ---", is_error=True)
        
    
    
    
    def assess_risk_action(self):
        if not self.dl_model or not self.cph_model: messagebox.showerror("Error", "Model not fully trained."); return
        
        input_values_from_gui = {}
        try:
            for feature_name, widget in self.prediction_input_widgets.items():
                value_str = widget.get()
                try: input_values_from_gui[feature_name] = float(value_str)
                except (ValueError, TypeError): input_values_from_gui[feature_name] = np.nan
        except Exception as e: messagebox.showerror("Input Error", f"Error reading input values: {e}"); return

        input_df = pd.DataFrame([input_values_from_gui])

        input_dl_processed = self._preprocess_features(input_df, self.trained_dl_feature_names, self.trained_feature_medians_dl, self.scaler_dl, self.scaled_columns_dl, self.num_imputer_dl, fit_transform=False)
        
        self.dl_model.eval()
        
        with torch.no_grad():
            new_dl_risk_score = self.dl_model(torch.tensor(input_dl_processed.values, dtype=torch.float32)).numpy().flatten()[0]

        cph_input_dict = {self.DL_RISK_SCORE_COL_NAME: new_dl_risk_score}
        
        if self.trained_cph_linear_feature_names:
            input_cph_linear_processed = self._preprocess_features(input_df, self.trained_cph_linear_feature_names, self.trained_feature_medians_cph_linear, self.scaler_cph_linear, self.scaled_columns_cph_linear, self.num_imputer_cph_linear, fit_transform=False)
            for col in input_cph_linear_processed.columns: cph_input_dict[col] = input_cph_linear_processed[col].iloc[0]
        
        input_cph_df = pd.DataFrame([cph_input_dict])
        
        try:
            pred_horizon_years = float(self.time_horizon_var.get())
            
            pred_horizon_days = pred_horizon_years * 365.25
            
            survival_prob = self.cph_model.predict_survival_function(input_cph_df, times=[pred_horizon_days]).iloc[0,0]
            
            risk = 1.0 - survival_prob

            self.risk_prob_label.config(text=f"{pred_horizon_years:.0f}-Year Risk: {risk*100:.1f}%")
            
            if risk > 0.20: self.risk_interpretation_label.config(text="Higher Risk", foreground=STYLE_CONFIG["error_text_color"])
            
            elif risk > 0.10: self.risk_interpretation_label.config(text="Moderate Risk", foreground="#FFA000")
            
            else: self.risk_interpretation_label.config(text="Lower Risk", foreground="#388E3C")

            patient_survival_curve = self.cph_model.predict_survival_function(input_cph_df)
            
            self.update_plots(patient_survival_data=patient_survival_curve)

        except Exception as e:
            self.log_training_message(f"Prediction error: {e}", is_error=True); self.log_training_message(traceback.format_exc(), is_error=True)
            messagebox.showerror("Prediction Error", f"Could not make CPH prediction: {e}");

    
    
    #same plotting like beofre
    def update_plots(self, clear_only=False, patient_survival_data=None):
        for ax in [self.ax_cph_coeffs, self.ax_risk_dist, self.ax_baseline_survival, self.ax_survival_curve]: ax.clear()
        
        title_font = {'color': STYLE_CONFIG["fg_header"], 'fontsize': STYLE_CONFIG["font_size_normal"] + 2, 'weight': 'bold'}
        
        label_font = {'color': STYLE_CONFIG["fg_text"], 'fontsize': STYLE_CONFIG["font_size_normal"]}
        
        tick_color = STYLE_CONFIG["fg_text"]
        
        self.ax_cph_coeffs.set_title("CPH Model: Log(Hazard Ratios)", fontdict=title_font)
        
        if not clear_only and self.cph_model and not self.cph_model.params_.empty:
            self.cph_model.plot(ax=self.ax_cph_coeffs)
            self.ax_cph_coeffs.set_xlabel("Log(Hazard Ratio)", fontdict=label_font)
            legend = self.ax_cph_coeffs.get_legend()
            if legend:
                legend.remove()
        else:
            self.ax_cph_coeffs.text(0.5, 0.5, "Train model for coefficients.", ha='center', va='center', color=tick_color)
        self.ax_cph_coeffs.tick_params(colors=tick_color); self.ax_cph_coeffs.set_facecolor(STYLE_CONFIG["bg_entry"])


        self.ax_risk_dist.set_title("CPH: Predicted Risk Distribution (Test)", fontdict=title_font)
        
        if not clear_only and self.cph_model and self.test_cph_df_for_metrics is not None:
            try:
                scores = self.cph_model.predict_partial_hazard(self.test_cph_df_for_metrics)
                
                events = self.test_cph_df_for_metrics[self.target_event_col_var.get()]
                
                sns.histplot(scores[events == 0], label='Censored', kde=True, ax=self.ax_risk_dist, color="skyblue", stat="density", element="step")
                
                sns.histplot(scores[events == 1], label='Event', kde=True, ax=self.ax_risk_dist, color="salmon", stat="density", element="step")
                
                self.ax_risk_dist.set_xlabel("Predicted Partial Hazard", fontdict=label_font); self.ax_risk_dist.set_ylabel("Density", fontdict=label_font)
                
                self.ax_risk_dist.legend(facecolor=STYLE_CONFIG["bg_entry"], edgecolor=STYLE_CONFIG["border_color"], labelcolor=tick_color)
            except Exception as e: self.ax_risk_dist.text(0.5, 0.5, "Plotting Error", color=STYLE_CONFIG["error_text_color"])
        else:
            self.ax_risk_dist.text(0.5, 0.5, "Train model for risk distribution.", ha='center', va='center', color=tick_color)
        
        self.ax_risk_dist.tick_params(colors=tick_color); self.ax_risk_dist.set_facecolor(STYLE_CONFIG["bg_entry"])

        self.ax_baseline_survival.set_title("CPH: Baseline Survival Function", fontdict=title_font)
        
        if not clear_only and self.cph_model and not self.cph_model.baseline_survival_.empty:
            self.cph_model.baseline_survival_.plot(ax=self.ax_baseline_survival, color=STYLE_CONFIG["accent_color"], legend=False)
            
            self.ax_baseline_survival.set_xlabel(f"Time", fontdict=label_font); self.ax_baseline_survival.set_ylabel("Survival Probability", fontdict=label_font)
            
            self.ax_baseline_survival.set_ylim(0, 1.05)
        else:
            self.ax_baseline_survival.text(0.5, 0.5, "Train model for baseline survival.", ha='center', va='center', color=tick_color)
        
        self.ax_baseline_survival.tick_params(colors=tick_color); self.ax_baseline_survival.set_facecolor(STYLE_CONFIG["bg_entry"])
        
        self.ax_survival_curve.set_title("Patient Survival Prediction", fontdict=title_font)
        
        if not clear_only and patient_survival_data is not None:
            patient_survival_data.plot(ax=self.ax_survival_curve, color='red', legend=False)
            
            self.ax_survival_curve.set_xlabel(f"Time", fontdict=label_font); self.ax_survival_curve.set_ylabel("Survival Probability", fontdict=label_font)
            
            self.ax_survival_curve.set_ylim(0, 1.05)
        else:
             self.ax_survival_curve.text(0.5, 0.5, "Assess risk for a patient.", ha='center', va='center', color=tick_color)
        
        self.ax_survival_curve.tick_params(colors=tick_color); self.ax_survival_curve.set_facecolor(STYLE_CONFIG["bg_entry"])
        
        try: self.fig.tight_layout(pad=2.5)
        
        except Exception: pass
        
        self.canvas.draw()

    
    
    def show_more_plots_window(self):
        if self.more_plots_window is not None and self.more_plots_window.winfo_exists():
            self.more_plots_window.lift(); return
        
        if not self.cph_model or self.test_cph_df_for_metrics is None:
            messagebox.showinfo("No Data", "Model must be trained to view performance plots.")
            return

        self.more_plots_window = tk.Toplevel(self.root)
        
        self.more_plots_window.title("Model Performance Diagnostic Plots (Test Set)")
        
        self.more_plots_window.geometry("1000x800")
        
        self.more_plots_window.configure(bg=STYLE_CONFIG["bg_root"])


        notebook = ttk.Notebook(self.more_plots_window)
        
        notebook.pack(pady=10, padx=10, fill="both", expand=True)

        time_col = self.time_to_event_col_var.get()
        
        event_col = self.target_event_col_var.get()
        
        test_df = self.test_cph_df_for_metrics
        
        train_df = self.train_cph_df_for_metrics

        #quartiles here
        tab1 = ttk.Frame(notebook, style="Content.TFrame")
        
        notebook.add(tab1, text='Survival by Risk')
        
        fig1 = plt.Figure(figsize=(8, 6), dpi=100, facecolor=STYLE_CONFIG["bg_widget"])
        
        ax1 = fig1.add_subplot(111)
        
        try:
            risk_scores = self.cph_model.predict_partial_hazard(test_df)
            
            test_df_copy = test_df.copy()
            
            test_df_copy['risk_group'] = pd.qcut(risk_scores, 4, labels=["Q1 (Low Risk)", "Q2", "Q3", "Q4 (High Risk)"])
            
            #i added observed false ti the grouoby object, to avoid the warning
            for group, grouped_df in test_df_copy.groupby('risk_group',observed=False):
                kmf = KaplanMeierFitter()
                
                kmf.fit(grouped_df[time_col], event_observed=grouped_df[event_col], label=group)
                
                kmf.plot_survival_function(ax=ax1)
            
            ax1.set_title("Survival Curves by Predicted Risk Quartile", color=STYLE_CONFIG["fg_header"])
            
            ax1.set_xlabel(f"Time ({time_col})", color=STYLE_CONFIG["fg_text"])
            
            ax1.set_ylabel("Survival Probability", color=STYLE_CONFIG["fg_text"])
            
            ax1.tick_params(colors=STYLE_CONFIG["fg_text"])
            
            ax1.legend(title="Risk Group")
        except Exception as e:
            ax1.text(0.5, 0.5, f"Could not generate plot:\n{e}", ha='center', va='center', color='red')
        
        FigureCanvasTkAgg(fig1, master=tab1).get_tk_widget().pack(fill=tk.BOTH, expand=True)

        #AUC
        tab2 = ttk.Frame(notebook, style="Content.TFrame")
        
        notebook.add(tab2, text='Time-Dependent AUC')
        
        fig2 = plt.Figure(figsize=(8, 6), dpi=100, facecolor=STYLE_CONFIG["bg_widget"])
        
        ax2 = fig2.add_subplot(111)
        try:
            #issue was here now fixed
            y_train_sksurv = np.array(
                list(zip(train_df[event_col].astype(bool), train_df[time_col].astype(float))),
                dtype=[('event', '?'), ('time', '<f8')]
            )
            y_test_sksurv = np.array(
                list(zip(test_df[event_col].astype(bool), test_df[time_col].astype(float))),
                dtype=[('event', '?'), ('time', '<f8')]
            )
            
            covariates = self.cph_model.params_.index
            
            risk_scores_test = self.cph_model.predict_partial_hazard(test_df[covariates]).values.ravel()
            
            valid_times = np.percentile(y_train_sksurv[y_train_sksurv['event']]['time'], [10, 80])
            
            times_to_eval = np.linspace(valid_times[0], valid_times[1], 100)
            
            times, auc = cumulative_dynamic_auc(
                y_train_sksurv, y_test_sksurv, risk_scores_test, times_to_eval
            )

            ax2.plot(times, auc, marker="o", color=STYLE_CONFIG["accent_color"], markersize=4, linestyle='-')
            
            ax2.set_title("Time-Dependent Area Under ROC", color=STYLE_CONFIG["fg_header"])
            
            ax2.set_xlabel(f"Time ({time_col})", color=STYLE_CONFIG["fg_text"])
            
            ax2.set_ylabel("AUC", color=STYLE_CONFIG["fg_text"])
            
            ax2.axhline(0.5, color="grey", linestyle="--")
            
            ax2.set_ylim(0.4, 1.0)
            
            ax2.tick_params(colors=STYLE_CONFIG["fg_text"])
            
            integrated_auc = np.trapz(auc, times) / (times[-1] - times[0])
            self.log_training_message(
                f"Integrated AUC over time: {integrated_auc:.4f}"
            )

        except Exception as e:
            ax2.text(0.5, 0.5, f"Could not generate AUC plot:\n{e}", ha='center', va='center', color='red')
            self.log_training_message(f"AUC plot error: {traceback.format_exc()}", is_error=True)
        
        FigureCanvasTkAgg(fig2, master=tab2).get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        #calib curve
        tab3 = ttk.Frame(notebook, style="Content.TFrame")
        
        notebook.add(tab3, text='Calibration')
        
        fig3 = plt.Figure(figsize=(8, 6), dpi=100, facecolor=STYLE_CONFIG["bg_widget"])
        
        ax3 = fig3.add_subplot(111)
        
        try:
            horizon_years = float(self.time_horizon_var.get())
            
            horizon_days = horizon_years * 365.25
            
            predicted_survival = self.cph_model.predict_survival_function(test_df, times=[horizon_days]).iloc[0]
            
            y_true_at_horizon = (test_df[time_col] <= horizon_days) & (test_df[event_col] == 1)
            
            prob_pred, prob_true = calibration_curve(
                y_true_at_horizon,
                1 - predicted_survival, #event prob. predicted here
                n_bins=10, strategy='quantile'
            )
            
            ax3.plot(prob_pred, prob_true, marker='o', label='Model Calibration', color=STYLE_CONFIG["accent_color"])
            
            ax3.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Perfect Calibration')
            
            ax3.set_title(f"Calibration Curve at {horizon_years}-Year Horizon", color=STYLE_CONFIG["fg_header"])
            
            ax3.set_xlabel("Mean Predicted Event Probability", color=STYLE_CONFIG["fg_text"])
            
            ax3.set_ylabel("Fraction of Observed Events", color=STYLE_CONFIG["fg_text"])
            
            ax3.tick_params(colors=STYLE_CONFIG["fg_text"])
            
            ax3.legend()
        except Exception as e:
            ax3.text(0.5, 0.5, f"Could not generate calibration plot:\n{e}", ha='center', va='center', color='red')
            self.log_training_message(f"Calibration plot error: {traceback.format_exc()}", is_error=True)
        
        FigureCanvasTkAgg(fig3, master=tab3).get_tk_widget().pack(fill=tk.BOTH, expand=True)

        #PH here
        tab4 = ttk.Frame(notebook, style="Content.TFrame")
        
        notebook.add(tab4, text='PH Assumption Test')
        
        ph_text = scrolledtext.ScrolledText(tab4, wrap=tk.WORD, bg=STYLE_CONFIG["bg_entry"], font=("Courier", 9))
        
        ph_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        ph_text.insert(tk.END, "Running Proportional Hazards Assumption Test...\n\n")
        
        try:
            ph_results = self.cph_model.check_assumptions(self.train_cph_df_for_metrics, show_plots=False, p_value_threshold=0.05)
            
            ph_text.insert(tk.END, "--- Statistical Test Summary ---\n")
            
            ph_text.insert(tk.END, ph_results.to_string())
            
            violations = ph_results[ph_results['p'] < 0.05]
            
            if not violations.empty:
                ph_text.insert(tk.END, "\n\n--- Potential Violations (p < 0.05) ---\n")
                ph_text.insert(tk.END, violations.to_string())
            else:
                ph_text.insert(tk.END, "\n\nNo significant violations of the proportional hazards assumption were detected.")
        
        except Exception as e:
            ph_text.insert(tk.END, f"\n\nError during test: {e}\n")
            ph_text.insert(tk.END, traceback.format_exc())
        
        ph_text.config(state=tk.DISABLED)

    
    
    #about contents here
    def show_about_dialog(self):
        messagebox.showinfo("About Advanced Dynamic CVD Predictor",
                            "Advanced Dynamic CVD Predictor -- MLP-CPH Enhanced\n\n"
                            "A MLP model generates a non-linear risk score. A CPH model incorporates this score along with other linear predictors for survival analysis.\n\n"
                            "Developed by ODAT project.")

#main fun init here
if __name__ == "__main__":
    root = tk.Tk()
    app = DynamicCVDApp(root)
    root.mainloop()
