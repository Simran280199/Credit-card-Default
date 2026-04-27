# 🚀 Complete Deployment Guide — NEXUS Credit Intelligence
## GitHub + Streamlit Cloud | Step-by-Step

---

## ✅ Why the Old App Failed on Streamlit Cloud

| Problem | Cause | Fix Applied |
|---|---|---|
| `ModuleNotFoundError: joblib` | `requirements.txt` missing `joblib` | Added `joblib==1.4.2` |
| `model.pkl` = 82 MB | 100 trees, no compression | Retrained: 50 trees + `compress=3` → **1.85 MB** |
| `gdown` / `tensorflow` / `xgboost` | Unnecessary heavy deps | Removed completely |
| GitHub 25 MB file limit | Huge pickle | Model now 1.85 MB ✅ |

---

## 📁 Your Final Project Structure

```
your-repo/                          ← GitHub repository root
│
├── models/
│   ├── model.pkl                   ← 1.85 MB ✅ (fits GitHub)
│   ├── scaler.pkl                  ← < 1 KB
│   └── meta.json                   ← 2 KB
│
├── app.py                          ← Streamlit app
├── requirements.txt                ← Cloud dependencies
└── README.md                       ← (optional)
```

> ⚠️ Do NOT include: `myenv28/`, `Credit_Card_Default.csv`, `.ipynb`, old `model.pkl` (82MB)

---

## 🔧 STEP-BY-STEP COMMANDS

### STEP 1 — Install Git (if not already)
```bash
# Check if git is installed
git --version

# If not installed, download from: https://git-scm.com/download/win
```

---

### STEP 2 — Create GitHub Repository

1. Go to **https://github.com** → Sign in
2. Click **"New"** (green button, top left)
3. Repository name: `credit-card-default`  
4. Set to **Public** (required for free Streamlit Cloud)
5. ✅ Check **"Add a README file"**
6. Click **"Create repository"**

---

### STEP 3 — Prepare Local Folder

Open **VS Code Terminal** (`Ctrl + ~`) and run:

```bash
# Navigate to your project folder
cd "C:\Users\simra\cc default"

# Verify these files exist
dir
# You should see: app.py, requirements.txt, models/ folder
```

---

### STEP 4 — Initialize Git and Push to GitHub

```bash
# Step 4a: Initialize git repo
git init

# Step 4b: Set your identity (first time only)
git config --global user.email "your-email@gmail.com"
git config --global user.name "Your Name"

# Step 4c: Connect to your GitHub repo
# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/credit-card-default.git

# Step 4d: Stage all required files
git add app.py
git add requirements.txt
git add models/model.pkl
git add models/scaler.pkl
git add models/meta.json

# Step 4e: Commit
git commit -m "Initial commit: NEXUS Credit Intelligence App"

# Step 4f: Push to GitHub
git branch -M main
git push -u origin main
```

> 💡 **GitHub will ask for your password** — use a **Personal Access Token** not your password:
> Go to GitHub → Settings → Developer Settings → Personal Access Tokens → Generate New Token (classic)
> Give it `repo` scope → Copy token → Paste as password

---

### STEP 5 — Verify on GitHub

1. Go to `https://github.com/YOUR_USERNAME/credit-card-default`
2. Confirm you see:
   - `app.py`
   - `requirements.txt`
   - `models/` folder with `model.pkl` (should show ~1.85 MB)

---

### STEP 6 — Deploy on Streamlit Cloud

1. Go to **https://share.streamlit.io**
2. Sign in with your **GitHub account**
3. Click **"New app"**
4. Fill in:
   - **Repository:** `YOUR_USERNAME/credit-card-default`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **"Deploy!"**
6. Wait ~2–3 minutes for first build
7. Your app will be live at:  
   `https://YOUR_USERNAME-credit-card-default-app-XXXXX.streamlit.app`

---

### STEP 7 — If Deployment Fails (Troubleshooting)

#### Error: Module not found
```
Check requirements.txt has all packages listed correctly.
Your requirements.txt should only contain:
  streamlit==1.35.0
  pandas==2.2.2
  numpy==1.26.4
  scikit-learn==1.4.2
  imbalanced-learn==0.12.3
  plotly==5.22.0
  joblib==1.4.2
```

#### Error: model.pkl not found
```
Check that models/ folder was pushed to GitHub.
Run: git status
If models/ is missing, run:
  git add models/
  git commit -m "add models folder"
  git push
```

#### Error: File too large
```
If any file is still > 25MB, GitHub will reject it.
Check: models/model.pkl should be ~1.85MB only.
Run: dir models\ (Windows) to verify sizes.
```

---

### STEP 8 — Updating the App Later

Whenever you change `app.py` or any file:

```bash
cd "C:\Users\simra\cc default"
git add app.py
git commit -m "Update: describe your change here"
git push
```
Streamlit Cloud **automatically redeploys** within ~60 seconds.

---

## 📌 Quick Reference — Files to Push vs Exclude

| File / Folder | Push to GitHub? | Reason |
|---|---|---|
| `app.py` | ✅ YES | Main application |
| `requirements.txt` | ✅ YES | Cloud dependencies |
| `models/model.pkl` | ✅ YES | 1.85 MB — fits GitHub |
| `models/scaler.pkl` | ✅ YES | Tiny file |
| `models/meta.json` | ✅ YES | Tiny file |
| `myenv28/` | ❌ NO | Virtual environment — never push |
| `Credit_Card_Default.csv` | ❌ NO | 2.7 MB data not needed at runtime |
| `*.ipynb` | ❌ NO | Notebook not needed for app |
| `credit_card_default_project.zip` | ❌ NO | Archive not needed |
| Old `model.pkl` (82 MB) | ❌ NO | Too large for GitHub |

---

## 🛡️ Add .gitignore (Recommended)

Create a file named `.gitignore` in your project folder with this content:

```
myenv28/
__pycache__/
*.ipynb
*.csv
*.zip
.env
```

Then:
```bash
git add .gitignore
git commit -m "Add gitignore"
git push
```

---

## ✅ Final Checklist Before Deploying

- [ ] `app.py` is updated (premium dark theme version)
- [ ] `requirements.txt` has only 7 packages listed (no gdown/tensorflow/xgboost)
- [ ] `models/model.pkl` is **1.85 MB** (not 82 MB)
- [ ] `models/scaler.pkl` is present
- [ ] `models/meta.json` is present
- [ ] GitHub repo is **Public**
- [ ] All 5 files are visible on GitHub before deploying

---

*NEXUS Credit Intelligence — Deployment Guide v2.0*
