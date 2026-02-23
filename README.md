# 🩺 MediPulseAI — Diabetes Risk Analyzer

A premium AI-powered web application that predicts diabetes risk using a trained neural network model, with optional user authentication and cloud-persisted analysis history.

---

## ✨ Features

- 🤖 **AI Prediction** — Diabetes risk analysis using an ANN trained with SMOTE balancing
- 📊 **Probability Score** — Visual risk percentage with animated progress bar
- 👤 **Optional Authentication** — Use the analyzer as a guest, or sign in/sign up for history
- 📋 **Analysis History** — Every prediction saved per user in MongoDB Atlas
- 🌙 **Dark / Light Mode** — Persistent theme preference
- 📱 **Mobile Responsive** — Glassmorphism UI works on all screen sizes

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask, Flask-Login |
| ML Model | TensorFlow / Keras (ANN) |
| Preprocessing | scikit-learn (LabelEncoder, StandardScaler) |
| Database | MongoDB Atlas (pymongo) |
| Frontend | HTML, Tailwind CSS, Vanilla JS |
| Auth | Werkzeug password hashing |

---

## 🚀 Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/diabetes-app.git
cd diabetes-app
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables
Create a `.env` file in the project root:
```env
MONGO_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/medipulse?retryWrites=true&w=majority
SECRET_KEY=your-secret-key-here
```
> ⚠️ If your password contains special characters like `@`, encode them (e.g. `@` → `%40`)

### 4. Run the app
```bash
python app.py
```
Open [http://localhost:5000](http://localhost:5000)

---

## 📁 Project Structure

```
diabetes-app/
├── app.py                          # Flask app, routes, MongoDB logic
├── .env                            # Environment variables (never commit this)
├── .env.example                    # Template for .env
├── requirements.txt                # Python dependencies
├── diabetes_ann_smote_model.keras  # Trained ANN model
├── le_gender.pkl                   # Gender label encoder
├── le_smoking.pkl                  # Smoking history label encoder
├── scaler.pkl                      # StandardScaler
├── static/
│   └── style.css                   # Custom styles
└── templates/
    ├── index.html                  # Main analyzer page
    ├── login.html                  # Sign in page
    ├── register.html               # Sign up page
    └── history.html                # Analysis history page
```

---

## 🗄 MongoDB Collections

| Collection | Purpose |
|---|---|
| `users` | Stores usernames and hashed passwords |
| `history` | Stores per-user prediction history (max 50 records) |

---

## 🔐 Default Accounts

| Username | Password |
|---|---|
| `admin` | `admin123` |
| `doctor` | `doctor123` |

> These are seeded automatically on first run if the `users` collection is empty. Change them after deployment.

---

## 🌐 Deployment

Recommended platform: **[Render.com](https://render.com)** (supports Flask + TensorFlow, persistent server)

> ⚠️ Do **not** deploy to Vercel — TensorFlow exceeds Vercel's 250MB bundle limit.

Set the following environment variables in your hosting dashboard:
- `MONGO_URI`
- `SECRET_KEY`

---

## 📄 License

MIT License — free to use and modify.

## Project BY
P PremKumar
