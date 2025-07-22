# ✨ AI-Powered Spam Email Detector (Fullstack App)

Hi, I’m **Praise Ogooluwa Bakare** 👋

This is a fullstack AI-powered spam email detector that I built from scratch—frontend and backend included.

The frontend is built with **React + Vite**, styled using **Tailwind CSS** and **shadcn/ui**, and deployed on **Vercel**.  
The backend is a **FastAPI** application powered by a **machine learning model** for classifying spam emails, deployed on **Render**.

---

## 🌐 Live Links

- **Frontend (React/Vercel)**: _Coming Soon_

- **Backend API (FastAPI/Render)**: _Coming Soon_

---

## 🚀 Tech Stack

### Frontend

- React + Vite

- TypeScript

- Tailwind CSS

- shadcn/ui

### Backend

- Python + FastAPI

- Scikit-learn (Naive Bayes)

- TF-IDF Vectorizer

- langdetect

- `.eml` email file support

- CORS enabled


---

## 🔍 Features

### 💡 Spam Detection Capabilities
- **Spam Classification** (SPAM / HAM)

- **Keyword Scanning** for 50+ known spam triggers

- **Sender Domain Risk Analysis**

- **Bot/Auto-generated Text Detection**

- **Language Detection (55+ languages)**

- **EML File Upload Support**

- **Smart AI-Based Explanations** with prediction confidence

---

## 🛠 Backend Setup (FastAPI)

### ⚙️ Installation

``bash
# Clone the backend repo
git clone <backend_repo_url>

cd backend

# Create virtual environment

python -m venv venv

source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the spam detection model
python create_model.py

# Run the server
uvicorn app:app --reload

---
### 🔗 API Endpoints

POST /predict
Predict spam likelihood for raw email text.

Request:
{
  "text": "Congratulations! You've won $1,000,000!",
  "sender": "winner@lottery.tk"
}

Response:
{
  "label": "SPAM",
  "confidence": 96.2,
  "spammy_keywords": ["congratulations", "won"],
  "language": "en",
  "sender_risk": "High Risk",
  "bot_likeness": "Likely Bot-Written",
  "explanation": "This email is very likely spam based on suspicious keywords and sender domain."
}

POST /upload_eml
Upload a .eml file for spam analysis.

GET /health
Health check for model and API.

GET /
Overview of API endpoints and version.
---

### 🧠 AI Model Details

Algorithm: Multinomial Naive Bayes

Features: TF-IDF (unigrams + bigrams)

Training Data: Custom spam/ham dataset

Performance: ~90% accuracy

Explanations: Includes keyword, domain risk, and bot analysis

🧪 Test the API
Curl:
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"text":"Click here to claim your reward","sender":"scammer@promo.ru"}'

Python:

import requests

res = requests.post("http://localhost:8000/predict", json={
    "text": "Hi, are we still meeting tomorrow?",
    "sender": "colleague@company.com"
})
print(res.json())
---

### 🎨 Frontend Setup (React + Vite)

# Clone the frontend repo
git clone <your_frontend_repo_url>
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
---

### 🔧 Customization
✏️ Add Custom Spam Keywords
Edit the SPAM_KEYWORDS list in app.py:

SPAM_KEYWORDS = [
    'click here',  
    'win money',
    'urgent',
    'verify account',
    # your custom additions here
]
---
### 🔐 Add New Risky Domains
Update the HIGH_RISK_PATTERNS list:

HIGH_RISK_PATTERNS = [
    '.tk', '.ru', '.ml', 'tempmail',
    # add more here
]
---
### 📁 Project Structure
project-root/
├── backend/

│   ├── app.py

│   ├── model.py

│   ├── model.pkl

│   ├── requirements.txt#
│   └── README.md

├── frontend/

│   ├── index.html

│   ├── style.css

│   ├── script.js (or React app files)

│   └── README.md

└── README.md (this file)
---
### 🚀 Deployment

Backend (Render)

Deploy backend/ on Render

Use uvicorn app:app --host=0.0.0.0 --port=10000 as your start command

Set create_model.py to run during deploy if needed

Frontend (Vercel)
Push the frontend/ directory to GitHub

Connect repo on Vercel

Set root to frontend if deploying from monorepo
---
### 📄 License

This project is provided for educational, demonstration, and practical use. Feel free to fork, modify, or build upon it.
---
### 🙋‍♂️ About Me

I’m Praise Ogooluwa Bakare, a data scientist and backend developer with a growing passion for building intelligent applications and scalable backend systems.

Let’s connect on GitHub or collaborate on something powerful.
