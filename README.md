<h1 align="center">Hi, I'm Gimhan Inupa 👨‍💻📸</h1>
<h3 align="center">Crafting realities in code and capturing moments in time.</h3>

<p align="center">
  <img src="https://komarev.com/ghpvc/?username=gimhaninupa&label=Profile%20views&color=00ADB5&style=flat" alt="gimhaninupa" />
</p>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&pause=1000&color=F7941D&center=true&vCenter=true&multiline=true&width=900&lines=Full+Stack+Dev+%7C+AI+%26+ML+Enthusiast+%7C+Photographer" alt="Typing SVG" />
</p>

---

- 🔭 I’m currently working on **Full Stack Web Apps & AI Models**
- 👨‍💻 Reach me via: [Gimhan Inupa](https://gimhaninupa.com)
- 🌐 View my projects & photography gallery at [Gimhan Inupa](https://gimhaninupa.com)
- 💡 More about me on [LinkedIn](https://www.linkedin.com/in/gimhan-inupa-samaraweera-199233375/)
- ⚡ **Fun fact**: I debug code by day and capture light by night 🌙

---

### Connect with me:

<p align="center">
  <a href="https://www.linkedin.com/in/gimhan-inupa-samaraweera-199233375/" target="_blank">
    <img src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" height="30" width="40" alt="LinkedIn" />
  </a>
</p>

---

### Languages & Tools

<p align="center">
  <img src="https://skillicons.dev/icons?i=react,angular,flutter,laravel,nextjs,tailwind,html,css,js,ts,java,cpp,python,c,mysql,mongodb,firebase,aws,azure,gcp,docker,tensorflow,git,github,vscode" />
</p>

---

### GitHub Trophies

<p align="center">
  <img src="https://github-trophies.vercel.app/?username=gimhaninupa&theme=gruvbox" alt="GitHub Trophies">
</p>

---

# 🚀 AI-Powered Career Intelligence System

This repository hosts the **AI-Powered Career Intelligence System**, an intelligent workspace that enables job seekers to analyze, match, and optimize their resumes against current job datasets. The project uses advanced Machine Learning and Natural Language Processing (NLP) models to parse resumes, predict professional domains, extract key technical competencies, semantically match resumes with target roles, and rewrite CV bullet points to maximize relevance.

## 🌟 Key Features

- 📄 **Multi-Format Resume Parser**: Supports uploading resumes in **PDF**, **DOCX**, or **TXT** formats. Text content is automatically extracted on upload.
- 🧠 **SpaCy Named Entity Recognition (NER)**: Automatically extracts and highlights technical and soft skills from candidate resumes.
- 🎯 **Semantic Job Matching**: Leverages `SentenceTransformers` embeddings combined with a K-Nearest Neighbors (KNN) model to rank the top 5 job listings that match a user's resume semantically.
- 📊 **Career Field Classifier**: Uses TF-IDF vectorization and a Logistic Regression classifier trained on industry job datasets to classify resumes and suggest job fields.
- ✨ **Generative CV Optimizer (FLAN-T5)**: Harnesses a sequence-to-sequence transformer model to rewrite or align specific CV bullet points with targeted job requirements.

---

## 🛠️ Tech Stack

### Frontend
- **Framework**: [React](https://react.dev/) + [Vite](https://vite.dev/) (fast HMR and building)
- **Styling**: [Tailwind CSS](https://tailwindcss.com/)
- **Icons**: [Lucide React](https://lucide.dev/)

### Backend
- **Framework**: [Flask](https://flask.palletsprojects.com/) + [Flask-CORS](https://flask-cors.readthedocs.io/)
- **NLP & Embeddings**: [SpaCy](https://spacy.io/) (`en_core_web_sm`), [SentenceTransformers](https://www.sbert.net/)
- **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/), [PyTorch](https://pytorch.org/)
- **Generative AI**: [Transformers (Hugging Face - FLAN-T5)](https://huggingface.co/docs/transformers)
- **File Parsing**: `pypdf`, `python-docx`

---

## 💻 How to Run the Project on Localhost

Follow these step-by-step instructions to get the application running locally on your computer.

### 📋 Prerequisites
Make sure you have the following installed on your machine:
- **Node.js** (v18.0.0 or higher) & **npm**
- **Python** (v3.9 or higher) & **pip**

---

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/gimhaninupa/AI-Powered-Career-Intelligence-System-.git
cd AI-Powered-Career-Intelligence-System-
```

---

### 2️⃣ Backend Setup (Flask Server)
Navigate to the backend directory and set up a virtual environment:

```bash
# Move into the backend folder
cd backend

# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
# On Windows (PowerShell/CMD):
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# Install the Python dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py
```

> **Note**: On the first startup, the server will automatically download the necessary NLP models (`en_core_web_sm` for SpaCy, SentenceTransformer models, and FLAN-T5 for translation/generation). This may take a few minutes depending on your internet connection.
>
> The backend server will be listening at **`http://localhost:5000`**.

---

### 3️⃣ Frontend Setup (React App)
Open a new terminal session, navigate to the frontend directory, and run the React app using Vite:

```bash
# Move into the frontend folder
cd frontend

# Install Node modules
npm install

# Start the local development server
npm run dev
```

> **Note**: The frontend application will start up and run at **`http://localhost:5173`** (or the URL printed in the terminal). Open this address in your web browser to interact with the system!
