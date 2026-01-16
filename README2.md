# ■ DevOps & MLOps Lab Manual (Ubuntu) – Complete README

This repository contains step-by-step implementation of DevOps and MLOps lab tasks (Questions 1–9).





# ■ QUESTION 1: Git + GitHub Registration Form (Branch + Merge)


## ■ Aim
Create an event registration form and manage changes using Git branching and merging.
## ■ Steps
### Step 1: Create folder
```bash
mkdir user-registration-form
cd user-registration-form
```
### Step 2: Create `index.html`
```bash
nano index.html
```
Paste this code:
```html
<!DOCTYPE html>
<html>
<head>
 <title>Event Registration</title>
</head>
<body>
 <h2>Event Registration Form</h2>
 <form>
 <label>Name:</label><br>
 <input type="text" name="name" required><br><br>
 <label>Email:</label><br>
 <input type="email" name="email" required><br><br>
 <label>Phone:</label><br>
 <input type="text" name="phone" required><br><br>
 <button type="submit">Register</button>
 </form>
</body>
</html>
```
### Step 3: Initialize git + commit
```bash
git init
git add .
git commit -m "Initial registration form"
```
### Step 4: Push to GitHub
```bash
git branch -M main
git remote add origin <GitHub-Repo-URL>
git push -u origin main
```
### Step 5: Create branch
```bash
git checkout -b update-form
```
### Step 6: Update `index.html` (Add Department field)
Add this block **before Submit button**:
```html
<label>Department:</label><br>
<input type="text" name="dept" required><br><br>
```
### Step 7: Commit + merge + push
```bash
git add .
git commit -m "Added Department field"
git checkout main
git merge update-form
git push origin main
```
■ **Result:** Registration form updated using branch and merge process.
---






# ■ QUESTION 2: Jenkins CI (Freestyle Job)


## ■ Aim
Jenkins pulls website code from GitHub and triggers build automatically on push.
## ■ Steps
### Step 1: Install Jenkins
```bash
sudo apt update
sudo apt install jenkins -y
sudo systemctl enable --now jenkins
```
### Step 2: Open Jenkins
Open browser:
```
http://localhost:8080
```
Get admin password:
```bash
sudo cat /var/lib/jenkins/secrets/initialAdminPassword
```
### Step 3: Create Freestyle Job
Jenkins Dashboard → **New Item** → **Freestyle Project**
Name: `student-portal-ci`
### Step 4: Configure GitHub
Job → Configure → Source Code Management → **Git**
- Repository URL:
```
https://github.com/<username>/<repo>.git
```
- Branch:
```
*/main
```
### Step 5: Add Credentials
Manage Jenkins → Credentials → (global) → Add Credentials
- Scope: **Global**
- Username: GitHub ID
- Password: GitHub password/token
### Step 6: Poll SCM (*****)
Build Triggers → ■ Poll SCM
Schedule:
```txt
* * * * *
```
### Step 7: Build Step (Execute Shell)
Build Steps → Add build step → **Execute Shell**
```bash
echo "Build Started"
ls
echo "Build Finished"
```
### Step 8: Test
Make small change in HTML file and push:
```bash
git add .
git commit -m "Updated HTML"
git push origin main
```
■ **Result:** Jenkins automatically triggers build on code update.
---







# ■ QUESTION 3: Dockerize Simple Web Application (HTML)



## ■ Aim
Create a simple web app, write Dockerfile, build Docker image, run container using port mapping, and perform dock## ■ Complete Steps
### Step 1: Start Docker service
```bash
sudo systemctl start docker
sudo systemctl enable docker
docker --version
```
### Step 2: Create project folder
```bash
mkdir appDocker
cd appDocker
```
### Step 3: Create `index.html`
```bash
sudo nano index.html
```
Paste:
```html
<!DOCTYPE html>
<html>
<head>
 <title>Docker Web App</title>
</head>
<body>
 <h1>Hello from Docker HTML App!</h1>
</body>
</html>
```
### Step 4: Create `Dockerfile`
```bash
sudo nano Dockerfile
```
Paste:
```dockerfile
FROM nginx:alpine
COPY index.html /usr/share/nginx/html/index.html
EXPOSE 80
```
### Step 5: Build image (tag must be lowercase)
```bash
docker build -t htmlapp:1.0 .
docker images
```
### Step 6: Run container
```bash
docker run -d -p 8081:80 --name html-container htmlapp:1.0
docker ps
```
Open:
```
http://localhost:8081
```
### Step 7: Stop and remove container + image
```bash
docker stop html-container
docker rm html-container
docker rmi htmlapp:1.0
```
■ **Result:** HTML web app deployed inside Docker container successfully.
---
# ■ QUESTION 4: Deploy Docker Image on Kubernetes (Minikube)
## ■ Aim
Deploy Docker image on Kubernetes, expose using NodePort, and scale deployment to 3 replicas.
## ■ Steps
### Step 1: Verify image
```bash
docker images
```
### Step 2: Start minikube
```bash
minikube start --driver=docker
```
(Optional if memory warning):
```bash
minikube start --driver=docker --memory=2048
```
### Step 3: Load image into minikube
```bash
minikube image load app:1.0
```
### Step 4: Create deployment yaml
```bash
sudo nano app.yaml
```
Paste:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
 name: app-deploy
spec:
 replicas: 1
 selector:
 matchLabels:
 app: app
 template:
 metadata:
 labels:
 app: app
 spec:
 containers:
 - name: app
 image: app:1.0
 imagePullPolicy: Never
 ports:
 - containerPort: 80
```
### Step 5: Apply deployment
```bash
kubectl apply -f app.yaml
```
### Step 6: Verify pods + deployments
```bash
kubectl get pods
kubectl get deployments
```
### Step 7: Expose NodePort service
```bash
kubectl expose deployment app-deploy --type=NodePort --port=80
```
If service already exists:
```bash
kubectl get svc
```
### Step 8: Access service URL
```bash
kubectl get svc
minikube service app-deploy --url
```
### Step 9: Scale deployment to 3 replicas
```bash
kubectl scale deployment app-deploy --replicas=3
kubectl get pods
```
■ **Result:** Application deployed on Kubernetes using NodePort and scaled to 3 replicas.
---






# ■ QUESTION 5: ML Environment Setup (requirements.txt + Notebook)


## ■ Aim
Create ML environment using requirements.txt, install packages, verify setup and document in Jupyter notebook.
## ■ Steps
```bash
mkdir ml-env
cd ml-env
```
Create requirements file:
```bash
nano requirements.txt
```
Paste:
```txt
numpy
pandas
scikit-learn
matplotlib
jupyter
```
Create venv and install:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Verify:
```bash
python3 -c "import numpy, pandas, sklearn; print('Environment OK')"
```
Open Notebook:
```bash
jupyter notebook
```
Git commit:
```bash
git init
git add .
git commit -m "ML environment setup"
```
■ **Result:** ML environment set successfully and documented.
---




# ■ QUESTION 6: Docker Compose (App + Redis)


## ■ Aim
Run multi-container app using Docker Compose with Flask app and Redis service.
## ■ Steps
```bash
mkdir compose-app
cd compose-app
```
Create `app.py`:
```bash
nano app.py
```
Paste:
```python
import redis
from flask import Flask
app = Flask(__name__)
r = redis.Redis(host="redis", port=6379)
@app.route("/")
def home():
 r.incr("visits")
 return f"Visits: {r.get('visits').decode()}"
if __name__ == "__main__":
 app.run(host="0.0.0.0", port=5000)
```
Create requirements:
```bash
nano requirements.txt
```
Paste:
```txt
flask
redis
```
Create Dockerfile:
```bash
nano Dockerfile
```
Paste:
```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py .
EXPOSE 5000
CMD ["python", "app.py"]
```
Create docker-compose:
```bash
nano docker-compose.yml
```
Paste:
```yaml
version: "3.8"
services:
 app:
 build: .
 ports:
 - "5000:5000"
 depends_on:
 - redis
 redis:
 image: redis:alpine
```
Run:
```bash
docker compose up -d --build
docker compose ps
```
Open:
```
http://localhost:5000
```
■ **Result:** Multi-container system runs successfully using Docker Compose.
---





# ■ QUESTION 7: DVC Data Versioning + Pipeline
## ■ Aim
Track raw and processed datasets using DVC and create reproducible pipeline using `dvc.yaml`.
## ■ Steps
```bash
mkdir ques7
cd ques7
git init
dvc init
```
Raw dataset:
```bash
mkdir data
cd data
mkdir raw
cd raw
gedit data.csv
```
Paste:
```csv
name,marks
A,80
B,90
C,70
D,85
```
Create processed folder:
```bash
cd ..
mkdir processed
cd ..
```
Create script:
```bash
mkdir scripts
cd scripts
gedit clean.py
```
Paste:
```python
import pandas as pd
df = pd.read_csv("data/raw/data.csv")
df["marks"] = df["marks"] + 5
df.to_csv("data/processed/clean.csv", index=False)
print("■ Cleaned dataset saved to data/processed/clean.csv")
```
Create DVC stage:
```bash
cd ..
dvc stage add -n clean \
-d scripts/clean.py -d data/raw/data.csv \
-o data/processed/clean.csv \
python scripts/clean.py
```
Run pipeline:
```bash
dvc repro
```
Git commit:
```bash
git add dvc.yaml dvc.lock data/processed/.gitignore
git commit -m "Added DVC pipeline for dataset cleaning"
```
■ **Result:** DVC pipeline created and executed successfully.
---





# ■ QUESTION 8: MLflow Experiment Tracking



## ■ Aim
Perform experiment tracking using MLflow by training ML model and logging metrics and parameters.
## ■ Code (train.py)
```python
import mlflow
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2)
for c in [0.1, 1.0]:
 with mlflow.start_run():
 m = LogisticRegression(C=c, max_iter=200)
 m.fit(Xtr, ytr)
 acc = accuracy_score(yte, m.predict(Xte))
 mlflow.log_param("C", c)
 mlflow.log_metric("accuracy", acc)
 mlflow.sklearn.log_model(m, "model")
```
Run:
```bash
python train.py
mlflow ui
```
Open link shown.
■ **Result:** MLflow tracks runs and shows comparison.
---





# ■ QUESTION 9: ONNX Export + Benchmark
## ■ Aim
Export ML model to ONNX format and benchmark ONNX Runtime vs scikit-learn.
## ■ Code (onnx_test.py)
```python
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from skl2onnx import to_onnx
import onnxruntime as ort
# Train model
X = np.random.rand(1000, 1)
y = 3 * X.squeeze() + 5
model = LinearRegression()
model.fit(X, y)
# Export ONNX
onnx_model = to_onnx(model, X)
with open("model.onnx", "wb") as f:
 f.write(onnx_model.SerializeToString())
# ONNX runtime session
sess = ort.InferenceSession("model.onnx")
# Benchmark sklearn
t0 = time.time()
for _ in range(10000):
 model.predict(X)
sk_time = time.time() - t0
# Benchmark onnx
t0 = time.time()
for _ in range(10000):
 sess.run(None, {"X": X})
onnx_time = time.time() - t0
print("scikit-learn time:", sk_time)
print("ONNX Runtime time:", onnx_time)
```
■ **Result:** Model successfully converted to ONNX and inference performance compared.
---
