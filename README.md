QUESTION 1: Git + GitHub Registration Form (Branch + Merge)
Aim
Create an event registration form and manage changes using Git branching and merging.

Step 1: Create folder
mkdir user-registration-form
cd user-registration-form

Step 2: Create index.html (FULL)
nano index.html
PASTE THIS FULL CODE:
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

Step 3: Git init + commit
git init
git add .
git commit -m "Initial registration form"

Step 4: Push to GitHub
git branch -M main
git remote add origin <GitHub-Repo-URL>
git push -u origin main

Step 5: Create branch
git checkout -b update-form

Step 6: Update index.html (ADD Department field)
Edit index.html and add this block BEFORE the submit button:
<label>Department:</label><br>
<input type="text" name="dept" required><br><br>

Step 7: Commit + merge + push
git add .
git commit -m "Added Department field"
git checkout main
git merge update-form
git push origin main





QUESTION 2: Jenkins CI (Freestyle Job) – Simple Steps

Aim
Jenkins pulls website code from GitHub and triggers build automatically on push.


## QUESTION 2: Jenkins CI using WAR + Freestyle Job + Poll SCM (*****)

###  Aim
Configure Jenkins Freestyle Job that automatically pulls website code from GitHub and builds the project whenever a change is pushed (using Poll SCM schedule `* * * * *`).

---

# PART A: Jenkins Installation using WAR (JDK 21)

##  Step 1: Install Java 21

sudo apt update
sudo apt install openjdk-21-jdk -y
java -version


## Step 2: Download Jenkins WAR file

mkdir -p ~/jenkins
cd ~/jenkins
wget https://get.jenkins.io/war-stable/latest/jenkins.war
ls


## Step 3: Run Jenkins WAR

java -jar jenkins.war --httpPort=8080

 Jenkins will run at:

http://localhost:8080


##  Step 4: Unlock Jenkins

In new terminal:

cat ~/.jenkins/secrets/initialAdminPassword

Paste it in Jenkins setup page → Install Suggested Plugins → Create Admin user.

---

# PART B: Jenkins Freestyle Job Setup

## Step 5: Create Freestyle Job

Jenkins Dashboard → **New Item**
Select: **Freestyle project**
Name: `student-portal-ci`
Click: **OK**

---

##  Step 6: Add GitHub Credentials (GLOBAL)

Jenkins Dashboard →
**Manage Jenkins → Credentials → (global) → Add Credentials**

Fill:

* Kind: **Username with password**
* Scope: **Global**
* Username: GitHub Username
* Password: GitHub Password / Token
* ID: `github-cred`
* Description: GitHub Credentials

Click  Save

---

##  Step 7: Configure GitHub Repository in Job

Job → Configure → **Source Code Management**

 Select: Git
Repository URL:

https://github.com/<your-username>/<repo-name>.git


Credentials:
Select: `github-cred`

Branch:
*/main

---

#  PART C: Poll SCM Trigger (*****)

## Step 8: Enable Poll SCM

Job → Configure → **Build Triggers**

 Tick: **Poll SCM**

In Schedule box paste:
* * * * *


 Meaning: Jenkins checks GitHub every **1 minute** for changes.

---

# PART D: Build Steps (Execute Shell)

##  Step 9: Add Build Step → Execute Shell

Job → Configure → **Build Steps**
Click: **Add build step**
Select: **Execute shell**

Paste this command:
echo "Starting Student Portal Jenkins"
echo "Build Time"
date
echo "Current Directory"
pwd
echo "Files in Workspace"
ls -la
echo "Build Completed Successfully"


Click  Save

---

# PART E: Test (Auto Build)

##  Step 10: First Manual Build

Job page → Click:  **Build Now**
Then check:  **Console Output**

---

##  Step 11: Make Change + Push to GitHub (Trigger Build Automatically)

Edit HTML file (example):

bash
nano index.html


Make small change and push:

bash
git add .
git commit -m "Updated HTML page"
git push origin main


Within 1 minute, Jenkins will automatically trigger build because Poll SCM is `* * * * *`.

---

### Result

Jenkins Freestyle Job successfully configured using WAR installation, GitHub global credentials, Poll SCM trigger (`* * * * *`) and Execute Shell build step. Auto build is triggered whenever changes are pushed to GitHub.







Question 3: Dockerize Simple Web Application (HTML) + Docker Commands

 Aim
Create a simple web application (HTML), write a Dockerfile, build a Docker image, run the container with port mapping, and perform Docker operations such as listing images/containers, stopping container, removing container and image.

---

# ==========================================
# CREATE SIMPLE WEB APPLICATION USING DOCKER
# ==========================================

# ---------- STEP 1: CHECK & INSTALL DOCKER ----------
docker --version

# If Docker is NOT installed
sudo apt update
sudo apt install docker.io -y

sudo systemctl start docker
sudo systemctl enable docker

# ---------- STEP 2: CREATE PROJECT DIRECTORY ----------
mkdir html-app
cd html-app

# ---------- STEP 3: CREATE HTML FILE ----------
cat <<EOF > index.html
<!DOCTYPE html>
<html>
<head>
    <title>Docker Web App</title>
</head>
<body>
    <h1>Hello from Docker!</h1>
    <p>This web app is running inside a Docker container.</p>
</body>
</html>
EOF

# ---------- STEP 4: CREATE DOCKERFILE ----------
cat <<EOF > Dockerfile
FROM nginx:latest
COPY index.html /usr/share/nginx/html/index.html
EOF

# ---------- STEP 5: BUILD DOCKER IMAGE ----------
sudo docker build -t html-app:1.0 .

# ---------- STEP 6: LIST DOCKER IMAGES ----------
sudo docker images

# ---------- STEP 7: RUN DOCKER CONTAINER (PORT MAPPING) ----------
sudo docker run -d -p 8080:80 --name html-container html-app:1.0

# Open browser and access:
# http://localhost:8080

# ---------- STEP 8: LIST RUNNING CONTAINERS ----------
sudo docker ps

# ---------- STEP 9: LIST ALL CONTAINERS ----------
sudo docker ps -a

# ---------- STEP 10: STOP RUNNING CONTAINER ----------
sudo docker stop html-container

# ---------- STEP 11: REMOVE CONTAINER ----------
sudo docker rm html-container

# ---------- STEP 12: REMOVE DOCKER IMAGE ----------
sudo docker rmi html-app:1.0

 Result

A simple HTML application is successfully containerized using Docker, executed using port mapping in browser, and Docker operations (list, stop, remove) are performed using Docker commands.











##  Question 4: Deploy Docker Image on Kubernetes (Minikube) + NodePort + Scale

###  Aim
Deploy the previously created Docker image on Kubernetes using a Deployment YAML file. Verify the pod is running, expose the app using a NodePort service, access it using node IP + port, and scale the deployment to 3 replicas.

---

### Step 1: Verify Docker Image
Check that your Docker image is available (example: `app:1.0`):

docker images
Step 2: Start Minikube using Docker Driver
minikube start --driver=docker


(If memory warning occurs, use lower memory)

minikube start --driver=docker --memory=2048

 Step 3: Load Docker Image into Minikube
minikube image load app:1.0

 Step 4: Create Deployment YAML (app.yaml)

Create file:

sudo nano app.yaml


Paste this content:

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

 Step 5: Apply Deployment
kubectl apply -f app.yaml

 Step 6: Verify Pod and Deployment
kubectl get pods
kubectl get deployments

 Step 7: Expose Application (NodePort Service)
kubectl expose deployment app-deploy --type=NodePort --port=80


 If you get error: services "app-deploy" already exists, it means service is already created. Just check service using:

kubectl get svc

Step 8: Get Service URL and Access Application
kubectl get svc
minikube service app-deploy --url


Open the shown URL in browser (example):

http://127.0.0.1:43749


Note: When using docker driver, keep terminal open while accessing service.

Step 9: Scale Deployment to 3 Replicas
kubectl scale deployment app-deploy --replicas=3
kubectl get pods

ANOTHER METHOD 

# -------------------------------
# QUESTION 4 : Docker -> Kubernetes (Minikube)
# -------------------------------

# 1) Create folder
mkdir mlops
cd mlops

# 2) Create HTML file
gedit html.html
# PASTE THIS:
# <h1> Hello From Docker </h1>

# 3) Create Dockerfile
gedit Dockerfile
# PASTE THIS:
# FROM nginx:latest
# COPY html.html /usr/share/nginx/html/

# 4) Build Docker image
docker build -t app:1.0 .

# 5) Check image
docker images

# 6) Start Minikube (Docker driver)
minikube start --driver=docker

# (If memory warning comes, run this instead)
# minikube start --driver=docker --memory=2048

# 7) Load docker image into minikube
minikube image load app:1.0

# 8) Create Deployment YAML file
gedit app.yaml

# PASTE THIS FULL YAML:
# apiVersion: apps/v1
# kind: Deployment
# metadata:
#   name: app-deploy
# spec:
#   replicas: 1
#   selector:
#     matchLabels:
#       app: myapp
#   template:
#     metadata:
#       labels:
#         app: myapp
#     spec:
#       containers:
#       - name: app
#         image: app:1.0
#         imagePullPolicy: Never

# 9) Apply YAML
kubectl apply -f app.yaml

# 10) Check pods and deployments
kubectl get pods
kubectl get deployments

# 11) Expose deployment (NodePort Service)
kubectl expose deployment app-deploy --type=NodePort --port=80

# If service already exists error comes, ignore and continue
kubectl get svc

# 12) Get URL and access application
minikube service app-deploy --url

# NOTE:
# If using docker driver, keep terminal open while accessing URL in browser.

# 13) Scale replicas to 3
kubectl scale deployment app-deploy --replicas=3

# 14) Verify scaled pods
kubectl get pods

# IMPORTANT:
# If you typed kubectle by mistake, correct command is kubectl






QUESTION 5: ML Environment Setup (FULL requirements.txt + notebook steps)

Aim
Create requirements.txt, install packages, verify environment, document in notebook, commit to Git.

Step 1: Create folder
mkdir ml-env
cd ml-env

Step 2: requirements.txt (FULL)
nano requirements.txt
numpy
pandas
scikit-learn
matplotlib
jupyter

Step 3: venv + install
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Step 4: Verify
python3 -c "import numpy, pandas, sklearn; print('Environment OK')"

Step 5: Jupyter Notebook content to write
Run:
jupyter notebook
In notebook, write:
1) pip install -r requirements.txt
2) import numpy, pandas, sklearn
3) print versions:
import numpy as np, pandas as pd, sklearn
print(np.__version__, pd.__version__, sklearn.__version__)

Step 6: Git commit
git init
git add .
git commit -m "ML environment setup"







QUESTION 6: Docker Compose (FULL app.py + Dockerfile + compose file)

Redis Question (Docker Compose) – SAME steps as your screenshots (Copy Paste)
# Install docker + docker-compose
sudo apt update
sudo apt install docker-compose docker.io -y

# Start docker
sudo systemctl start docker
sudo systemctl enable docker

# Create folder
mkdir ques8
cd ques8

# Create required files
gedit app.py
gedit Dockerfile
gedit docker-compose.yml

1) app.py (PASTE THIS EXACTLY)

Open app.py and paste:

import redis

result = redis.Redis(host="redis", port=6379)
result.incr("count")

print("Count :", result.get("count").decode())


Save and close.

2) Dockerfile (PASTE THIS EXACTLY)

Open Dockerfile and paste:

FROM python:3.10
RUN pip install redis
COPY app.py .
CMD ["python", "app.py"]


Save and close.

3) docker-compose.yml (PASTE THIS EXACTLY)

 IMPORTANT: YAML must use spaces only, not TAB.

Open docker-compose.yml and paste:

services:
  app:
    build: .
    depends_on: [redis]

  redis:
    image: redis
    volumes: [redis-data:/data]

volumes:
  redis-data:

 Save and close.

 4) Run Docker Compose Build + Up (Same as Screenshot)
docker-compose up --build


 Expected output:

It will create network + volume

Build python image

Run redis container

Run app container

It prints:

Count : 1


Run again (optional):

docker-compose up --build


Then output increments:

Count : 2

 If you get YAML error (same as screenshot)

Error:

found character '\t' that cannot start any token

Fix:

Open file:

gedit docker-compose.yml


Remove TAB spaces (press Backspace)

Use only spaces

Save
Then run again:

docker-compose up --build

 Stop Containers (Optional)

Press:

CTRL + C


Then run:

docker-compose down







##  Question 7: DVC Data Versioning + Pipeline (Raw → Processed) using dvc.yaml + dvc repro

###  Aim
Implement data ingestion, cleaning, and versioning using DVC by tracking raw and processed datasets, creating a reproducible pipeline with `dvc.yaml`, and validating reproducibility using `dvc repro`.

---

##  Complete Steps (Copy-Paste)

### 1) Create Project Folder + Initialize Git and DVC
mkdir ques7
cd ques7

git init
dvc init

2) Create Raw Data Folder and Dataset
mkdir data
cd data

mkdir raw
cd raw


Create raw CSV file:

gedit data.csv


Paste this inside data.csv:

name,marks
A,80
B,90
C,70
D,85


Go back:

cd ..
mkdir processed
cd ..

3) Create Scripts Folder and Cleaning Script
mkdir scripts
cd scripts


Create Python script:

gedit clean.py


Paste this inside clean.py:

import pandas as pd

# Read raw dataset
df = pd.read_csv("data/raw/data.csv")

# Cleaning / processing (example: add 5 marks)
df["marks"] = df["marks"] + 5

# Save processed dataset
df.to_csv("data/processed/clean.csv", index=False)

print(" Cleaned dataset saved to data/processed/clean.csv")


Go back:

cd ..

4) Create DVC Stage (Pipeline)
dvc stage add -n clean \
-d scripts/clean.py -d data/raw/data.csv \
-o data/processed/clean.csv \
python scripts/clean.py


This command will automatically create:

dvc.yaml

dvc.lock

5) Run Pipeline
dvc repro

6) Add DVC Files to Git and Commit
git add dvc.yaml dvc.lock data/processed/.gitignore
git commit -m "Added DVC pipeline for dataset cleaning"

7) Check Files
ls


Expected:

data  dvc.yaml  dvc.lock  scripts

✅ Result

Raw dataset is tracked using DVC, processed dataset is generated using reproducible DVC pipeline (dvc.yaml), and pipeline successfully runs using dvc repro.





Questions 8:

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# C values between 0.1 and 1.0
for c in [0.1, 1.0]:
    with mlflow.start_run():
        m = LogisticRegression(C=c, max_iter=200)
        m.fit(X_train, y_train)

        acc_score = accuracy_score(y_test, m.predict(X_test))

        mlflow.log_param("C", c)
        mlflow.log_metric("accuracy", acc_score)
        mlflow.sklearn.log_model(m, "model")


then run the python code and after that run mlflow ui and click the link it shows




Question 9:

# Train a model, Export it to ONNX, Runtime Session, Benchmark

from sklearn.linear_model import LinearRegression
import numpy as np

X = np.random.rand(1000, 1)
y = 3 * X + 5

model = LinearRegression()
model.fit(X, y)

from skl2onnx import to_onnx

onnx_model = to_onnx(model, X)

with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

import onnxruntime as ort

sess = ort.InferenceSession("model.onnx")

import time

t0 = time.time()
for _ in range(10000):
    model.predict(X)
sktime = time.time() - t0

t0 = time.time()
for _ in range(10000):
    sess.run(None, {"X": X})
onnxTime = time.time() - t0

print("Scikit-Learn time:", sktime)
print("ONNX time:", onnxTime)
