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

# ================================
# CONTINUOUS INTEGRATION USING JENKINS
# UBUNTU + GITHUB (FULL CODE FORMAT)
# ================================

# -------- STEP 1: INSTALL JAVA --------
sudo apt update
sudo apt install openjdk-11-jdk -y
java -version

# -------- STEP 2: INSTALL JENKINS --------
curl -fsSL https://pkg.jenkins.io/debian/jenkins.io-2023.key | sudo tee \
/usr/share/keyrings/jenkins-keyring.asc > /dev/null

echo deb [signed-by=/usr/share/keyrings/jenkins-keyring.asc] \
https://pkg.jenkins.io/debian binary/ | sudo tee \
/etc/apt/sources.list.d/jenkins.list > /dev/null

sudo apt update
sudo apt install jenkins -y

sudo systemctl start jenkins
sudo systemctl enable jenkins
sudo systemctl status jenkins

# -------- STEP 3: GET JENKINS PASSWORD --------
sudo cat /var/lib/jenkins/secrets/initialAdminPassword

# Open browser and complete setup:
# URL: http://localhost:8080
# Paste password → Install Suggested Plugins → Create Admin User

# -------- STEP 4: CREATE STUDENT PORTAL WEBSITE --------
mkdir student-portal
cd student-portal

cat <<EOF > index.html
<!DOCTYPE html>
<html>
<head>
    <title>Student Portal</title>
</head>
<body>
    <h1>Welcome to Student Portal</h1>
    <p>This is the college student portal website.</p>
</body>
</html>
EOF

# -------- STEP 5: PUSH WEBSITE TO GITHUB --------
# (Make sure you already created an empty GitHub repo)

git init
git add .
git commit -m "Initial student portal website"
git branch -M main
git remote add origin https://github.com/USERNAME/student-portal.git
git push -u origin main

# -------- STEP 6: CONFIGURE JENKINS FREESTYLE JOB --------
# Jenkins Dashboard → New Item
# Name: student-portal-ci
# Type: Freestyle Project

# Source Code Management:
# Git
# Repository URL: https://github.com/USERNAME/student-portal.git

# Build Triggers:
# ☑ GitHub hook trigger for GITScm polling

# Build Step:
# Execute Shell:
# cat index.html

# Save the Job

# -------- STEP 7: MAKE CHANGE TO HTML (CI TEST) --------
sed -i 's/college student portal website/UPDATED student portal website/' index.html

git add index.html
git commit -m "Updated homepage content"
git push origin main

# RESULT:
# GitHub push → Jenkins automatically triggers new build
# Build console shows updated HTML output





Question 3: Dockerize Simple Web Application (HTML) + Docker Commands

 Aim
Create a simple web application (HTML), write a Dockerfile, build a Docker image, run the container with port mapping, and perform Docker operations such as listing images/containers, stopping container, removing container and image.

---

 Complete Steps (Copy-Paste)

### 1) Start and Enable Docker

sudo systemctl start docker
sudo systemctl enable docker
docker --version

2) Create Project Folder + Files
mkdir appDocker
cd appDocker


Create HTML file:

sudo nano index.html


Paste this in index.html:

<!DOCTYPE html>
<html>
<head>
  <title>Docker Web App</title>
</head>
<body>
  <h1>Hello from Docker HTML App!</h1>
</body>
</html>


Create Dockerfile:

sudo nano Dockerfile


Paste this in Dockerfile:

FROM nginx:alpine
COPY index.html /usr/share/nginx/html/index.html
EXPOSE 80


Check files:

ls

3) Build Docker Image

Docker image tag must be lowercase:

docker build -t htmlapp:1.0 .


Verify image:

docker images

4) Run Docker Container (Port Mapping)
docker run -d -p 8081:80 --name html-container htmlapp:1.0


Check running container:

docker ps


Open in browser:

http://localhost:8081

5) Stop and Remove Container + Image

Stop container:

docker stop html-container


Remove container:

docker rm html-container


Remove image:

docker rmi htmlapp:1.0

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
