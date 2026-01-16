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

Step 1: Install Jenkins (simple)
sudo apt update
sudo apt install jenkins -y
sudo systemctl enable --now jenkins

Step 2: Open Jenkins
Open browser:
http://localhost:8080
Get admin password:
sudo cat /var/lib/jenkins/secrets/initialAdminPassword

Step 3: Create Freestyle Job
Dashboard → New Item → Freestyle project Name: student-portal-ci

Step 4: Configure Git
Job → Configure:
Source Code Management → Git
Repository URL: https://github.com/<username>/<repo>.git
Branch: */main

Step 5: Auto Trigger
Build Triggers:
✓ GitHub hook trigger for GITScm polling

Step 6: GitHub Webhook
GitHub Repo → Settings → Webhooks → Add webhook
Payload URL: http://<your-ip>:8080/github-webhook/
Content type: application/json
Events: Just the push event
Save webhook.

Step 7: Test
Make a small change in HTML:
Example: change heading text.
Then push:
git add .
git commit -m "Updated HTML"
git push origin main
Result: Jenkins starts build automatically






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



then run the python code and after that run mlflow ui and click the link it shows




Question 9:

import numpy as np
import time
from sklearn.linear_model import LinearRegression
from skl2onnx import to_onnx
import onnxruntime as ort

# 1. Train scikit-learn model
X = np.random.rand(1000, 1)
y = 3 * X.squeeze() + 5

model = LinearRegression()
model.fit(X, y)

# 2. Export model to ONNX
onnx_model = to_onnx(model, X)
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# 3. Create ONNX Runtime session
sess = ort.InferenceSession("model.onnx")

# 4. Benchmark scikit-learn
t0 = time.time()
for _ in range(10000):
    model.predict(X)
sk_time = time.time() - t0

# 5. Benchmark ONNX Runtime
t0 = time.time()
for _ in range(10000):
    sess.run(None, {"X": X})
onnx_time = time.time() - t0

# 6. Print results
print("scikit-learn time:", sk_time)
print("ONNX Runtime time:", onnx_time)
