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






QUESTION 3: Docker Web App (FULL Dockerfile + commands)

Aim
Build and run a Docker container for a simple web app and use basic Docker commands.

Step 1: Install Docker
sudo apt update
sudo apt install docker.io -y
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
newgrp docker

Step 2: Create index.html
mkdir docker-webapp
cd docker-webapp
nano index.html
index.html content (FULL)
<h1>Hello from Docker Web App!</h1>

Step 3: Create Dockerfile (FULL)
nano Dockerfile
FROM nginx:alpine
COPY index.html /usr/share/nginx/html/index.html
EXPOSE 80

Step 4: Build + Run
docker build -t mywebapp:1.0 .
docker run -d -p 8081:80 --name web1 mywebapp:1.0
Open: http://localhost:8081
Docker commands
docker ps
docker images
docker stop web1
docker rm web1
docker rmi mywebapp:1.0






QUESTION 4: Kubernetes (Deployment + Service YAML FULL)

Aim
Deploy Docker image in Kubernetes and expose it using NodePort, then scale replicas.

Step 1: Start minikube
minikube start

Step 2: Use minikube docker and build image
eval $(minikube docker-env)
docker build -t mywebapp:1.0 .

Step 3: deployment.yaml (FULL)
nano deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
 name: webapp-deployment
spec:
 replicas: 1
 selector:
 matchLabels:
 app: webapp
 template:
 metadata:
 labels:
 app: webapp
 spec:
 containers:
 - name: webapp
 image: mywebapp:1.0
 imagePullPolicy: Never
 ports:
 - containerPort: 80
   
Step 4: Apply + verify
kubectl apply -f deployment.yaml
kubectl get pods

Step 5: service.yaml (FULL)
nano service.yaml
apiVersion: v1
kind: Service
metadata:
 name: webapp-service
spec:
 type: NodePort
 selector:
 app: webapp
 ports:
 - port: 80
 targetPort: 80
 nodePort: 30007

Step 6: Apply service + access
kubectl apply -f service.yaml
kubectl get svc
minikube ip
Open: http://<minikube-ip>:30007

Step 7: Scale to 3
kubectl scale deployment webapp-deployment --replicas=3
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

Aim
Run multi-container app using Docker Compose (Flask app + Redis).
Step 1: Create folder
mkdir compose-app
cd compose-app

Step 2: app.py (FULL)
nano app.py
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
 
Step 3: requirements.txt (FULL)
nano requirements.txt
flask
redis

Step 4: Dockerfile (FULL)
nano Dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py .
EXPOSE 5000
CMD ["python", "app.py"]

Step 5: docker-compose.yml (FULL)
nano docker-compose.yml
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


Step 6: Run + verify
docker compose up -d --build
docker compose ps
Open: http://localhost:5000



QUESTION 7: DVC (FULL raw/processed pipeline)

Aim
Track datasets using DVC and reproduce pipeline using dvc repro.

Step 1: Install DVC
pip install dvc

Step 2: Init project
mkdir dvc-project
cd dvc-project
git init
dvc init

Step 3: Create raw dataset
mkdir -p data
nano data/raw.csv
name,marks
A,80
B,90

Step 4: Track raw with DVC
dvc add data/raw.csv
git add data/raw.csv.dvc .gitignore
git commit -m "Track raw dataset"

Step 5: Cleaning script (FULL)
mkdir -p src
nano src/clean.py
import pandas as pd
df = pd.read_csv("data/raw.csv")
df["marks"] = df["marks"] + 5
df.to_csv("data/processed.csv", index=False)
print("Processed dataset saved.")

Step 6: Create DVC stage
dvc stage add -n clean_data -d src/clean.py -d data/raw.csv -o data/processed.csv python src/cleanStep 7: Run repro
dvc repro




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
