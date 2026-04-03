# Docker Hub Push Guide (Start from `(base) delusional@l40s:~$`)

This guide explains exactly how to push your local Docker image `dlops-ass5:latest` to Docker Hub.

## Prerequisites

- You are at terminal prompt: `(base) delusional@l40s:~$`
- Docker is installed and working with `sudo docker ...`
- You already built the image locally: `dlops-ass5:latest`
- You have a Docker Hub account

## 1) Go to your assignment folder

Run:

```bash
cd /home/delusional/Priyadip/Code
```

Why: keeps commands in the expected project path.

## 2) Check local image exists

Run:

```bash
sudo docker image ls | grep -E '^dlops-ass5\s+latest'
```

Expected: a row like `dlops-ass5   latest   <IMAGE_ID> ...`

If nothing appears: build image first:

```bash
sudo docker build -t dlops-ass5 .
```

## 3) Set your Docker Hub username once

Replace `YOUR_DOCKERHUB_USERNAME` with your real Docker Hub username.

```bash
export DH_USER="YOUR_DOCKERHUB_USERNAME"
```

Check value:

```bash
echo "$DH_USER"
```

## 4) Login to Docker Hub

Run:

```bash
sudo docker login
```

Enter Docker Hub username and password/access token when prompted.

## 5) Tag local image for Docker Hub repo

Run:

```bash
sudo docker tag dlops-ass5:latest $DH_USER/dlops-ass5:latest
```

Why: Docker Hub requires image names in `username/repository:tag` format.

## 6) Push `latest` tag

Run:

```bash
sudo docker push $DH_USER/dlops-ass5:latest
```

This may take time because your image is large (~8+ GB).

## 7) Optional: add assignment-specific tag and push

Run:

```bash
sudo docker tag dlops-ass5:latest $DH_USER/dlops-ass5:assignment-5
sudo docker push $DH_USER/dlops-ass5:assignment-5
```

Why: clearer traceability for submission.

## 8) Verify local tags

Run:

```bash
sudo docker image ls | grep "$DH_USER/dlops-ass5"
```

Expected: at least `latest` (and `assignment-5` if pushed).

## 9) Verify on Docker Hub website

Run:

```bash
echo "https://hub.docker.com/r/$DH_USER/dlops-ass5"
```

Open the printed URL in browser and check tags.

## 10) If push fails, rerun push (resume upload)

Run again:

```bash
sudo docker push $DH_USER/dlops-ass5:latest
```

Docker reuses already uploaded layers.

## Common Errors and Fixes

### Error: `denied: requested access to the resource is denied`

Fix:

1. Confirm you logged in: `sudo docker login`
2. Confirm tag uses your username: `echo $DH_USER`
3. Confirm image name is exactly `$DH_USER/dlops-ass5:latest`

### Error: `unauthorized`

Fix: login again, use Docker Hub access token instead of password.

### Error: slow or interrupted push

Fix: rerun `sudo docker push $DH_USER/dlops-ass5:latest`.

## One-shot command chain (optional)

Replace username first, then run:

```bash
export DH_USER="YOUR_DOCKERHUB_USERNAME" && \
cd /home/delusional/Priyadip/Code && \
sudo docker image ls | grep -E '^dlops-ass5\s+latest' && \
sudo docker login && \
sudo docker tag dlops-ass5:latest $DH_USER/dlops-ass5:latest && \
sudo docker push $DH_USER/dlops-ass5:latest
```
