# 🔐 GitHub Secrets Setup Guide

## Current Status
- ✅ Workflow fixed (upload-artifact v3 → v4)
- ⏳ Secrets need to be added manually

## 📋 What You Need to Do

### Step 1: Add GitHub Secrets Manually

Go to your repository secrets page:
**https://github.com/febinrenu/ml-loan/settings/secrets/actions**

### Step 2: Click "New repository secret" and add these 4 secrets:

#### Required Secrets:

1. **DOCKER_USERNAME**
   - Click "New repository secret"
   - Name: `DOCKER_USERNAME`
   - Value: Your Docker Hub username (example: `febinrenu`)
   - Click "Add secret"

2. **DOCKER_PASSWORD**
   - Click "New repository secret"
   - Name: `DOCKER_PASSWORD`
   - Value: Your Docker Hub Personal Access Token
   - Click "Add secret"
   
   **How to get Docker Hub token:**
   - Go to https://hub.docker.com/settings/security
   - Click "New Access Token"
   - Name it "GitHub Actions"
   - Copy the token and paste it here

3. **RENDER_API_KEY** (Optional - for deployment)
   - Click "New repository secret"
   - Name: `RENDER_API_KEY`
   - Value: Your Render API key
   - Click "Add secret"
   
   **How to get Render API key:**
   - Go to https://dashboard.render.com/u/settings#api-keys
   - Click "Create API Key"
   - Copy and paste it

4. **RENDER_SERVICE_ID** (Optional - for deployment)
   - Click "New repository secret"
   - Name: `RENDER_SERVICE_ID`
   - Value: Your Render service ID (format: `srv-xxxxxxxxxxxxx`)
   - Click "Add secret"
   
   **How to get Render Service ID:**
   - Create a web service on Render first
   - The service ID is in the URL: `https://dashboard.render.com/web/srv-xxxxxxxxxxxxx`

---

## 🎯 Quick Test Options

### Option A: Run Test Job Only (No Secrets Needed)

The test job will run automatically and should **PASS** without any secrets:
- ✅ Runs preprocessing
- ✅ Trains models
- ✅ Runs unit tests
- ✅ Uploads artifacts

**Build and Deploy jobs will be skipped** if secrets are missing (this is normal).

### Option B: Enable Full CI/CD (Requires All Secrets)

Once you add all 4 secrets:
1. Go to **Actions** tab
2. Click on the latest workflow run
3. Click **Re-run all jobs**
4. All jobs should pass! 🎉

---

## 📊 What Each Job Does

| Job | Needs Secrets? | What It Does |
|-----|----------------|--------------|
| **Test** | ❌ No | Runs your ML pipeline and tests |
| **Build** | ✅ Yes (Docker) | Builds Docker image and pushes to Docker Hub |
| **Deploy** | ✅ Yes (Render) | Deploys your app to Render cloud |

---

## ✅ Success Checklist

- [x] Workflow updated to v4
- [x] Workflow pushed to GitHub
- [ ] Add DOCKER_USERNAME secret
- [ ] Add DOCKER_PASSWORD secret
- [ ] Add RENDER_API_KEY secret (optional)
- [ ] Add RENDER_SERVICE_ID secret (optional)
- [ ] Go to Actions tab and verify workflow runs

---

## 🆘 If You Need Help

**Test job failing?**
- Check if `loan.csv` is in your repository
- Check if `requirements.txt` has all dependencies

**Build job skipped?**
- This is normal if Docker secrets are not added
- Add DOCKER_USERNAME and DOCKER_PASSWORD to enable it

**Deploy job skipped?**
- This is normal if Render secrets are not added
- Add RENDER_API_KEY and RENDER_SERVICE_ID to enable it

---

## 🚀 Next Steps

1. **Right now**: Go add secrets at https://github.com/febinrenu/ml-loan/settings/secrets/actions
2. **Then**: Check the Actions tab to see your workflow run
3. **Test job should pass** even without secrets!
4. **Build/Deploy** will work once you add the respective secrets

---

**Current Workflow Status:**
- Upload artifact v3 → v4: ✅ Fixed
- Pushed to GitHub: ✅ Done
- Waiting for: Your manual secret addition
