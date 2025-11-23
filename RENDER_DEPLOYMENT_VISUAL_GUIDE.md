# Render Docker Deployment - Visual Step-by-Step Guide

## 📋 Complete Step-by-Step Process

### STEP 1: Prepare Your Code ✅

**1.1 Verify Dockerfile**
```
✅ Check: Dockerfile exists in project root
✅ Check: Dockerfile uses ${PORT} variable (already updated)
```

**1.2 Commit and Push to GitHub**
```bash
# Open terminal in your project directory
git add .
git commit -m "Ready for Render Docker deployment"
git push origin main
```

**What happens**: Your code is now on GitHub and ready for deployment.

---

### STEP 2: Create Render Account 🔐

**2.1 Go to Render**
- Open browser: https://render.com
- Click **"Get Started for Free"** or **"Sign Up"**

**2.2 Sign Up Options**
- Option A: Sign up with GitHub (Recommended - easier)
- Option B: Sign up with email

**2.3 Connect GitHub** (if not using GitHub signup)
- After signup, click **"New +"** → **"Connect GitHub"**
- Authorize Render to access your repositories
- Select repositories (or "All repositories")

**What happens**: Render can now access your GitHub repositories.

---

### STEP 3: Create Web Service 🚀

**3.1 Start Creation**
- In Render dashboard, click **"New +"** (top right corner)
- Select **"Web Service"** from dropdown

**Visual Guide:**
```
Render Dashboard
┌─────────────────────────┐
│  [New +]  [Dashboard]   │
│    ↓                     │
│  Web Service  ← Click    │
│  Background Worker       │
│  Static Site             │
│  PostgreSQL              │
└─────────────────────────┘
```

**3.2 Select Repository**
- You'll see list of your GitHub repositories
- Find: **"Summative-MLOP-Classification-Pipeline"**
- Click **"Connect"** button

**What happens**: Render connects to your repository.

---

### STEP 4: Configure Docker Deployment 🐳

**4.1 Basic Settings**

Fill in these fields:

```
Name: brain-tumor-classifier
     ↓
Region: Oregon (US West) [or closest to you]
     ↓
Branch: main
     ↓
Root Directory: [Leave empty]
```

**4.2 IMPORTANT: Select Docker**

Look for **"Environment"** or **"Build Method"** section:

```
Build Method:
○ Nixpacks (Auto-detected)
○ Docker  ← SELECT THIS ONE
○ Pip
```

**Why Docker?** Because you have a Dockerfile and want to use it.

**4.3 Docker Settings**

Render will auto-detect:
- **Dockerfile Path**: `./Dockerfile` ✅
- **Docker Context**: `.` ✅

You can leave these as default.

**What happens**: Render knows to use your Dockerfile for building.

---

### STEP 5: Configure Environment Variables 🔧

**5.1 Find Environment Section**

Scroll down to **"Environment Variables"** section.

**5.2 Add Variables**

Click **"Add Environment Variable"** and add:

```
Key: PYTHONUNBUFFERED
Value: 1
```

**Note**: `PORT` is automatically set by Render, no need to add it.

**Visual:**
```
Environment Variables
┌─────────────────────────────┐
│ Key              Value       │
├─────────────────────────────┤
│ PORT             (auto)      │
│ PYTHONUNBUFFERED 1           │
└─────────────────────────────┘
```

**What happens**: Your app will have proper logging.

---

### STEP 6: Choose Instance Type 💰

**6.1 Select Plan**

```
Instance Type:
○ Free (512 MB RAM) - May not work with TensorFlow
○ Starter ($7/month) - Recommended minimum ← SELECT
○ Standard ($25/month) - Best for production
```

**Recommendation**: Choose **Starter** ($7/month) because:
- Free tier has only 512 MB RAM (TensorFlow needs more)
- Free tier apps sleep after 15 minutes
- Starter gives better performance

**6.2 Auto-Deploy**

```
Auto-Deploy: Yes ← Keep this checked
```

**What happens**: App will redeploy automatically when you push to GitHub.

---

### STEP 7: Deploy! 🎯

**7.1 Review Settings**

Before clicking, verify:
- ✅ Repository selected
- ✅ Docker selected as build method
- ✅ Environment variables added
- ✅ Instance type selected

**7.2 Start Deployment**

Click **"Create Web Service"** button (bottom of page)

**7.3 Watch Build Process**

You'll see build logs:

```
Step 1/8 : FROM python:3.9-slim
Step 2/8 : WORKDIR /app
Step 3/8 : RUN apt-get update...
Step 4/8 : COPY requirements.txt .
Step 5/8 : RUN pip install...
Step 6/8 : COPY . .
Step 7/8 : RUN mkdir -p models...
Step 8/8 : CMD streamlit run...
```

**Build Time**: 5-15 minutes (first time may take longer)

**What happens**: 
1. Render pulls your code
2. Builds Docker image using your Dockerfile
3. Installs all dependencies
4. Starts your application

---

### STEP 8: Verify Deployment ✅

**8.1 Check Status**

Once build completes, you'll see:
```
Status: Live ✅
URL: https://brain-tumor-classifier.onrender.com
```

**8.2 Test Your App**

1. Click the URL or copy it
2. Open in browser
3. Test features:
   - ✅ Home page loads
   - ✅ Prediction works
   - ✅ Upload works
   - ✅ All pages accessible

**8.3 Check Logs**

Click **"Logs"** tab to see:
- Application startup messages
- Any errors or warnings
- Request logs

**What happens**: Your app is live and accessible worldwide!

---

## 🎯 Quick Reference

### Before Deployment Checklist

- [ ] Code pushed to GitHub
- [ ] Dockerfile exists and uses `${PORT}`
- [ ] Model file exists (`models/brain_tumor_model.h5`)
- [ ] Render account created
- [ ] GitHub connected to Render

### During Deployment

- [ ] Selected **Docker** as build method
- [ ] Added environment variables
- [ ] Selected appropriate instance type
- [ ] Build completes successfully

### After Deployment

- [ ] App URL is accessible
- [ ] All features work
- [ ] No errors in logs
- [ ] Performance is acceptable

---

## 🆘 Troubleshooting

### Build Fails?
1. Check build logs for error
2. Verify Dockerfile syntax
3. Ensure all files are in repository

### App Crashes?
1. Check application logs
2. Verify model file exists
3. Check memory usage (may need upgrade)

### Port Error?
1. Ensure Dockerfile uses `${PORT}`
2. Don't hardcode port number

---

## 📞 Need Help?

- **Render Docs**: https://render.com/docs
- **Docker Docs**: https://docs.docker.com
- **Check Logs**: Always check logs first!

---

## ✅ Success!

Once deployed, your app will be:
- 🌐 Accessible at: `https://your-app-name.onrender.com`
- 🔄 Auto-deploys on git push
- 📊 Monitored by Render
- 🔒 HTTPS enabled automatically

**Congratulations! Your app is live! 🎉**

