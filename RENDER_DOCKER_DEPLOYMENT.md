# Render Docker Deployment Guide - Step by Step

This guide will walk you through deploying your Brain Tumor Classification app on Render using Docker.

## Prerequisites Checklist

Before starting, ensure you have:
- [ ] GitHub account
- [ ] Code pushed to a GitHub repository
- [ ] Render account (sign up at https://render.com)
- [ ] Dockerfile in your project root
- [ ] Model file (`models/brain_tumor_model.h5`) in repository or accessible

---

## Step 1: Prepare Your Code for Deployment

### 1.1 Verify Dockerfile Exists
- Check that `Dockerfile` is in your project root directory
- Ensure it's committed to Git

### 1.2 Update Dockerfile for Render (if needed)

Your Dockerfile should use the PORT environment variable. The CMD line should be:

```dockerfile
CMD streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true
```

Or if using JSON format:
```dockerfile
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]
```

### 1.3 Create .dockerignore (Optional but Recommended)

Create a `.dockerignore` file to exclude unnecessary files:

```
.git
.gitignore
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
venv/
env/
.venv
.vscode/
.idea/
*.ipynb
.ipynb_checkpoints/
*.log
.DS_Store
```

### 1.4 Commit and Push to GitHub

```bash
# Add all files
git add .

# Commit changes
git commit -m "Prepare for Docker deployment on Render"

# Push to GitHub
git push origin main
```

---

## Step 2: Create Render Account and Connect GitHub

### 2.1 Sign Up for Render
1. Go to **https://render.com**
2. Click **"Get Started for Free"** or **"Sign Up"**
3. Sign up using:
   - Email and password, OR
   - GitHub account (recommended - easier integration)

### 2.2 Connect GitHub Account
1. After signing up, click **"New +"** in the top right
2. Select **"Connect GitHub"** or **"Connect Repository"**
3. Authorize Render to access your GitHub repositories
4. Select the repositories you want to deploy (or select "All repositories")

---

## Step 3: Create a New Web Service on Render

### 3.1 Start Creating Web Service
1. In Render dashboard, click **"New +"** button (top right)
2. Select **"Web Service"** from the dropdown menu

### 3.2 Connect Your Repository
1. You'll see a list of your GitHub repositories
2. Find and click on **"Summative-MLOP-Classification-Pipeline"** (or your repo name)
3. Click **"Connect"** button

---

## Step 4: Configure Docker Deployment

### 4.1 Basic Settings
Fill in the following:

- **Name**: `brain-tumor-classifier` (or your preferred name)
- **Region**: Choose closest to your users (e.g., `Oregon (US West)`)
- **Branch**: `main` (or your default branch name)
- **Root Directory**: Leave **empty** (or `.` if needed)

### 4.2 Build & Deploy Settings

**IMPORTANT: Select Docker for deployment**

1. Look for **"Environment"** or **"Build Method"** section
2. Select **"Docker"** (not "Nixpacks" or "Pip")
3. This tells Render to use your Dockerfile

### 4.3 Docker Configuration

Render will automatically detect your Dockerfile, but verify:

- **Dockerfile Path**: Should be `./Dockerfile` (or just `Dockerfile`)
- **Docker Context**: Should be `.` (current directory)

### 4.4 Start Command (Optional)

If Render asks for a start command, you can leave it empty since Dockerfile has CMD.
Or use:
```
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

---

## Step 5: Configure Environment Variables

### 5.1 Add Environment Variables

Click **"Advanced"** or scroll to **"Environment Variables"** section.

Add these variables:

| Key | Value | Required |
|-----|-------|----------|
| `PORT` | `8501` | No (Render sets this automatically) |
| `PYTHONUNBUFFERED` | `1` | Recommended (for better logging) |

**Note**: Render automatically sets `$PORT` environment variable, but you can set a default.

### 5.2 Optional Environment Variables

You can also add:
- `PYTHON_VERSION`: `3.9` (if you want to specify Python version)

---

## Step 6: Choose Instance Type

### 6.1 Select Plan

Choose an instance type based on your needs:

- **Free**: 
  - 512 MB RAM
  - Limited CPU
  - **Warning**: May not be enough for TensorFlow
  - App sleeps after 15 minutes of inactivity
  
- **Starter** ($7/month): 
  - 512 MB RAM
  - Better performance
  - **Recommended minimum** for ML apps
  
- **Standard** ($25/month):
  - 2 GB RAM
  - Best for production
  - **Recommended** for TensorFlow applications

### 6.2 Auto-Deploy Settings

- **Auto-Deploy**: Set to **"Yes"** (deploys automatically on git push)
- **Pull Request Previews**: Optional (set to "Yes" if you want PR previews)

---

## Step 7: Deploy Your Application

### 7.1 Create the Service
1. Review all settings
2. Click **"Create Web Service"** button at the bottom
3. Render will start building your Docker image

### 7.2 Monitor the Build Process

You'll see the build logs in real-time:

1. **Building Docker Image**: 
   - Render pulls your code
   - Builds Docker image using your Dockerfile
   - Installs dependencies
   - This may take 5-15 minutes

2. **Deploying**:
   - Starts the container
   - Runs your application
   - Health checks

### 7.3 Watch for Errors

Common issues to watch for:
- ❌ **Build fails**: Check Dockerfile syntax
- ❌ **Dependencies fail**: Check requirements.txt
- ❌ **Port binding error**: Ensure Dockerfile uses $PORT
- ❌ **Model not found**: Verify model file is in repository

---

## Step 8: Verify Deployment

### 8.1 Get Your App URL

Once deployment completes, you'll see:
- **Status**: "Live" (green)
- **URL**: `https://your-app-name.onrender.com`

### 8.2 Test Your Application

1. Click on the URL or copy it
2. Test the application:
   - ✅ Home page loads
   - ✅ Prediction feature works
   - ✅ Upload feature works
   - ✅ All pages accessible

### 8.3 Check Logs

1. In Render dashboard, click **"Logs"** tab
2. Check for any errors or warnings
3. Verify application started successfully

---

## Step 9: Post-Deployment Configuration (Optional)

### 9.1 Custom Domain (Optional)

1. Go to **Settings** → **Custom Domains**
2. Add your domain name
3. Follow DNS configuration instructions

### 9.2 Environment Variables Updates

You can add/update environment variables anytime:
1. Go to **Environment** tab
2. Add or modify variables
3. Click **"Save Changes"**
4. App will automatically redeploy

### 9.3 Monitoring

- Check **Metrics** tab for:
  - CPU usage
  - Memory usage
  - Request count
  - Response times

---

## Troubleshooting Common Issues

### Issue 1: Build Fails

**Symptoms**: Build process stops with error

**Solutions**:
1. Check build logs for specific error
2. Verify Dockerfile syntax is correct
3. Ensure all files are in repository
4. Check requirements.txt has all dependencies
5. Verify Python version compatibility

### Issue 2: App Crashes on Startup

**Symptoms**: App builds but crashes when starting

**Solutions**:
1. Check application logs
2. Verify model file exists: `models/brain_tumor_model.h5`
3. Check file paths are correct
4. Verify PORT environment variable is used correctly
5. Check memory usage (may need to upgrade plan)

### Issue 3: Model Not Loading

**Symptoms**: App starts but model fails to load

**Solutions**:
1. Verify model file is in repository
2. Check model file path in code
3. Ensure model file is not too large (use Git LFS if >100MB)
4. Check file permissions
5. Review error logs for specific error message

### Issue 4: Port Binding Error

**Symptoms**: Error about port already in use

**Solutions**:
1. Ensure Dockerfile uses `${PORT}` or `$PORT` environment variable
2. Don't hardcode port number
3. Use: `--server.port=${PORT:-8501}` in CMD

### Issue 5: Out of Memory

**Symptoms**: App crashes or becomes unresponsive

**Solutions**:
1. Upgrade to Starter or Standard plan
2. Optimize model loading (lazy loading)
3. Reduce batch sizes
4. Use model quantization

---

## Quick Reference Commands

### Local Docker Testing (Before Deploying)

Test your Docker setup locally:

```bash
# Build Docker image
docker build -t brain-tumor-app .

# Run Docker container
docker run -p 8501:8501 -e PORT=8501 brain-tumor-app

# Test in browser
# Open http://localhost:8501
```

### Git Commands

```bash
# Check status
git status

# Add files
git add .

# Commit
git commit -m "Your commit message"

# Push to GitHub
git push origin main
```

---

## Deployment Checklist

Use this checklist before deploying:

- [ ] Dockerfile exists and is correct
- [ ] Dockerfile uses PORT environment variable
- [ ] All code is committed to Git
- [ ] Code is pushed to GitHub
- [ ] Model file exists (or download mechanism in place)
- [ ] requirements.txt is complete
- [ ] .dockerignore is configured (optional)
- [ ] Render account created
- [ ] GitHub connected to Render
- [ ] Web service created with Docker option
- [ ] Environment variables configured
- [ ] Instance type selected
- [ ] Deployment successful
- [ ] Application tested and working

---

## Important Notes

### Free Tier Limitations

- **512 MB RAM**: May be insufficient for TensorFlow
- **Sleep after inactivity**: App sleeps after 15 minutes
- **Build time limit**: 90 minutes
- **Database**: SQLite is ephemeral (resets on restart)

### Recommendations

1. **For Development**: Free tier is okay for testing
2. **For Production**: Use Starter ($7/month) or Standard ($25/month)
3. **For Large Models**: Use Git LFS or download from cloud storage
4. **For Database**: Use Render PostgreSQL (free tier available)

---

## Support and Resources

- **Render Documentation**: https://render.com/docs
- **Docker Documentation**: https://docs.docker.com
- **Streamlit Deployment**: https://docs.streamlit.io/streamlit-community-cloud
- **Render Status**: https://status.render.com

---

## Summary

**Quick Deployment Steps:**
1. Push code to GitHub
2. Create Render account → Connect GitHub
3. New Web Service → Select repository
4. Choose **Docker** as build method
5. Configure environment variables
6. Select instance type
7. Deploy and test!

Your app will be live at: `https://your-app-name.onrender.com`

