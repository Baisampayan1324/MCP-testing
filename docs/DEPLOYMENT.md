# 🚀 Deployment Guide for Research Paper Summarizer

## 📋 Pre-Deployment Checklist

### Performance Optimizations Applied:
✅ **Lightweight Models**: Using `sshleifer/distilbart-cnn-6-6` (80% faster)  
✅ **CPU Optimization**: Optimized for CPU-only inference  
✅ **Memory Management**: Reduced chunk size and batch processing  
✅ **File Size Limits**: 5MB max per file for deployment  
✅ **Caching**: Streamlit resource caching for models  

## 🌐 Deployment Options

### **Option 1: Render.com (Recommended)**

#### Steps:
1. **Create GitHub Repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/research-paper-summarizer.git
   git push -u origin main
   ```

2. **Deploy to Render**
   - Go to [render.com](https://render.com)
   - Connect your GitHub account
   - Select "New Web Service"
   - Choose your repository
   - Use these settings:
     - **Build Command**: `pip install -r deployment/requirements-minimal.txt`
     - **Start Command**: `cd apps && streamlit run app_deploy.py --server.port $PORT --server.address 0.0.0.0`
     - **Python Version**: `3.11.10`
     - **Environment**: `ENVIRONMENT=deployment`

#### Render Benefits:
- ✅ 512MB RAM on free tier
- ✅ Better for ML applications
- ✅ Automatic HTTPS
- ✅ Custom domains
- ✅ GitHub integration

---

### **Option 2: Heroku**

#### Steps:
1. **Install Heroku CLI**
   ```bash
   # Download from https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Create Heroku App**
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

3. **Set Environment Variables**
   ```bash
   heroku config:set ENVIRONMENT=deployment
   heroku config:set STREAMLIT_SERVER_HEADLESS=true
   ```

#### Heroku Limitations:
- ⚠️ 512MB RAM limit (may cause issues)
- ⚠️ 30-second request timeout
- ⚠️ Slug size limit (500MB)

---

## ⚡ Performance Expectations

### **Current System (Local)**:
- 📄 **6-page PDF**: ~259 seconds (4.3 minutes)
- 🧠 **Model Loading**: ~30 seconds
- 💾 **Memory Usage**: ~2-4GB RAM

### **Optimized Deployment**:
- 📄 **6-page PDF**: ~60-90 seconds
- 🧠 **Model Loading**: ~15 seconds (cached)
- 💾 **Memory Usage**: ~400-500MB RAM
- 🚀 **Speed Improvement**: ~3x faster

---

## 🔧 Additional Optimizations

### **For Even Better Performance**:

1. **Use GPU-enabled hosting** (Paid plans):
   - Google Cloud Run
   - AWS Lambda (with container)
   - Azure Container Instances

2. **Model Optimizations**:
   ```python
   # Switch to even lighter models
   "embedding_model": "sentence-transformers/paraphrase-MiniLM-L3-v2"  # 61MB
   "summarization_model": "sshleifer/distilbart-xsum-6-6"              # 244MB
   ```

3. **Preprocessing Pipeline**:
   - Pre-process documents offline
   - Store embeddings in cloud database
   - Use vector databases (Pinecone, Weaviate)

---

## 🧪 Test Deployment Locally

```bash
# Test the deployment version locally
ENVIRONMENT=deployment streamlit run app_deploy.py

# Monitor performance
pip install psutil
python -c "import psutil; print(f'RAM: {psutil.virtual_memory().percent}%')"
```

---

## 📊 Monitoring & Scaling

### **Free Tier Limitations**:
- **Render**: 750 hours/month, sleeps after 15min inactivity
- **Heroku**: 550 hours/month, sleeps after 30min inactivity

### **Upgrade Recommendations**:
- For production use: **Render Pro ($7/month)** or **Heroku Hobby ($7/month)**
- For high traffic: Consider **AWS/GCP** with auto-scaling

---

## 🚀 Quick Deploy Commands

### **Render Deployment**:
```bash
# Add files to git
git add requirements-deploy.txt app_deploy.py render.yaml
git commit -m "Add deployment files"
git push origin main

# Deploy automatically via Render dashboard
```

### **Heroku Deployment**:
```bash
# Create and deploy
heroku create your-app-name
git add Procfile requirements-deploy.txt runtime.txt app_deploy.py
git commit -m "Add Heroku deployment files"
git push heroku main

# Check logs
heroku logs --tail
```

---

## 🎯 Success Metrics

After deployment, your app should:
- ✅ Load in under 30 seconds
- ✅ Process 5MB PDFs in 60-90 seconds
- ✅ Handle 3-5 concurrent users
- ✅ Stay within 512MB RAM limit
- ✅ Provide reliable Q&A functionality

**Ready to deploy? Choose your platform and follow the steps above!** 🚀
