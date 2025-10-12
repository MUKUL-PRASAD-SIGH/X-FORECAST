# 🚀 X-FORECAST Deployment Guide

## 🌐 **Quick Deploy Options**

### **1. Railway (RECOMMENDED - Full Stack)**
```bash
# 1. Push to GitHub
git add .
git commit -m "Deploy X-FORECAST"
git push origin main

# 2. Deploy on Railway
# - Go to railway.app
# - Connect GitHub repo
# - Deploy automatically
# - Backend runs on Railway domain
```

### **2. Vercel (Frontend + Serverless Backend)**
```bash
# 1. Install Vercel CLI
npm i -g vercel

# 2. Deploy
vercel --prod

# 3. Frontend + API routes deployed
```

### **3. Netlify (Frontend Only)**
```bash
# 1. Build frontend
cd frontend
npm run build

# 2. Deploy to Netlify
# - Drag & drop build folder to netlify.com
# - Or connect GitHub repo
```

### **4. Render (Full Stack)**
```bash
# 1. Push to GitHub
git push origin main

# 2. Deploy on Render
# - Connect GitHub repo
# - Auto-deploy backend + frontend
```

## 📋 **Pre-Deployment Checklist**

### **Required Files:**
- ✅ `requirements.txt` - Python dependencies
- ✅ `package.json` - Frontend dependencies  
- ✅ `vercel.json` - Vercel config
- ✅ `netlify.toml` - Netlify config
- ✅ `railway.json` - Railway config

### **Environment Setup:**
```bash
# Install dependencies
py -m pip install -r requirements.txt
cd frontend && npm install
```

## 🎯 **Deployment Steps**

### **Option A: Railway (Easiest)**
1. **Create GitHub repo**
2. **Push code**: `git push origin main`
3. **Go to**: railway.app
4. **Connect repo** → Auto-deploy
5. **Access**: `your-app.railway.app`

### **Option B: Vercel**
1. **Install CLI**: `npm i -g vercel`
2. **Deploy**: `vercel --prod`
3. **Access**: `your-app.vercel.app`

### **Option C: Netlify**
1. **Build**: `cd frontend && npm run build`
2. **Deploy**: Drag build folder to netlify.com
3. **Access**: `your-app.netlify.app`

## 🔧 **Configuration**

### **Backend URL Update:**
Update frontend API calls to production URL:
```typescript
// In frontend/src/components/MainDashboard.tsx
const API_URL = process.env.NODE_ENV === 'production' 
  ? 'https://your-backend.railway.app' 
  : 'http://localhost:8000';
```

### **CORS Setup:**
Backend already configured for production:
```python
# In src/api/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 🚀 **One-Click Deploy**

### **Railway Deploy Button:**
[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template?template=https://github.com/yourusername/X_FORECAST)

### **Vercel Deploy Button:**
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/yourusername/X_FORECAST)

### **Netlify Deploy Button:**
[![Deploy to Netlify](https://www.netlify.com/img/deploy/button.svg)](https://app.netlify.com/start/deploy?repository=https://github.com/yourusername/X_FORECAST)

## 📊 **Production Features**

### **What Works in Production:**
- ✅ SuperX Corporation dashboard (no login required)
- ✅ AI chatbot with vector RAG
- ✅ CSV data upload and processing
- ✅ Real-time metrics and analytics
- ✅ Cyberpunk UI with animations
- ✅ Multi-tenant architecture ready

### **Performance:**
- **Backend**: FastAPI + Uvicorn (production ready)
- **Frontend**: React + TypeScript (optimized build)
- **AI**: Sentence transformers + FAISS (efficient)
- **Database**: SQLite (embedded, no external DB needed)

## 🔍 **Troubleshooting**

### **Common Issues:**
1. **Dependencies**: Install all requirements.txt packages
2. **CORS**: Update allowed origins for production
3. **File paths**: Use relative paths for deployment
4. **Environment**: Set NODE_ENV=production

### **Quick Fixes:**
```bash
# Backend not starting
py -m pip install fastapi uvicorn sentence-transformers faiss-cpu

# Frontend build fails
cd frontend && npm install && npm run build

# AI not working
py -m pip install sentence-transformers faiss-cpu scikit-learn
```

## 🎉 **Success!**

After deployment, your X-FORECAST system will be live with:
- **Public URL** for global access
- **SuperX Corporation** dashboard ready
- **AI chatbot** trained on retail data
- **File upload** for custom datasets
- **Production-grade** performance

**Ready to deploy? Choose your platform and follow the steps above!** 🚀