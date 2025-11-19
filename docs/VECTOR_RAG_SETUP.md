# 🤖 Vector RAG System Setup Guide

## 🚀 **What is Vector RAG?**

The X-FORECAST platform now includes a **Real Vector RAG (Retrieval-Augmented Generation)** system that creates personalized AI chatbots for each company login.

### ✅ **Key Features:**
- **Real Vector Embeddings**: Uses sentence-transformers for actual semantic search
- **FAISS Indexing**: Fast similarity search across company documents
- **Multi-Tenant**: Separate AI knowledge base per company
- **Dynamic Learning**: Updates when you upload new CSV files
- **Personalized Responses**: AI trained on YOUR specific company data

## 📦 **Required Dependencies**

### **CORE (Required for Vector RAG):**
```bash
pip install sentence-transformers faiss-cpu scikit-learn
```

### **FULL (All features):**
```bash
pip install -r requirements.txt
```

## 🏢 **How It Works**

### **1. Company Login**
- Each company gets isolated vector database
- Company dataset automatically loaded into embeddings
- FAISS index built for fast similarity search

### **2. Data Processing**
- CSV files converted to searchable documents
- Product-level, category-level, and time-based embeddings
- Vector similarity for intelligent retrieval

### **3. Personalized Responses**
- Query → Vector search → Relevant documents → AI response
- Company-specific context in every answer
- Confidence scoring based on similarity

## 🧪 **Test Companies & Datasets**

### **SuperX Corporation (Retail)**
- **Login**: admin@superx.com / admin123
- **Dataset**: Stationery, office supplies, pencils, pens
- **AI Focus**: Retail analytics, inventory management

### **TechCorp Industries (Manufacturing)**
- **Login**: john@techcorp.com / john123  
- **Dataset**: Industrial robots, CNC machines, automation
- **AI Focus**: Manufacturing optimization, equipment forecasting

### **HealthPlus Medical (Healthcare)**
- **Login**: sarah@healthplus.com / sarah123
- **Dataset**: Medical equipment, MRI scanners, ventilators
- **AI Focus**: Healthcare analytics, medical device insights

## 🔧 **Setup Instructions**

### **1. Install Dependencies**
```powershell
# Core AI packages
py -m pip install sentence-transformers faiss-cpu scikit-learn

# Full system
py -m pip install fastapi uvicorn pydantic pandas numpy PyJWT passlib bcrypt
```

### **2. Create Test Users**
```powershell
py create_multiple_test_users.py
```

### **3. Test Vector RAG**
```powershell
py test_real_rag.py
```

### **4. Start System**
```powershell
# Backend
py -m uvicorn src.api.main:app --reload --port 8000

# Frontend (new terminal)
cd frontend && npm start
```

## 🧪 **Testing Personalization**

### **Login with Different Companies:**
1. **SuperX**: Ask "What products do you have?"
   - Gets: Apsara Pencils, Parker Pens, Notebooks
2. **TechCorp**: Ask "What products do you have?"
   - Gets: Industrial Robots, CNC Machines, 3D Printers
3. **HealthPlus**: Ask "What products do you have?"
   - Gets: MRI Scanners, Ventilators, Medical Equipment

### **Upload New Data:**
- Upload CSV file → Vector embeddings updated automatically
- Ask same question → Gets updated response with new data

## 📊 **Vector RAG Architecture**

```
CSV Upload → Data Preprocessing → Vector Embeddings → FAISS Index
                                                           ↓
User Query → Query Embedding → Similarity Search → Document Retrieval → AI Response
```

## 🔍 **Troubleshooting**

### **"ModuleNotFoundError: sentence_transformers"**
```bash
pip install sentence-transformers
```

### **"ModuleNotFoundError: faiss"**
```bash
pip install faiss-cpu
```

### **Vector RAG not working**
```bash
# Test the system
py test_real_rag.py

# Check if datasets exist
ls data/*.csv
```

### **Responses not personalized**
- Ensure you're logged in with correct company
- Check if company dataset loaded successfully
- Verify vector embeddings created

## 🚀 **Production Deployment**

### **For GitHub Users:**
1. Clone repository
2. Install: `pip install sentence-transformers faiss-cpu scikit-learn`
3. Run: `py create_multiple_test_users.py`
4. Start: `py -m uvicorn src.api.main:app --reload --port 8000`
5. Test: Login with any test company and ask questions

### **Scalability:**
- ✅ Unlimited companies supported
- ✅ Automatic vector index management
- ✅ Persistent SQLite storage
- ✅ Memory-efficient FAISS indexing
- ✅ Real-time data updates

## 🎯 **Next Steps**

1. **Upload Your Data**: Replace sample datasets with real company data
2. **Customize Responses**: Modify response templates in `real_vector_rag.py`
3. **Add More Companies**: Register new users and upload their datasets
4. **Scale Up**: Deploy to cloud with proper authentication

**The Vector RAG system is now production-ready for multi-tenant AI chatbots!** 🚀