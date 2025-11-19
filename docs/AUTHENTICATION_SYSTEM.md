# X-FORECAST Authentication System

## 🚀 BABY STEPS - Get This Working in 5 Minutes!

### For Gmail Users (vol670668@gmail.com)

#### Step 1: Install Python Packages (30 seconds)
```bash
pip install bcrypt python-dotenv PyJWT
```

#### Step 2: Set Up Gmail App Password (2 minutes)
1. **Go to Gmail Security**: https://myaccount.google.com/security
2. **Enable 2-Step Verification** (if not already enabled):
   - Click "2-Step Verification" 
   - Follow prompts to add your phone number
   - Complete verification

3. **Create App Password**:
   - Go back to Security page
   - Click "2-Step Verification" again
   - Scroll down and click "App passwords"
   - Select App: "Mail"
   - Select Device: "Other" → Type "X-FORECAST"
   - Click "Generate"
   - **COPY THE 16-CHARACTER PASSWORD** (like: `abcd efgh ijkl mnop`)

#### Step 3: Configure Email (1 minute)
```bash
# Copy the example file
cp .env.example .env
```

Edit `.env` file:
```bash
EMAIL_USER=vol670668@gmail.com
EMAIL_PASSWORD=abcdefghijklmnop  # Your 16-char app password (no spaces!)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
FROM_EMAIL=vol670668@gmail.com
```

#### Step 4: Test It Works (1 minute)
```bash
# Start backend
python -m uvicorn src.api.main:app --reload --port 8000

# In another terminal, start frontend
cd frontend && npm start
```

Go to http://localhost:3000 and try registering!

### 🆘 Quick Fixes for Common Issues

**"Authentication failed" error?**
- ✅ Use the 16-character app password, NOT your regular Gmail password
- ✅ Remove any spaces from the app password in .env file
- ✅ Double-check EMAIL_USER=vol670668@gmail.com is correct

**"App passwords" option missing?**
- ✅ Enable 2-Factor Authentication first at https://myaccount.google.com/security
- ✅ Wait 5 minutes after enabling 2FA
- ✅ Refresh the page and look under "Signing in to Google"

**Email not sending?**
- ✅ Check spam folder for verification emails
- ✅ Test with: `curl -X POST http://localhost:8000/api/v1/auth/register`
- ✅ Look at backend console for error messages

**Backend won't start?**
- ✅ Run: `pip install bcrypt python-dotenv PyJWT`
- ✅ Make sure you're in the project root directory
- ✅ Check Python version: `python --version` (need 3.8+)

### 🧪 Quick Test Script
Create `test_gmail.py` to test your Gmail setup:
```python
import smtplib
from email.mime.text import MimeText

# Your Gmail credentials
EMAIL_USER = "vol670668@gmail.com"
EMAIL_PASSWORD = "your-16-char-app-password"  # Replace with actual app password

try:
    msg = MimeText("Test from X-FORECAST!")
    msg['Subject'] = "Gmail Test"
    msg['From'] = EMAIL_USER
    msg['To'] = EMAIL_USER
    
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(EMAIL_USER, EMAIL_PASSWORD)
    server.send_message(msg)
    server.quit()
    
    print("✅ SUCCESS! Gmail is working!")
except Exception as e:
    print(f"❌ ERROR: {e}")
    print("Check your app password and 2FA settings")
```

Run it: `python test_gmail.py`

---

## Overview
The X-FORECAST platform now includes a production-ready authentication system with email verification, password security, and comprehensive user management.

## 🚀 Complete Setup Guide (10 Minutes)

### Prerequisites
- Python 3.8+ installed
- Node.js 16+ installed
- Gmail account (or other email provider)

### Step-by-Step Setup

#### 1️⃣ Install Python Dependencies
```bash
# Navigate to your project root
cd your-project-directory

# Install required packages
pip install bcrypt python-dotenv PyJWT sqlite3
```

#### 2️⃣ Set Up Environment Variables
```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your details
# (See detailed Gmail setup below)
```

#### 3️⃣ Initialize Database
```bash
# The database will be created automatically when you first run the app
# Location: users.db in your project root
```

#### 4️⃣ Start the Backend Server
```bash
# Start the FastAPI server
python -m uvicorn src.api.main:app --reload --port 8000
```

#### 5️⃣ Start the Frontend
```bash
# In a new terminal, navigate to frontend directory
cd frontend

# Install dependencies (if not already done)
npm install

# Start the React development server
npm start
```

#### 6️⃣ Test the System
1. Open http://localhost:3000
2. Click "Register" to create a new account
3. Fill in the registration form
4. Check your email for verification link
5. Click the verification link
6. Login with your credentials

### 🎯 Quick Test Without Email
If you want to test without setting up email first:

1. **Disable email verification** temporarily by editing `src/auth/user_management.py`:
```python
# In the register_user method, change this line:
cursor.execute('''
    INSERT INTO users 
    (user_id, email, password_hash, company_name, business_type, 
     verification_code, verification_expires, email_verified)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
''', (user_id, email, password_hash, company_name, business_type, 
      verification_code, verification_expires, True))  # Set to True
```

2. **Register and login** normally - no email verification required
3. **Set up email later** for production use

## Features Implemented

### ✅ Frontend (React/TypeScript)
- **Cyberpunk-styled UI**: Modern, responsive login/registration forms
- **Password Confirmation**: Ensures passwords match during registration
- **Real-time Validation**: Client-side validation for email format and password strength
- **Error Handling**: Clear error messages and loading states
- **Form Switching**: Seamless toggle between login and registration

### ✅ Backend Security
- **bcrypt Password Hashing**: Industry-standard password encryption
- **JWT Authentication**: Secure token-based authentication
- **Account Lockout**: Protection against brute force attacks (5 failed attempts = 30min lock)
- **Password Requirements**: Enforced strong password policy
- **Email Verification**: Required email verification before account activation
- **Password Reset**: Secure password reset via email

### ✅ Database (SQLite)
- **User Management**: Comprehensive user profiles with business information
- **Security Tracking**: Failed login attempts, account locks, verification status
- **Multi-tenant Support**: Company-specific data isolation
- **Audit Trail**: Creation timestamps and activity tracking

### ✅ Email Integration
- **SMTP Support**: Configurable email service (Gmail, Outlook, etc.)
- **HTML Templates**: Professional email templates for verification and reset
- **Security**: Time-limited verification codes and reset tokens

## Password Requirements
- Minimum 8 characters
- At least one uppercase letter (A-Z)
- At least one lowercase letter (a-z)
- At least one number (0-9)
- At least one special character (!@#$%^&*(),.?\":{}|<>)

## Security Features

### Account Protection
- **Email Verification**: Users must verify email before login
- **Account Lockout**: 5 failed login attempts locks account for 30 minutes
- **Password Strength**: Enforced strong password requirements
- **Secure Tokens**: Time-limited verification and reset codes

### Data Protection
- **bcrypt Hashing**: Passwords hashed with salt using bcrypt
- **JWT Tokens**: Secure authentication tokens with expiration
- **User Isolation**: Company-specific data directories and RAG systems
- **Input Validation**: Server-side validation for all inputs

## API Endpoints

### Authentication
- `POST /api/v1/auth/register` - User registration with email verification
- `POST /api/v1/auth/login` - User authentication with security checks
- `POST /api/v1/auth/verify-email` - Email verification
- `POST /api/v1/auth/forgot-password` - Request password reset
- `POST /api/v1/auth/reset-password` - Reset password with code
- `POST /api/v1/auth/verify-token` - Validate JWT token
- `GET /api/v1/auth/profile` - Get user profile

### RAG Integration
- `POST /api/v1/auth/initialize-rag` - Initialize company-specific AI system

## 📧 Email Setup Guide - Step by Step

### 🚀 Quick Start (5 Minutes)

#### Step 1: Install Required Python Packages
```bash
# In your project root directory
pip install bcrypt python-dotenv
```

#### Step 2: Create Environment File
```bash
# Copy the example file
cp .env.example .env
```

#### Step 3: Gmail Setup (Most Common)

##### 3.1 Enable 2-Factor Authentication
1. Go to [Google Account Settings](https://myaccount.google.com/)
2. Click **Security** in the left sidebar
3. Under "Signing in to Google", click **2-Step Verification**
4. Follow the setup process (use your phone number)
5. ✅ **Verify it's enabled** - you should see "2-Step Verification: On"

##### 3.2 Generate App-Specific Password
1. Still in **Security** settings
2. Under "Signing in to Google", click **App passwords**
   - If you don't see this option, make sure 2FA is enabled first
3. Click **Select app** → Choose **Mail**
4. Click **Select device** → Choose **Other (Custom name)**
5. Type: `X-FORECAST Authentication`
6. Click **Generate**
7. 📋 **Copy the 16-character password** (looks like: `abcd efgh ijkl mnop`)

##### 3.3 Update .env File
Open your `.env` file and update these lines:
```bash
# Replace with your actual Gmail address
EMAIL_USER=your.email@gmail.com

# Replace with the 16-character app password (no spaces)
EMAIL_PASSWORD=abcdefghijklmnop

# Keep these as-is for Gmail
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
FROM_EMAIL=noreply@xforecast.com
```

#### Step 4: Test Email Setup
Create a test file `test_email.py`:
```python
import os
from dotenv import load_dotenv
import smtplib
from email.mime.text import MimeText

# Load environment variables
load_dotenv()

def test_email():
    try:
        # Get email config
        smtp_server = os.getenv("SMTP_SERVER")
        smtp_port = int(os.getenv("SMTP_PORT"))
        email_user = os.getenv("EMAIL_USER")
        email_password = os.getenv("EMAIL_PASSWORD")
        
        print(f"Testing email with: {email_user}")
        
        # Create test message
        msg = MimeText("Test email from X-FORECAST!")
        msg['Subject'] = "X-FORECAST Email Test"
        msg['From'] = email_user
        msg['To'] = email_user  # Send to yourself
        
        # Send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(email_user, email_password)
        server.send_message(msg)
        server.quit()
        
        print("✅ Email sent successfully!")
        print("Check your inbox for the test email.")
        
    except Exception as e:
        print(f"❌ Email failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check your Gmail app password is correct")
        print("2. Ensure 2FA is enabled on your Google account")
        print("3. Verify your email address is correct")

if __name__ == "__main__":
    test_email()
```

Run the test:
```bash
python test_email.py
```

### 🔧 Alternative Email Providers

#### Outlook/Hotmail Setup
```bash
# In your .env file
SMTP_SERVER=smtp-mail.outlook.com
SMTP_PORT=587
EMAIL_USER=your.email@outlook.com
EMAIL_PASSWORD=your-outlook-password
```

#### Yahoo Mail Setup
```bash
# In your .env file
SMTP_SERVER=smtp.mail.yahoo.com
SMTP_PORT=587
EMAIL_USER=your.email@yahoo.com
EMAIL_PASSWORD=your-yahoo-app-password
```

#### Custom SMTP Server
```bash
# In your .env file
SMTP_SERVER=mail.yourdomain.com
SMTP_PORT=587
EMAIL_USER=noreply@yourdomain.com
EMAIL_PASSWORD=your-server-password
```

### 🚨 Troubleshooting Common Issues

#### "Authentication failed" Error
- ✅ **Check**: 2FA is enabled on Gmail
- ✅ **Check**: Using app password, not regular password
- ✅ **Check**: App password has no spaces
- ✅ **Check**: Email address is correct

#### "Connection refused" Error
- ✅ **Check**: SMTP server and port are correct
- ✅ **Check**: Internet connection is working
- ✅ **Check**: Firewall isn't blocking port 587

#### "App passwords" option missing
- ✅ **Enable 2FA first** - App passwords only appear after 2FA is set up
- ✅ **Wait 5 minutes** after enabling 2FA
- ✅ **Refresh the page** and check Security settings again

#### Emails not being received
- ✅ **Check spam folder** - automated emails often go to spam
- ✅ **Test with personal email** first before using company email
- ✅ **Verify FROM_EMAIL** is set correctly

### 📱 Production Email Services

For production, consider using dedicated email services:

#### SendGrid (Recommended)
```bash
# More reliable for production
SMTP_SERVER=smtp.sendgrid.net
SMTP_PORT=587
EMAIL_USER=apikey
EMAIL_PASSWORD=your-sendgrid-api-key
```

#### Mailgun
```bash
SMTP_SERVER=smtp.mailgun.org
SMTP_PORT=587
EMAIL_USER=your-mailgun-username
EMAIL_PASSWORD=your-mailgun-password
```

#### Amazon SES
```bash
SMTP_SERVER=email-smtp.us-east-1.amazonaws.com
SMTP_PORT=587
EMAIL_USER=your-ses-username
EMAIL_PASSWORD=your-ses-password
```

## Database Schema

### Users Table
```sql
CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    company_name TEXT NOT NULL,
    business_type TEXT NOT NULL,
    subscription_tier TEXT DEFAULT 'basic',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT 1,
    email_verified BOOLEAN DEFAULT 0,
    verification_code TEXT,
    verification_expires TIMESTAMP,
    reset_code TEXT,
    reset_expires TIMESTAMP,
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP
);
```

### Business Profiles Table
```sql
CREATE TABLE business_profiles (
    user_id TEXT PRIMARY KEY,
    company_name TEXT NOT NULL,
    business_type TEXT NOT NULL,
    industry TEXT,
    data_sources TEXT,
    model_config TEXT,
    storage_path TEXT,
    FOREIGN KEY (user_id) REFERENCES users (user_id)
);
```

## User Flow

### Registration Process
1. User fills registration form with validation
2. Server validates input and creates account (unverified)
3. Verification email sent with secure link
4. User clicks verification link
5. Account activated and RAG system initialized
6. User can now login

### Login Process
1. User enters credentials
2. Server validates email/password
3. Checks account status (verified, not locked)
4. Generates JWT token
5. Initializes RAG system if needed
6. Returns authentication token

### Password Reset Process
1. User requests password reset
2. Server generates secure reset code
3. Reset email sent with time-limited link
4. User enters new password
5. Password updated and account unlocked

## Production Deployment

### Security Checklist
- [ ] Change JWT secret key
- [ ] Configure production email service
- [ ] Set up SSL/HTTPS
- [ ] Configure rate limiting
- [ ] Set up monitoring and logging
- [ ] Regular security updates
- [ ] Backup database regularly

### Environment Setup
1. Copy `.env.example` to `.env`
2. Configure email service credentials
3. Set production URLs
4. Update JWT secret key
5. Configure database path

## Error Handling
- **Invalid Credentials**: Clear error messages without revealing user existence
- **Account Locked**: Informative message with unlock time
- **Email Not Verified**: Prompt to check email
- **Expired Tokens**: Clear expiration messages
- **Network Issues**: Timeout handling and retry suggestions

## 🛠️ Common Setup Issues & Solutions

### Backend Won't Start
**Error**: `ModuleNotFoundError: No module named 'bcrypt'`
```bash
# Solution: Install missing packages
pip install bcrypt python-dotenv PyJWT
```

**Error**: `ImportError: cannot import name 'user_manager'`
```bash
# Solution: Check file paths and imports
# Make sure you're running from the project root directory
```

### Frontend Won't Connect
**Error**: `Network Error` or `CORS Error`
```bash
# Solution 1: Check backend is running on port 8000
curl http://localhost:8000/api/v1/status

# Solution 2: Check CORS settings in src/api/main.py
# Make sure localhost:3000 is in allowed origins
```

### Database Issues
**Error**: `sqlite3.OperationalError: no such table: users`
```bash
# Solution: Delete existing database and restart
rm users.db
python -m uvicorn src.api.main:app --reload --port 8000
```

### Email Not Working
**Error**: `Authentication failed (535)`
```bash
# Solution: Check Gmail app password setup
# 1. Ensure 2FA is enabled
# 2. Generate new app password
# 3. Use app password (not regular password)
# 4. Remove spaces from app password
```

### Registration/Login Issues
**Error**: `Password validation failed`
```bash
# Solution: Check password requirements
# - Minimum 8 characters
# - At least one uppercase letter
# - At least one lowercase letter  
# - At least one number
# - At least one special character
```

### Development vs Production
**Development Mode** (Quick testing):
- Set `email_verified = True` in database
- Skip email verification temporarily
- Use SQLite database

**Production Mode** (Full security):
- Enable email verification
- Use PostgreSQL/MySQL database
- Set up proper SMTP service
- Configure SSL/HTTPS
- Use environment variables for secrets

## 📋 Deployment Checklist

### Before Going Live
- [ ] ✅ Email service configured and tested
- [ ] ✅ Strong JWT secret key set
- [ ] ✅ Database backed up
- [ ] ✅ HTTPS/SSL configured
- [ ] ✅ Environment variables secured
- [ ] ✅ CORS configured for production domain
- [ ] ✅ Rate limiting enabled
- [ ] ✅ Monitoring and logging set up

### Security Hardening
- [ ] ✅ Change default JWT secret
- [ ] ✅ Use production email service (SendGrid, etc.)
- [ ] ✅ Enable account lockout (already implemented)
- [ ] ✅ Set up password complexity rules (already implemented)
- [ ] ✅ Configure session timeouts
- [ ] ✅ Add request rate limiting
- [ ] ✅ Set up security headers
- [ ] ✅ Regular security updates

## 🎯 Testing Your Setup

### Manual Testing Steps
1. **Registration Flow**:
   - Register new user → Check email → Verify → Login ✅

2. **Password Security**:
   - Try weak password → Should be rejected ✅
   - Try mismatched passwords → Should be rejected ✅

3. **Account Security**:
   - Try 5 wrong passwords → Account should lock ✅
   - Wait 30 minutes → Should unlock ✅

4. **Email Features**:
   - Request password reset → Check email ✅
   - Use reset link → Should work ✅

### Automated Testing
Create `test_auth.py`:
```python
import requests
import time

BASE_URL = "http://localhost:8000/api/v1/auth"

def test_registration():
    data = {
        "email": "test@example.com",
        "password": "TestPass123!",
        "confirmPassword": "TestPass123!",
        "company_name": "Test Company",
        "business_type": "retail"
    }
    
    response = requests.post(f"{BASE_URL}/register", json=data)
    print(f"Registration: {response.status_code} - {response.json()}")

def test_login():
    data = {
        "email": "test@example.com", 
        "password": "TestPass123!"
    }
    
    response = requests.post(f"{BASE_URL}/login", json=data)
    print(f"Login: {response.status_code} - {response.json()}")

if __name__ == "__main__":
    test_registration()
    time.sleep(1)
    test_login()
```

## Future Enhancements
- [ ] Two-Factor Authentication (2FA)
- [ ] Social Login (Google, Microsoft)
- [ ] Advanced password policies
- [ ] Session management
- [ ] Audit logging
- [ ] GDPR compliance features
- [ ] Advanced rate limiting
- [ ] Captcha integration