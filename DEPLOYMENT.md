# 🚀 Deployment Guide for Streamlit Cloud

This guide shows you how to deploy your CHUNKS AI Chatbot to Streamlit Cloud safely without exposing your API keys.

## 🔒 Important Security Notes

**❌ NEVER** commit API keys to your GitHub repository!
**✅ ALWAYS** use Streamlit's secrets management for deployment.

## 📝 Step-by-Step Deployment

### 1. Prepare Your Repository

```bash
# Make sure .gitignore is protecting your secrets
git add .
git commit -m "Initial commit - ready for deployment"
git push origin main
```

Your `.gitignore` file already protects:
- `.env` (local API keys)
- `.streamlit/secrets.toml` (deployment secrets)
- `streamlit_chat_sessions_v2.json` (user data)

### 2. Deploy to Streamlit Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**

2. **Click "New App"**

3. **Connect Your GitHub Repository:**
   - Repository: `your-username/chunks-ai-chatbot-v2`
   - Branch: `main` 
   - Main file path: `app.py`

4. **Click "Deploy"** (it will fail first - that's expected!)

### 3. Add Your API Keys Securely

1. **In Streamlit Cloud Dashboard:**
   - Go to your app settings
   - Click **"Secrets"**

2. **Add your secrets in TOML format:**
```toml
# Copy and paste this, replacing with your actual keys:

OPENROUTER_API_KEY = "sk-or-v1-your-actual-key-here"
FIREWORKS_API_KEY = "fw_your-actual-key-here"
NOVITA_API_KEY = "sk_your-actual-key-here"
DEEPGRAM_API_KEY = "your-actual-deepgram-key"
HF_TOKEN = "hf_your-actual-token-here"
```

3. **Click "Save"**

4. **Your app will automatically restart and work!**

## 🔄 How It Works

The app uses a smart function that checks for API keys in this order:
1. **Local development**: Reads from `.env` file
2. **Streamlit Cloud**: Reads from secrets management
3. **Fallback**: Shows error if no keys found

```python
def get_api_key(key_name):
    # First try local .env file
    env_key = os.getenv(key_name)
    if env_key:
        return env_key
    
    # Then try Streamlit secrets
    try:
        return st.secrets[key_name]
    except (KeyError, FileNotFoundError):
        return None
```

## 🌐 Your App URLs

After deployment, you'll get URLs like:
- **Main App**: `https://your-app-name.streamlit.app`
- **Settings**: Access through Streamlit Cloud dashboard

## 🛠️ Updating Your Deployed App

### Code Changes:
```bash
git add .
git commit -m "Update features"
git push origin main
# App auto-updates!
```

### API Key Changes:
1. Go to Streamlit Cloud dashboard
2. Click your app → "Settings" → "Secrets"
3. Update the keys
4. Click "Save"

## 🔍 Testing Your Deployment

1. **Visit your app URL**
2. **Check that it loads without errors**
3. **Test model selection and filtering**
4. **Send a test message**
5. **Verify all features work**

## 🆘 Troubleshooting

### App Won't Start
- Check that all required secrets are added
- Verify OPENROUTER_API_KEY is present
- Check app logs in Streamlit Cloud

### 401/403 Errors
- Verify API keys are correct in secrets
- Check that keys have sufficient credits/permissions
- Use "✅ Ready to Use" filter to avoid external key requirements

### Performance Issues
- Streamlit Cloud has resource limits
- Consider caching model data
- Use free models for testing

## 🎯 Best Practices

1. **Never commit secrets** - Always use Streamlit's secrets management
2. **Test locally first** - Use `.env` file for development
3. **Use free models** - For public demos, consider free models
4. **Monitor usage** - Keep track of API costs
5. **Update regularly** - Keep dependencies updated

## 📊 Resource Limits

Streamlit Cloud (free tier):
- **RAM**: 1GB
- **CPU**: Shared
- **Storage**: Limited
- **Bandwidth**: Fair use policy

For heavy usage, consider upgrading to Streamlit Cloud for Teams.

## 🔗 Useful Links

- [Streamlit Cloud](https://share.streamlit.io)
- [Streamlit Secrets Management](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app/connect-to-data-sources/secrets-management)
- [OpenRouter Dashboard](https://openrouter.ai/activity)
- [Fireworks AI Console](https://app.fireworks.ai)

---

**Happy Deploying! 🚀**