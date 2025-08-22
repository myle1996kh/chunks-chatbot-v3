# 🚀 CHUNKS AI Chatbot v2.0 - All Models Edition

A powerful multi-provider AI chatbot with voice capabilities, authentication, and access to 483+ AI models from OpenRouter, Fireworks AI, and Novita AI.

## ✨ Features

### 🤖 Multi-Provider AI Support
- **OpenRouter**: Access to 483+ models (GPT-4, Claude, Gemini, Llama, and more)
- **Fireworks AI**: High-performance inference for popular models
- **Novita AI**: Additional model variety and capabilities
- **Smart Model Filtering**: Filter by provider, popularity, free models, or search

### 🎤 Advanced Voice Capabilities
- **Speech-to-Text**: Convert voice input to text using Deepgram
- **Text-to-Speech**: AI voice responses with Deepgram TTS
- **Voice Cloning**: Create custom voices with Speechify
- **Auto-Play**: Automatic voice playback of AI responses
- **Voice Database**: Organize and manage custom voice clones

### 🔐 Secure Authentication
- Login system with user management
- Session timeout protection
- User credentials stored in secure configuration
- Clean sidebar with user info and logout

### 💬 Enhanced Chat Experience
- Session management with persistent chat history
- System prompt templates for different conversation styles
- Real-time streaming responses
- Message formatting with code highlighting
- Copy-to-clipboard functionality

### 🎯 Smart Defaults
- Pre-configured for optimal user experience
- Voice input enabled by default
- Speechify auto-play ready
- OpenAI models filtered by default
- GPT-5-Chat as default model

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Streamlit
- API keys for desired services

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/myle1996kh/chunks-chatbot-v3.git
cd chunks-chatbot-v3
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure API keys and authentication**
Create `.streamlit/secrets.toml`:
```toml
# API Keys (replace with your actual keys)
OPENROUTER_API_KEY = "your_openrouter_key_here"
FIREWORKS_API_KEY = "your_fireworks_key_here"  
NOVITA_API_KEY = "your_novita_key_here"
DEEPGRAM_API_KEY = "your_deepgram_key_here"
SPEECHIFY_API_KEY = "your_speechify_key_here"

# Authentication
[auth]
users = [
    {username = "admin", password = "your_secure_password", name = "Administrator"},
    {username = "user1", password = "another_password", name = "User One"}
]
session_timeout = 60
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Access the app**
- Open your browser to `http://localhost:8501`
- Login with the credentials you configured
- Start chatting with AI models!

## 🔑 API Keys Setup

### Required Services

| Service | Purpose | Required For |
|---------|---------|--------------|
| **OpenRouter** | Access to 483+ AI models | Main chat functionality |
| **Deepgram** | Speech-to-text and text-to-speech | Voice input/output |
| **Speechify** | Voice cloning and custom TTS | Custom voice creation |
| **Fireworks AI** | High-performance model inference | Alternative AI provider |
| **Novita AI** | Additional model variety | Alternative AI provider |

### Getting API Keys

1. **OpenRouter**: Sign up at [openrouter.ai](https://openrouter.ai) - Access to 483+ models
2. **Deepgram**: Get key at [deepgram.com](https://deepgram.com) - Voice processing
3. **Speechify**: Sign up at [speechify.com](https://speechify.com) - Voice cloning
4. **Fireworks AI**: Register at [fireworks.ai](https://fireworks.ai) - Fast inference
5. **Novita AI**: Get access at [novita.ai](https://novita.ai) - Model variety

## 🎯 Usage Guide

### Basic Chat
1. Login with your credentials
2. Select AI provider (OpenRouter recommended)
3. Choose your model (GPT-5-Chat is default)
4. Start typing or use voice input
5. Enjoy AI responses with optional voice playback

### Voice Features
1. **Voice Input**: Click microphone to record speech
2. **Voice Output**: Enable TTS for AI responses  
3. **Voice Cloning**: Create custom voices in Speechify section
4. **Auto-Play**: AI responses play automatically

### Model Selection
- **Filter by Provider**: OpenAI, Anthropic, Google, Meta
- **Filter by Type**: Popular, Free, All models
- **Search**: Find specific models by name
- **Cost Calculator**: See token pricing for each model

### Session Management
- **Multiple Sessions**: Create and switch between conversations
- **Persistent History**: Chat history saved automatically
- **System Prompts**: Choose conversation styles
- **Export/Import**: Backup your chat sessions

## 🏗️ Architecture

### Core Components
- **app.py**: Main Streamlit application (2400+ lines)
- **auth.py**: Authentication and session management
- **voice_database.py**: SQLite-based voice clone storage
- **simple_voice_components.py**: Voice UI components

### Data Storage
- **Chat Sessions**: JSON file storage
- **Voice Clones**: SQLite database
- **User Auth**: Streamlit secrets configuration
- **API Keys**: Multi-source priority system

### AI Provider Integration
- **OpenRouter**: Dynamic model loading with 483+ options
- **Fireworks/Novita**: Static model catalogs
- **Streaming**: Real-time response generation
- **Fallbacks**: Graceful error handling

## 🔧 Configuration

### Default Settings
The app comes pre-configured with optimal defaults:
- OpenAI filter active
- GPT-5-Chat selected
- Voice input enabled
- Speechify auto-play enabled
- 60-minute session timeout

### Customization
- Modify `.streamlit/secrets.toml` for users and API keys
- Adjust model defaults in `app.py`
- Customize voice settings per user preference
- Configure system prompts for different conversation styles

## 🛠️ Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run with auto-reload
streamlit run app.py --logger.level=debug

# Validate code
python -m py_compile app.py
python -m py_compile auth.py
```

### Project Structure
```
chunks-chatbot-v3/
├── app.py                      # Main application
├── auth.py                     # Authentication system
├── voice_database.py           # Voice clone management
├── simple_voice_components.py  # Voice UI components
├── requirements.txt            # Python dependencies
├── .streamlit/
│   └── secrets.toml           # API keys and auth config
├── voice_clones.db            # Voice clone database
└── streamlit_chat_sessions_v2.json  # Chat history
```

## 🔒 Security

### Authentication
- Secure login system with configurable users
- Session timeout protection
- No API keys in source code
- User credentials in encrypted configuration

### API Key Management
- Multi-source priority: Session → Browser → Environment → Secrets
- Secure storage in `.streamlit/secrets.toml`
- No keys committed to repository
- Graceful fallbacks for missing keys

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- 2GB RAM minimum (4GB recommended)
- Internet connection for AI model access
- Modern web browser for Streamlit interface

### Dependencies
See `requirements.txt` for full list:
- `streamlit>=1.28.0` - Web interface
- `requests>=2.28.0` - API communications
- `python-dotenv>=0.19.0` - Environment management
- `qdrant-client>=1.6.0` - Vector database support
- `sentence-transformers>=2.2.0` - Text embeddings

## 📝 License

This project is open source. Please ensure you comply with the terms of service of all integrated AI providers when using their APIs.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section in the app
- Review API provider documentation for service-specific issues

## 🎉 Acknowledgments

Built with:
- [Streamlit](https://streamlit.io) - Web framework
- [OpenRouter](https://openrouter.ai) - AI model access
- [Deepgram](https://deepgram.com) - Voice processing
- [Speechify](https://speechify.com) - Voice cloning
- [Fireworks AI](https://fireworks.ai) - Fast inference
- [Novita AI](https://novita.ai) - Additional models

---

**🚀 Ready to chat with 483+ AI models? Clone, configure, and launch your advanced AI assistant today!**