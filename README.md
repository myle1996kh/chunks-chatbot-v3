# 🚀 CHUNKS AI Chatbot v2.0 - All Models Edition

A powerful multi-provider AI chatbot built with Streamlit that supports 483+ AI models from OpenRouter, Fireworks AI, and Novita AI. Features advanced model filtering, voice chat, session management, and smart error handling.

## ✨ Features

### 🤖 Multi-Provider AI Support
- **🌐 OpenRouter**: Access to 483+ AI models including GPT-4, Claude, Llama, and more
- **🔥 Fireworks AI**: 8 high-performance models including GPT OSS and Qwen models
- **⭐ Novita AI**: 6 specialized models including DeepSeek and GLM models

### 🔍 Smart Model Filtering
- **✅ Ready to Use**: Filter models that work without additional API keys
- **🆓 Free Models**: Find completely free models
- **🏷️ Provider Filtering**: Filter by OpenAI, Anthropic, Google, Meta, etc.
- **🔍 Search**: Search models by name or provider
- **🧪 Model Testing**: Test any model with 3 tokens before using

### 🎤 Voice Features
- **🔊 Browser Speech API**: Free, built-in voice synthesis
- **🎵 Google TTS**: High-quality free voice generation
- **🎤 Deepgram TTS**: Premium voice with multiple models

### 💬 Session Management
- **📁 Multiple Sessions**: Create and manage multiple chat sessions
- **🎯 Custom System Prompts**: Set different personalities for each session
- **💾 Auto-Save**: Automatically saves all conversations
- **📋 Copy/Share**: Easy copy buttons for all messages

### 🛡️ Enhanced Error Handling
- **401/403 Errors**: Clear guidance for API key issues
- **Model Access**: Smart detection of models requiring additional keys
- **Connection Issues**: Helpful troubleshooting messages

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd chunks-ai-chatbot-v2
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up Environment Variables
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
```

### 4. Configure API Keys
Edit the `.env` file and add your API keys:

```env
# Required
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Optional (for additional providers)
FIREWORKS_API_KEY=fw_your-key-here
NOVITA_API_KEY=sk_your-key-here
DEEPGRAM_API_KEY=your-key-here  # For premium voice features
HF_TOKEN=hf_your-token-here     # For HuggingFace integration
```

### 5. Run the Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🔑 Getting API Keys

### OpenRouter (Required)
1. Visit [OpenRouter](https://openrouter.ai/keys)
2. Create an account
3. Generate an API key
4. Add credits to your account

### Fireworks AI (Optional)
1. Visit [Fireworks AI](https://app.fireworks.ai)
2. Sign up for an account
3. Generate an API key in settings

### Novita AI (Optional)
1. Visit [Novita AI](https://novita.ai)
2. Create an account
3. Get your API key from the dashboard

### Deepgram (Optional - for premium voice)
1. Visit [Deepgram](https://deepgram.com)
2. Sign up for an account
3. Get your API key from the console

## 🎯 Usage Tips

### Model Selection
- **Start with "✅ Ready to Use"** filter to see models that work immediately
- **Use "🆓 Free Models"** to find completely free options
- **Test models** with the 🧪 testing feature before full conversations

### Best Free Models
- `meta-llama/llama-3.2-3b-instruct` - Good general purpose
- `mistralai/mistral-7b-instruct` - Fast and efficient
- `microsoft/phi-3-mini-4k-instruct` - Compact and smart

### Voice Chat
1. Enable voice in sidebar settings
2. Choose your preferred voice method
3. Click 🔊 on any AI message to hear it
4. Use browser speech for best free experience

### Session Management
- Create different sessions for different topics
- Use custom system prompts to change AI personality
- Sessions auto-save - your conversations persist

## 🛠️ Troubleshooting

### API Key Issues
- **401 Error**: Your API key is invalid or expired
- **403 Error**: Model requires additional API key integration
- **Solution**: Use "✅ Ready to Use" filter or add required keys

### Voice Issues
- Ensure browser allows audio
- Try Chrome/Edge for best speech synthesis support
- Check volume settings
- Refresh page if voice stops working

### Model Loading Issues
- Check internet connection
- Verify API keys are correct
- Try clearing browser cache
- Use fallback models if loading fails

## 📁 Project Structure

```
chunks-ai-chatbot-v2/
├── app.py                 # Main application
├── requirements.txt       # Python dependencies
├── .env                  # Your API keys (not in git)
├── .env.example          # Example environment file
├── .gitignore           # Git ignore file
├── README.md            # This file
└── streamlit_chat_sessions_v2.json  # Auto-generated session data
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🆘 Support

If you encounter issues:
1. Check the troubleshooting section
2. Verify your API keys are correct
3. Try the model testing feature
4. Create an issue on GitHub with error details

## 🌟 Features Coming Soon

- [ ] Image analysis with vision models
- [ ] File upload and analysis
- [ ] Export conversations
- [ ] Custom model temperature per session
- [ ] Conversation templates
- [ ] Model performance analytics

---

**Happy Chatting! 🚀**

Built with ❤️ using Streamlit and powered by multiple AI providers.