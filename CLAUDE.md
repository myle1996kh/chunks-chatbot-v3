# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
streamlit run app.py
```

### Installation
```bash
pip install -r requirements.txt
```

### Code Validation
```bash
python -m py_compile app.py
python -m py_compile auth.py
python -m py_compile voice_database.py
```

## Core Architecture

### Main Application Structure
This is a **Streamlit-based AI chatbot** that integrates multiple AI providers (OpenRouter, Fireworks, Novita) with voice capabilities (Deepgram, Speechify) and authentication.

**Key Components:**
- `app.py` - Main Streamlit application (2400+ lines)
- `auth.py` - Authentication system using secrets.toml
- `voice_database.py` - SQLite-based voice clone management  
- `simple_voice_components.py` - Streamlit UI components for voice features

### Multi-Provider AI Architecture
The `MultiProviderAI` class in `app.py` handles:
- **OpenRouter**: 483+ models with dynamic loading and filtering
- **Fireworks AI**: Static model catalog with streaming support
- **Novita AI**: Static model catalog

**API Key Management**: Multi-source priority system:
1. Session state (user input)
2. Browser localStorage (JavaScript)
3. Environment variables
4. Streamlit secrets.toml

### Voice System Architecture
Two main voice services:
- **DeepgramVoice**: Speech-to-text and text-to-speech
- **SpeechifyVoice**: Voice cloning and custom TTS

**Voice Database**: SQLite database (`voice_clones.db`) manages:
- Voice clone metadata (name, category, description, tags)
- Creation timestamps and usage tracking
- Search and filtering capabilities

### Authentication System
**AuthManager** class provides:
- User credentials stored in `.streamlit/secrets.toml`
- Session timeout management (default 60 minutes)
- Login/logout with session state persistence
- User info display in sidebar

### Session Management
**SessionManager** class handles:
- Chat session persistence in `streamlit_chat_sessions_v2.json`
- Message history and system prompts
- Session naming and organization

### Context System
**FastJSONContext** class provides:
- Conversation context storage without vector database
- User pattern analysis
- Style adaptation based on conversation history

## Configuration

### Secrets Configuration
The application requires `.streamlit/secrets.toml` with:
```toml
# API Keys (at root level)
OPENROUTER_API_KEY = "your_key"
DEEPGRAM_API_KEY = "your_key"
SPEECHIFY_API_KEY = "your_key"

# Authentication
[auth]
users = [
    {username = "admin", password = "password", name = "Admin"}
]
session_timeout = 60
```

### Default Settings
The application uses **session state** to maintain defaults:
- OpenAI model filter (index 3)
- GPT-5-Chat as default model
- Voice input enabled
- Speechify TTS with auto-play enabled

### Model Filtering System
OpenRouter models are filtered by:
- Provider (OpenAI, Anthropic, Google, Meta)
- Model type (free, popular, all)
- Search terms (name, description, provider)
- External API key requirements

## Key Implementation Details

### Streamlit State Management
Critical session state variables:
- `voice_input_enabled`, `voice_enabled`, `auto_voice_response` - Voice settings
- `model_filter_index`, `voice_method_index` - UI defaults
- `current_session_id`, `chat_sessions` - Session management
- Authentication state (`authenticated`, `username`, `login_time`)

### Voice Processing Flow
1. **Input**: Audio recording → Deepgram STT → Text
2. **Processing**: Text → AI provider → Response
3. **Output**: Response → Speechify/Deepgram TTS → Audio playback

### Error Handling Patterns
- API key validation with multi-source fallback
- Graceful model loading failures with static fallbacks
- Voice processing errors with user-friendly messages
- Authentication timeout with automatic logout

### Performance Considerations
- OpenRouter models cached with `@st.cache_data`
- Streaming responses for better UX
- Lazy loading of voice components
- Session-based state persistence to avoid re-initialization

## Database Schema

### Voice Clones Table
```sql
CREATE TABLE voice_clones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    voice_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    category TEXT DEFAULT 'Personal',
    description TEXT,
    tags TEXT,  -- JSON array
    created_at TEXT,
    last_used TEXT,
    usage_count INTEGER DEFAULT 0
);
```

## Security Notes
- API keys stored in secrets.toml (gitignored)
- Authentication required before app access
- Session timeout enforcement
- No API keys in source code or commits