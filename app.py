# ===================================================================
# CHUNKS AI CHATBOT V2.0 - ALL MODELS EDITION
# ===================================================================
# Enhanced version with ALL 483+ OpenRouter models + Fireworks + Novita
# Features: Dynamic model loading, filtering, search, cost calculator

# ===================================================================
# IMPORTS AND SETUP
# ===================================================================
import streamlit as st
import json
import datetime
from typing import Dict, List, Any, Optional, Tuple
import uuid
import requests
import base64
import io
import wave
import tempfile
import os
from pathlib import Path
import time
from collections import defaultdict
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import numpy as np
import threading
import queue

# Load environment variables
load_dotenv()

# ===================================================================
# API KEYS CONFIGURATION WITH BROWSER STORAGE
# ===================================================================

# JavaScript functions for browser local storage
def add_local_storage_js():
    """Add JavaScript functions for browser local storage"""
    st.markdown("""
    <script>
    // Save API key to browser local storage (encrypted)
    function saveAPIKey(keyName, keyValue) {
        if (keyValue && keyValue.trim() !== '') {
            // Simple encoding (not encryption, just obfuscation)
            const encoded = btoa(keyValue);
            localStorage.setItem('chunks_' + keyName, encoded);
            return true;
        }
        return false;
    }
    
    // Load API key from browser local storage
    function loadAPIKey(keyName) {
        try {
            const encoded = localStorage.getItem('chunks_' + keyName);
            if (encoded) {
                return atob(encoded);
            }
        } catch (e) {
            console.log('Error loading key:', e);
        }
        return null;
    }
    
    // Clear all saved API keys
    function clearAllAPIKeys() {
        const keys = ['OPENROUTER_API_KEY', 'FIREWORKS_API_KEY', 'NOVITA_API_KEY', 'DEEPGRAM_API_KEY'];
        keys.forEach(key => localStorage.removeItem('chunks_' + key));
    }
    
    // Check if API key is saved
    function hasAPIKey(keyName) {
        return localStorage.getItem('chunks_' + keyName) !== null;
    }
    </script>
    """, unsafe_allow_html=True)

def get_api_key_multi_source(key_name):
    """Get API key from multiple sources with priority order"""
    
    # Priority 1: Check if user has input the key in current session
    session_key = f"user_input_{key_name}"
    if session_key in st.session_state and st.session_state[session_key]:
        return st.session_state[session_key]
    
    # Priority 2: Check browser local storage via session state
    browser_key = f"browser_{key_name}"
    if browser_key in st.session_state and st.session_state[browser_key]:
        return st.session_state[browser_key]
    
    # Priority 3: Environment variables (for local development)
    env_key = os.getenv(key_name)
    if env_key and env_key != f"{key_name.lower()}-placeholder":
        return env_key
    
    # Priority 4: Streamlit secrets (for deployment)
    try:
        secret_key = st.secrets[key_name]
        if secret_key and not secret_key.startswith("your-actual-") and not secret_key.endswith("-here"):
            return secret_key
    except (KeyError, FileNotFoundError, AttributeError):
        pass
    
    return None

# Initialize API keys (will be updated via UI)
OPENROUTER_API_KEY = get_api_key_multi_source("OPENROUTER_API_KEY")
FIREWORKS_API_KEY = get_api_key_multi_source("FIREWORKS_API_KEY")
NOVITA_API_KEY = get_api_key_multi_source("NOVITA_API_KEY")
DEEPGRAM_API_KEY = get_api_key_multi_source("DEEPGRAM_API_KEY")

# ===================================================================
# STATIC MODEL CATALOGS (Fireworks & Novita)
# ===================================================================

FIREWORKS_MODELS = {
    "openai/gpt-oss-120b": {
        "name": "OpenAI GPT OSS 120B",
        "parameters": "117B",
        "description": "High reasoning, production use cases",
        "fireworks_model": "accounts/fireworks/models/gpt-oss-120b"
    },
    "openai/gpt-oss-20b": {
        "name": "OpenAI GPT OSS 20B", 
        "parameters": "21B",
        "description": "Lower latency, specialized use cases",
        "fireworks_model": "accounts/fireworks/models/gpt-oss-20b"
    },
    "qwen3-235b-a22b-instruct-2507": {
        "name": "Qwen3 235B Instruct",
        "parameters": "235B total, 22B active",
        "description": "Large reasoning model",
        "fireworks_model": "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507"
    },
    "qwen3-235b-a22b-thinking-2507": {
        "name": "Qwen3 235B Thinking",
        "parameters": "235B total, 22B active",
        "description": "Advanced reasoning and thinking",
        "fireworks_model": "accounts/fireworks/models/qwen3-235b-a22b-thinking-2507"
    },
    "qwen3-coder-480b-a35b-instruct": {
        "name": "Qwen3 Coder 480B",
        "parameters": "480B total, 35B active",
        "description": "Specialized for coding tasks",
        "fireworks_model": "accounts/fireworks/models/qwen3-coder-480b-a35b-instruct"
    },
    "qwen3-30b-a3b": {
        "name": "Qwen3 30B",
        "parameters": "30B total, 3B active",
        "description": "Efficient smaller model",
        "fireworks_model": "accounts/fireworks/models/qwen3-30b-a3b"
    },
    "kimi-k2-instruct": {
        "name": "Kimi K2 Instruct",
        "parameters": "1T total, 32B active",
        "description": "High-capacity instruct model",
        "fireworks_model": "accounts/fireworks/models/kimi-k2-instruct"
    },
    "deepseek-r1-0528": {
        "name": "DeepSeek R1",
        "parameters": "Unknown",
        "description": "Advanced reasoning model",
        "fireworks_model": "accounts/fireworks/models/deepseek-r1-0528"
    }
}

NOVITA_MODELS = {
    "deepseek-v3-0324": {
        "name": "DeepSeek V3",
        "parameters": "685B MoE",
        "description": "Mixture-of-experts model with function calling",
        "novita_model": "deepseek/deepseek-v3-0324"
    },
    "glm-4.5v": {
        "name": "GLM 4.5V",
        "parameters": "Unknown",
        "description": "Visual reasoning and function calling",
        "novita_model": "zai-org/glm-4.5v"
    },
    "glm-4.5": {
        "name": "GLM 4.5",
        "parameters": "Unknown",
        "description": "Advanced reasoning capabilities",
        "novita_model": "zai-org/glm-4.5"
    },
    "openai-gpt-oss-120b": {
        "name": "OpenAI GPT OSS 120B (Novita)",
        "parameters": "117B MoE",
        "description": "Structured outputs and function calling",
        "novita_model": "openai/gpt-oss-120b"
    },
    "openai-gpt-oss-20b": {
        "name": "OpenAI GPT OSS 20B (Novita)",
        "parameters": "20B",
        "description": "Faster responses, good quality",
        "novita_model": "openai/gpt-oss-20b"
    },
    "qwen3-coder-480b": {
        "name": "Qwen3 Coder 480B (Novita)",
        "parameters": "480B total, 35B active",
        "description": "Coding and development specialist",
        "novita_model": "qwen/qwen3-coder-480b-a35b-instruct"
    }
}

# ===================================================================
# SYSTEM PROMPT TEMPLATES
# ===================================================================

SYSTEM_PROMPT_TEMPLATES = {
    "default": {
        "name": "🤖 Default Assistant",
        "prompt": "You are a helpful AI assistant."
    },
    "mirror_chatbot": {
        "name": "🪞 Mirror Chatbot",
        "prompt": """You are a reflective conversation partner that helps users discover insights about themselves.

- Mirror back the user's thinking patterns and strengths you observe
- Ask questions that invite deeper self-reflection  
- Be concise but go beneath surface level
- When user shares accomplishments, help them see the deeper capabilities shown
- Avoid giving direct advice - instead reflect back what you notice about their approach
- Match the user's depth level and invite them slightly deeper
- Point out contradictions gently (like "I know nothing" vs building complex systems)"""
    },
    "depth_archaeologist": {
        "name": "🏺 The Depth Archaeologist", 
        "prompt": """You dig beneath surface statements to uncover deeper truths about the person.

- When someone says "I don't know," explore what they actually do know
- Look for contradictions between self-perception and demonstrated abilities
- Ask questions that reveal hidden assumptions or beliefs
- Reflect back the deeper motivations behind their stated goals
- Help them see the gap between how they see themselves vs. their actions
- Be concise but penetrating in your observations"""
    },
    "resonance_detector": {
        "name": "📡 The Resonance Detector",
        "prompt": """You are attuned to what energizes and drains the user, helping them discover their natural resonance.

- Notice when their language becomes more animated or engaged
- Identify topics/activities that seem to flow naturally for them
- Reflect back moments when they seem most authentic
- Ask about what feels effortless vs. forced in their experience
- Help them recognize their own patterns of curiosity and interest
- Point out when they're in flow vs. when they're struggling"""
    },
    "capability_mirror": {
        "name": "💪 The Capability Mirror",
        "prompt": """You specialize in showing people capabilities they possess but don't recognize.

- Look for evidence of skills in their stories and examples
- Translate their accomplishments into broader competencies
- Challenge limiting self-beliefs with concrete evidence
- Help them see transferable skills across different domains
- Reflect back the sophistication of their actual thinking
- Ask questions that reveal unconscious expertise"""
    },
    "frequency_matcher": {
        "name": "📻 The Frequency Matcher",
        "prompt": """You adapt to and amplify the user's current mental/emotional frequency while gently inviting deeper exploration.

- Match their communication style and energy level
- Sense the "vibe" they're operating from and meet them there
- Gradually invite slightly deeper exploration without forcing it
- Reflect back the quality of presence they bring to the conversation
- Help them notice their own shifts in awareness or understanding
- Be responsive to their readiness for different levels of depth"""
    }
}

# ===================================================================
# DYNAMIC OPENROUTER MODEL LOADING
# ===================================================================

# Note: OpenRouter model caching is now handled by Streamlit's @st.cache_data decorator

@st.cache_data(ttl=3600)  # Cache for 1 hour - major performance boost!
def fetch_all_openrouter_models():
    """Fetch ALL available models from OpenRouter API (483+ models)"""
    # Note: Streamlit cache_data replaces manual caching logic
    
    # Get current API key for caching
    current_key = get_api_key_multi_source("OPENROUTER_API_KEY")
    if not current_key:
        return {}
        
    try:
        headers = {
            "Authorization": f"Bearer {current_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=15)
        
        if response.status_code == 200:
            models_data = response.json()
            loaded_models = {}
            
            for model_info in models_data.get("data", []):
                model_id = model_info.get("id", "")
                if not model_id:
                    continue
                
                name = model_info.get("name", model_id)
                context_length = model_info.get("context_length", 0)
                
                # Extract pricing info
                pricing = model_info.get("pricing", {})
                prompt_price = float(pricing.get("prompt", "0") or "0")
                completion_price = float(pricing.get("completion", "0") or "0")
                
                # Determine if it's free
                is_free = prompt_price == 0 and completion_price == 0
                
                # Get provider from model ID
                provider = model_id.split("/")[0] if "/" in model_id else "unknown"
                
                # Create model entry
                loaded_models[model_id] = {
                    "name": name,
                    "parameters": f"{context_length//1000}k context" if context_length >= 1000 else f"{context_length} context" if context_length else "Unknown",
                    "description": f"{provider.title()} model" + (" (FREE)" if is_free else f" (${prompt_price:.6f}/${completion_price:.6f})"),
                    "openrouter_model": model_id,
                    "context_length": context_length,
                    "is_free": is_free,
                    "provider": provider,
                    "prompt_price": prompt_price,
                    "completion_price": completion_price
                }
            
            return loaded_models
            
    except Exception as e:
        st.error(f"❌ Failed to load OpenRouter models: {e}")
        # Return fallback popular models
        fallback_models = {
            "openai/gpt-4o": {
                "name": "GPT-4 Omni", "parameters": "Unknown", 
                "description": "OpenAI model (fallback)", "openrouter_model": "openai/gpt-4o"
            },
            "openai/gpt-4o-mini": {
                "name": "GPT-4 Omni Mini", "parameters": "Unknown",
                "description": "OpenAI model (fallback)", "openrouter_model": "openai/gpt-4o-mini"
            },
            "anthropic/claude-3.5-sonnet": {
                "name": "Claude 3.5 Sonnet", "parameters": "Unknown",
                "description": "Anthropic model (fallback)", "openrouter_model": "anthropic/claude-3.5-sonnet"
            }
        }
        return fallback_models

def filter_openrouter_models(models: Dict, filter_type: str = "all", search_term: str = ""):
    """Filter OpenRouter models by various criteria"""
    if not models:
        return {}
    
    # Models that typically require additional API keys (only for "no_external_keys" filter)
    REQUIRES_EXTERNAL_KEYS = [
        "openai/", "gpt-", "chatgpt", "dall-e",  # OpenAI models
        "google/", "gemini", "palm", "bard",     # Google models  
        "cohere/",                               # Cohere models
        "ai21/",                                 # AI21 models
        "anthropic/claude-3-5-sonnet-20241022", # Some specific Anthropic models
        "anthropic/claude-3-5-haiku-20241022",
    ]
    
    # Best/Popular models to highlight
    POPULAR_MODELS = [
        "gpt-4o", "gpt-4o-mini", "gpt-4-turbo",
        "claude-3-5-sonnet", "claude-3-5-haiku", "claude-3-opus",
        "gemini-pro", "gemini-1.5-pro", "gemini-flash",
        "llama-3.1-8b", "llama-3.1-70b", "llama-3.1-405b",
        "mistral-7b", "mistral-large", "mixtral-8x7b",
        "phi-3-mini", "qwen-2.5-72b"
    ]
    
    filtered = {}
    
    for model_id, model_info in models.items():
        # Apply search filter
        if search_term:
            search_lower = search_term.lower()
            if not (search_lower in model_id.lower() or 
                   search_lower in model_info.get("name", "").lower() or
                   search_lower in model_info.get("provider", "").lower()):
                continue
        
        # Apply filter based on type
        if filter_type == "no_external_keys":
            # Exclude models requiring external API keys
            requires_external = False
            model_id_lower = model_id.lower()
            for pattern in REQUIRES_EXTERNAL_KEYS:
                if pattern.lower() in model_id_lower:
                    requires_external = True
                    break
            if requires_external:
                continue
        elif filter_type == "free":
            if not model_info.get("is_free", False):
                continue
        elif filter_type == "openai":
            # Show ALL OpenAI models (including those requiring keys)
            if not ("openai" in model_id.lower() or "gpt" in model_id.lower()):
                continue
        elif filter_type == "anthropic":
            # Show ALL Anthropic models
            if not ("anthropic" in model_id.lower() or "claude" in model_id.lower()):
                continue
        elif filter_type == "google":
            # Show ALL Google models  
            if not ("google" in model_id.lower() or "gemini" in model_id.lower() or "palm" in model_id.lower()):
                continue
        elif filter_type == "meta":
            # Show ALL Meta models
            if not ("meta" in model_id.lower() or "llama" in model_id.lower()):
                continue
        elif filter_type == "popular":
            # Show curated popular models
            is_popular = any(popular_model.lower() in model_id.lower() for popular_model in POPULAR_MODELS)
            if not is_popular:
                continue
        
        filtered[model_id] = model_info
    
    return filtered

# ===================================================================
# MULTI-PROVIDER AI CLASS (Enhanced)
# ===================================================================

class MultiProviderAI:
    def __init__(self):
        self.fireworks_url = "https://api.fireworks.ai/inference/v1/chat/completions"
        self.novita_url = "https://api.novita.ai/v3/openai/chat/completions"
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Connection pooling for faster API responses
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3
        )
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)
    
    def get_current_api_keys(self):
        """Get current API keys from all sources"""
        return {
            'fireworks': get_api_key_multi_source("FIREWORKS_API_KEY"),
            'novita': get_api_key_multi_source("NOVITA_API_KEY"),
            'openrouter': get_api_key_multi_source("OPENROUTER_API_KEY")
        }
        
    def get_available_models(self, provider: str) -> Dict[str, Dict]:
        """Get available models for the specified provider"""
        if provider == "fireworks":
            return FIREWORKS_MODELS
        elif provider == "novita":
            return NOVITA_MODELS
        elif provider == "openrouter":
            return fetch_all_openrouter_models()
        else:
            return {}
    
    def chat_completion(self, messages: List[Dict[str, str]], provider: str, model: str, temperature: float = 0.7, max_tokens: int = 512) -> str:
        """Send chat completion request to the specified provider"""
        start_time = time.time()
        
        try:
            if provider == "fireworks":
                result = self._fireworks_chat(messages, model, temperature, max_tokens)
            elif provider == "novita":
                result = self._novita_chat(messages, model, temperature, max_tokens)
            elif provider == "openrouter":
                result = self._openrouter_chat(messages, model, temperature, max_tokens)
            else:
                result = f"⚠️ **Unknown provider: {provider}**\\n\\nSupported providers: fireworks, novita, openrouter"
            
            return result
            
        except Exception as e:
            return f"⚠️ **Error in chat completion:**\\n{str(e)}"
    
    def _fireworks_chat(self, messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int) -> str:
        """Handle Fireworks AI chat completion"""
        api_keys = self.get_current_api_keys()
        fireworks_key = api_keys['fireworks']
        
        if not fireworks_key:
            return "⚠️ **Fireworks API key required!** Please add your Fireworks API key in the sidebar."
        
        headers = {"Authorization": f"Bearer {fireworks_key}", "Content-Type": "application/json"}
        model_info = FIREWORKS_MODELS.get(model, {})
        fireworks_model = model_info.get("fireworks_model", model)
        
        payload = {
            "model": fireworks_model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens, "stream": False
        }
        
        try:
            response = self.session.post(self.fireworks_url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"⚠️ **Fireworks API Error {response.status_code}**\\n\\n{response.text[:200]}"
        except Exception as e:
            return f"⚠️ **Fireworks Connection Error**\\n\\n{str(e)}"
    
    def _novita_chat(self, messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int) -> str:
        """Handle Novita AI chat completion"""
        api_keys = self.get_current_api_keys()
        novita_key = api_keys['novita']
        
        if not novita_key:
            return "⚠️ **Novita API key required!** Please add your Novita API key in the sidebar."
        
        headers = {"Authorization": f"Bearer {novita_key}", "Content-Type": "application/json"}
        model_info = NOVITA_MODELS.get(model, {})
        novita_model = model_info.get("novita_model", model)
        
        payload = {
            "model": novita_model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens, "stream": False
        }
        
        try:
            response = self.session.post(self.novita_url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"⚠️ **Novita API Error {response.status_code}**\\n\\n{response.text[:200]}"
        except Exception as e:
            return f"⚠️ **Novita Connection Error**\\n\\n{str(e)}"
    
    def _openrouter_chat(self, messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int) -> str:
        """Handle OpenRouter AI chat completion"""
        api_keys = self.get_current_api_keys()
        openrouter_key = api_keys['openrouter']
        
        if not openrouter_key:
            return "⚠️ **OpenRouter API key required!** Please add your OpenRouter API key in the sidebar."
        
        # Debug: Check if key is being read correctly
        if len(openrouter_key) < 20:
            return f"⚠️ **API Key Error**\\n\\nAPI key appears truncated: {openrouter_key[:10]}..."
        
        headers = {
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://chunks-chatbot-v2.com",
            "X-Title": "CHUNKS Multi-Provider Chatbot v2.0"
        }
        
        openrouter_models = fetch_all_openrouter_models()
        model_info = openrouter_models.get(model, {})
        openrouter_model = model_info.get("openrouter_model", model)
        
        payload = {
            "model": openrouter_model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens, "stream": False
        }
        
        try:
            response = self.session.post(self.openrouter_url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            elif response.status_code == 401:
                # Get more details about the 401 error
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", {}).get("message", "Unknown error")
                    return f"⚠️ **OpenRouter API Error 401**\\n\\n{error_msg}\\n\\nAPI Key starts with: {openrouter_key[:15]}...\\n\\nPlease verify your key at https://openrouter.ai/keys"
                except:
                    return f"⚠️ **OpenRouter API Error 401**\\n\\nAPI key authentication failed.\\n\\nAPI Key starts with: {openrouter_key[:15]}...\\n\\nPlease check your OpenRouter API key at https://openrouter.ai/keys"
            elif response.status_code == 403:
                # Parse the error message to provide better guidance
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", {}).get("message", "")
                    if "OpenAI is requiring a key" in error_msg:
                        return "⚠️ **OpenAI Integration Required**\\n\\nThis model requires your OpenAI API key. Go to https://openrouter.ai/settings/integrations and add your OpenAI key, or try a different model like:\\n• meta-llama models\\n• mistralai models\\n• anthropic models"
                    elif "Google" in error_msg:
                        return "⚠️ **Google Integration Required**\\n\\nThis model requires your Google API key. Go to https://openrouter.ai/settings/integrations and add your Google key."
                    else:
                        return f"⚠️ **Access Denied**\\n\\nYou don't have access to this model. Try a different model or check your account limits."
                except:
                    return f"⚠️ **OpenRouter API Error 403**\\n\\nAccess denied. Try a different model or check your account status."
            else:
                return f"⚠️ **OpenRouter API Error {response.status_code}**\\n\\n{response.text[:200]}"
        except Exception as e:
            return f"⚠️ **OpenRouter Connection Error**\\n\\n{str(e)}"
    
    def chat_completion_stream(self, messages: List[Dict[str, str]], provider: str, model: str, temperature: float = 0.7, max_tokens: int = 512):
        """Stream chat completion responses with typewriter effect"""
        try:
            if provider == "openrouter":
                yield from self._openrouter_stream(messages, model, temperature, max_tokens)
            else:
                # For non-streaming providers, simulate streaming
                response = self.chat_completion(messages, provider, model, temperature, max_tokens)
                yield from self._simulate_stream(response)
        except Exception as e:
            yield f"⚠️ **Error in streaming chat:**\\n{str(e)}"
    
    def _openrouter_stream(self, messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int):
        """Stream OpenRouter responses"""
        api_key = get_api_key_multi_source("OPENROUTER_API_KEY")
        if not api_key:
            yield "⚠️ **OpenRouter API key required**"
            return
            
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://chunks-ai-chatbot.streamlit.app",
            "X-Title": "CHUNKS AI Chatbot v2.0"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        try:
            response = self.session.post(self.openrouter_url, headers=headers, json=payload, stream=True)
            
            if response.status_code == 200:
                import json
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]  # Remove 'data: ' prefix
                            if data_str.strip() == '[DONE]':
                                break
                            try:
                                data = json.loads(data_str)
                                if 'choices' in data and len(data['choices']) > 0:
                                    delta = data['choices'][0].get('delta', {})
                                    if 'content' in delta and delta['content']:
                                        yield delta['content']
                            except json.JSONDecodeError:
                                continue
            else:
                yield f"⚠️ **Streaming Error {response.status_code}**\\n\\n{response.text[:200]}"
                
        except Exception as e:
            yield f"⚠️ **Streaming Connection Error**\\n\\n{str(e)}"
    
    def _simulate_stream(self, response: str):
        """Simulate streaming for non-streaming providers"""
        import time
        words = response.split(' ')
        for i, word in enumerate(words):
            if i == 0:
                yield word
            else:
                yield ' ' + word
            time.sleep(0.05)  # Simulate typing delay

# ===================================================================
# VOICE AND SESSION MANAGEMENT
# ===================================================================

class DeepgramVoice:
    def __init__(self):
        self.stt_url = "https://api.deepgram.com/v1/listen"
        self.tts_url = "https://api.deepgram.com/v1/speak"
    
    def get_current_api_key(self):
        """Get current Deepgram API key"""
        return get_api_key_multi_source("DEEPGRAM_API_KEY")
        
    def speech_to_text(self, audio_file_path: str) -> str:
        """Convert speech to text using Deepgram STT"""
        api_key = self.get_current_api_key()
        if not api_key:
            return "[Speech-to-Text requires Deepgram API key - add it in the sidebar]"
            
        headers = {"Authorization": f"Token {api_key}", "Content-Type": "audio/wav"}
        params = {"model": "nova-2", "language": "en-US", "smart_format": "true", "punctuate": "true"}
        
        try:
            with open(audio_file_path, "rb") as audio:
                response = requests.post(self.stt_url, headers=headers, params=params, data=audio)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")
            else:
                return f"[STT Error: {response.status_code}]"
        except Exception as e:
            return f"[STT Error: {str(e)}]"
    
    def text_to_speech(self, text: str, voice_model: str = "aura-asteria-en") -> bytes:
        """Convert text to speech using Deepgram TTS"""
        api_key = self.get_current_api_key()
        if not api_key:
            return None
            
        headers = {"Authorization": f"Token {api_key}", "Content-Type": "application/json"}
        tts_url_with_model = f"https://api.deepgram.com/v1/speak?model={voice_model}&encoding=linear16&sample_rate=24000"
        
        # Handle long text by taking more characters but with reasonable limit
        # Deepgram has API limits, so we use 2000 chars (4x previous limit)
        max_chars = 2000
        processed_text = text[:max_chars]
        
        # If text was truncated, try to end at a sentence boundary
        if len(text) > max_chars:
            # Find last complete sentence within limit
            sentences = processed_text.split('. ')
            if len(sentences) > 1:
                processed_text = '. '.join(sentences[:-1]) + '.'
        
        payload = {"text": processed_text}
        
        try:
            response = requests.post(tts_url_with_model, headers=headers, json=payload, timeout=15)
            if response.status_code == 200:
                return response.content
            else:
                print(f"Deepgram TTS Error {response.status_code}: {response.text}")
                return None
        except Exception as e:
            print(f"Deepgram TTS Exception: {str(e)}")
            return None

class SessionManager:
    def __init__(self, sessions_file: str = "streamlit_chat_sessions_v2.json"):
        self.sessions_file = Path(sessions_file)
        self.load_sessions()
    
    def load_sessions(self):
        """Load sessions from file"""
        try:
            if self.sessions_file.exists():
                with open(self.sessions_file, 'r', encoding='utf-8') as f:
                    st.session_state.chat_sessions = json.load(f)
            else:
                st.session_state.chat_sessions = {}
        except Exception as e:
            print(f"Error loading sessions: {e}")
            st.session_state.chat_sessions = {}
    
    def save_sessions(self):
        """Save sessions to file"""
        try:
            with open(self.sessions_file, 'w', encoding='utf-8') as f:
                json.dump(st.session_state.chat_sessions, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving sessions: {e}")
    
    def create_session(self, name: str = None) -> str:
        """Create a new chat session"""
        session_id = str(uuid.uuid4())
        if not name:
            name = f"Chat {datetime.datetime.now().strftime('%m/%d %H:%M')}"
        
        st.session_state.chat_sessions[session_id] = {
            "name": name, "created_at": datetime.datetime.now().isoformat(),
            "messages": [], "system_prompt": SYSTEM_PROMPT_TEMPLATES["mirror_chatbot"]["prompt"],
            "model": "openai/gpt-5-chat"
        }
        self.save_sessions()
        return session_id
    
    def get_session_list(self) -> List[Tuple[str, str]]:
        """Returns list of (session_id, display_name) tuples"""
        return [(sid, data["name"]) for sid, data in st.session_state.chat_sessions.items()]
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to the session"""
        if session_id in st.session_state.chat_sessions:
            st.session_state.chat_sessions[session_id]["messages"].append({
                "role": role, "content": content, "timestamp": datetime.datetime.now().isoformat()
            })
            self.save_sessions()
    
    def update_system_prompt(self, session_id: str, prompt: str):
        """Update system prompt for session"""
        if session_id in st.session_state.chat_sessions:
            st.session_state.chat_sessions[session_id]["system_prompt"] = prompt
            self.save_sessions()
    
    def rename_session(self, session_id: str, new_name: str) -> bool:
        """Rename a session"""
        if session_id in st.session_state.chat_sessions and new_name.strip():
            st.session_state.chat_sessions[session_id]["name"] = new_name.strip()
            self.save_sessions()
            return True
        return False

class FastJSONContext:
    """Fast JSON-only context management - no vector DB overhead"""
    
    def __init__(self):
        self.initialized = True  # Always ready
        print("✅ Fast JSON Context initialized - no model loading needed")
    
    def initialize(self):
        """Always ready - no initialization needed"""
        return True
    
    def add_conversation_context(self, session_id: str, user_message: str, ai_response: str):
        """No-op - context is handled via JSON sessions only"""
        pass  # Do nothing - all context comes from session JSON
    
    def get_relevant_context(self, session_id: str, query: str, limit: int = 3):
        """Fast JSON-only context retrieval"""
        try:
            session = st.session_state.chat_sessions.get(session_id, {})
            messages = session.get("messages", [])
            
            # Get recent messages (last 20 messages for fast keyword search)
            recent_messages = messages[-20:]
            
            contexts = []
            query_lower = query.lower()
            
            # Simple keyword matching for recent messages (very fast)
            for i in range(0, len(recent_messages) - 1, 2):
                if i + 1 < len(recent_messages):
                    user_msg = recent_messages[i]
                    ai_msg = recent_messages[i + 1]
                    
                    if (user_msg.get("role") == "user" and 
                        ai_msg.get("role") == "assistant"):
                        
                        context_text = f"User: {user_msg['content']}\nAI: {ai_msg['content']}"
                        
                        # Simple relevance scoring based on keyword overlap
                        if (query_lower in user_msg['content'].lower() or 
                            query_lower in ai_msg['content'].lower()):
                            contexts.append({
                                "context": context_text,
                                "timestamp": user_msg.get("timestamp", ""),
                                "score": 0.8  # High score for recent matches
                            })
            
            return contexts[:limit]
            
        except Exception as e:
            print(f"Error getting JSON context: {e}")
            return []
    
    def get_user_patterns(self, session_id: str, limit: int = 10):
        """Fast analysis of user communication patterns from JSON"""
        try:
            session = st.session_state.chat_sessions.get(session_id, {})
            messages = session.get("messages", [])
            
            # Get recent user messages only
            user_messages = [msg["content"] for msg in messages[-20:] if msg.get("role") == "user"]
            
            if not user_messages:
                return {"message_count": 0, "communication_style": "Unknown"}
            
            return {
                "message_count": len(user_messages),
                "communication_style": self._analyze_style(user_messages)
            }
            
        except Exception as e:
            print(f"Error analyzing user patterns: {e}")
            return {"message_count": 0, "communication_style": "Unknown"}
    
    def _analyze_style(self, messages: List[str]) -> str:
        """Simple analysis of user communication style"""
        if not messages:
            return "Unknown"
            
        avg_length = sum(len(msg.split()) for msg in messages) / len(messages)
        question_ratio = sum(1 for msg in messages if '?' in msg) / len(messages)
        
        if avg_length > 20 and question_ratio > 0.3:
            return "Detailed and Inquisitive"
        elif avg_length > 20:
            return "Detailed and Explanatory"
        elif question_ratio > 0.3:
            return "Concise and Inquisitive"
        else:
            return "Concise and Direct"

# ===================================================================
# RESPONSE FORMATTING
# ===================================================================

def format_ai_response(response_text: str) -> str:
    """Enhanced formatting for AI responses with highlighting and code containers"""
    import re
    
    # Detect and format code blocks
    response_text = _format_code_blocks(response_text)
    
    # Highlight important information
    response_text = _highlight_important_text(response_text)
    
    # Format lists and structure
    response_text = _format_lists_and_structure(response_text)
    
    return response_text

def _format_code_blocks(text: str) -> str:
    """Detect and properly format code blocks"""
    import re
    
    # Patterns for different code indicators
    patterns = [
        # Already formatted code blocks
        (r'```(\w+)?\n(.*?)\n```', r'```\1\n\2\n```'),
        
        # Inline code
        (r'`([^`]+)`', r'`\1`'),
        
        # Multi-line code without proper formatting
        (r'\n((?:    |\t)[^\n]+(?:\n(?:    |\t)[^\n]+)*)', lambda m: f'\n```\n{m.group(1).strip()}\n```'),
        
        # Common code patterns
        (r'\b(def|class|import|from|if|for|while|function|const|let|var)\s+([^\n]+)', 
         r'`\1 \2`'),
    ]
    
    for pattern, replacement in patterns:
        if callable(replacement):
            text = re.sub(pattern, replacement, text, flags=re.MULTILINE | re.DOTALL)
        else:
            text = re.sub(pattern, replacement, text, flags=re.MULTILINE | re.DOTALL)
    
    return text

def _highlight_important_text(text: str) -> str:
    """Add highlighting for important information"""
    import re
    
    # First, protect problematic markdown patterns
    text = _protect_markdown_issues(text)
    
    # Patterns for highlighting
    patterns = [
        # Important indicators
        (r'\b(IMPORTANT|WARNING|NOTE|ERROR|SUCCESS)\b:?', r'**🔥 \1:**'),
        
        # Performance improvements
        (r'\b(\d+%)\s+(faster|improvement|better|reduction)', r'**✅ \1 \2**'),
        
        # File paths and commands
        (r'\b([a-zA-Z]:[\\/][^\s]+|\/[^\s]+\.[a-zA-Z0-9]+)', r'`\1`'),
        
        # Commands
        (r'\b(npm|pip|git|python|streamlit)\s+([^\n]+)', r'`\1 \2`'),
        
        # URLs
        (r'(https?://[^\s]+)', r'[\1](\1)'),
        
        # Bullet points enhancement (only at start of line)
        (r'^-\s+(.+)', r'• \1'),
        (r'^\*\s+(.+)', r'• \1'),
    ]
    
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
    
    return text

def _protect_markdown_issues(text: str) -> str:
    """Fix common markdown rendering issues"""
    import re
    
    # Fix isolated asterisks that break markdown
    # Pattern: *"text"* should become "text" (bold) or just text
    text = re.sub(r'\*"([^"]+)"\*', r'**"\1"**', text)
    
    # Fix single asterisks that aren't meant for italics
    text = re.sub(r'(?<!\*)\*(?!\*|\s)([^*]+?)(?<!\s)\*(?!\*)', r'**\1**', text)
    
    # Fix broken emphasis patterns
    text = re.sub(r'\*([^*\n]{1,3})\*', r'**\1**', text)
    
    return text

def _format_lists_and_structure(text: str) -> str:
    """Improve list formatting and structure"""
    import re
    
    # Convert numbered lists to better format
    text = re.sub(r'^\d+\.\s+', r'**\g<0>**', text, flags=re.MULTILINE)
    
    # Add spacing around headers
    text = re.sub(r'^(#{1,6})\s*(.+)$', r'\n\1 \2\n', text, flags=re.MULTILINE)
    
    return text

# Initialize components
deepgram_voice = DeepgramVoice()
session_manager = SessionManager()
multi_ai = MultiProviderAI()
vector_db = FastJSONContext()  # Fast JSON-only context (no vector DB loading)

# ===================================================================
# CHAT FUNCTIONS
# ===================================================================


def generate_voice_gtts(text: str) -> Optional[str]:
    """Generate voice using Google TTS"""
    if not text.strip():
        return None
    
    try:
        from gtts import gTTS
        tts = gTTS(text=text[:500], lang='en', slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tts.save(tmp_file.name)
            return tmp_file.name
    except ImportError:
        st.warning("📦 **Install gTTS for free voice:** `pip install gtts`")
        return None
    except Exception as e:
        st.error(f"Free voice generation error: {str(e)}")
        return None

# ===================================================================
# STREAMLIT PAGE CONFIGURATION
# ===================================================================

st.set_page_config(
    page_title="CHUNKS AI Chatbot v2.0 - All Models",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Enhanced for v2
st.markdown("""
<style>
    .main-header {
        text-align: center; padding: 1.5rem; background: #000000; color: #ffffff;
        border-radius: 10px; margin-bottom: 2rem; border: 2px solid #333333;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .chat-message {
        padding: 1rem; border-radius: 10px; margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd; margin-left: 20%;
    }
    .assistant-message {
        background-color: #f5f5f5; margin-right: 5%; width: 90%;
        border-left: 3px solid #4a4a4a;  /* Added prominent border */
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);  /* Enhanced shadow */
        font-weight: 500;  /* Slightly bolder text */
        padding: 1.2rem;  /* Increased padding */
    }
    .model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;
    }
    .provider-badge {
        display: inline-block; padding: 0.25rem 0.5rem; border-radius: 15px;
        font-size: 0.8rem; font-weight: bold; margin-right: 0.5rem;
    }
    .openai-badge { background: #74aa9c; }
    .anthropic-badge { background: #d4af37; }
    .google-badge { background: #4285f4; }
    .meta-badge { background: #1877f2; }
    .free-badge { background: #22c55e; }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🚀 CHUNKS AI Chatbot v2.0</h1>
    <p>🤖 Cross-Model AI in One Platform • 🔥 Multiple Providers • 🎤 Voice Enabled</p>
</div>
""", unsafe_allow_html=True)

# ===================================================================
# SIDEBAR - ENHANCED MODEL SELECTION
# ===================================================================

with st.sidebar:
    # Add JavaScript for local storage
    add_local_storage_js()
    
    # Logo and Header
    col1, col2 = st.columns([1, 3])
    with col1:
        # Display CHUNKS logo
        st.image("logo.png", width=100)
    with col2:
        st.markdown("### 🚀 CHUNKS AI CHATBOT v2.0")
    
    # ===================================================================
    # API KEY MANAGEMENT INTERFACE
    # ===================================================================
    
    def get_key_status_icon(key):
        """Get status icon for API key"""
        if key and len(key) > 10:
            return "✅"
        return "❌"
    
    def mask_api_key(key):
        """Mask API key for display"""
        if not key:
            return "Not set"
        if len(key) <= 8:
            return "*" * len(key)
        return key[:4] + "*" * (len(key) - 8) + key[-4:]
    
    with st.expander("🔑 API Key Management", expanded=not OPENROUTER_API_KEY):
        st.markdown("**Enter your API keys to access AI models:**")
        
        # Load existing keys from browser storage on first load
        st.markdown("""
        <script>
        // Load saved keys into Streamlit
        const keyNames = ['OPENROUTER_API_KEY', 'FIREWORKS_API_KEY', 'NOVITA_API_KEY', 'DEEPGRAM_API_KEY'];
        keyNames.forEach(keyName => {
            const savedKey = loadAPIKey(keyName);
            if (savedKey) {
                // This would need to be handled differently in Streamlit
                console.log('Loaded key for:', keyName);
            }
        });
        </script>
        """, unsafe_allow_html=True)
        
        # OpenRouter (Required)
        st.markdown("**🌐 OpenRouter (Required for 483+ models):**")
        openrouter_key = st.text_input(
            "OpenRouter API Key",
            value=mask_api_key(OPENROUTER_API_KEY) if OPENROUTER_API_KEY else "",
            type="password",
            key="openrouter_input",
            placeholder="sk-or-v1-your-key-here",
            help="Get your API key from https://openrouter.ai/keys"
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("💾 Save", key="save_openrouter", help="Save to browser"):
                if openrouter_key and not openrouter_key.startswith("*"):
                    st.session_state["user_input_OPENROUTER_API_KEY"] = openrouter_key
                    st.markdown("""
                    <script>
                    saveAPIKey('OPENROUTER_API_KEY', '%s');
                    </script>
                    """ % openrouter_key, unsafe_allow_html=True)
                    st.success("✅ OpenRouter key saved!")
                    st.rerun()
        
        with col2:
            if st.button("🧪 Test", key="test_openrouter"):
                test_key = st.session_state.get("user_input_OPENROUTER_API_KEY") or OPENROUTER_API_KEY
                if test_key:
                    with st.spinner("Testing..."):
                        headers = {"Authorization": f"Bearer {test_key}"}
                        try:
                            response = requests.get("https://openrouter.ai/api/v1/models", 
                                                  headers=headers, timeout=5)
                            if response.status_code == 200:
                                st.success("✅ Key works!")
                            else:
                                st.error("❌ Invalid key")
                        except:
                            st.error("❌ Connection error")
        
        with col3:
            st.markdown(f"{get_key_status_icon(OPENROUTER_API_KEY)} Status")
        
        # Optional providers
        st.markdown("---")
        st.markdown("**Optional Providers:**")
        
        # Fireworks AI
        st.markdown("**🔥 Fireworks AI:**")
        fireworks_key = st.text_input(
            "Fireworks API Key",
            value=mask_api_key(FIREWORKS_API_KEY) if FIREWORKS_API_KEY else "",
            type="password",
            key="fireworks_input",
            placeholder="fw_your-key-here"
        )
        
        if st.button("💾 Save Fireworks", key="save_fireworks"):
            if fireworks_key and not fireworks_key.startswith("*"):
                st.session_state["user_input_FIREWORKS_API_KEY"] = fireworks_key
                st.success("✅ Fireworks key saved!")
        
        # Novita AI  
        st.markdown("**⭐ Novita AI:**")
        novita_key = st.text_input(
            "Novita API Key",
            value=mask_api_key(NOVITA_API_KEY) if NOVITA_API_KEY else "",
            type="password",
            key="novita_input",
            placeholder="sk_your-key-here"
        )
        
        if st.button("💾 Save Novita", key="save_novita"):
            if novita_key and not novita_key.startswith("*"):
                st.session_state["user_input_NOVITA_API_KEY"] = novita_key
                st.success("✅ Novita key saved!")
        
        # Deepgram (Voice)
        st.markdown("**🎤 Deepgram (Voice):**")
        deepgram_key = st.text_input(
            "Deepgram API Key",
            value=mask_api_key(DEEPGRAM_API_KEY) if DEEPGRAM_API_KEY else "",
            type="password",
            key="deepgram_input",
            placeholder="your-deepgram-key"
        )
        
        if st.button("💾 Save Deepgram", key="save_deepgram"):
            if deepgram_key and not deepgram_key.startswith("*"):
                st.session_state["user_input_DEEPGRAM_API_KEY"] = deepgram_key
                st.success("✅ Deepgram key saved!")
        
        # Clear all keys
        st.markdown("---")
        if st.button("🗑️ Clear All Saved Keys", key="clear_all_keys"):
            # Clear session state keys
            keys_to_clear = ["user_input_OPENROUTER_API_KEY", "user_input_FIREWORKS_API_KEY", 
                           "user_input_NOVITA_API_KEY", "user_input_DEEPGRAM_API_KEY"]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Clear browser storage
            st.markdown("""
            <script>
            clearAllAPIKeys();
            </script>
            """, unsafe_allow_html=True)
            st.success("🗑️ All keys cleared!")
            st.rerun()
    
    # Update global API keys with user input
    OPENROUTER_API_KEY = get_api_key_multi_source("OPENROUTER_API_KEY")
    FIREWORKS_API_KEY = get_api_key_multi_source("FIREWORKS_API_KEY")
    NOVITA_API_KEY = get_api_key_multi_source("NOVITA_API_KEY")
    DEEPGRAM_API_KEY = get_api_key_multi_source("DEEPGRAM_API_KEY")
        
    # Initialize session state - auto-create new session on page load
    if "current_session_id" not in st.session_state or "page_loaded" not in st.session_state:
        # Always create a new session when the page loads
        st.session_state.current_session_id = session_manager.create_session()
        st.session_state.page_loaded = True
    
    # Session Management
    with st.expander("💬 Sessions", expanded=False):
        session_options = session_manager.get_session_list()
        if session_options:
            current_session_name = next((name for sid, name in session_options if sid == st.session_state.current_session_id), "Select Session")
            selected_session = st.selectbox(
                "Active Session", options=[sid for sid, name in session_options],
                format_func=lambda x: next((name for sid, name in session_options if sid == x), x),
                index=next((i for i, (sid, name) in enumerate(session_options) if sid == st.session_state.current_session_id), 0)
            )
            
            if selected_session != st.session_state.current_session_id:
                st.session_state.current_session_id = selected_session
                st.rerun()
        
        # Session Rename Feature
        if session_options:
            st.markdown("**✏️ Rename Session:**")
            current_session_name = st.session_state.chat_sessions[st.session_state.current_session_id]["name"]
            new_name = st.text_input("Session name", value=current_session_name, key="rename_session_input")
            
            col_rename1, col_rename2 = st.columns(2)
            with col_rename1:
                if st.button("💾 Save Name", key="save_session_name"):
                    if new_name.strip() and new_name != current_session_name:
                        if session_manager.rename_session(st.session_state.current_session_id, new_name):
                            st.success("✅ Session renamed!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to rename session")
            with col_rename2:
                if st.button("↩️ Reset", key="reset_session_name"):
                    st.rerun()
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ New Session"):
                new_session_id = session_manager.create_session()
                st.session_state.current_session_id = new_session_id
                st.rerun()
        with col2:
            if st.button("🗑️ Delete Session") and len(session_options) > 1:
                del st.session_state.chat_sessions[st.session_state.current_session_id]
                session_manager.save_sessions()
                st.session_state.current_session_id = list(st.session_state.chat_sessions.keys())[0]
                st.rerun()
    
    # System Prompt
    with st.expander("🎯 System Prompt", expanded=False):
        current_session = st.session_state.chat_sessions.get(st.session_state.current_session_id, {})
        current_prompt = current_session.get("system_prompt", SYSTEM_PROMPT_TEMPLATES["mirror_chatbot"]["prompt"])
        
        # Role Selection Dropdown
        st.markdown("**🎭 Choose AI Role:**")
        role_options = list(SYSTEM_PROMPT_TEMPLATES.keys())
        role_names = [SYSTEM_PROMPT_TEMPLATES[key]["name"] for key in role_options]
        
        # Find current selection based on prompt content
        current_selection = 0
        for i, key in enumerate(role_options):
            if SYSTEM_PROMPT_TEMPLATES[key]["prompt"] == current_prompt:
                current_selection = i
                break
        
        selected_role_key = st.selectbox(
            "Select Role Template:",
            options=role_options,
            format_func=lambda x: SYSTEM_PROMPT_TEMPLATES[x]["name"],
            index=current_selection,
            key="role_selector"
        )
        
        # Show selected template prompt
        template_prompt = SYSTEM_PROMPT_TEMPLATES[selected_role_key]["prompt"]
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🎭 Apply Selected Role", use_container_width=True):
                session_manager.update_system_prompt(st.session_state.current_session_id, template_prompt)
                st.success(f"✅ Applied {SYSTEM_PROMPT_TEMPLATES[selected_role_key]['name']}!")
                st.rerun()
        
        with col2:
            if st.button("🎲 Custom", use_container_width=True):
                st.session_state.show_custom_prompt = True
                st.rerun()
        
        # Custom prompt editor (toggle)
        if st.session_state.get("show_custom_prompt", False):
            st.markdown("**✏️ Custom System Prompt:**")
            custom_prompt = st.text_area(
                "System Instructions",
                value=current_prompt,
                height=120,
                key="custom_prompt_editor"
            )
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("💾 Save Custom Prompt"):
                    session_manager.update_system_prompt(st.session_state.current_session_id, custom_prompt)
                    st.session_state.show_custom_prompt = False
                    st.success("Custom prompt saved!")
                    st.rerun()
            with col2:
                if st.button("❌ Cancel"):
                    st.session_state.show_custom_prompt = False
                    st.rerun()
        else:
            # Show current prompt preview
            st.markdown("**📋 Current Prompt Preview:**")
            st.text_area(
                "Current System Instructions",
                value=current_prompt,
                height=80,
                disabled=True,
                key="prompt_preview"
            )
    
    # Enhanced Provider & Model Selection
    with st.expander("🤖 AI Provider & Model", expanded=True):
        
        # Provider Selection
        provider_choice = st.selectbox(
            "Select AI Provider",
            options=["openrouter", "fireworks", "novita"],
            format_func=lambda x: {
                "fireworks": "🔥 Fireworks AI (8 models)",
                "novita": "⭐ Novita AI (6 models)", 
                "openrouter": "🌐 OpenRouter (483+ models)"
            }.get(x, x)
        )
        
        # Model Filtering (Only for OpenRouter)
        if provider_choice == "openrouter":
            st.markdown("**🔍 Model Filtering:**")
            
            col1, col2 = st.columns(2)
            with col1:
                model_filter = st.selectbox(
                    "Filter by:",
                    options=["free", "popular", "openai", "anthropic", "google", "meta"],
                    index=2,  # Default to "openai" to show GPT-5 Chat
                    format_func=lambda x: {
                        "free": "🆓 Free Models", 
                        "popular": "⭐ Popular Models",
                        "openai": "🤖 OpenAI",
                        "anthropic": "🧠 Anthropic",
                        "google": "🔍 Google",
                        "meta": "🦙 Meta"
                    }.get(x, x)
                )
            
            with col2:
                search_term = st.text_input("🔍 Search models:", placeholder="e.g., gpt-4, claude")
            
            # Load and filter models
            with st.spinner("Loading models..."):
                all_openrouter_models = fetch_all_openrouter_models()
                filtered_models = filter_openrouter_models(all_openrouter_models, model_filter, search_term)
            
            if filtered_models:
                
                # Get current session's model for default selection
                current_session = st.session_state.chat_sessions.get(st.session_state.current_session_id, {})
                current_model = current_session.get("model", "openai/gpt-5-chat")
                
                # Find default index - prefer current session's model, fallback to gpt-5-chat
                default_index = 0
                model_options = list(filtered_models.keys())
                if current_model in model_options:
                    default_index = model_options.index(current_model)
                elif "openai/gpt-5-chat" in model_options:
                    default_index = model_options.index("openai/gpt-5-chat")
                
                # Model selection with enhanced display
                model_choice = st.selectbox(
                    "Select Model",
                    options=model_options,
                    index=default_index,
                    format_func=lambda x: f"{filtered_models[x]['name']} ({filtered_models[x]['parameters']})" if x in filtered_models else x
                )
                
                # Display model info
                if model_choice and model_choice in filtered_models:
                    model_info = filtered_models[model_choice]
                    provider_name = model_info.get('provider', '').title()
                    
                    st.markdown(f"""
                    <div class="model-card">
                        <h4>{model_info['name']}</h4>
                        <p><strong>Provider:</strong> {provider_name}</p>
                        <p><strong>Context:</strong> {model_info['parameters']}</p>
                        <p><strong>Price:</strong> {'FREE' if model_info.get('is_free') else f"${model_info.get('prompt_price', 0):.6f}/${model_info.get('completion_price', 0):.6f}"}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Inline Model Testing
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        # Create unique key based on provider and model
                        test_key = f"test_{provider_choice}_{model_choice.replace('/', '_').replace('-', '_').replace('.', '_')}"
                        if st.button("🧪 Test Model", key=test_key, help="Send 'Hi' to test if model works"):
                            with st.spinner("🔍 Testing model..."):
                                test_response = multi_ai.chat_completion(
                                    messages=[{"role": "user", "content": "Hi"}],
                                    provider=provider_choice, model=model_choice,
                                    temperature=0.7, max_tokens=3
                                )
                                if test_response.startswith("⚠️") or test_response.startswith("❌"):
                                    st.error(f"❌ **Test Failed:** {test_response[:100]}...")
                                else:
                                    st.success(f"✅ **Works!** Response: {test_response}")
                    with col2:
                        pass  # Reserved for status indicator
            else:
                st.warning("No models match your criteria. Try adjusting filters.")
                model_choice = None
        
        else:
            # Regular model selection for Fireworks/Novita
            available_models = multi_ai.get_available_models(provider_choice)
            if available_models:
                model_choice = st.selectbox(
                    "Select Model",
                    options=list(available_models.keys()),
                    format_func=lambda x: f"{available_models[x]['name']} ({available_models[x]['parameters']})"
                )
                
                if model_choice in available_models:
                    model_info = available_models[model_choice]
                    st.info(f"**{model_info['name']}**\\n\\n**Parameters:** {model_info['parameters']}\\n\\n**Description:** {model_info['description']}")
                    
                    # Inline Model Testing
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        # Create unique key based on provider and model  
                        test_key = f"test_{provider_choice}_{model_choice.replace('/', '_').replace('-', '_').replace('.', '_')}"
                        if st.button("🧪 Test Model", key=test_key, help="Send 'Hi' to test if model works"):
                            with st.spinner("🔍 Testing model..."):
                                test_response = multi_ai.chat_completion(
                                    messages=[{"role": "user", "content": "Hi"}],
                                    provider=provider_choice, model=model_choice,
                                    temperature=0.7, max_tokens=3
                                )
                                if test_response.startswith("⚠️") or test_response.startswith("❌"):
                                    st.error(f"❌ **Test Failed:** {test_response[:100]}...")
                                else:
                                    st.success(f"✅ **Works!** Response: {test_response}")
                    with col2:
                        pass  # Reserved for status indicator
            else:
                st.error("No models available for selected provider")
                model_choice = None
        
        # Advanced Parameters
        st.markdown("**⚙️ Parameters:**")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.5, 0.1)
        max_tokens = st.slider("Max Tokens", 1, 4096, 512, 1)
    
    
    # Voice Settings
    voice_enabled = False
    voice_method = "Browser Speech API (FREE, Best!)"
    voice_model = "aura-asteria-en"
    
    with st.expander("🎤 Voice Settings", expanded=False):
        voice_enabled = st.checkbox("Enable Voice Chat")
        if voice_enabled:
            voice_method = st.radio(
                "Voice Method",
                options=["Browser Speech API (FREE, Best!)", "Google TTS (FREE, Good Quality)", "Deepgram TTS (Requires API Key)"]
            )
            if "Deepgram" in voice_method:
                voice_model = st.selectbox(
                    "Deepgram Voice Model",
                    options=["aura-asteria-en", "aura-luna-en", "aura-stella-en", "aura-athena-en", "aura-hera-en", "aura-orion-en"]
                )
            
            if "Deepgram" in voice_method:
                st.info(f"🎤 **Active:** Deepgram TTS with {voice_model}")
            elif "Google" in voice_method:
                st.info("🎵 **Active:** Google TTS (gTTS library)")
            else:
                st.info("🔊 **Active:** Browser Speech Synthesis")
    
    # Context Management Settings  
    with st.expander("🧠 Context Settings", expanded=False):
        st.success("✅ **Fast JSON Context Active**")
        st.markdown("**Lightning-fast conversation context from JSON storage**")
        
        # Show current session context stats
        current_session = st.session_state.chat_sessions.get(st.session_state.current_session_id, {})
        if current_session.get("messages"):
            user_patterns = vector_db.get_user_patterns(st.session_state.current_session_id)
            if user_patterns.get('message_count', 0) > 0:
                st.markdown(f"""
                **Current Session Context:**
                - Messages analyzed: {user_patterns.get('message_count', 0)}
                - Communication style: {user_patterns.get('communication_style', 'Unknown')}
                """)
        
        st.info("🚀 **Performance Optimized:** No vector DB loading delays - 2-3 second responses!")
        
        # Show context source
        st.markdown("""
        **Context Source:** Recent conversation history (last 20 messages)  
        **Search Method:** Fast keyword matching  
        **Advantages:** Instant startup, no model downloads, 2-3s responses
        """)
        
        # Option to re-enable vector DB (advanced users)
        if st.checkbox("🔬 Advanced: Enable Vector DB (slower but more semantic search)"):
            st.warning("⚠️ This will require downloading AI models and will slow responses to 10+ seconds")
            st.info("Vector DB provides better semantic understanding but requires internet connection and model downloads")
    
    # Response Formatting Settings
    with st.expander("🎨 Response Formatting", expanded=False):
        enable_formatting = st.checkbox(
            "✨ Enhanced Response Formatting", 
            value=st.session_state.get("enable_formatting", True),
            help="Adds highlighting, code blocks, and better structure to AI responses"
        )
        st.session_state.enable_formatting = enable_formatting
        
        if enable_formatting:
            st.success("✅ **Enhanced formatting active**")
            st.markdown("""
            **Features:**
            • Code highlighting with syntax containers
            • **Bold** text for important information  
            • `Command` and file path styling
            • • Improved bullet points
            • Performance metrics highlighting
            """)
        else:
            st.info("📝 **Plain text mode active**")
            st.markdown("AI responses will show as plain text without formatting")
            
        if st.button("🔄 Refresh Chat Display"):
            st.rerun()

# ===================================================================
# MAIN CHAT INTERFACE
# ===================================================================

# Display current provider and model status
if provider_choice and model_choice:
    provider_names = {"fireworks": "🔥 Fireworks AI", "novita": "⭐ Novita AI", "openrouter": "🌐 OpenRouter"}
    provider_name = provider_names.get(provider_choice, "❌ Unknown Provider")
    
    # Check API status with dynamic keys
    current_api_keys = {
        "fireworks": get_api_key_multi_source("FIREWORKS_API_KEY"),
        "novita": get_api_key_multi_source("NOVITA_API_KEY"),
        "openrouter": get_api_key_multi_source("OPENROUTER_API_KEY")
    }
    api_status = "✅ Connected" if current_api_keys.get(provider_choice) else "⚠️ Add API Key in Sidebar"
    
    if provider_choice == "openrouter":
        available_models = fetch_all_openrouter_models()
        model_info = available_models.get(model_choice, {})
    else:
        model_info = multi_ai.get_available_models(provider_choice).get(model_choice, {})
    
    model_display_name = model_info.get("name", model_choice)
    
    st.info(f"🤖 **Model:** {model_display_name} | {provider_name}: {api_status}")
    
    # Show simple confirmation
    st.success(f"🎯 **Ready to Chat!**")
    
else:
    # Show welcome message if no API keys are configured
    if not get_api_key_multi_source("OPENROUTER_API_KEY"):
        st.warning("🔑 **Welcome to CHUNKS AI Chatbot v2.0!** Please add your API keys in the sidebar to get started.")
        st.info("💡 **Quick Start:** Add your OpenRouter API key in the sidebar to access 483+ AI models instantly!")
    else:
        st.warning("⚠️ Please select a provider and model to continue")

# Voice troubleshooting
if voice_enabled:
    with st.expander("🔧 Voice Help & Troubleshooting"):
        st.markdown("""
        ### 🔊 Browser Speech Issues?
        **If 🔊 buttons don't work:**
        1. ✅ Check browser volume - Make sure it's not muted
        2. ✅ Allow audio permissions - Chrome may ask for permission
        3. ✅ Try Chrome/Edge - Best support for speech synthesis  
        4. ✅ Refresh the page - Sometimes helps reset audio
        5. ✅ Close other audio apps - They may block speech
        """)

# Initialize messages in session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sync with session manager (backward compatibility)
current_session = st.session_state.chat_sessions.get(st.session_state.current_session_id, {})
session_messages = current_session.get("messages", [])

# If session_state.messages is empty but we have session messages, load them
if not st.session_state.messages and session_messages:
    st.session_state.messages = [
        {"role": msg["role"], "content": msg["content"]} 
        for msg in session_messages
    ]

# Display chat messages using native st.chat_message with enhanced formatting
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            # Option to disable formatting if it causes issues
            if st.session_state.get("enable_formatting", True):
                try:
                    formatted_content = format_ai_response(message["content"])
                    st.markdown(formatted_content)
                except:
                    # Fallback to plain text if formatting fails
                    st.write(message["content"])
            else:
                st.write(message["content"])
        else:
            # Keep user messages simple
            st.write(message["content"])
        
        # Add voice and copy buttons for AI responses
        if message["role"] == "assistant":
            col1, col2, col3 = st.columns([8, 1, 1])
            
            with col2:
                message_text = message["content"][:300]
                message_key = f"speak_{i}_{hash(message_text[:20])}"
                
                if st.button("🔊", key=message_key, help="Click to hear this message"):
                    if voice_enabled and voice_method:
                        if "Deepgram" in voice_method:
                            with st.spinner("🎤 Generating Deepgram voice..."):
                                audio_content = deepgram_voice.text_to_speech(message_text, voice_model)
                                if audio_content:
                                    st.audio(audio_content, format="audio/mp3", autoplay=True)
                                    st.success("🔊 ✅")
                                else:
                                    st.error("❌ Deepgram TTS failed - check API key")
                        elif "Google TTS" in voice_method:
                            voice_file = generate_voice_gtts(message_text)
                            if voice_file:
                                st.audio(voice_file, format="audio/mp3", autoplay=True)
                                st.success("🔊 ✅")
                            else:
                                st.error("❌ Google TTS failed - install gtts")
                        else:
                            clean_text = message_text.replace('"', '').replace("'", "").replace('\n', ' ').replace('*', '')
                            st.markdown(f"""
                            <script>
                                setTimeout(function() {{
                                    if ('speechSynthesis' in window) {{
                                        window.speechSynthesis.cancel();
                                        const utterance = new SpeechSynthesisUtterance("{clean_text[:200]}");
                                        utterance.rate = 1.0; utterance.volume = 1.0; utterance.lang = 'en-US';
                                        window.speechSynthesis.speak(utterance);
                                    }}
                                }}, 200);
                            </script>
                            """, unsafe_allow_html=True)
                            st.success("🔊 Playing with Browser Speech!")
                    else:
                        st.warning("⚠️ Enable voice chat in settings to use different voice models")
            
            with col3:
                copy_key = f"copy_{i}_{hash(message_text[:20])}"
                if st.button("📋", key=copy_key, help="Copy message text"):
                    copy_text = message["content"].replace('"', '\\"').replace('\n', '\\n')
                    st.markdown(f"""
                    <script>
                        navigator.clipboard.writeText("{copy_text}").then(function() {{
                            console.log('✅ Text copied to clipboard');
                        }});
                    </script>
                    """, unsafe_allow_html=True)
                    st.success("📋 Copied!")

# Clear Chat Button
col1, col2 = st.columns([4, 1])
with col2:
    if st.button("Clear Chat 🗑️", use_container_width=True):
        # Clear both session_state and session manager
        st.session_state.messages = []
        if st.session_state.current_session_id in st.session_state.chat_sessions:
            st.session_state.chat_sessions[st.session_state.current_session_id]["messages"] = []
            session_manager.save_sessions()
        st.rerun()

# Modern Chat Input with Real-time Display
if prompt := st.chat_input("Type your message here..."):
    if provider_choice and model_choice:
        # Add user message to session_state immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Also add to session manager for persistence
        session_manager.add_message(st.session_state.current_session_id, "user", prompt)
        
        # IMMEDIATELY show user message by rerunning
        st.rerun()
        
# Separate section: Handle AI response generation when we have a new user message
if (st.session_state.messages and 
    st.session_state.messages[-1]["role"] == "user" and 
    not st.session_state.get("processing_response", False)):
    
    if provider_choice and model_choice:
        # Mark as processing to avoid duplicate responses
        st.session_state.processing_response = True
        
        # Get the latest user message
        latest_user_message = st.session_state.messages[-1]["content"]
        
        # Get relevant context from vector database
        relevant_contexts = vector_db.get_relevant_context(st.session_state.current_session_id, latest_user_message, limit=3)
        user_patterns = vector_db.get_user_patterns(st.session_state.current_session_id)
        
        # Enhanced system prompt with user context
        session = st.session_state.chat_sessions.get(st.session_state.current_session_id, {})
        system_prompt = session.get("system_prompt", SYSTEM_PROMPT_TEMPLATES["mirror_chatbot"]["prompt"])
        
        enhanced_prompt = system_prompt
        if relevant_contexts:
            context_info = "\\n\\n**RELEVANT CONVERSATION CONTEXT:**\\n"
            for i, ctx in enumerate(relevant_contexts, 1):
                context_info += f"{i}. {ctx['context']} (similarity: {ctx['score']:.2f})\\n"
            enhanced_prompt += context_info
            
        if user_patterns and user_patterns.get('communication_style'):
            enhanced_prompt += f"\\n\\n**USER COMMUNICATION STYLE:** {user_patterns['communication_style']}"
            if user_patterns.get('message_count', 0) > 0:
                enhanced_prompt += f" (based on {user_patterns['message_count']} previous messages)"
        
        # Build messages from session_state (exclude the message we're responding to from context)
        messages = [{"role": "system", "content": enhanced_prompt}]
        for msg in st.session_state.messages[-8:-1]:  # Get recent context, exclude the latest user message
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add the current user message
        messages.append({"role": "user", "content": latest_user_message})
        
        # Show a status indicator while generating
        with st.container():
            with st.chat_message("assistant"):
                with st.spinner("🤖 Thinking..."):
                    try:
                        # Stream AI response 
                        full_response = st.write_stream(
                            multi_ai.chat_completion_stream(
                                messages=messages,
                                provider=provider_choice,
                                model=model_choice,
                                temperature=temperature,
                                max_tokens=max_tokens
                            )
                        )
                        
                        # Store original response for session management and vector DB
                        # But display will use formatting via the display loop above
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        
                        # Also add to session manager for persistence (original content)
                        session_manager.add_message(st.session_state.current_session_id, "assistant", full_response)
                        vector_db.add_conversation_context(st.session_state.current_session_id, latest_user_message, full_response)
                        
                        # Clear processing flag
                        st.session_state.processing_response = False
                        
                        # Rerun to show the complete conversation
                        st.rerun()
                        
                    except Exception as e:
                        error_msg = f"⚠️ **Error generating response:** {str(e)}"
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        st.session_state.processing_response = False
                        st.error(error_msg)
                        st.rerun()
    else:
        st.error("⚠️ Please select a provider and model first!")
        st.session_state.processing_response = False

# ===================================================================
# FOOTER
# ===================================================================
