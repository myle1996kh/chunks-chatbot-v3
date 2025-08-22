"""
Simple Voice Clone Components
Simplified interface for voice clone creation and selection
"""

import streamlit as st
import tempfile
import os
import datetime
from typing import Optional, List, Dict
from voice_database import VoiceCloneDatabase

def render_simple_voice_selector(db: VoiceCloneDatabase, key: str = "simple_voice_selector") -> Optional[str]:
    """Render a simple voice clone selector - just a dropdown list"""
    
    # Get all voice clones
    voice_clones = db.get_voice_clones()
    
    if not voice_clones:
        st.info("No voice clones available. Create one below.")
        return None
    
    # Create simple options for selectbox
    voice_options = [("Select a voice...", None)]
    for voice in voice_clones:
        voice_options.append((voice['name'], voice['voice_id']))
    
    selected_voice = st.selectbox(
        "Select Voice Clone:",
        options=[voice_id for _, voice_id in voice_options],
        format_func=lambda x: next((name for name, vid in voice_options if vid == x), "Select a voice..."),
        key=f"{key}_select"
    )
    
    # Update usage count when voice is selected
    if selected_voice:
        db.update_voice_usage(selected_voice)
    
    return selected_voice

def render_simple_voice_creator(db: VoiceCloneDatabase, speechify_voice, key: str = "simple_voice_creator"):
    """Render a simple voice clone creation interface"""
    
    st.markdown("### 🎤 Create New Voice Clone")
    
    # Simple creation form
    voice_name = st.text_input(
        "Voice Name:",
        placeholder="Enter a name for this voice",
        key=f"{key}_name"
    )
    
    uploaded_audio = st.file_uploader(
        "Upload Audio Sample:",
        type=['wav', 'mp3', 'ogg'],
        help="Upload a 10-30 second audio sample",
        key=f"{key}_upload"
    )
    
    if uploaded_audio and voice_name.strip():
        if st.button("🎭 Create Voice Clone", key=f"{key}_create"):
            with st.spinner("🎤 Creating voice clone..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(uploaded_audio.getbuffer())
                    temp_audio_path = tmp_file.name
                
                try:
                    # Create voice clone using Speechify API
                    voice_id = speechify_voice.create_voice_clone(voice_name.strip(), temp_audio_path)
                    
                    if voice_id and not voice_id.startswith("["):
                        # Save to database
                        success = db.add_voice_clone(
                            voice_id=voice_id,
                            name=voice_name.strip(),
                            category="Personal",  # Default category
                            description=f"Created on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
                        )
                        
                        if success:
                            st.success(f"✅ Voice clone '{voice_name}' created successfully!")
                            st.info(f"🆔 Voice ID: `{voice_id}`")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error("❌ Failed to save voice clone to database")
                    else:
                        st.error(f"❌ Voice cloning failed: {voice_id}")
                
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(temp_audio_path)
                    except:
                        pass
    
    elif uploaded_audio and not voice_name.strip():
        st.warning("⚠️ Please enter a voice name")
    elif voice_name.strip() and not uploaded_audio:
        st.warning("⚠️ Please upload an audio file")

def render_simple_voice_manager(db: VoiceCloneDatabase):
    """Render a simple voice clone management interface"""
    
    st.markdown("### 🎭 Your Voice Clones")
    
    # Get all voice clones
    voice_clones = db.get_voice_clones()
    
    if not voice_clones:
        st.info("No voice clones created yet.")
        return
    
    # Show simple list with delete option
    for voice in voice_clones:
        col1, col2 = st.columns([4, 1])
        
        with col1:
            # Simple voice info
            usage_text = f" • {voice['usage_count']} uses" if voice['usage_count'] > 0 else ""
            created_date = datetime.datetime.fromisoformat(voice['created_date']).strftime('%Y-%m-%d')
            
            st.markdown(f"""
            **{voice['name']}**  
            Created: {created_date}{usage_text}  
            ID: `{voice['voice_id'][:20]}...`
            """)
        
        with col2:
            if st.button("🗑️", key=f"delete_{voice['voice_id']}", help=f"Delete {voice['name']}"):
                if db.delete_voice_clone(voice['voice_id']):
                    st.success(f"Voice clone '{voice['name']}' deleted!")
                    st.rerun()
                else:
                    st.error("Failed to delete voice clone")
        
        st.divider()

def render_simple_voice_section(db: VoiceCloneDatabase, speechify_voice):
    """Render the complete simple voice section"""
    
    # Voice creation
    render_simple_voice_creator(db, speechify_voice)
    
    st.markdown("---")
    
    # Voice management
    render_simple_voice_manager(db)