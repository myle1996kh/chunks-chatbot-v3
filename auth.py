import streamlit as st
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class AuthManager:
    def __init__(self):
        self.session_timeout = st.secrets.get("auth", {}).get("session_timeout", 60)  # minutes
        
    def get_users(self) -> List[Dict]:
        """Get users from secrets.toml"""
        try:
            return st.secrets["auth"]["users"]
        except KeyError:
            st.error("No users configured in secrets.toml")
            return []
    
    def verify_credentials(self, username: str, password: str) -> Optional[Dict]:
        """Verify username and password against secrets.toml"""
        users = self.get_users()
        for user in users:
            if user["username"] == username and user["password"] == password:
                return user
        return None
    
    def login(self, username: str, password: str) -> bool:
        """Attempt to log in user"""
        user = self.verify_credentials(username, password)
        if user:
            st.session_state.authenticated = True
            st.session_state.username = user["username"]
            st.session_state.user_name = user["name"]
            st.session_state.login_time = datetime.now()
            return True
        return False
    
    def logout(self):
        """Log out current user"""
        for key in ["authenticated", "username", "user_name", "login_time"]:
            if key in st.session_state:
                del st.session_state[key]
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated and session is valid"""
        if not st.session_state.get("authenticated", False):
            return False
        
        # Check session timeout
        login_time = st.session_state.get("login_time")
        if login_time:
            time_diff = datetime.now() - login_time
            if time_diff > timedelta(minutes=self.session_timeout):
                self.logout()
                return False
        
        return True
    
    def render_login_form(self):
        """Render login form"""
        st.title("🔐 Login Required")
        st.markdown("Please log in to access the Chunks AI Chatbot")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login")
            
            if login_button:
                if username and password:
                    if self.login(username, password):
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.error("Please enter both username and password")
    
    def render_user_info(self):
        """Render user info and logout button in sidebar"""
        if self.is_authenticated():
            with st.sidebar:
                st.markdown("---")
                st.markdown(f"**Logged in as:** {st.session_state.user_name}")
                st.markdown(f"**Username:** {st.session_state.username}")
                
                # Session timeout still active in background, but no longer displayed
                
                if st.button("Logout", type="secondary"):
                    self.logout()
                    st.rerun()

def require_auth(func):
    """Decorator to require authentication for a function"""
    def wrapper(*args, **kwargs):
        auth_manager = AuthManager()
        if not auth_manager.is_authenticated():
            auth_manager.render_login_form()
            return None
        return func(*args, **kwargs)
    return wrapper