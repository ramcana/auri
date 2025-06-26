import os
import json
import logging
import time
from typing import Dict, List, Any, Optional
import tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default system prompts for different conversation modes
DEFAULT_SYSTEM_PROMPTS = {
    "default": "You are a helpful, conversational assistant. Speak clearly and naturally. Respond like you're having a friendly conversation, not reading from a script.",
    "casual": "You're a friendly assistant chatting informally. Keep your responses brief and conversational.",
    "tech_support": "You are a calm technical assistant helping users with their devices. Provide clear step-by-step instructions.",
    "concise": "You are a helpful assistant that provides brief, to-the-point responses."
}

# Maximum number of tokens to keep in conversation history
MAX_HISTORY_TOKENS = int(os.getenv("MAX_HISTORY_TOKENS", "2000"))

# Maximum number of message pairs (user + assistant) to keep in history
MAX_HISTORY_PAIRS = int(os.getenv("MAX_HISTORY_PAIRS", "5"))

# Get encoding for token counting
def get_encoding():
    try:
        return tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
    except:
        try:
            return tiktoken.encoding_for_model("gpt-3.5-turbo")
        except:
            logger.warning("Tiktoken not available or model not found, using approximate token counting")
            return None

class ConversationManager:
    """Manages conversation history and context for the voice chatbot"""
    
    def __init__(self, session_id: str, mode: str = "default"):
        """
        Initialize a new conversation manager
        
        Args:
            session_id: Unique identifier for this conversation session
            mode: Conversation mode (default, casual, tech_support, etc.)
        """
        self.session_id = session_id
        self.mode = mode
        self.system_prompt = DEFAULT_SYSTEM_PROMPTS.get(mode, DEFAULT_SYSTEM_PROMPTS["default"])
        self.conversation_history = []
        self.last_activity_time = time.time()
        self.encoding = get_encoding()
        
        # Initialize with system prompt
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string"""
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Approximate token count (rough estimate: 4 chars per token)
            return len(text) // 4
    
    def get_total_tokens(self) -> int:
        """Get the total number of tokens in the conversation history"""
        if not self.conversation_history:
            return 0
            
        total = 0
        for message in self.conversation_history:
            total += self.count_tokens(message.get("content", ""))
        return total
    
    def trim_history(self):
        """Trim conversation history to stay within token limits"""
        # Always keep system prompt
        if len(self.conversation_history) <= 1:
            return
            
        # First check if we need to trim based on token count
        while self.get_total_tokens() > MAX_HISTORY_TOKENS and len(self.conversation_history) > 3:
            # Remove the oldest user-assistant pair (2 messages after system prompt)
            self.conversation_history.pop(1)  # Remove user message
            if len(self.conversation_history) > 1:
                self.conversation_history.pop(1)  # Remove assistant message
        
        # Then check if we need to trim based on message pair count
        # Count how many pairs we have (excluding system prompt)
        user_messages = sum(1 for msg in self.conversation_history if msg["role"] == "user")
        
        while user_messages > MAX_HISTORY_PAIRS:
            # Remove oldest user-assistant pair
            for i, msg in enumerate(self.conversation_history):
                if msg["role"] == "user":
                    self.conversation_history.pop(i)  # Remove user message
                    if i < len(self.conversation_history) and self.conversation_history[i]["role"] == "assistant":
                        self.conversation_history.pop(i)  # Remove assistant message
                    user_messages -= 1
                    break
    
    def add_user_message(self, message: str):
        """Add a user message to the conversation history"""
        self.conversation_history.append({"role": "user", "content": message})
        self.last_activity_time = time.time()
        self.trim_history()
    
    def add_assistant_message(self, message: str):
        """Add an assistant message to the conversation history"""
        self.conversation_history.append({"role": "assistant", "content": message})
        self.last_activity_time = time.time()
        self.trim_history()
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history"""
        return self.conversation_history
    
    def change_mode(self, new_mode: str):
        """Change the conversation mode and update system prompt"""
        if new_mode in DEFAULT_SYSTEM_PROMPTS:
            self.mode = new_mode
            self.system_prompt = DEFAULT_SYSTEM_PROMPTS[new_mode]
            
            # Update the system prompt in history
            if self.conversation_history and self.conversation_history[0]["role"] == "system":
                self.conversation_history[0]["content"] = self.system_prompt
            else:
                # Insert system prompt at the beginning
                self.conversation_history.insert(0, {"role": "system", "content": self.system_prompt})
    
    def clear_history(self, keep_system_prompt: bool = True):
        """Clear conversation history"""
        if keep_system_prompt and self.conversation_history and self.conversation_history[0]["role"] == "system":
            system_prompt = self.conversation_history[0]
            self.conversation_history = [system_prompt]
        else:
            self.conversation_history = [{"role": "system", "content": self.system_prompt}]
    
    def is_inactive(self, timeout_seconds: int = 300) -> bool:
        """Check if the conversation has been inactive for the specified timeout"""
        return (time.time() - self.last_activity_time) > timeout_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation state to a dictionary for serialization"""
        return {
            "session_id": self.session_id,
            "mode": self.mode,
            "system_prompt": self.system_prompt,
            "conversation_history": self.conversation_history,
            "last_activity_time": self.last_activity_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationManager':
        """Create a ConversationManager instance from a dictionary"""
        manager = cls(data["session_id"], data["mode"])
        manager.system_prompt = data["system_prompt"]
        manager.conversation_history = data["conversation_history"]
        manager.last_activity_time = data["last_activity_time"]
        return manager


class SessionManager:
    """Manages multiple conversation sessions"""
    
    def __init__(self):
        self.sessions: Dict[str, ConversationManager] = {}
        self.cleanup_interval = 3600  # 1 hour
        self.last_cleanup = time.time()
    
    def get_session(self, session_id: str, mode: str = "default") -> ConversationManager:
        """Get or create a conversation session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationManager(session_id, mode)
        return self.sessions[session_id]
    
    def cleanup_inactive_sessions(self, timeout_seconds: int = 3600):
        """Remove inactive sessions"""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
            
        inactive_sessions = []
        for session_id, manager in self.sessions.items():
            if manager.is_inactive(timeout_seconds):
                inactive_sessions.append(session_id)
        
        for session_id in inactive_sessions:
            logger.info(f"Removing inactive session: {session_id}")
            self.sessions.pop(session_id, None)
        
        self.last_cleanup = current_time
    
    def save_sessions(self, file_path: str):
        """Save all sessions to a file"""
        try:
            sessions_data = {
                session_id: manager.to_dict() 
                for session_id, manager in self.sessions.items()
            }
            
            with open(file_path, 'w') as f:
                json.dump(sessions_data, f)
            
            logger.info(f"Saved {len(self.sessions)} sessions to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save sessions: {e}")
    
    def load_sessions(self, file_path: str):
        """Load sessions from a file"""
        try:
            if not os.path.exists(file_path):
                logger.info(f"Sessions file not found: {file_path}")
                return
                
            with open(file_path, 'r') as f:
                sessions_data = json.load(f)
            
            for session_id, session_data in sessions_data.items():
                self.sessions[session_id] = ConversationManager.from_dict(session_data)
            
            logger.info(f"Loaded {len(self.sessions)} sessions from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load sessions: {e}")


# Global session manager instance
session_manager = SessionManager()
