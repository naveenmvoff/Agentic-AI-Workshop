from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

class EditCommand(BaseModel):
    """Model for editing commands"""
    command: str = Field(..., min_length=1, description="Natural language editing command")

class ParsedCommand(BaseModel):
    """Model for parsed command structure"""
    action: str = Field(..., description="Action type: create_layout, update_css, update_content, undo")
    target: str = Field(..., description="Target element selector")
    props: Dict[str, Any] = Field(default_factory=dict, description="Properties to apply")

class WebsiteState(BaseModel):
    """Model for website state"""
    layout: str = Field(..., description="HTML layout")
    props: Dict[str, Any] = Field(default_factory=dict, description="CSS properties")

class AgentResponse(BaseModel):
    """Model for agent responses"""
    status: str = Field(..., description="Response status: success, error")
    parsed_command: Optional[ParsedCommand] = None
    action_taken: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    current_state: Optional[WebsiteState] = None
    history_length: Optional[int] = None
    error: Optional[str] = None

class SessionInfo(BaseModel):
    """Model for session information"""
    current_state: WebsiteState
    history_length: int
    rag_enabled: bool

class ApiResponse(BaseModel):
    """Generic API response model"""
    status: str = Field(..., description="Response status")
    message: str = Field(default="", description="Response message")
    data: Dict[str, Any] = Field(default_factory=dict, description="Response data")

class HelpInfo(BaseModel):
    """Model for help information"""
    supported_actions: List[str]
    example_commands: List[str]
    tips: List[str]


# from pydantic import BaseModel

# class EditCommand(BaseModel):
#     command: str
