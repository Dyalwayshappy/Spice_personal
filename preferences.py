from typing import Optional
from spice.domain.world_state import WorldState

def get_user_vibe(state: Optional[WorldState] = None) -> str:
    
    
    if state is None:
        return "professional"

    user_context = state.entities.get("physical.user")
    
    if user_context:
        mode = getattr(user_context, "mode", "desk")
        
       
        if mode in ["heading_out", "in_transit", "heading_to_airport"]:
            return "urgent"
            
      
        if mode == "in_meeting":
            return "focused"

    return "professional"
