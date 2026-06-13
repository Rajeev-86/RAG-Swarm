from langchain_core.tools import tool

def create_tool(source_agent: str):
    @tool
    def send_peer_query(msg: str) -> str:
        """doc"""
        return msg
    
    send_peer_query.name = f"send_peer_query_from_{source_agent}"
    return send_peer_query

t = create_tool("financial")
print(t.name)
