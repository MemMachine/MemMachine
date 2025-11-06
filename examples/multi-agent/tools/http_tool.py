from strands import tool
import httpx

@tool
def http_request(method: str = "GET", url: str = "", headers: dict = None, json_data: dict = None) -> dict:
    """
    Make an HTTP request to any URL.
    
    Args:
        method: HTTP method (GET, POST, etc.)
        url: The URL to request
        headers: Optional headers dict
        json_data: Optional JSON body for POST/PUT
    
    Returns:
        Response as dict with status_code, text, and json (if applicable)
    """
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.request(
                method=method.upper(),
                url=url,
                headers=headers,
                json=json_data
            )
            
            result = {
                "status_code": response.status_code,
                "text": response.text,
                "headers": dict(response.headers)
            }
            
            try:
                result["json"] = response.json()
            except:
                pass
                
            return result
    except Exception as e:
        return {"error": str(e), "status_code": 0}

