import os
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
import requests
from backend.config import AUTH0_DOMAIN, AUTH0_AUDIENCE
from functools import lru_cache

security = HTTPBearer()

# Only enable debug logging in development
_DEBUG = os.environ.get("AUTH_DEBUG", "").lower() in ("1", "true", "yes")


@lru_cache()
def get_jwks():
    """Get Auth0 public keys for JWT verification."""
    url = f"https://{AUTH0_DOMAIN}/.well-known/jwks.json"
    response = requests.get(url)
    return response.json()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify Auth0 JWT token and return user info."""
    token = credentials.credentials
    
    try:
        # Get signing key
        jwks = get_jwks()
        unverified_header = jwt.get_unverified_header(token)
        
        rsa_key = {}
        for key in jwks["keys"]:
            if key["kid"] == unverified_header["kid"]:
                rsa_key = {
                    "kty": key["kty"],
                    "kid": key["kid"],
                    "use": key["use"],
                    "n": key["n"],
                    "e": key["e"]
                }
        
        if not rsa_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unable to find appropriate key"
            )
        
        # Verify token — handle multiple audiences
        unverified_payload = jwt.get_unverified_claims(token)
        token_audiences = unverified_payload.get('aud', [])
        
        if isinstance(token_audiences, list):
            if AUTH0_AUDIENCE not in token_audiences:
                raise JWTError("Invalid audience")
        else:
            if token_audiences != AUTH0_AUDIENCE:
                raise JWTError("Invalid audience")
        
        # Verify signature and claims
        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=["RS256"],
            issuer=f"https://{AUTH0_DOMAIN}/",
            options={"verify_aud": False}  # We already verified audience above
        )
        
        return payload
        
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

def get_current_user(token_payload: dict = Depends(verify_token)):
    """Extract user info from verified token."""
    return {
        "user_id": token_payload.get("sub"),
        "email": token_payload.get("email"),
        "name": token_payload.get("name")
    }
