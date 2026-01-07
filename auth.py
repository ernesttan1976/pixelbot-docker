"""
Simple authentication module - single password for one user.
"""
import os
from functools import wraps
from flask import redirect, url_for, session, g, request
from werkzeug.security import check_password_hash, generate_password_hash


# Configuration from environment variables
# Set APP_PASSWORD in your .env file
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")

# If password hash is provided, use it; otherwise generate from plain password
PASSWORD_HASH = os.environ.get("PASSWORD_HASH", "")

# Generate password hash if we have a plain password but no hash
if APP_PASSWORD and not PASSWORD_HASH:
    PASSWORD_HASH = generate_password_hash(APP_PASSWORD)


def verify_password(password):
    """Verify the provided password."""
    if not PASSWORD_HASH:
        return False, "Password authentication is not configured"
    
    if not password:
        return False, "Password is required"
    
    if check_password_hash(PASSWORD_HASH, password):
        return True, "Password is correct"
    
    return False, "Invalid password"


def login_required(f):
    """Decorator to require authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if user is logged in
        if "authenticated" not in session or not session["authenticated"]:
            return redirect(url_for("login"))
        
        # Set user info in Flask's request context for compatibility
        g.user_id = "user"
        g.user_email = "user@example.com"
        
        # Create a mock user object for compatibility with existing code
        g.user = type(
            "User",
            (),
            {
                "id": "user",
                "email": "user@example.com",
                "first_name": "User",
                "last_name": "",
            },
        )()
        
        return f(*args, **kwargs)
    
    return decorated_function


def login_user():
    """Log in a user by setting session variables."""
    session["authenticated"] = True
    session.permanent = True  # Make session persistent


def logout_user():
    """Log out a user by clearing session."""
    session.clear()


def is_authenticated():
    """Check if current user is authenticated."""
    return session.get("authenticated", False)


def get_current_user():
    """Get current user info from session."""
    if not is_authenticated():
        return None
    
    return {
        "email": "user@example.com",
        "id": "user",
        "first_name": "User",
        "last_name": "",
    }

