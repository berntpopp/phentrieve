from fastapi import APIRouter

# Create router without any tags or prefixes - these will be set in main.py
router = APIRouter()


@router.get("/")
async def health_check():
    """
    Health check endpoint for container orchestration
    """
    return {"status": "ok"}
