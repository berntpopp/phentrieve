from fastapi import APIRouter

router = APIRouter(
    prefix="/health",
    tags=["health"],
)


@router.get("/")
async def health_check():
    """
    Health check endpoint for container orchestration
    """
    return {"status": "ok"}
