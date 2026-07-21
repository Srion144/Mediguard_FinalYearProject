"""
Week 1 (scaffold) + Week 3 (real /predict endpoint) for Track B.

Run this from inside the `backend/` folder:
    uvicorn app.main:app --reload --port 8000

Then check:
    http://127.0.0.1:8000/         -> health check
    http://127.0.0.1:8000/docs     -> interactive API docs (FastAPI gives you this for free)
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.predict import router as predict_router

app = FastAPI(title="MediGuard AI API", version="0.1.0")

# Lets the React dev server (running on a different port) call this API
# from the browser. Vite's default dev port is 5173.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"status": "ok", "service": "MediGuard AI backend"}


app.include_router(predict_router)
